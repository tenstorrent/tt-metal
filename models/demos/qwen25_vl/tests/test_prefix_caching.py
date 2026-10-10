# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Resumed (cached-prefix) prefill for Qwen2.5-VL, the model-side half of vLLM automatic prefix caching.

A full prefill of a prompt fills the paged KV cache; a second prefill of the same prompt with
``start_pos=[P]`` must then run only the suffix and produce the same next-token logits. The same must
hold for a prompt that shares the first P tokens but ends differently, and for a prompt whose image
tokens lie entirely inside the cached prefix (the vLLM wrapper then gets no pixels and feeds zero
placeholder rows for those tokens, which the resumed prefill never reads).

Needs the real checkpoint: HF_MODEL (and TT_CACHE_PATH for the weight cache), as the CI unit job sets.
"""
import os

import pytest
import torch
from loguru import logger
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

import ttnn
from models.demos.qwen25_vl.demo.demo import prepare_generator_args
from models.demos.qwen25_vl.tt.common import (
    get_block_size,
    merge_vision_tokens,
    multimodal_rope_from_hf,
    preprocess_inputs_prefill,
)
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model_config import qwen25_vl_mesh_shape
from models.tt_transformers.tt.model_config import DecodersPrecision

CONTEXT = " ".join(
    f"Clause {i}: the committee resolved that item {i} shall be reviewed every {i % 7 + 1} weeks and reported."
    for i in range(1, 40)
)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _assert_same_next_token(name, full, resumed, min_pcc=0.99):
    pcc = _pcc(full, resumed)
    logger.info(f"{name}: argmax full={full.argmax().item()} resumed={resumed.argmax().item()} pcc={pcc:.5f}")
    assert full.argmax().item() == resumed.argmax().item(), f"{name}: resumed prefill changed the next token"
    assert pcc > min_pcc, f"{name}: logits PCC {pcc:.4f} < {min_pcc}"


class _Harness:
    def __init__(self, mesh_device):
        max_seq_len = 4096
        model_args_list, model_list, _, self.kv_cache_list, self.page_table = prepare_generator_args(
            data_parallel=1,
            mesh_device=mesh_device,
            instruct=True,
            batch_size=1,
            optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
            max_seq_len=max_seq_len,
            page_params={"page_block_size": 32, "page_max_num_blocks": 1024},
            dtype=ttnn.bfloat8_b,
            use_paged_kv_cache=True,
        )
        self.model_args = model_args_list[0]
        self.generator = Generator(
            model_list,
            model_args_list,
            mesh_device,
            processor=self.model_args.processor,
            tokenizer=self.model_args.tokenizer,
        )
        ref_model_name = self.model_args.CKPT_DIR
        config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(ref_model_name)
        self.reference_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ref_model_name, config=config, torch_dtype="auto", device_map="cpu"
        )
        self.visual = getattr(self.reference_model, "visual", None) or self.reference_model.model.visual
        self.processor = AutoProcessor.from_pretrained(ref_model_name)
        self.pad_token_id = self.model_args.tokenizer.pad_token_id
        self.block_size = get_block_size(self.kv_cache_list[0])

    def encode(self, question, image=None):
        content = ([{"type": "image", "image": image}] if image is not None else []) + [
            {"type": "text", "text": CONTEXT + " " + question}
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image] if image is not None else None, return_tensors="pt")
        if "pixel_values" in inputs:
            image_embeds = self.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
            image_embeds = getattr(image_embeds, "pooler_output", image_embeds)
        else:
            image_embeds = torch.tensor([], dtype=torch.bfloat16)
        text_embeds = self.reference_model.model.language_model.embed_tokens(inputs.input_ids)
        input_embeds = merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, self.reference_model.config)
        input_prefill_pt, decoding_pos, _ = preprocess_inputs_prefill(
            input_embeds,
            self.model_args,
            inputs.attention_mask,
            pad_embedding=self.reference_model.model.language_model.embed_tokens(torch.tensor(self.pad_token_id)),
        )
        cos, sin, _ = multimodal_rope_from_hf(
            inputs, input_embeds, self.reference_model, self.model_args, pad_token_id=self.pad_token_id
        )
        return inputs.input_ids[0], input_prefill_pt, (cos, sin), decoding_pos

    def prefill(self, input_prefill_pt, rot_mats, decoding_pos, start_pos=None):
        return self.generator.prefill_forward_text(
            input_prefill_pt,
            rot_mats=rot_mats,
            page_table=self.page_table,
            kv_cache=self.kv_cache_list,
            prompt_lens=decoding_pos,
            enable_trace=False,
            start_pos=start_pos,
        )


def _shared_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return i
    return n


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [qwen25_vl_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_resumed_prefill_matches_full_prefill(mesh_device, qwen25_vl_mesh_device, device_params):
    if not os.environ.get("HF_MODEL"):
        pytest.skip("needs the real checkpoint (HF_MODEL)")
    h = _Harness(qwen25_vl_mesh_device)
    bs = h.block_size

    # 1) Same prompt: prefill fully, then again from a cached offset inside the context.
    ids_a, pt_a, rot_a, pos_a = h.encode("How often is item 12 reviewed?")
    full_a = h.prefill(pt_a, rot_a, pos_a)
    cached = (pos_a[0] // 2 // bs) * bs
    assert 0 < cached < pos_a[0]
    resumed_a = h.prefill(pt_a, rot_a, pos_a, start_pos=[cached])
    _assert_same_next_token(f"same prompt, {cached}/{pos_a[0]} cached", full_a, resumed_a)

    # 2) Different question after the same context: the shared blocks come from prompt A's prefill.
    ids_b, pt_b, rot_b, pos_b = h.encode("Which clause mentions a 4-week cadence first?")
    full_b = h.prefill(pt_b, rot_b, pos_b)  # reference, also overwrites the cache with B
    h.prefill(pt_a, rot_a, pos_a)  # restore A's KV; its first `shared` blocks equal B's
    shared = (_shared_prefix_len(ids_a.tolist(), ids_b.tolist()) // bs) * bs
    assert shared >= bs, "prompts must share at least one KV block"
    resumed_b = h.prefill(pt_b, rot_b, pos_b, start_pos=[shared])
    _assert_same_next_token(f"shared context, {shared}/{pos_b[0]} cached", full_b, resumed_b)

    # 3) Image inside the cached prefix: the cached image rows are never read, so zero placeholders
    #    (what the vLLM wrapper substitutes when the pixels are stripped) must give the same logits.
    image = Image.new("RGB", (448, 448), (250, 250, 240))
    draw = ImageDraw.Draw(image)
    for i in range(0, 448, 28):
        draw.text((8, i), f"line {i // 28}: invoice item {i // 28} amount {i * 3}", fill=(10, 10, 10))
    ids_i, pt_i, rot_i, pos_i = h.encode("What text is on line 3?", image=image)
    image_token_id = h.reference_model.config.image_token_id
    image_positions = (ids_i == image_token_id).nonzero().flatten()
    assert len(image_positions) > 0
    # Resume from the middle of the text context, which follows the image. The generator floors the
    # offset to the chunked-SDPA alignment; zero only the image rows below the floored offset, exactly
    # the rows a resumed prefill never reads.
    cached_i = (pos_i[0] // 2 // bs) * bs
    (aligned_i,) = h.generator._align_cached_prefix_offsets([cached_i], [pos_i[0]], h.kv_cache_list[0])
    assert (
        aligned_i > image_positions[-1].item()
    ), f"image (ends at {image_positions[-1].item()}) must lie inside the cached prefix ({aligned_i})"
    full_i = h.prefill(pt_i, rot_i, pos_i)
    pt_i_placeholder = pt_i.clone()
    pt_i_placeholder[0, :aligned_i][(ids_i[:aligned_i] == image_token_id)] = 0
    resumed_i = h.prefill(pt_i_placeholder, rot_i, pos_i, start_pos=[cached_i])
    _assert_same_next_token(f"cached image, {cached_i}/{pos_i[0]} cached", full_i, resumed_i)
