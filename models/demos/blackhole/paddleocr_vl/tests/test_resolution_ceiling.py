# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for the repeat-loop failure mode at the resolution ceiling.

Commit ``dd4f214a4b2`` found that images above the supported resolution make the
decoder loop a phrase indefinitely and hit the token cap instead of stopping: at
1605632 pixels one printed page in five transcribed itself twice, and a dense page
at 3211264 looped a phrase 141 times, emitting 27140 characters for an
8228-character page. The vision tower was not at fault -- short text at that same
grid came back exact -- so the fault is the decoder's stop behaviour once the
image exceeds the resolution the model was trained at.

The deployment now serves at 1204224 (the top bucket, 6144 patches / 1280 image
tokens). This test locks that boundary in: nothing should silently widen the
bucket table and reintroduce the loop without this failing first.

Run::

    MESH_DEVICE=P150 TT_VISIBLE_DEVICES=3 \
    TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 \
    pytest models/demos/blackhole/paddleocr_vl/tests/test_resolution_ceiling.py -s
"""

from __future__ import annotations

import os
import re

import torch
from PIL import Image

import ttnn
from models.demos.blackhole.paddleocr_vl.tt.common import multimodal_rope_from_hf, splice_image_embeddings
from models.demos.blackhole.paddleocr_vl.tt.model import Transformer
from models.demos.blackhole.paddleocr_vl.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.paddleocr_vl.tt.vision.vision_model_config import VisionModelArgs
from models.demos.blackhole.paddleocr_vl.tt.weight_mapping import map_vision_state_dict
from models.demos.qwen3_vl.tt.generator import Generator as VLGenerator
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.model_config import ModelArgs

DEMO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
OCR_PROMPT = "OCR:"
SAMPLE_NAME = "page_table_large"  # bucket 6144, the largest currently-supported bucket
MAX_NEW_TOKENS = 256

# A phrase this long repeating 3+ times back-to-back is the "looped a phrase 141
# times" failure mode from dd4f214a4b2, not a legitimate table/list repetition.
_RUNAWAY_REPEAT = re.compile(r"(.{12,}?)\1{2,}", re.DOTALL)


def _to_torch_logits(x, mesh, vocab_size: int, row: int = -1) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    return x.float().reshape(-1, vocab_size)[row]


def test_largest_bucket_terminates_without_looping(device, hf_goldens):
    samples = {s["name"]: s for s in hf_goldens["samples"]}
    if SAMPLE_NAME not in samples:
        from models.demos.blackhole.paddleocr_vl.tests._artifacts import missing_artifact

        missing_artifact(f"{SAMPLE_NAME!r} not found in hf_goldens.json")
    sample = samples[SAMPLE_NAME]

    vision_args = VisionModelArgs(device, instruct=True, max_batch_size=1, max_seq_len=8192)
    text_args = ModelArgs(device, instruct=True, max_batch_size=1, max_seq_len=8192)

    full_sd = vision_args.load_state_dict()
    device_sd, host_sd = map_vision_state_dict(full_sd, vision_head_dim=vision_args.head_dim, strict=True)

    tower = DropInVisionTransformer(
        model_args=vision_args,
        device_state_dict=device_sd,
        host_state_dict=host_sd,
        dtype=ttnn.bfloat8_b,
    )
    text_model = Transformer(
        args=text_args,
        dtype=ttnn.bfloat8_b,
        mesh_device=device,
        state_dict=device_sd,
        weight_cache_path=text_args.weight_cache_path(ttnn.bfloat8_b),
    )
    generator = VLGenerator(text_model, text_args, device, tokenizer=text_args.tokenizer)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(text_args.CKPT_DIR)
    ref = AutoModelForImageTextToText.from_pretrained(text_args.CKPT_DIR, dtype=torch.bfloat16)
    ref.eval()
    embed_tokens = ref.model.language_model.embed_tokens
    image_token_id = ref.config.image_token_id
    pad_token_id = getattr(ref.config, "pad_token_id", 0) or 0
    tokenizer = text_args.tokenizer

    img = Image.open(os.path.join(DEMO, sample["image"])).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    grid = inputs["image_grid_thw"]

    image_embeds = tower(inputs["pixel_values"].to(torch.bfloat16), grid)
    with torch.no_grad():
        text_embeds = embed_tokens(input_ids.unsqueeze(0))[0]
    merged = splice_image_embeddings(input_ids, text_embeds, image_embeds, image_token_id)

    prompt_len = merged.shape[0]
    padded_len = get_padded_prefill_len(prompt_len)
    cos, sin, rope_deltas = multimodal_rope_from_hf(
        input_ids,
        grid,
        ref,
        text_args,
        pad_token_id=pad_token_id,
        min_positions=max(padded_len, prompt_len + MAX_NEW_TOKENS + 1),
    )
    if padded_len > prompt_len:
        merged = torch.cat([merged, torch.zeros(padded_len - prompt_len, merged.shape[-1], dtype=merged.dtype)], dim=0)
    embeds_tt = ttnn.from_torch(
        merged.unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    logits = generator.prefill_forward_single_user_text(
        ttnn.unsqueeze(embeds_tt, 0) if len(embeds_tt.shape) == 2 else embeds_tt,
        page_table=None,
        user_id=0,
        last_token_idx=prompt_len - 1,
        rot_mats=(cos, sin),
        kv_cache=None,
    )
    generator.update_rope_deltas([int(rope_deltas.reshape(-1)[0].item())])

    eos = set(tokenizer.all_special_ids or [])
    token = int(_to_torch_logits(logits, device, text_args.vocab_size).argmax().item())
    generated = [token]
    hit_eos = token in eos

    for step in range(MAX_NEW_TOKENS - 1):
        if token in eos:
            break
        out = generator.decode_forward(
            torch.tensor([[token]], dtype=torch.int32),
            start_pos=torch.tensor([prompt_len + step], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=False,
        )
        lg = out[0] if isinstance(out, tuple) else out
        token = int(_to_torch_logits(lg, device, text_args.vocab_size).argmax().item())
        generated.append(token)
        if token in eos:
            hit_eos = True

    generated_text = tokenizer.decode([t for t in generated if t not in eos], skip_special_tokens=True)
    print(f"generated {len(generated)} tokens, hit_eos={hit_eos}")
    print(f"text: {generated_text[:200]!r}")

    assert hit_eos, (
        f"generation ran to the {MAX_NEW_TOKENS}-token cap without stopping on its own "
        f"-- this is the dd4f214a4b2 repeat-loop symptom, not a legitimate long page"
    )
    repeat = _RUNAWAY_REPEAT.search(generated_text)
    assert repeat is None, f"runaway repetition detected: {repeat.group(1)[:60]!r} repeats 3+ times consecutively"
