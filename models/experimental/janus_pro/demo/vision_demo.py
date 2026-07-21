# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Janus-Pro multimodal perf demo: full TT pipeline (vision tower + LLaMA decoder)
generating text from an image + prompt, with warmup and BenchmarkProfiler perf
capture (TTFT + decode tok/s/user).

Modeled on ``models/demos/multimodal/gemma3/demo/vision_demo.py``. Scoped to
batch1 (notrace + trace); the CI save/verify path is intentionally absent —
manual-run only.

Requires real Janus-Pro weights (HF_MODEL=deepseek-community/Janus-Pro-7B);
dummy weights produce garbage text.
"""

import os

import pytest
import torch
from loguru import logger
from PIL import Image as PIL_Image

import ttnn
from models.experimental.janus_pro.tt.janus_pro_e2e_model import JanusMultimodalGenerator, TtJanusProModel
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import create_submeshes


def create_multimodal_model(mesh_device, max_batch_size, max_seq_len, dtype=ttnn.bfloat8_b):
    model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, cache_hf=True)
    state_dict = model_args.load_state_dict()
    model = TtJanusProModel(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        vision_dtype=ttnn.bfloat16,
    )
    return model_args, model


def _greedy(logits):
    # logits: [B, seq, vocab] or [B, vocab]; take the last position and argmax.
    if logits.dim() == 3:
        logits = logits[:, -1]
    return torch.argmax(logits, dim=-1).reshape(-1)


SAMPLE_DIR = "models/tt_transformers/demo/sample_prompts/llama_models"

# Default single-image scenarios mirror gemma3 vision_demo's non-multi-image set:
# a generative prompt and an OCR/reading prompt.
SINGLE_IMAGE_SCENARIOS = [
    ("haiku", ["dog.jpg"], "Write a haiku for this image."),
    ("ocr", ["ocr_image.jpeg"], "What is the full text of this image? Do OCR"),
]

# Multi-image-in-one-prompt: the Janus fusion path coalesces a per-image list of
# vision features (see JanusMultimodalGenerator.encode_vision_for_prefill and
# TtJanusProModel._coalesce_vision_embeddings), so images are fed as a list of
# single-image pixel tensors, one placeholder block per image.
MULTI_IMAGE_SCENARIO = ("multi", ["dog.jpg", "ocr_image.jpeg"], "Describe each of these images in one sentence.")


def _run_scenario(
    generator, processor, tokenizer, model_args, images, prompt, max_gen_len, enable_trace, max_batch_size
):
    content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]

    # The Janus HF processor does not expand its single <image_placeholder> into the
    # per-image token block the decoder needs (tokenize=True returns a malformed batch),
    # so build input_ids manually like test_e2e: one image_token_id run of
    # mm_tokens_per_image per image, in prompt order. masked_scatter then fuses the
    # vision features onto those placeholder positions.
    image_token = processor.image_token
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    num_image_tokens = model_args[0].mm_tokens_per_image
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    segments = text.split(image_token)
    assert len(segments) == len(images) + 1, f"expected {len(images)} image placeholders, got {len(segments) - 1}"
    ids = []
    for i, seg in enumerate(segments):
        ids += tokenizer(seg, add_special_tokens=(i == 0))["input_ids"]
        if i < len(images):
            ids += [image_token_id] * num_image_tokens
    input_ids = torch.tensor(ids, dtype=torch.long)

    # Processor stacks all images as [N, 3, H, W]; the generator expects a per-image
    # list of single-image tensors (each encoded separately, then coalesced in order).
    pixel_values = processor(text=text, images=images, return_tensors="pt")["pixel_values"]
    vision_images = [pixel_values[i : i + 1] for i in range(pixel_values.shape[0])]
    prefill_len = int(input_ids.shape[0])

    # Pad tokens to leave room for generation (Generator slot layout: [B, total_len]).
    total_len = prefill_len + max_gen_len
    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((max_batch_size, total_len), pad_id, dtype=torch.long)
    tokens[0, :prefill_len] = input_ids
    prefill_lens = torch.tensor([prefill_len], dtype=torch.long)
    total_lens = prefill_lens + max_gen_len
    stop_tokens = set(getattr(model_args[0].tokenizer, "stop_tokens", []) or [])

    profiler = BenchmarkProfiler()
    profiler.start("inference_prefill")
    prefill_out = generator.prefill_forward(
        vision_images,  # per-image list of [1, 3, H, W] tensors
        [None] * len(vision_images),  # vision_masks (unused)
        tokens,
        None,  # xattn_caches (unused)
        total_lens,
        prefill_lens,
    )
    profiler.end("inference_prefill")
    next_token = _greedy(prefill_out)
    tokens[0, prefill_len] = next_token[0]

    num_decoded = 0
    profiler.start("inference_decode")
    for gen_idx in range(max_gen_len - 1):
        position_id = prefill_lens + gen_idx
        logits, _ = generator.decode_forward(
            next_token.reshape(max_batch_size, 1), position_id, enable_trace=enable_trace
        )
        next_token = _greedy(logits)
        tokens[0, position_id[0] + 1] = next_token[0]
        num_decoded += 1
        if int(next_token[0]) in stop_tokens:
            break
    profiler.end("inference_decode")

    ttft = profiler.get_duration("inference_prefill")
    decode_time = profiler.get_duration("inference_decode")
    decode_t_s_u = num_decoded / decode_time if decode_time > 0 else 0.0
    text = tokenizer.decode(tokens[0, prefill_len : prefill_len + max_gen_len].tolist())
    return {
        "text": text,
        "prefill_len": prefill_len,
        "ttft": ttft,
        "decode_t_s_u": decode_t_s_u,
        "num_decoded": num_decoded,
    }


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("max_gen_len", [50])
@pytest.mark.parametrize("multi_image", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("enable_trace", [False, True], ids=["notrace", "trace"])
@pytest.mark.timeout(1200)
def test_multimodal_demo_text(mesh_device, enable_trace, multi_image, max_gen_len, reset_seeds):
    max_batch_size = 1
    # 7B weights nearly fill a single device's DRAM; the full 4096 context OOMs there.
    # 2048 covers single- and multi-image prefill (~600 / ~1200 tokens) + generation.
    max_seq_len = 2048

    # Build one model per submesh (data_parallel=1 here); Generator expects lists.
    submeshes = create_submeshes(mesh_device, 1)
    model_args_i, model_i = create_multimodal_model(submeshes[0], max_batch_size, max_seq_len)
    model_args, model = [model_args_i], [model_i]

    generator = JanusMultimodalGenerator(model, model_args, mesh_device)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_args[0].CKPT_DIR, local_files_only=os.getenv("CI") == "true", use_fast=True
    )
    tokenizer = processor.tokenizer

    # A single custom scenario can be forced via env; otherwise run the parametrized set.
    if os.environ.get("JANUS_DEMO_IMAGE") or os.environ.get("JANUS_DEMO_PROMPT"):
        scenarios = [
            (
                "custom",
                [os.environ.get("JANUS_DEMO_IMAGE", os.path.join(SAMPLE_DIR, "dog.jpg"))],
                os.environ.get("JANUS_DEMO_PROMPT", "Describe this image."),
            )
        ]
    elif multi_image:
        name, fns, prompt = MULTI_IMAGE_SCENARIO
        scenarios = [(name, [os.path.join(SAMPLE_DIR, fn) for fn in fns], prompt)]
    else:
        scenarios = [
            (name, [os.path.join(SAMPLE_DIR, fn) for fn in fns], prompt) for name, fns, prompt in SINGLE_IMAGE_SCENARIOS
        ]

    # host greedy decode → no on-device sampling; trace only affects the decode step.
    # Only prefill is warmed: warmup_model_decode drives the paged-attention path
    # (page_table sized by num_blocks), but this demo decodes non-paged, and num_blocks=0
    # there divides by zero (SIGFPE). The decode loop below runs the non-paged path directly.
    generator.warmup_model_prefill(
        kv_cache=None, enable_trace=enable_trace, can_sample_on_device=False, greedy_only=True
    )
    logger.info("Warmup complete")

    for name, img_paths, prompt in scenarios:
        images = [PIL_Image.open(p).convert("RGB") for p in img_paths]
        result = _run_scenario(
            generator, processor, tokenizer, model_args, images, prompt, max_gen_len, enable_trace, max_batch_size
        )
        logger.info(
            f"[{name}] prompt={prompt!r} prefill_tokens={result['prefill_len']} "
            f"TTFT={result['ttft'] * 1000:.1f}ms decode={result['decode_t_s_u']:.2f} tok/s/user"
        )
        print(
            f"\n=== Janus-Pro [{name}] (trace={enable_trace}) ===\n"
            f"prompt: {prompt}\n{result['text']}\n"
            f"TTFT={result['ttft'] * 1000:.1f} ms  decode={result['decode_t_s_u']:.2f} tok/s/user\n"
        )
