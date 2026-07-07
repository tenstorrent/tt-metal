# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal Janus-Pro vision demo: full TT pipeline (vision tower + LLaMA decoder)
generating text from an image + prompt.

Modeled on ``models/demos/multimodal/gemma3/demo/vision_demo.py`` but trimmed to
the functional path (no perf-benchmark / CI-target machinery): build the model,
encode one image + prompt with the HF Janus processor, prefill, then greedily
decode a few tokens on host and print the result.

Requires real Janus-Pro weights (HF_MODEL=deepseek-community/Janus-Pro-7B);
dummy weights produce garbage text.
"""

import os
import time

import pytest
import torch
from loguru import logger
from PIL import Image as PIL_Image

import ttnn
from models.experimental.janus_pro.tt.janus_pro_e2e_model import JanusMultimodalGenerator, TtJanusProModel
from models.experimental.janus_pro.tt.model_config import ModelArgs
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
def test_janus_vision_demo(mesh_device, max_gen_len, reset_seeds):
    max_batch_size = 1
    max_seq_len = 4096

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

    # ---- one image + prompt ----
    img_path = os.environ.get("JANUS_DEMO_IMAGE", "models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg")
    image = PIL_Image.open(img_path).convert("RGB")
    prompt = os.environ.get("JANUS_DEMO_PROMPT", "Describe this image.")

    conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    prefill_len = input_ids.shape[1]

    logger.info(f"Prompt: {prompt!r} | prefill tokens: {prefill_len}")

    # Pad tokens to leave room for generation (Generator slot layout: [B, total_len]).
    total_len = prefill_len + max_gen_len
    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((max_batch_size, total_len), pad_id, dtype=torch.long)
    tokens[0, :prefill_len] = input_ids[0]
    prefill_lens = torch.tensor([prefill_len], dtype=torch.long)
    total_lens = prefill_lens + max_gen_len

    stop_tokens = set(getattr(model_args[0].tokenizer, "stop_tokens", []) or [])

    # ---- prefill ----
    t0 = time.perf_counter()
    prefill_out = generator.prefill_forward(
        [pixel_values],  # vision_images
        [None],  # vision_masks (unused)
        tokens,
        None,  # xattn_caches (unused)
        total_lens,
        prefill_lens,
    )
    next_token = _greedy(prefill_out)
    tokens[0, prefill_len] = next_token[0]
    logger.info(f"Prefill done in {(time.perf_counter() - t0) * 1000:.1f} ms; first token: {next_token.tolist()}")

    # ---- decode loop ----
    for gen_idx in range(max_gen_len - 1):
        position_id = prefill_lens + gen_idx
        logits, _ = generator.decode_forward(next_token.reshape(max_batch_size, 1), position_id, enable_trace=False)
        next_token = _greedy(logits)
        tokens[0, position_id[0] + 1] = next_token[0]
        if int(next_token[0]) in stop_tokens:
            break

    text = tokenizer.decode(tokens[0, prefill_len : prefill_len + max_gen_len].tolist())
    logger.info(f"Generated: {text}")
    print(f"\n=== Janus-Pro output ===\n{text}\n")
