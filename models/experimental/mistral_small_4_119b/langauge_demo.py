# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B text generation demo.

Runs prefill on a prompt, then decodes up to --max-new-tokens additional tokens,
printing each token as it is generated.

Run::

    export MESH_DEVICE=T3K          # or P150x4, single, etc.
    python models/experimental/mistral_small_4_119b/demo.py \
        --prompt "The capital of France is" \
        --max-new-tokens 32 \
        --n-layers 36
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HF_MODEL_ID,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mistral-Small-4-119B generation demo")
    p.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    p.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate after prefill")
    p.add_argument(
        "--n-layers",
        type=int,
        default=EXPECTED_NUM_LAYERS,
        help=f"Number of decoder layers to load (1..{EXPECTED_NUM_LAYERS})",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="KV cache capacity (defaults to prompt_len + max-new-tokens + 64)",
    )
    return p.parse_args()


# ── State-dict prefixes ────────────────────────────────────────────────────────


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


# ── Mesh device open/close ─────────────────────────────────────────────────────


def _open_mesh_device() -> ttnn.MeshDevice:
    rows, cols = mesh_device_request_param()
    if (rows, cols) != (1, 1):
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        trace_region_size=30_000_000,
        num_command_queues=1,
    )
    logger.info(f"Opened {rows}×{cols} mesh device ({device.get_num_devices()} chips)")
    return device


# ── Generation loop ────────────────────────────────────────────────────────────


def _precompute_rope_table(
    rotary_cls,
    text_config,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (cos, sin) for every position in [0, max_seq_len) once, on CPU.

    HF's Mistral4RotaryEmbedding uses ``x`` only to inherit dtype/device, so we
    pass a 1-element bf16 tensor instead of doing a full embedding lookup.
    """
    rotary = rotary_cls(text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(dummy, pos_ids)
    return cos, sin


def generate(
    model: TtMistral4TextModel,
    tokenizer,
    rotary_cls,
    text_config,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {prompt!r}  →  {seq_len} tokens")

    # Precompute (cos, sin) for every position we'll touch and upload once.
    # Per-step lookups are then on-device slices keyed on ``current_pos``.
    total_positions = seq_len + max_new_tokens
    logger.info(f"Precomputing RoPE table for {total_positions} positions…")
    cos_full, sin_full = _precompute_rope_table(rotary_cls, text_config, total_positions)
    model.cache_rope_tables(cos_full, sin_full)

    # Prefill — on-device argmax returns the token id directly.
    logger.info(f"Running prefill (seq_len={seq_len})…")
    t0 = time.perf_counter()
    next_id_int = model.prefill_next_token(input_ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Prefill done in {prefill_ms:.0f} ms")

    generated_ids = [next_id_int]
    print(f"\n{prompt}", end="", flush=True)
    print(tokenizer.decode([next_id_int], skip_special_tokens=False), end="", flush=True)

    # Decode loop — on-device argmax keeps full logits off the host.
    next_id = torch.tensor([[next_id_int]], dtype=torch.long)
    decode_times = []
    for step in range(1, max_new_tokens):
        current_pos = seq_len + step - 1

        t_dec = time.perf_counter()
        tok_id = model.decode_next_token(next_id, current_pos)
        decode_times.append((time.perf_counter() - t_dec) * 1000)

        generated_ids.append(tok_id)
        print(tokenizer.decode([tok_id], skip_special_tokens=False), end="", flush=True)

        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break

        next_id = torch.tensor([[tok_id]], dtype=torch.long)

    print()  # newline after generation

    if decode_times:
        avg_ms = sum(decode_times) / len(decode_times)
        logger.info(
            f"Generated {len(generated_ids)} tokens | " f"decode avg {avg_ms:.0f} ms/tok ({1000/avg_ms:.1f} tok/s)"
        )

    return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    try:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding
    except ImportError as e:
        sys.exit(f"transformers with Mistral4 support required: {e}")

    # Config
    logger.info(f"Loading HF config from {HF_MODEL_ID!r}…")
    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"Failed to load HF config: {e}")
    text = cfg.text_config

    # State dict
    logger.info(f"Loading state dict ({args.n_layers} layers)…")
    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(args.n_layers))
    except (FileNotFoundError, OSError) as e:
        sys.exit(f"Checkpoint load failed: {e}")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        sys.exit(f"Tokenizer load failed: {e}")

    # Mesh device
    mesh_device = _open_mesh_device()
    try:
        # Compute KV cache capacity
        prompt_tokens = tokenizer(args.prompt, return_tensors="pt").input_ids.shape[1]
        max_seq_len = args.max_seq_len or (prompt_tokens + args.max_new_tokens + 64)

        # Build model
        logger.info(f"Building TtMistral4TextModel ({args.n_layers} layers, max_seq_len={max_seq_len})…")
        model = TtMistral4TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            text_config=text,
            num_decoder_layers=args.n_layers,
            max_seq_len=max_seq_len,
        )

        # Free the host-side state dict before generation — weights are already
        # on device, so we no longer need the bf16 copies (saves ~1 GB CPU RAM).
        del state_dict

        # Run generation
        with torch.no_grad():
            generate(
                model=model,
                tokenizer=tokenizer,
                rotary_cls=Mistral4RotaryEmbedding,
                text_config=text,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
