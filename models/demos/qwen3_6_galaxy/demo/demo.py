#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive (greedy) text-generation demo for Qwen3.6-27B on BH GLX 8x4.

Run:
    cd /home/tt-admin/ssinghal/tt-metal
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python models/demos/qwen3_6_galaxy/demo/demo.py \\
        --prompt "The capital of France is" --num-tokens 10

Mirrors the loop in tests/test_decode_loop.py::test_greedy_decode_5_tokens_after_prefill
but as a standalone CLI script with configurable prompt and token count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from safetensors.torch import load_file as load_st

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config
from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

_DEFAULT_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _load_global_weights(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


def _load_layer_weights(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer_idx}"
    keys = [k for k in weight_map if k.startswith(pfx + ".")]
    files = sorted({weight_map[k] for k in keys})
    raw = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in keys:
            if k in shard:
                raw[k] = shard[k].float()
    return {k[len(pfx) + 1 :]: v for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--num-tokens", type=int, default=10, help="Number of tokens to greedy-generate")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=64,
        help="Number of decoder layers (full 64-layer model by default)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=pathlib.Path,
        default=_DEFAULT_SNAPSHOT,
        help="HF snapshot directory (defaults to local cached Qwen3.6-27B)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[demo] loading tokenizer from {args.snapshot_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot_dir), trust_remote_code=True)

    print(f"[demo] tokenizing prompt: {args.prompt!r}")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    print(f"[demo] prompt is {T_prompt} tokens: {input_ids[0].tolist()}")

    # Tile-align prefill length
    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    print(f"[demo] loading weights for {args.num_layers} decoder layers ...")
    global_weights = _load_global_weights(args.snapshot_dir)
    layers_weights = [_load_layer_weights(args.snapshot_dir, i) for i in range(args.num_layers)]

    print("[demo] opening BH GLX 8x4 mesh ...")
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    try:
        with open(args.snapshot_dir / "config.json") as f:
            config = Qwen36Config(json.load(f))
        model_args = TtQwen36ModelArgs(mesh)

        print(f"[demo] building TtQwen36Transformer ({args.num_layers} layers) ...")
        t0 = time.time()
        model = TtQwen36Transformer(
            mesh_device=mesh,
            args=model_args,
            global_weights=global_weights,
            layers_weights=layers_weights,
            num_layers=args.num_layers,
        )
        print(f"[demo] model built in {time.time()-t0:.1f}s")

        print("[demo] prefill ...")
        t0 = time.time()
        tt_logits, kv_caches, dn_states, conv_states = model.forward_prefill(input_ids_padded, return_caches=True)
        print(f"[demo] prefill done in {time.time()-t0:.2f}s")

        # First generated token comes from the prefill logits at the last real position.
        last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]
        generated = [int(last_logits.argmax().item())]
        first_text = tokenizer.decode(generated)
        print(f"[demo] token 1 (from prefill): id={generated[-1]} text={first_text!r}")

        # Greedy decode loop.
        current_pos = T_padded
        for step in range(1, args.num_tokens):
            next_id = torch.tensor([[generated[-1]]])
            t0 = time.time()
            step_logits, kv_caches, dn_states, conv_states = model.forward_decode(
                next_id,
                current_pos=current_pos,
                kv_caches=kv_caches,
                dn_states=dn_states,
                conv_states=conv_states,
            )
            dt = time.time() - t0
            step_last = step_logits[0, 0, : config.vocab_size]
            if torch.isnan(step_last).any() or torch.isinf(step_last).any():
                print(f"[demo] NaN/Inf at step {step+1}; stopping")
                break
            tok_id = int(step_last.argmax().item())
            generated.append(tok_id)
            print(
                f"[demo] token {step+1} (decode step {step}): id={tok_id} text={tokenizer.decode([tok_id])!r} ({dt:.2f}s)"
            )
            current_pos += 1

        full_text = tokenizer.decode(generated)
        print()
        print(f"[demo] generated {len(generated)} tokens: {full_text!r}")
        print(f"[demo] full sequence: {(args.prompt + full_text)!r}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
