#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a ttml GRPOTrainer checkpoint to HuggingFace safetensors layout.

The ``save_checkpoint`` path in ``ttml.trainers.grpo_trainer`` writes ``checkpoint.pkl``, with ttml-side
keys (``Qwen3/blocks/{i}/self_attn/q_proj/weight`` ...) and a fused
``kv_proj`` in the meta-permuted Q/K row layout ``[r0, i0, r1, i1, ...]`` (per
head). ``AutoModelForCausalLM.from_pretrained`` expects the standard HF Qwen3
layout: ``model.layers.{i}.self_attn.q_proj.weight``, separate ``k_proj`` and
``v_proj``, and the "half-then-half" row layout ``[r0, ..., r_{d/2-1}, i0, ...,
i_{d/2-1}]`` that HF's rotary embedding assumes. This script does the one-time
key rename, ``kv_proj`` split, and row re-permutation of Q/K projections and
Q/K-Norm gammas so eval can consume GRPO checkpoints.

Usage:
    python convert_ttml_ckpt_to_hf.py <src_ckpt_dir> <dst_dir> [--dtype bf16|fp32]

Example (each GRPO checkpoint of a run):
    for step in 20 40 60 80 100; do
        python convert_ttml_ckpt_to_hf.py \\
            generated/tt-train/grpo_gsm8k_run/<utc>/checkpoints/grpo_step_${step} \\
            generated/tt-train/converted/grpo-ckpt-${step}
    done
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from safetensors.torch import save_file
from ttml.models.qwen3.weights import build_weight_mapping_single, repermute_norm_weights, repermute_proj_rows


# Standard tt-transformers assumes the top-level dir prefix in the ttml
# checkpoint is ``Qwen3`` (see grpo_trainer.save_checkpoint). If a future
# checkpoint uses a different prefix, --root-prefix overrides.
DEFAULT_ROOT_PREFIX = "Qwen3"

# Files copied verbatim from src to dst so the resulting dir is
# HF-``from_pretrained``-ready.
_COPY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="Source ttml checkpoint directory.")
    p.add_argument("dst", type=Path, help="Destination directory (created if missing).")
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Output tensor dtype. bf16 shrinks the file 2x and matches "
        "how Qwen3-0.6B is stored on HF Hub. Default: bf16.",
    )
    p.add_argument(
        "--root-prefix",
        default=DEFAULT_ROOT_PREFIX,
        help=f"ttml top-level prefix in the checkpoint keys (default: {DEFAULT_ROOT_PREFIX}).",
    )
    return p.parse_args()


def _squeeze2d(t: torch.Tensor, key: str) -> torch.Tensor:
    """ttml stores 2D weights as (1, 1, out, in) and 1D gammas as (1, 1, 1, dim)
    or (1, 1, dim, 1). Return the HF-expected (out, in) for Linear weights or
    (dim,) for RMSNorm gammas. Check the more-specific 1D shapes first --
    (1, 1, 1, dim) also matches the (1, 1, out, in) pattern with out=1."""
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 1 and t.shape[2] == 1:
        return t.reshape(t.shape[3])
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 1 and t.shape[3] == 1:
        return t.reshape(t.shape[2])
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 1:
        return t.reshape(t.shape[2], t.shape[3])
    if t.dim() == 2 or t.dim() == 1:
        return t
    raise ValueError(f"Unexpected tensor rank {t.dim()} shape {tuple(t.shape)} for key {key!r}")


def _load_ttml_state(path: Path) -> dict[str, torch.Tensor]:
    """Read the model parameters of a ttml ``checkpoint.pkl``: a header record, then one array per tensor."""
    with open(path, "rb") as f:
        names = pickle.load(f)["manifest"]["model"]["named_parameters"]
        return {name: torch.from_numpy(np.asarray(pickle.load(f), dtype=np.float32)) for name in names}


def _copy_side_files(src: Path, dst: Path, files: Iterable[str]) -> None:
    for name in files:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def convert(src: Path, dst: Path, dtype: torch.dtype, root_prefix: str) -> None:
    if not src.is_dir():
        raise SystemExit(f"src {src} is not a directory")
    st_path = src / "checkpoint.pkl"
    cfg_path = src / "config.json"
    if not st_path.exists():
        raise SystemExit(f"missing {st_path}")
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    n_kv = int(cfg["num_key_value_heads"])
    n_q = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    kv_dim = n_kv * head_dim
    hidden = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    tie = bool(cfg.get("tie_word_embeddings", False))

    print(f"[convert] src={src}")
    print(f"[convert] dst={dst}")
    print(
        f"[convert] arch: {n_layers} layers, hidden={hidden}, head_dim={head_dim}, "
        f"n_q_heads={n_q}, n_kv_heads={n_kv} (kv_dim={kv_dim}), tie_word_embeddings={tie}"
    )
    print(f"[convert] output dtype: {dtype}")

    ttml = _load_ttml_state(st_path)
    print(f"[convert] loaded {len(ttml)} ttml tensors from checkpoint.pkl")

    def get(key: str) -> torch.Tensor:
        if key not in ttml:
            available = sorted(ttml.keys())[:10]
            raise SystemExit(f"[convert] missing ttml key {key!r}; first 10 available: {available}")
        return ttml[key]

    def cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).contiguous()

    hf: dict[str, torch.Tensor] = {}

    # Top-level: embedding and final norm.
    fc = _squeeze2d(get(f"{root_prefix}/fc/weight"), "fc/weight")
    if fc.shape != (cfg["vocab_size"], hidden):
        raise SystemExit(
            f"[convert] fc/weight shape {tuple(fc.shape)} != " f"(vocab_size, hidden) = ({cfg['vocab_size']}, {hidden})"
        )
    # Tied path: HF materializes lm_head from embed_tokens at model init, so we
    # do NOT save lm_head.weight. Saving both would cause safetensors to hold
    # two copies unnecessarily.
    mapping, transforms = build_weight_mapping_single(SimpleNamespace(**cfg), root_prefix, tie)
    for hf_name, ttml_name in mapping.items():
        t = _squeeze2d(get(ttml_name), ttml_name)
        match transforms.get(hf_name):
            case ("unpermute_proj", num_heads):
                t = repermute_proj_rows(t, num_heads)
            case ("unpermute_norm",):
                t = repermute_norm_weights(t)
            case ("combine_kv", num_heads, v_name):
                if t.shape[0] != 2 * kv_dim:
                    raise SystemExit(
                        f"[convert] {ttml_name} shape {tuple(t.shape)}: expected 2 * kv_dim = {2 * kv_dim} rows"
                    )
                hf[v_name] = cast(t[kv_dim:])
                t = repermute_proj_rows(t[:kv_dim], num_heads)
        hf[hf_name] = cast(t)

    leftover = sorted(set(ttml) - set(mapping.values()))
    if leftover:
        raise SystemExit(
            f"[convert] {len(leftover)} ttml key(s) were not consumed by the "
            f"conversion. This usually means the checkpoint has extra tensors this "
            f"script does not know about (bias terms, MoE experts, sliding-window "
            f"gates, etc.). First 10: {leftover[:10]}"
        )

    # Write.
    dst.mkdir(parents=True, exist_ok=True)
    out_st = dst / "model.safetensors"
    save_file(hf, str(out_st))
    print(f"[convert] wrote {len(hf)} HF tensors -> {out_st}")

    _copy_side_files(src, dst, _COPY_FILES)
    print(f"[convert] copied side files: " f"{[n for n in _COPY_FILES if (src / n).exists()]}")

    print("[convert] done. Point eval_gsm8k.py at this directory to run pass@k on the checkpoint.")


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    convert(args.src, args.dst, dtype=dtype, root_prefix=args.root_prefix)


if __name__ == "__main__":
    main()
