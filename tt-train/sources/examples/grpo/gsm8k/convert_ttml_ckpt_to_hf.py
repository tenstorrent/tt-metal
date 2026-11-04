#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a ttml GRPOTrainer checkpoint to HuggingFace safetensors layout.

The ``save_checkpoint`` path in ``ttml.trainers.grpo_trainer`` writes ttml-side
keys (``Qwen3/blocks/{i}/self_attn/q_proj/weight`` ...) with a fused
``kv_proj`` in the meta-permuted Q/K row layout ``[r0, i0, r1, i1, ...]`` (per
head). ``AutoModelForCausalLM.from_pretrained`` expects the standard HF Qwen3
layout: ``model.layers.{i}.self_attn.q_proj.weight``, separate ``k_proj`` and
``v_proj``, and the "half-then-half" row layout ``[r0, ..., r_{d/2-1}, i0, ...,
i_{d/2-1}]`` that HF's rotary embedding assumes. This script does the one-time
key rename, ``kv_proj`` split, and row re-permutation of Q/K projections and
Q/K-Norm gammas so eval can consume GRPO checkpoints.

Usage:
    python convert_ttml_ckpt_to_hf.py <src_ckpt_dir> <dst_dir> [--dtype bf16|fp32]

Example (each GRPO checkpoint of the current run):
    for step in 100 200 300; do
        python convert_ttml_ckpt_to_hf.py \\
            generated/tt-train/grpo_gsm8k_run/<utc>/checkpoints/grpo_step_${step} \\
            /localdev/ichovpan/converted/grpo-ckpt-${step}/qwen3-0.6b-base-think-sft
    done

The destination directory's basename is intentionally set to
``qwen3-0.6b-base-think-sft`` so ``ModelArgs.model_name`` (derived from the last
path segment of ``HF_MODEL``) hits the existing ``LOCAL_HF_PARAMS`` entry when
eval_gsm8k.py points at the converted directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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
        help=f"ttml top-level prefix in the safetensors keys (default: {DEFAULT_ROOT_PREFIX}).",
    )
    return p.parse_args()


def _repermute_proj_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """ttml interleaved ``[r0, i0, r1, i1, ...]`` -> HF ``[real_half, imag_half]`` per head.

    Inverse of ``ttml.models.qwen3.weights.unpermute_proj_rows``. Inlined here
    so this script does not depend on the ttml package.
    """
    rows, cols = weight.shape
    if rows % num_heads != 0:
        raise ValueError(f"rows={rows} not divisible by num_heads={num_heads}")
    D = rows // num_heads
    w = weight.view(num_heads, D, cols)
    reals = w[:, 0::2, :]  # even positions along head_dim -> real half
    imags = w[:, 1::2, :]  # odd positions -> imag half
    return torch.cat([reals, imags], dim=1).reshape(rows, cols).contiguous()


def _repermute_norm_weights(weight: torch.Tensor) -> torch.Tensor:
    """ttml ``[x1, y1, x2, y2, ...]`` -> HF ``[x1, x2, ..., y1, y2, ...]``.

    Inverse of ``ttml.models.qwen3.weights.unpermute_norm_weights`` for a
    per-head-dim RMSNorm gamma.
    """
    head_dim = weight.shape[0]
    if head_dim % 2 == 0:
        w = weight.view(head_dim // 2, 2)
        return w.t().contiguous().flatten()
    return weight


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
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state


def _copy_side_files(src: Path, dst: Path, files: Iterable[str]) -> None:
    for name in files:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def convert(src: Path, dst: Path, dtype: torch.dtype, root_prefix: str) -> None:
    if not src.is_dir():
        raise SystemExit(f"src {src} is not a directory")
    st_path = src / "model.safetensors"
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
    print(f"[convert] loaded {len(ttml)} ttml tensors from safetensors")

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
    hf["model.embed_tokens.weight"] = cast(fc)
    if not tie:
        # Untied: expect a separate tok_emb tensor; fc is lm_head only.
        tok = _squeeze2d(get(f"{root_prefix}/tok_emb/weight"), "tok_emb/weight")
        hf["model.embed_tokens.weight"] = cast(tok)
        hf["lm_head.weight"] = cast(fc)
    # Tied path: HF materializes lm_head from embed_tokens at model init, so we
    # do NOT save lm_head.weight. Saving both would cause safetensors to hold
    # two copies unnecessarily.

    hf["model.norm.weight"] = cast(_squeeze2d(get(f"{root_prefix}/ln_fc/weight"), "ln_fc/weight"))

    # Per-layer.
    for i in range(n_layers):
        tp = f"{root_prefix}/blocks/{i}"
        hp = f"model.layers.{i}"

        hf[f"{hp}.input_layernorm.weight"] = cast(_squeeze2d(get(f"{tp}/input_layernorm/weight"), "input_layernorm"))
        hf[f"{hp}.post_attention_layernorm.weight"] = cast(
            _squeeze2d(get(f"{tp}/post_attention_layernorm/weight"), "post_attention_layernorm")
        )

        # Attention projections. Q/K row layout is ttml's meta-permuted
        # [r0, i0, r1, i1, ...] per head; HF Qwen3 expects "half-then-half"
        # [r0..r_{d/2-1}, i0..i_{d/2-1}] so rotary math lines up. Re-permute
        # Q/K rows (Q with n_q heads, K with n_kv heads) and Q/K-norm gammas
        # (per-head-dim), leaving V and O untouched.
        q = _squeeze2d(get(f"{tp}/self_attn/q_proj/weight"), "q_proj")
        hf[f"{hp}.self_attn.q_proj.weight"] = cast(_repermute_proj_rows(q, n_q))
        hf[f"{hp}.self_attn.o_proj.weight"] = cast(_squeeze2d(get(f"{tp}/self_attn/o_proj/weight"), "o_proj"))

        qn = _squeeze2d(get(f"{tp}/self_attn/q_norm/weight"), "q_norm")
        kn = _squeeze2d(get(f"{tp}/self_attn/k_norm/weight"), "k_norm")
        hf[f"{hp}.self_attn.q_norm.weight"] = cast(_repermute_norm_weights(qn))
        hf[f"{hp}.self_attn.k_norm.weight"] = cast(_repermute_norm_weights(kn))

        # Fused kv_proj (ttml, K rows then V rows) -> split into HF k_proj / v_proj.
        # K rows use the meta-permuted layout; V rows do not (V is not RoPE'd).
        kv = _squeeze2d(get(f"{tp}/self_attn/kv_proj/weight"), "kv_proj")
        if kv.shape != (2 * kv_dim, hidden):
            raise SystemExit(
                f"[convert] layer {i} kv_proj shape {tuple(kv.shape)} != "
                f"(2 * kv_dim, hidden) = ({2 * kv_dim}, {hidden})"
            )
        k_rows = kv[:kv_dim, :].contiguous()
        v_rows = kv[kv_dim:, :].contiguous()
        hf[f"{hp}.self_attn.k_proj.weight"] = cast(_repermute_proj_rows(k_rows, n_kv))
        hf[f"{hp}.self_attn.v_proj.weight"] = cast(v_rows)

        # MLP.
        hf[f"{hp}.mlp.gate_proj.weight"] = cast(_squeeze2d(get(f"{tp}/mlp/gate_proj/weight"), "gate_proj"))
        hf[f"{hp}.mlp.up_proj.weight"] = cast(_squeeze2d(get(f"{tp}/mlp/up_proj/weight"), "up_proj"))
        hf[f"{hp}.mlp.down_proj.weight"] = cast(_squeeze2d(get(f"{tp}/mlp/down_proj/weight"), "down_proj"))

    # Any leftover ttml keys mean an unhandled architectural piece.
    consumed_ttml_keys: set[str] = set()
    consumed_ttml_keys.add(f"{root_prefix}/fc/weight")
    consumed_ttml_keys.add(f"{root_prefix}/ln_fc/weight")
    if not tie:
        consumed_ttml_keys.add(f"{root_prefix}/tok_emb/weight")
    for i in range(n_layers):
        tp = f"{root_prefix}/blocks/{i}"
        for suffix in (
            "input_layernorm/weight",
            "post_attention_layernorm/weight",
            "self_attn/q_proj/weight",
            "self_attn/o_proj/weight",
            "self_attn/q_norm/weight",
            "self_attn/k_norm/weight",
            "self_attn/kv_proj/weight",
            "mlp/gate_proj/weight",
            "mlp/up_proj/weight",
            "mlp/down_proj/weight",
        ):
            consumed_ttml_keys.add(f"{tp}/{suffix}")

    leftover = sorted(set(ttml.keys()) - consumed_ttml_keys)
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
