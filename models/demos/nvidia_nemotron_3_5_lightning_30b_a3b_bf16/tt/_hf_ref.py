# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-A access: the HuggingFace reference for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

Two jobs, and ONLY these two:

  1. hand the TT pipeline the torch submodules it extracts weights from at
     BUILD time (allowed HF usage #2), and
  2. compute the GOLDEN output for the PCC gate inside
     `tt/pipeline.py::_hf_reference_text_generation` (allowed HF usage #3).

Nothing here is ever called from the TT hot path.

DEPTH CAP
---------
The checkpoint is 31.58e9 parameters. Its 23 MoE blocks alone hold
128 x 2 x 2688 x 1856 x 23 = 29.4e9 parameters (58.8 GB bf16); expert-parallel
at TP=2 halves that to ~29 GB *per chip* against ~12 GB of Wormhole DRAM. No
TP degree this model permits (TP>2 is blocked by num_key_value_heads=2, see
kernel_findings.json) makes a resident 52-layer build fit on 4 chips.

So the on-device gate runs a DEPTH-CAPPED model, and the golden is the SAME
checkpoint capped to the SAME depth -- TT and HF compute the same function, so
the PCC comparison is exact-in-scope rather than approximate. `layers=None`
still means every layer; the caller chooses.

The first 7 blocks are `[mamba, moe, mamba, moe, mamba, attention, moe]` --
the shortest prefix carrying all three block types with enough of each to host
every graduated stub.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt._hf_compat import install_hf_compat

HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

# `layers_block_type[:7] == [mamba, moe, mamba, moe, mamba, attention, moe]`
DEFAULT_GATE_LAYERS = 7

_DEMO_DIR = Path(__file__).resolve().parents[1]
_CACHE_DIR = Path(os.environ.get("TT_NEMOTRON_REF_CACHE", _DEMO_DIR / "_captured" / "_hf_ref"))

_MODEL_CACHE: dict[int | None, object] = {}
_TOK = None


def get_tokenizer():
    """The Source-A tokenizer used to build every input in this package."""
    global _TOK
    if _TOK is None:
        install_hf_compat()
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    return _TOK


def _truncate(model, layers: int):
    """Cap the decoder stack to `layers` blocks, in place, and free the rest.

    Everything that is NOT a repeated block -- embeddings, final norm, lm_head --
    stays intact, so a capped build still exercises every distinct op the full
    model runs, just fewer times.
    """
    import torch.nn as nn

    cfg = model.config
    full = len(model.model.layers)
    if layers is None or layers >= full:
        return model
    keep = list(model.model.layers[:layers])
    model.model.layers = nn.ModuleList(keep)
    cfg.num_hidden_layers = layers
    cfg.layers_block_type = list(cfg.layers_block_type)[:layers]
    gc.collect()
    return model


def _cache_path(layers: int) -> Path:
    return _CACHE_DIR / f"depth{layers}"


def load_reference(layers: int | None = DEFAULT_GATE_LAYERS, dtype=torch.float32):
    """Return the HF reference model capped to `layers` blocks (None == all 52).

    A depth-capped model is cached to disk on first use so later runs skip the
    66 GB full-checkpoint read.
    """
    key = layers
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    install_hf_compat()
    from transformers import AutoConfig, AutoModelForCausalLM

    ckpt = _cache_path(layers) if layers is not None else None
    if ckpt is not None and (ckpt / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    else:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model = _truncate(model, layers)
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt, safe_serialization=True)

    model.config._attn_implementation = "eager"
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[key] = model
    return model


def block_types(model) -> list[str]:
    return [blk.block_type for blk in model.model.layers]


if __name__ == "__main__":  # prime the depth-capped cache
    import argparse

    ap = argparse.ArgumentParser(description="Prime the depth-capped HF reference cache.")
    ap.add_argument("--layers", type=int, default=DEFAULT_GATE_LAYERS)
    a = ap.parse_args()
    m = load_reference(a.layers, dtype=None)
    print(f"[ref] depth={len(m.model.layers)} block_types={block_types(m)}")
    print(f"[ref] params={sum(p.numel() for p in m.parameters())/1e9:.2f}B cached at {_cache_path(a.layers)}")
