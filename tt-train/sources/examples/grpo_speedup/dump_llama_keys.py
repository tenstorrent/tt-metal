#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dump state-dict keys for the same Llama checkpoint in three formats:

    1. hf              -- HuggingFace ``transformers`` (raw safetensors keys)
    2. tt-transformers -- Meta naming used by ``models/tt_transformers``
                          (run through ``convert_hf_to_meta``)
    3. ttml            -- ``Llama/blocks/{i}/...`` naming used by tt-train,
                          derived from the loader in
                          ``tt-train/sources/ttml/ttml/models/llama/safetensors_loader.py``

CPU-only: no mesh device, no on-device weight load. Just lists the keys and
their shapes side-by-side. Edit ``MODEL_ID`` / ``WEIGHT_TYING`` to switch
checkpoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# ttml routes the embedding to a different param when the LM head is tied to
# the embedding (see safetensors_loader.py). Llama-3.2-1B/3B have weight tying
# enabled in the in-repo configs (configs/model_configs/llama3_2_1B.yaml).
# Set this False for 8B/70B which ship a separate ``lm_head.weight``.
WEIGHT_TYING = True


# ---------------------------------------------------------------------------
# 1) HF format
# ---------------------------------------------------------------------------


def hf_keys(model_id: str) -> Dict[str, Tuple[int, ...]]:
    """Return raw HF state-dict keys -> shape, without ever materialising fp32."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[hf] loading {model_id} (CPU, meta-tensor scan)")
    # Use ``torch_dtype=torch.float16`` to keep memory low; we only read keys/shapes.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    sd = model.state_dict()
    out = {k: tuple(v.shape) for k, v in sd.items()}
    del model, sd
    return out


# ---------------------------------------------------------------------------
# 2) tt-transformers (Meta) format
# ---------------------------------------------------------------------------


def tt_transformers_keys(hf_state_shapes: Dict[str, Tuple[int, ...]], model_id: str) -> Dict[str, Tuple[int, ...]]:
    """Run the HF state dict through ``convert_hf_to_meta`` and return renamed keys.

    No mesh device required. ``convert_hf_to_meta`` only does key renaming +
    Q/K row permutation (which preserves shape).
    """
    import torch
    from transformers import AutoConfig

    from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys

    cfg = AutoConfig.from_pretrained(model_id)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads

    # Build a "shape only" state dict of zero tensors so the Q/K permutation
    # path can run without holding real weights in memory.
    fake_sd = {k: torch.zeros(s, dtype=torch.float16) for k, s in hf_state_shapes.items()}
    fake_sd = standardize_hf_keys(fake_sd)
    meta_sd = convert_hf_to_meta(fake_sd, head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
    return {k: tuple(v.shape) for k, v in meta_sd.items()}


# ---------------------------------------------------------------------------
# 3) ttml format
# ---------------------------------------------------------------------------


def ttml_keys(model_id: str, weight_tying: bool) -> List[str]:
    """Enumerate the ttml parameter names statically from the loader's mapping.

    Mirrors ``tt-train/sources/ttml/ttml/models/llama/safetensors_loader.py``,
    which is the source of truth for ttml param naming.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers

    keys: List[str] = []
    if weight_tying:
        keys.append("Llama/fc/weight")  # tied: embedding lives here
    else:
        keys.append("Llama/tok_emb/weight")
        keys.append("Llama/fc/weight")  # separate LM head

    for i in range(n_layers):
        keys.extend(
            [
                f"Llama/blocks/{i}/attention_norm/gamma",
                f"Llama/blocks/{i}/attention/q_linear/weight",
                f"Llama/blocks/{i}/attention/kv_linear/weight",  # k_proj + v_proj fused
                f"Llama/blocks/{i}/attention/out_linear/weight",
                f"Llama/blocks/{i}/mlp_norm/gamma",
                f"Llama/blocks/{i}/mlp/w1/weight",  # gate_proj
                f"Llama/blocks/{i}/mlp/w3/weight",  # up_proj
                f"Llama/blocks/{i}/mlp/w2/weight",  # down_proj
            ]
        )

    keys.append("Llama/ln_fc/gamma")
    return keys


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_keys(title: str, items) -> None:
    print()
    print("=" * 78)
    print(f" {title}  (count={len(items)})")
    print("=" * 78)
    if isinstance(items, dict):
        width = max(len(k) for k in items) if items else 0
        for k, shape in items.items():
            print(f"  {k.ljust(width)}  shape={shape}")
    else:
        for k in items:
            print(f"  {k}")


def main() -> None:
    print(f"Model: {MODEL_ID}")
    print(f"weight_tying (ttml): {WEIGHT_TYING}")

    hf = hf_keys(MODEL_ID)
    print_keys("HF (transformers / safetensors)", hf)

    tt = tt_transformers_keys(hf, MODEL_ID)
    print_keys("tt-transformers (Meta naming, after convert_hf_to_meta)", tt)

    ttml = ttml_keys(MODEL_ID, WEIGHT_TYING)
    print_keys("ttml (Llama/blocks/{i}/... — from safetensors_loader.py)", ttml)


if __name__ == "__main__":
    main()
