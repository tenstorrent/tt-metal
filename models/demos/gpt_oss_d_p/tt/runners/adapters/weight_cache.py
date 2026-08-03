# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tilized-weight cache completeness check for GPT-OSS.

Mirrors ``minimax_m3/tt/weight_cache.py``. The gpt-oss-120b bf16 checkpoint is large enough that
re-reading it on every run is prohibitive; every weight already loads from a per-tensor tilized
``.tensorbin`` cache via ``ttnn.as_tensor(cache_file_name=)``. On a cache hit the passed torch tensor
is ignored, so once the cache is populated we can pass an empty state_dict and skip the source read.

The MoE routed-expert biases are the one thing the TTNN cache does NOT persist (they are consumed
by the fused unified_routed_expert_moe kernel outside the tilized-weight path); ``tt/mlp.py`` writes
them to a small sidecar (``routed_expert_biases.pt``) on the real-weight build and reloads them here
on a cache-only build. This function checks the sidecar exists too, otherwise ``MLP.__init__`` raises
mid-build.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn

# ttnn.as_tensor(cache_file_name=...) appends _dtype_<DT>_layout_<L>.tensorbin. Match by prefix.
_DTYPE_TAG = {
    ttnn.bfloat16: "BFLOAT16",
    ttnn.bfloat8_b: "BFLOAT8_B",
    ttnn.bfloat4_b: "BFLOAT4_B",
}


def weight_cache_is_complete(
    weight_cache_path,
    hf_config,
    num_layers: int,
    expert_weight_dtype,
) -> bool:
    """True iff the tilized cache holds every tensor + the routed-expert bias sidecar that the model
    builds for this run. Conservative: any missing file returns False (caller then loads weights from
    source — slow but correct)."""
    if not weight_cache_path:
        return False
    root = Path(weight_cache_path)
    if not root.is_dir():
        return False

    edt = _DTYPE_TAG.get(expert_weight_dtype)
    if edt is None:
        logger.warning(f"[gpt-oss weight-cache] unknown expert dtype {expert_weight_dtype}; loading from source")
        return False

    rels = set()
    for dirpath, _, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for fn in files:
            rels.add(fn if rel_dir == "." else f"{rel_dir}/{fn}")

    def has(prefix: str) -> bool:
        return any(r.startswith(prefix) for r in rels)

    # Top-level: replicated embedding (bf16 ROW_MAJOR), final norm, padded lm_head.
    required = [
        "model.embed_tokens.weight",
        "norm/weight",
        "lm_head_padded_pow2.weight",
    ]
    for L in range(num_layers):
        base = f"model.layers.{L}"
        required += [
            f"{base}/input_layernorm/weight",
            f"{base}/post_attention_layernorm/weight",
            f"{base}/self_attn/wqkv",
            f"{base}/self_attn/wqkv_bias",
            # o_proj / o_proj_bias may pick up a "_padded" suffix (see attention/weights.py:70) —
            # prefix match covers both.
            f"{base}/self_attn/o_proj",
            f"{base}/self_attn/sinks_div_scale",
            f"{base}/mlp/router/weight",
            f"{base}/mlp/router/bias",
            # Per-expert routed weights (dtype-specific). Local index 0 is enough as a canary — if
            # experts_per_chip>0 the cache-populate step writes them all in one pass.
            f"{base}/mlp/experts_ep/layer_{L}.routed_expert.local_0_gate_dtype_{edt}",
        ]

    missing = [p for p in required if not has(p)]
    if missing:
        logger.info(
            f"[gpt-oss weight-cache] cache at {root} INCOMPLETE: {len(missing)} of {len(required)} entries "
            f"missing (e.g. {missing[:3]}); will load bf16 source."
        )
        return False

    # Bias sidecar (per layer, written by tt/mlp.py on a real-weights build).
    for L in range(num_layers):
        sidecar = root / f"model.layers.{L}" / "mlp" / "experts_ep" / "routed_expert_biases.pt"
        if not sidecar.exists():
            logger.info(f"[gpt-oss weight-cache] cache at {root} INCOMPLETE: bias sidecar missing at {sidecar}")
            return False

    logger.info(f"[gpt-oss weight-cache] cache at {root} complete for {num_layers} layers; skipping bf16 source read.")
    return True
