# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Estimate the largest **TT dense-KV** prompt size that fits on the current device mesh.

4K works on a typical single Blackhole + full Devstral checkpoint; 256K OOMs because one K/V
tensor needs ~500+ MiB while only ~30 MiB DRAM is free per bank after weights.

Usage (repo root)::

    python models/experimental/devstarl2_small/scripts/find_max_tt_context_tokens.py

    # Tune budget if 4K is stable but you want to push higher (MiB for **one** K or V tensor):
    export DEVSTRAL2_KV_PER_TENSOR_BUDGET_MB=32
    python models/experimental/devstarl2_small/scripts/find_max_tt_context_tokens.py

    # Generate messages JSON at the estimate (multimodal):
    python models/experimental/devstarl2_small/scripts/make_long_context_prompt.py \\
        --target-tokens max \\
        --output models/experimental/devstarl2_small/reference/messages_max_tt.json
"""

from __future__ import annotations

import argparse
import os

import ttnn
from loguru import logger

from models.experimental.devstarl2_small.devstral_utils import (
    devstral_dense_kv_tensor_bytes,
    devstral_estimate_max_prompt_tokens_dense_kv,
    devstral_model_args_for_kv_estimate,
    devstral_tt_kv_cache_max_seq_len,
    open_devstral_demo_mesh,
    tt_prefill_target_seqlen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate max TT prompt tokens (dense KV, single-chip).")
    parser.add_argument("--model-id", default=os.getenv("HF_MODEL", "mistralai/Devstral-Small-2-24B-Instruct-2512"))
    parser.add_argument("--mesh-width", type=int, default=1)
    parser.add_argument("--known-good", type=int, default=4096, help="Prompt size you know works (default 4096).")
    parser.add_argument(
        "--per-tensor-budget-mb",
        type=float,
        default=None,
        help="Max MiB for one K or V tensor (default: env DEVSTRAL2_KV_PER_TENSOR_BUDGET_MB or 30).",
    )
    args = parser.parse_args()

    mesh_device = open_devstral_demo_mesh(max(1, min(args.mesh_width, ttnn.get_num_devices())))
    try:
        model_args = devstral_model_args_for_kv_estimate(mesh_device, model_id=args.model_id)

        max_prompt = devstral_estimate_max_prompt_tokens_dense_kv(
            model_args,
            mesh_device,
            known_good_tokens=args.known_good,
            per_tensor_budget_mb=args.per_tensor_budget_mb,
        )
        cols = int(model_args.cluster_shape[1])
        kv_padded = tt_prefill_target_seqlen(max_prompt, int(model_args.n_kv_heads), cols)
        per_mib = devstral_dense_kv_tensor_bytes(model_args, kv_padded) / (1024 * 1024)

        need = max_prompt + 2048
        kv_alloc = devstral_tt_kv_cache_max_seq_len(model_args, need)

        bh = ttnn.device.is_blackhole(mesh_device)
        logger.info(
            f"Device: {'Blackhole' if bh else 'other'}, mesh_width={args.mesh_width}, "
            f"n_kv_heads={model_args.n_kv_heads}, head_dim={model_args.head_dim}, "
            f"n_layers={model_args.n_layers}."
        )
        logger.info(
            f"Estimated max TT prompt (dense KV): {max_prompt:,} tokens\n"
            f"  KV tensor (one of K/V): ~{per_mib:.1f} MiB at padded kv_len={kv_padded:,}\n"
            f"  devstral_tt_kv_cache_max_seq_len(need≈{need:,}): {kv_alloc:,}\n"
            f"Next: make_long_context_prompt.py --target-tokens {max_prompt} "
            f"(or --target-tokens max) and demo --messages-json <that file>."
        )
        if max_prompt <= args.known_good:
            logger.warning(
                "Estimate did not exceed --known-good; raise DEVSTRAL2_KV_PER_TENSOR_BUDGET_MB "
                "(e.g. 32) if your 4K run had more DRAM headroom."
            )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
