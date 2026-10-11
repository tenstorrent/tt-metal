# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapt larger prefill blocks to L1 limits and test role-specific minimal configs."""

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.sweep_optimized import OptimizedDecoder, main

linear = OptimizedDecoder._linear
mlp = OptimizedDecoder._mlp


def role_linear(layer, x, name):
    layer._profile_role = name
    return linear(layer, x, name)


def role_mlp(layer, x):
    layer._profile_role = "mlp.gate_up"
    if "mlp.gate_up" not in layer.mm_ck:
        layer.mm_ck["mlp.gate_up"] = layer.mm_ck["mlp.gate_proj"]
    return mlp(layer, x)


def role_config(layer, x):
    name = getattr(layer, "_profile_role", "")

    def get(k, default):
        return layer.policy.get(name + "." + k, layer.policy.get(k, default))

    bm = min(get("minimal_m", 4), x.padded_shape[-2] // 32)
    return ttnn.MinimalMatmulConfig(
        M_block_size=bm,
        K_block_size=get("minimal_k", 8),
        N_block_size=get("minimal_n", 32),
        subblock_h=2 if bm % 2 == 0 else 1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidates", nargs="+")
    args = p.parse_args()
    args.output = args.output.resolve()
    args.batch = 1
    args.policies = args.output.with_suffix(".policies.json")
    base = OptimizedDecoder.POLICY
    policies = {}
    for m, k, n in [(8, 4, 32), (4, 4, 32), (2, 4, 64), (2, 8, 64), (4, 2, 64), (4, 4, 48)]:
        policies[f"m{m}k{k}n{n}"] = base | {"minimal_m": m, "minimal_k": k, "minimal_n": n}
    policies["qkv16"] = base | {"qkv.minimal_n": 16}
    policies["mlp48"] = base | {"mlp.gate_up.minimal_k": 4, "mlp.gate_up.minimal_n": 48}
    for block in (2, 4, 8):
        policies[f"separate_mlp{block}"] = base | {
            "pack_mlp": False,
            "fuse_silu": True,
            "mlp.gate_proj.block": block,
            "mlp.up_proj.block": block,
        }
    policies["separate_attention"] = base | {"pack_attention_gate": False}
    args.policies.write_text(json.dumps(policies, indent=2))
    args.candidates = args.candidates or ["default"] + list(policies)
    OptimizedDecoder._linear = role_linear
    OptimizedDecoder._mlp = role_mlp
    OptimizedDecoder._minimal_config = role_config
    args.output.with_suffix(".probe.py").write_bytes(Path(__file__).read_bytes())
    main(args)
