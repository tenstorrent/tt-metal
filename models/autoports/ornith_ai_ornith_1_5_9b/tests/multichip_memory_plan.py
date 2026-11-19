# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact tile storage accounting for the delivered per-chip projection copies."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bf8(k, n):
    return math.ceil(k / 32) * math.ceil(n / 32) * 1088


rows = []
for kind, packed, count in [("linear_attention", 3136, 24), ("full_attention", 2560, 8)]:
    shapes = {
        "packed_input": (4096, packed),
        "mlp.gate_proj": (4096, 3072),
        "mlp.up_proj": (4096, 3072),
        "mlp.down_proj": (3072, 4096),
        "output": (1024, 4096),
        "packed_mlp": (4096, 6144),
    }
    prefill = sum(bf8(k, n) for k, n in shapes.values())
    decode = sum(bf8(k, math.ceil(n / 256) * 256) for name, (k, n) in shapes.items() if name != "packed_mlp")
    rows.append(
        {
            "kind": kind,
            "count": count,
            "interleaved_bytes_per_chip_per_layer": prefill,
            "dram_sharded_bytes_per_chip_per_layer": decode,
            "total_projection_bytes": prefill + decode,
            "shapes": shapes,
        }
    )
projection = sum(r["count"] * r["total_projection_bytes"] for r in rows)
# Budget both embedding and untied head at BF16, vocab-sharded, with no full model implementation.
embedding_head = 2 * 248320 * 4096 * 2 // 4
kv = 8 * 2 * 8192 * 1 * bf8(32, 256)
recurrent = 24 * 8 * 128 * 128 * 4
conv = 24 * 3 * 2048 * 2
# Every full layer currently owns cos+sin and combined decode lookup; sharing is optional.
rope = 8 * (2 * 262144 * 64 * 2 + 262144 * 128 * 2 + 262144 * 4)
# Tiled singleton norm/tap vectors pad to 32 rows: budget 2 MiB/layer,
# above the ~1.1 MiB linear and ~0.6 MiB full-attention auxiliary tensors.
other_weights = 32 * 2 * 2**20
# Full-context input + accumulated chunk outputs + concat result = 6 GiB.
# Reserve 2 GiB more for activations, traces and transient CCL allocations.
reserved = 8 * 2**30
margin = 2**30
used = projection + embedding_head + kv + recurrent + conv + rope + other_weights + reserved + margin
x = {
    "mesh": [1, 4],
    "context": 262144,
    "weight_dtype": "BFP8",
    "tile_bytes": 1088,
    "rows": rows,
    "projection_bytes_all_layers_per_chip": projection,
    "embedding_head_BF16_budget_per_chip": embedding_head,
    "norm_conv_gdn_constant_budget_per_chip": other_weights,
    "kv_bytes_all_layers_per_chip": kv,
    "recurrent_bytes_all_layers_per_chip": recurrent,
    "conv_bytes_all_layers_per_chip": conv,
    "unshared_rope_position_bytes_per_chip": rope,
    "activation_trace_ccl_reserve_bytes": reserved,
    "shared_persistent_collective_L1_bytes": 2 * 32 * 16384 * 2,
    "shared_collective_L1_bytes_per_core": 2 * 32 * 512 * 2,
    "l1_workspace_contract": "One MultichipCollectives per serial stack; shared across all layers",
    "allocator_reserve_bytes": margin,
    "total_budget_per_chip": used,
    "device_dram_bytes": 8 * 4278190080,
    "free_margin_bytes": 8 * 4278190080 - used,
    "capability_reduction": None,
    "status": "analytic_plan_runtime_validation_separate",
}
assert used < 8 * 4278190080
out = ROOT / "doc/multichip_decoder/memory_capacity_plan.json"
out.write_text(json.dumps(x, indent=2) + "\n")
print("MEMORY_PLAN_PASS", round(used / 2**30, 3), "GiB per 32 GiB chip")
