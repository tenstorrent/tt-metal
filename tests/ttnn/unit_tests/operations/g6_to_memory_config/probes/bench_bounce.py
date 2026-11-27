# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-only bench for the bounce path (Refinement 2). Not collected by the suite (no test_ prefix).

Run under scripts/run_safe_pytest.sh --profile --no-py-zones <this file>. Env:
  G6_CASES  comma list of SCENARIOS keys (default qwen2183)
  G6_KNOBS  comma list NAME=VALUE of descriptor-module knobs to override
  G6_ABLATE comma list of kernel defines (G6_ABLATE_READ, G6_ABLATE_WRITE); skips the value check
One op call per case, so CSV rows follow the case order.
"""

import os

import pytest
import torch
import ttnn

import ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor as descriptor_module
from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

SCENARIOS = {f"qwen{n}": CASES[("Qwen", n)][0] for n in (2178, 2179, 2180, 2181, 2182, 2183, 2184)}
SCENARIOS["bf16_interleaved"] = {
    "shape": [16, 128, 1536],
    "dtype": ttnn.bfloat16,
    "in_memory": {"buffer": "dram", "scheme": "interleaved"},
    "out_memory": {"buffer": "dram", "scheme": "interleaved"},
}

# 2183's volume, width-sharded over half the DRAM banks: 0-3 (one DRAM column) or 4-7 (the other).
for name, (first, last) in {"banks0_3": (0, 3), "banks4_7": (4, 7)}.items():
    SCENARIOS[name] = {
        "shape": [2560, 16384],
        "dtype": ttnn.bfloat4_b,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {
            "buffer": "dram",
            "scheme": "width_sharded",
            "grid": [[first, 0, last, 0]],
            "shard_shape": [2560, 4096],
            "orientation": "row_major",
        },
    }

# 2183's target layout at 2 KB (bf16) and 1088 B (bf8b) pages: DRAM interleaved -> DRAM width-sharded on 8 banks.
for name, dtype in {"bf16_ws": ttnn.bfloat16, "bf8b_ws": ttnn.bfloat8_b}.items():
    SCENARIOS[name] = {
        "shape": [2560, 8192],
        "dtype": dtype,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {
            "buffer": "dram",
            "scheme": "width_sharded",
            "grid": [[0, 0, 7, 0]],
            "shard_shape": [2560, 1024],
            "orientation": "row_major",
        },
    }

CASE_LIST = [c for c in os.environ.get("G6_CASES", "qwen2183").split(",") if c]


@pytest.mark.parametrize("case", CASE_LIST)
def test_bench(device, monkeypatch, case):
    for item in [k for k in os.environ.get("G6_KNOBS", "").split(",") if k]:
        name, value = item.split("=")
        monkeypatch.setattr(descriptor_module, name, int(value) if value.lstrip("-").isdigit() else value)
    ablate = tuple(d for d in os.environ.get("G6_ABLATE", "").split(",") if d)
    monkeypatch.setattr(descriptor_module, "ABLATE_DEFINES", ablate)
    scenario = SCENARIOS[case]
    torch.manual_seed(1234)
    tt_input = ttnn.from_torch(
        torch.randn(scenario["shape"], dtype=torch.float32),
        dtype=scenario["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=build_memory_config(scenario["in_memory"]),
    )
    out_config = build_memory_config(scenario["out_memory"])
    plan = plan_of(tt_input, out_config)
    print(
        f"BENCH {case}: regime={plan['regime']} order={plan['bounce_order']} noc_swap={plan['bounce_noc_swap']} block={plan['bounce_block_pages']} depth={plan['bounce_depth']}"
    )
    output = g6_to_memory_config(tt_input, out_config)
    ttnn.synchronize_device(device)
    if not ablate:
        assert torch.equal(ttnn.to_torch(output), ttnn.to_torch(tt_input))
