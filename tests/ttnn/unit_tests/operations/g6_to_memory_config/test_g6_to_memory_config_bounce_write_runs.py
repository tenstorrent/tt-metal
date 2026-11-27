# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounce knobs of the DRAM -> DRAM path: write runs, target-order assignment and bank stagger.

`BOUNCE_WRITE_RUN_PAGES` coalesces pages contiguous in the target into one NoC write (1 = one write per
page, 0 = the whole run, N = runs capped at N). `BOUNCE_TARGET_ORDER` assigns movers contiguous ranges of a
sharded target's storage order instead of page-id ranges. `BOUNCE_BANK_STAGGER` rotates each mover's walk so
movers start on different target shards. A knob may change speed, never values: every variant is checked
bit-exact, including ragged edge shards of DRAM width-, height- and block-sharded targets.
"""

import pytest
import torch
import ttnn

import ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor as descriptor_module
from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

SCENARIOS = {
    # bf4b DRAM interleaved -> DRAM width sharded on 8 banks; last shard holds 40 of 52 tile columns.
    "qwen2184": CASES[("Qwen", 2184)][0],
    "qwen2178": CASES[("Qwen", 2178)][0],
    # A target with no run longer than a page.
    "bf16_interleaved": {
        "shape": [16, 128, 1536],
        "dtype": ttnn.bfloat16,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {"buffer": "dram", "scheme": "interleaved"},
    },
    # 32 padded tile rows in shards of 5: the last of 7 shards holds 2 rows.
    "dram_height_ragged": {
        "shape": [1, 1, 1000, 512],
        "dtype": ttnn.bfloat16,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {
            "buffer": "dram",
            "scheme": "height_sharded",
            "grid": [[0, 0, 7, 0]],
            "shard_shape": [160, 512],
            "orientation": "row_major",
        },
    },
    # 32 tile columns in shards of 7: the last of 5 shards holds 4 columns.
    "dram_block_ragged": {
        "shape": [2, 64, 1024],
        "dtype": ttnn.bfloat8_b,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {
            "buffer": "dram",
            "scheme": "block_sharded",
            "grid": [[0, 0, 4, 0]],
            "shard_shape": [128, 224],
            "orientation": "row_major",
        },
    },
}

# variant: (BOUNCE_WRITE_RUN_PAGES, BOUNCE_TARGET_ORDER, BOUNCE_BANK_STAGGER)
VARIANTS = {
    "per_page": (1, False, False),
    "runs": (0, False, False),
    "runs_capped": (5, False, False),
    "stagger": (0, False, True),
    "target_order": (0, True, False),
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", list(SCENARIOS))
def test_bounce_knobs_exact(device, monkeypatch, case, variant):
    write_run_pages, target_order, stagger = VARIANTS[variant]
    monkeypatch.setattr(descriptor_module, "BOUNCE_WRITE_RUN_PAGES", write_run_pages)
    monkeypatch.setattr(descriptor_module, "BOUNCE_TARGET_ORDER", target_order)
    monkeypatch.setattr(descriptor_module, "BOUNCE_BANK_STAGGER", stagger)
    scenario = SCENARIOS[case]
    dram_grid = device.dram_grid_size()
    for _, _, ex, ey in scenario["out_memory"].get("grid", []):
        if ex >= dram_grid.x or ey >= dram_grid.y:
            pytest.skip(f"shard grid end ({ex},{ey}) outside this device's DRAM grid")

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
    assert plan["regime"] == "bounce"
    assert plan["bounce_write_run_pages"] == write_run_pages
    sharded_target = out_config.is_sharded()
    assert plan["bounce_order"] == ("target_storage" if target_order and sharded_target else "page")

    output = g6_to_memory_config(tt_input, out_config)
    assert output.memory_config() == out_config
    assert torch.equal(ttnn.to_torch(output), ttnn.to_torch(tt_input))
