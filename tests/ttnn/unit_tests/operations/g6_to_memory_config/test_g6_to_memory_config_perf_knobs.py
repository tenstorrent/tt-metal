# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block knobs of g6_to_memory_config (op_design.md -> Perf lamps L1 and L6), one regime each.

A knob may change speed, never values: every variant is checked bit-exact and its plan is checked to
carry the knob. One op call per parametrization, so under `run_safe_pytest.sh --profile` the CSV rows
follow the parametrize order. Shapes and placements are production ones (BB-22).
"""

import pytest
import torch
import ttnn

import ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor as descriptor_module
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

_BUFFER = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_SCHEME = {
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}


def interleaved(buffer):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, _BUFFER[buffer])


def sharded(buffer, scheme, grid, shard_shape):
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)) for sx, sy, ex, ey in grid]
    )
    spec = ttnn.ShardSpec(core_grid, tuple(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(_SCHEME[scheme], _BUFFER[buffer], spec)


# regime: (shape, input placement, target placement)
REGIME_CASES = {
    "pull_shard": (  # Qwen 2067
        [1, 1, 16, 5120],
        lambda: sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 640]),
        lambda: sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
    ),
    "pull_bank": ([1, 77, 4352], lambda: interleaved("dram"), lambda: interleaved("l1")),  # Qwen 2105
    "push_shard": (  # Qwen 2071, ragged last shard
        [1, 1, 16, 12928],
        lambda: sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 1664]),
        lambda: interleaved("dram"),
    ),
    "push_bank": ([1, 1, 16, 4352], lambda: interleaved("l1"), lambda: interleaved("dram")),  # Qwen 2064 input
    "bounce": ([16, 128, 1536], lambda: interleaved("dram"), lambda: interleaved("dram")),  # Qwen 2077 shape
}


def _bank_regimes_available(device):
    info = ttnn.reports.get_device_info(device)
    return info.l1_num_banks == info.num_compute_cores


@pytest.mark.parametrize("movers_per_owner,bounce_depth", [(1, 2), (2, 2), (2, 4)], ids=lambda v: str(v))
@pytest.mark.parametrize("regime", list(REGIME_CASES))
def test_knob_is_live_and_exact(device, monkeypatch, regime, movers_per_owner, bounce_depth):
    if regime.endswith("_bank") and not _bank_regimes_available(device):
        pytest.skip("bank regimes fall back to bounce on this device")
    monkeypatch.setattr(descriptor_module, "MOVERS_PER_OWNER", movers_per_owner)
    monkeypatch.setattr(descriptor_module, "BOUNCE_DEPTH", bounce_depth)

    shape, in_memory, out_memory = REGIME_CASES[regime]
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_memory(),
    )
    out_config = out_memory()

    plan = plan_of(tt_input, out_config)
    assert plan["regime"] == regime
    if regime == "bounce":
        assert plan["bounce_depth"] == bounce_depth
        assert plan["cb_bounce_pages"] == bounce_depth * plan["bounce_block_pages"]
    else:
        assert plan["movers_per_owner"] == movers_per_owner

    output = g6_to_memory_config(tt_input, out_config)
    assert torch.equal(ttnn.to_torch(output), ttnn.to_torch(tt_input))
