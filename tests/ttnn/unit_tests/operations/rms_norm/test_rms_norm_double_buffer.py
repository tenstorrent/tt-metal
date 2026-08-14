# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — the double-buffered input path (Refinement 4).

`IN_CB_DEPTH` was pinned at 1 through Refinement 1 because the two in-place
rewrites of x pack at `get_write_ptr(cb) + i*page`, and a compute thread that
never pushes `cb_input_tiles` keeps its write pointer at the CB BASE forever —
so a deeper CB wrote `x*r` into the wrong half as soon as the READ window moved
off base.  Refinement 4 lifts the pin by taking the pack index modulo the CB's
whole capacity (`IN_CAPACITY_TILES`), which is the same mechanism Refinement 2
introduced for a resident shard whose capacity exceeds one block.

WHY THIS FILE EXISTS.  The failure mode is silent and it is NOT covered by the
immutable acceptance suite: every shape in `test_rms_norm.py` has at most 8
tile-rows, so `max_rows_full == 1` on a 110-core grid and the depth-2 rung never
fires.  A shape has to be TALL (thousands of rows) before a core owns two
blocks.  And when the pack base is wrong the output is not garbage — it is the
right values missing one block's `1/rms` factor, i.e. an error of exactly the
row-RMS spread ~ 1/sqrt(2W), which sails past a loose tolerance.

So each case here asserts the PLAN first (that the program really is
double-buffered and really has more than one block on some core) and only then
the values — a pin that cannot silently stop testing what it names.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as _create_program_descriptor

PLAN_GLOBALS = _create_program_descriptor.__globals__

# Tall shapes: rows over-fill the grid, so a core owns several tile-rows and the
# block/depth ladder can afford a second input buffer.
DOUBLE_BUFFERED_SHAPES = [
    pytest.param((1, 1, 8192, 1024), id="prefill_w1024"),
    pytest.param((1, 1, 4096, 2048), id="tall_w2048"),
    pytest.param((1, 1, 2048, 2048), id="tall_w2048_half"),
]

# Tall, but the depth-2 block would be shorter than one in-flight NoC burst
# (DM_CHUNK_TILES), so the ladder must NOT take the deeper rung: it would buy one
# hidden read of a few tiles at the price of a whole extra set of per-block fixed
# costs.  Measured as a real regression on (1,1,2048,1024) before the guard.
NOT_WORTH_PREFETCHING = [
    pytest.param((1, 1, 2048, 1024), id="prefill_2048x1024"),
    pytest.param((1, 1, 1024, 512), id="tall_w512"),
    pytest.param((1, 1, 4064, 160), id="resilience_4064x160"),
]


def _plan_for(device, tensor, has_gamma):
    tile = ttnn.tile_size(tensor.dtype)
    bytes_ = {
        "in_tile": tile,
        "out_tile": tile,
        "gamma_tile": tile,
        "stat_tile": ttnn.tile_size(ttnn.float32),
        "bf16_tile": ttnn.tile_size(ttnn.bfloat16),
    }
    return PLAN_GLOBALS["_plan"](device, tensor, has_gamma=has_gamma, bytes_=bytes_)


@pytest.mark.parametrize("shape", DOUBLE_BUFFERED_SHAPES)
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_double_buffered_input_keeps_the_scale(device, shape, with_gamma):
    torch.manual_seed(7)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gamma = (
        ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if with_gamma
        else None
    )

    # ---- the pin: this shape must actually take the double-buffer rung ----
    plan = _plan_for(device, x, with_gamma)
    base, rem = divmod(plan["row_tiles"], plan["num_row_groups"])
    max_rows = base + (1 if rem else 0)
    assert plan["in_depth"] > 1, f"{shape}: expected a double-buffered input, plan={plan}"
    assert max_rows > plan["block_rows"] or max_rows >= plan["in_depth"], (
        f"{shape}: the second buffer has nothing to prefetch into "
        f"(max_rows={max_rows}, block_rows={plan['block_rows']})"
    )

    out = ttnn.to_torch(rms_norm(x, gamma=gamma)).to(torch.float32)

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    if with_gamma:
        expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rel_rms = ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()
    # A block that packed at the wrong base keeps its VALUES and loses its 1/rms
    # factor, so the error lands at the row-RMS spread ~ 1/sqrt(2W) — bound well
    # under that, not at some generic "close enough".
    dropped_scale_signature = (2.0 * shape[-1]) ** -0.5
    assert pcc > 0.999, f"{shape} gamma={with_gamma}: PCC {pcc}"
    assert rel_rms < 0.25 * dropped_scale_signature, (
        f"{shape} gamma={with_gamma}: rel-RMS {rel_rms:.4f} is within reach of the "
        f"dropped-scale signature {dropped_scale_signature:.4f}"
    )


@pytest.mark.parametrize("shape", NOT_WORTH_PREFETCHING)
def test_rms_norm_short_block_declines_the_second_buffer(device, shape):
    """A block shorter than one NoC burst must not split further for overlap."""
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = _plan_for(device, x, True)
    assert plan["in_depth"] == 1, f"{shape}: short block took the double-buffer rung, plan={plan}"


def test_rms_norm_depth_one_plan_is_unchanged_for_decode(device):
    """The decode regime must stay byte-identical to Refinement 3.

    One tile-row per core means `max_rows_full == 1`, so there is no block b+1 to
    prefetch and the ladder must skip the deeper rung entirely rather than spend
    L1 on a buffer that overlaps nothing.
    """
    for shape in [(1, 1, 32, 7168), (1, 1, 32, 1024), (1, 1, 64, 12288)]:
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        plan = _plan_for(device, x, True)
        assert plan["in_depth"] == 1, f"{shape}: decode must not take the double-buffer rung, plan={plan}"
