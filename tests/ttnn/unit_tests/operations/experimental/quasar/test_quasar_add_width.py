# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Element-WIDTH sweep for ttnn.experimental.quasar.add_ — the controlled memory-vs-compute discriminator.

BLOCKED ON QUASAR (2026-06-25): the bf8_b leg cannot run. The DFB Metal-2.0 fast path only admits
bf16/bf8_b (matches_metal_v2_slice's is_bf8_or_bf16), but bf8_b == DataFormat::Bfp8_b is NOT in
is_supported_quasar (tt_backend_api_types.cpp), so it TT_FATALs at program_spec.cpp:1496 before any
kernel runs (empty CSV). No second element width satisfies both gates, so the byte-varying experiment
is un-runnable on Quasar today. Kept for when a supported sub-2-byte width lands. Use the roofline mode
of tools/quasar_profiling/analyze_add_profile.py instead (the add's intensity pins it memory-bound).


An elementwise add does the SAME number of FPU ops per tile regardless of element width, but moves
~2x the bytes in bfloat16 (2 B/elem) vs bfloat8_b (1 B/elem). Both stay on the SAME Metal-2.0 DFB
fast path (matches_metal_v2_slice accepts bf8_b and bf16 — fp32 falls off to a different kernel, so
it is NOT usable for a controlled comparison; bf8_b-vs-bf16 is).

So if a per-role cycles/tile SLOPE scales with byte width (≈2x bf16 vs bf8_b) → that role is
MEMORY-bound; if it is ≈flat across widths → COMPUTE-bound. This sidesteps the stall-contamination
that makes the single-width slope ambiguous.

Single core, height-sharded, num_tiles tiles tall on one core (mirrors test_quasar_add_multitile.py).
Parametrized over (dtype, num_tiles).

Run with the profiler (one invocation per (dtype, num_tiles); see tools/quasar_profiling/run_width_sweep.sh).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# bf8_b carries a shared exponent per 16-elem block, so its reconstruction PCC is looser than bf16.
_PCC = {
    ttnn.bfloat16: 0.9997,
    ttnn.bfloat8_b: 0.992,
}


def _height_sharded_config(num_tiles):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))})
    shard = [num_tiles * 32, 32]  # num_tiles tiles tall, on one core
    return ttnn.create_sharded_memory_config(
        shard,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize("num_tiles", [1, 4, 16, 64])
def test_quasar_add_width(device, dtype, num_tiles):
    torch.manual_seed(0)
    shape = torch.Size([num_tiles * 32, 32])
    a_pt = torch.randn(shape, dtype=torch.bfloat16)
    b_pt = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = _height_sharded_config(num_tiles)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    out_tt = ttnn.experimental.quasar.add_(a_tt, b_tt, activations=[])

    golden = torch.add(a_pt, b_pt)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC[dtype])
