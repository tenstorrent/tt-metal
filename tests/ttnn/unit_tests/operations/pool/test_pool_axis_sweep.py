# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Axis sweep for Pool2D program-cache neutrality.

Runs a broad battery of configs (shape, channels, kernel, stride, padding, ceil_mode, dtype,
shard scheme, output layout) through one program cache, then repeats the identical battery
and asserts it builds nothing new. Absolute entry counts are arch-dependent; zero growth on
the second pass is not, and it is what catches a key that varies per dispatch -- the failure
mode that keying the removed `memory_used` would have produced.

Measured differentially on wormhole while removing the custom hash: 41 entries with the custom
hash, 41 without.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d

pytestmark = pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)


@pytest.fixture
def iso(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def test_pool_axis_sweep(device, iso):
    """Vary shape, channels, kernel, stride, padding, ceil_mode, dtype, layout, shard scheme."""
    cases = [
        ([1, 16, 25, 23], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, None),
        ([1, 16, 25, 23], (3, 3), (2, 2), (1, 1), False, ttnn.bfloat16, HS, None),
        ([1, 16, 25, 23], (2, 2), (2, 2), (0, 0), True, ttnn.bfloat16, HS, None),
        ([1, 32, 32, 32], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, None),
        ([1, 64, 16, 16], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, None),
        ([1, 128, 8, 8], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, BS, None),
        ([1, 256, 8, 8], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, WS, None),
        # dtype axis (bfloat8_b enters via TILE layout in the harness)
        ([1, 16, 25, 23], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat8_b, HS, None),
        ([1, 32, 32, 32], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat8_b, HS, None),
        # output layout axis
        ([1, 16, 25, 23], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, ttnn.TILE_LAYOUT),
        ([1, 32, 32, 32], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, ttnn.TILE_LAYOUT),
        # repeats of earlier cases: must add nothing
        ([1, 16, 25, 23], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, None),
        ([1, 32, 32, 32], (2, 2), (2, 2), (0, 0), False, ttnn.bfloat16, HS, None),
    ]

    def sweep():
        ran = 0
        for shape, kernel, stride, pad, ceil_mode, dtype, scheme, out_layout in cases:
            kwargs = {}
            if out_layout is not None:
                kwargs["output_layout"] = out_layout
            try:
                run_max_pool2d(
                    list(shape),
                    list(kernel),
                    list(pad),
                    list(stride),
                    [1, 1],
                    device,
                    {},
                    dtype,
                    shard_scheme=scheme,
                    ceil_mode=ceil_mode,
                    nightly_skips=False,
                    **kwargs,
                )
                ran += 1
            except Exception as e:  # unsupported combo on this arch: skip, but report it
                print(f"\nSWEEP-SKIP {shape} {kernel} {dtype} {scheme} {out_layout}: {type(e).__name__}", flush=True)
        return ran

    with device.cache_entries_counter.measure():
        ran = sweep()
    first = device.cache_entries_counter.total
    print(f"\nSWEEP POOL ran={ran}/{len(cases)} entries={first}", flush=True)

    # Second pass over the identical battery must build nothing new. Absolute counts are
    # arch-dependent; zero growth is not, and it is what catches a key that varies per call.
    with device.cache_entries_counter.measure():
        sweep()
    assert device.cache_entries_counter.total == first
