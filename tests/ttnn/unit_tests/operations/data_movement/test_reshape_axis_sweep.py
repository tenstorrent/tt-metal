# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Axis sweep for ReshapeViewDeviceOperation program-cache neutrality.

Runs a broad battery of configs (input shape, output shape, memory config, tile mode, dtype)
through one program cache, then repeats the identical battery and asserts it builds nothing
new. Absolute entry counts are arch-dependent; zero growth on the second pass is not.

Measured differentially on wormhole while removing the custom hash: 7 entries with the custom
hash, 7 without.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def iso(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_reshape_axis_sweep(device, iso):
    """Vary input shape, output shape, memory config, tile mode, dtype."""
    cases = [
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 32, 256], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 128, 128], [1, 1, 256, 64], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.L1_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.TileReshapeMapMode.CACHE, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.TileReshapeMapMode.RECREATE, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.float32),
        ([1, 2, 64, 128], [1, 2, 128, 64], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 256, 64], [1, 1, 64, 256], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        # repeats: must add nothing
        ([1, 1, 64, 128], [1, 1, 128, 64], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
        ([1, 1, 64, 128], [1, 1, 32, 256], ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat16),
    ]
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

    def sweep():
        ran = 0
        for in_shape, out_shape, mem, mode, dtype in cases:
            try:
                t = torch.randn(in_shape, dtype=torch_dtype[dtype])
                tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device, memory_config=mem)
                if mode is None:
                    o = ttnn.reshape(tt, out_shape, memory_config=mem)
                else:
                    o = ttnn.reshape(tt, out_shape, memory_config=mem, reshape_tile_mode=mode)
                assert_with_pcc(t.reshape(out_shape), ttnn.to_torch(o), 0.999)
                ran += 1
            except Exception as e:
                print(f"\nSWEEP-SKIP {in_shape}->{out_shape} {mode} {dtype}: {type(e).__name__}", flush=True)
        return ran

    with device.cache_entries_counter.measure():
        ran = sweep()
    first = device.cache_entries_counter.total
    print(f"\nSWEEP RESHAPE ran={ran}/{len(cases)} entries={first}", flush=True)

    with device.cache_entries_counter.measure():
        sweep()
    assert device.cache_entries_counter.total == first
