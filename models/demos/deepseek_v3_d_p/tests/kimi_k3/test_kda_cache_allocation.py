# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure persistent carry and slab allocations separately at production TP4 geometry."""

import json
from dataclasses import replace

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("layer_count", [1, 3])
@pytest.mark.parametrize("slots", [1, 2])
def test_cache_allocation(mesh_device, layer_count, slots):
    def allocated():
        ttnn.synchronize_device(mesh_device)
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return v.total_bytes_allocated_per_bank * v.num_banks

    layer = object.__new__(ttKDA)
    layer.device = mesh_device
    config = kimi_k3_kda_config()
    layer.config = replace(config, num_heads=config.num_heads // 4)
    geometry = KdaContractGeometry.from_kda_config(config, mesh_shape=(2, 4), sp_axis=0, tp_axis=1)
    # DRAMZeroFill keeps a small program-owned buffer. Warm it before measuring persistent model storage.
    warm_slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=tuple(range(layer_count)), num_slots=slots)
    ttnn.deallocate(warm_slabs.recurrent)
    ttnn.deallocate(warm_slabs.convolution)
    before = allocated()
    seed = layer.allocate_state()
    seed_bytes = allocated() - before
    ttnn.deallocate(seed.recurrent)
    ttnn.deallocate(seed.convolution)
    before = allocated()
    cache = KdaStateCache({i: layer for i in range(layer_count)}, num_slots=slots)
    cache_bytes = allocated() - before
    before_slabs = allocated()
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=tuple(range(layer_count)), num_slots=slots)
    cache.bind_slabs(slabs)
    slab_bytes = allocated() - before_slabs
    assert not hasattr(cache, "_zeros")
    print(
        "KDA_CACHE_ALLOCATION="
        + json.dumps(
            dict(
                layers=layer_count,
                slots=slots,
                state_bytes=seed_bytes,
                cache_bytes=cache_bytes,
                slab_bytes=slab_bytes,
                retained_zeros=False,
            )
        )
    )
    assert cache_bytes == seed_bytes * layer_count * slots
    cache.deallocate()
    ttnn.deallocate(slabs.recurrent)
    ttnn.deallocate(slabs.convolution)
    assert allocated() == before
