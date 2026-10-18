# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit cache stripes stay fixed when chunk lengths and trace metadata change."""

import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention.ring_prefill import init_sliding_ring_kv_cache
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata


@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 4 * 1024 * 1024})
def test_striped_cache_update(mesh_device):
    mesh = MeshConfig(mesh_device)
    capacity, heads, width = 2048, 8, 64
    cache = init_sliding_ring_kv_cache(mesh, heads // 4, width, capacity, num_users=4).k
    metadata = PrefillMetadata(mesh)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, (8, 4), dims=(2, 1))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, (8, 4), dims=(2, 1))
    cache_positions = torch.tensor(
        [g + r * 32 + t for r in range(8) for g in range(0, capacity, 256) for t in range(32)]
    )
    expected = torch.zeros(4, heads, capacity, width)
    trace_id = None
    source = None
    try:
        for index, (length, start, slot, valid) in enumerate(
            ((512, 256, 0, 768), (768, 256, 1, 961), (768, 1024, 0, 1600))
        ):
            natural = (
                (torch.arange(heads * length * width).reshape(1, heads, length, width) + index) % 15 - 7
            ).float() / 8
            positions = torch.tensor(
                [g + r * 32 + t for r in range(8) for g in range(0, length, 256) for t in range(32)]
            )
            host = ttnn.from_torch(
                natural[:, :, positions].contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            metadata.update(slot_idx=slot, kv_actual_global=start, valid_global=valid)
            if index == 2:
                ttnn.copy_host_to_device_tensor(host, source)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            else:
                if source is not None:
                    source.deallocate(True)
                source = ttnn.to_device(host, mesh_device)
                kwargs = dict(
                    slot_idx=metadata.slot_idx,
                    kv_actual_global=metadata.kv_actual_global,
                    valid_global=metadata.valid_global,
                    layer_idx=1,
                    num_layers=2,
                    cluster_axis=0,
                    stripe_size=32,
                )
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(cache, source, **kwargs)
                if index == 1:
                    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(cache, source, **kwargs)
                    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            written = ((valid + 31) // 32) * 32 - start
            expected[slot * 2 + 1, :, start : start + written] = natural[0, :, :written]
            actual = ttnn.to_torch(cache, mesh_composer=composer)[:, :, torch.argsort(cache_positions)]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
