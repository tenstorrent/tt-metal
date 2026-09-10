# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-head pad cleanup preserves every valid row and unrelated user/layer."""

import torch

import ttnn
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 32 * 1024 * 1024})
def test_zero_padding_all_heads(mesh_device):
    def scalar(value):
        return ttnn.from_torch(
            torch.tensor([value]).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    for heads in (1, 4):
        for layout, dtype in ((ttnn.TILE_LAYOUT, ttnn.bfloat8_b), (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)):
            cache = ttnn.from_torch(
                torch.ones(4, heads, 4096, 64, dtype=torch.bfloat16),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            slot, end = scalar(0), scalar(0)
            traced = layout == ttnn.TILE_LAYOUT

            def zero(slot_value, end_value):
                return (
                    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                        cache,
                        slot_value,
                        end_value,
                        layer_idx=1,
                        num_layers=2,
                        chunk_size_global=8192,
                        cluster_axis=0,
                        pad_align=128,
                    )
                    if traced
                    else ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                        cache,
                        slot_idx=slot_value,
                        valid_global=end_value,
                        layer_idx=1,
                        num_layers=2,
                        chunk_size_global=8192,
                        cluster_axis=0,
                        pad_align=128,
                    )
                )

            trace = None
            if traced:
                zero(slot, end)
                ttnn.synchronize_device(mesh_device)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                zero(slot, end)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                windows = {}
                for user, valid in ((0, 7000), (1, 8999), (0, 32768)):
                    if traced:
                        for target, value in ((slot, user), (end, valid)):
                            host = ttnn.from_torch(
                                torch.tensor([value]).reshape(1, 1, 1, 1),
                                dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                            )
                            ttnn.copy_host_to_device_tensor(host, target)
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    else:
                        zero(user, valid)
                        ttnn.synchronize_device(mesh_device)
                    windows.setdefault(user * 2 + 1, []).append((valid, (valid + 127) // 128 * 128))
                    for rank in range(8):
                        actual = ttnn.to_torch(ttnn.get_device_tensors(cache)[rank * 4]).float()
                        expected = torch.ones_like(actual)
                        local = torch.arange(4096)
                        positions = local // 1024 * 8192 + rank * 1024 + local % 1024
                        for batch, spans in windows.items():
                            for begin, stop in spans:
                                expected[batch, :, (positions >= begin) & (positions < stop)] = 0
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh_device, trace)
                for tensor in (cache, slot, end):
                    tensor.deallocate(True)
