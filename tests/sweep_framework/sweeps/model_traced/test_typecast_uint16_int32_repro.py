# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repro test for typecast UINT16 → INT32 PCC failures on mesh devices.

CI run #24231888365 showed 2 failures:
  - cb1abd306535: N300 (1x2), shape (1,1,32,64),  PCC 0.49
  - 2c4602d407c7: T3K  (1x8), shape (1,1,32,256), PCC 0.84

Both pass locally on single device and on 1x2 mesh.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


def get_mesh_device(mesh_shape):
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=79104,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )


def run_typecast_uint16_to_int32(device, shape, use_mesh_mapper=False, mesh_shape=None):
    torch.manual_seed(0)

    torch_input = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535).to(torch.int16)

    golden = torch_input.to(torch.int32) & 0xFFFF

    if use_mesh_mapper and mesh_shape is not None:
        dims_tuple = (2, None)
        mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=dims_tuple, mesh_shape=mesh_shape)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
    else:
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_output = ttnn.typecast(tt_input, ttnn.int32)

    if hasattr(device, "get_num_devices"):
        device_tensors = ttnn.get_device_tensors(tt_output)
        output = ttnn.to_torch(device_tensors[0])
    else:
        output = ttnn.to_torch(tt_output)

    golden_f32 = golden.to(torch.float32)
    output_f32 = output.to(torch.float32)

    passed, pcc_val = check_with_pcc(golden_f32, output_f32, 0.999)

    mismatch = (golden != output).sum().item()
    total = golden.numel()
    print(f"  Shape={shape}, mesh_mapper={use_mesh_mapper}, PCC={pcc_val}, mismatches={mismatch}/{total}")

    assert passed, f"PCC check failed: {pcc_val} (mismatches: {mismatch}/{total})"


class TestTypecastUint16Int32Repro:
    """Repro for CI typecast UINT16→INT32 failures on mesh devices."""

    def test_single_device(self):
        """Baseline: single device, no mesh mapper."""
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        try:
            run_typecast_uint16_to_int32(device, (1, 1, 32, 64))
            run_typecast_uint16_to_int32(device, (1, 1, 32, 256))
        finally:
            ttnn.close_device(device)

    def test_mesh_1x2_no_mapper(self):
        """Mesh 1x2, no mesh mapper (replicate path)."""
        device = get_mesh_device((1, 2))
        try:
            run_typecast_uint16_to_int32(device, (1, 1, 32, 64), use_mesh_mapper=False)
        finally:
            ttnn.close_mesh_device(device)

    def test_mesh_1x2_with_shard_mapper_shape_32x64(self):
        """Exact CI config: N300 1x2, ShardTensor2dMesh dims=(2, None), shape (1,1,32,64)."""
        device = get_mesh_device((1, 2))
        try:
            run_typecast_uint16_to_int32(device, (1, 1, 32, 64), use_mesh_mapper=True, mesh_shape=(1, 2))
        finally:
            ttnn.close_mesh_device(device)

    def test_mesh_1x2_with_shard_mapper_shape_32x256(self):
        """Exact CI config: shape (1,1,32,256) with ShardTensor2dMesh dims=(2, None) on 1x2."""
        device = get_mesh_device((1, 2))
        try:
            run_typecast_uint16_to_int32(device, (1, 1, 32, 256), use_mesh_mapper=True, mesh_shape=(1, 2))
        finally:
            ttnn.close_mesh_device(device)

    def test_mesh_1x2_replicate_mapper(self):
        """Mesh 1x2 with ReplicateTensorToMesh (all devices get same data)."""
        device = get_mesh_device((1, 2))
        try:
            torch.manual_seed(0)
            shape = (1, 1, 32, 64)
            torch_input = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535).to(torch.int16)
            golden = torch_input.to(torch.int32) & 0xFFFF

            tt_input = ttnn.from_torch(
                torch_input,
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            tt_output = ttnn.typecast(tt_input, ttnn.int32)
            device_tensors = ttnn.get_device_tensors(tt_output)
            output = ttnn.to_torch(device_tensors[0])

            golden_f32 = golden.to(torch.float32)
            output_f32 = output.to(torch.float32)
            passed, pcc_val = check_with_pcc(golden_f32, output_f32, 0.999)
            mismatch = (golden != output).sum().item()
            total = golden.numel()
            print(f"  Replicate mapper: PCC={pcc_val}, mismatches={mismatch}/{total}")
            assert passed, f"PCC check failed: {pcc_val}"
        finally:
            ttnn.close_mesh_device(device)

    def test_mesh_1x2_check_both_devices(self):
        """Check output from BOTH devices in the mesh to see if one is corrupted."""
        device = get_mesh_device((1, 2))
        try:
            torch.manual_seed(0)
            shape = (1, 1, 32, 64)
            torch_input = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535).to(torch.int16)
            golden = torch_input.to(torch.int32) & 0xFFFF

            mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=(2, None), mesh_shape=(1, 2))
            tt_input = ttnn.from_torch(
                torch_input,
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

            tt_output = ttnn.typecast(tt_input, ttnn.int32)
            device_tensors = ttnn.get_device_tensors(tt_output)

            for i, dt in enumerate(device_tensors):
                out = ttnn.to_torch(dt)
                golden_f32 = golden.to(torch.float32)
                out_f32 = out.to(torch.float32)
                passed, pcc_val = check_with_pcc(golden_f32, out_f32, 0.999)
                mismatch = (golden != out).sum().item()
                total = golden.numel()
                print(f"  Device {i}: shape={out.shape}, PCC={pcc_val}, mismatches={mismatch}/{total}")
                assert passed, f"Device {i} PCC check failed: {pcc_val} (mismatches: {mismatch}/{total})"
        finally:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
