# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the DRAMZeroFill micro op.

Allocates a DRAM tensor with the exact KV cache NdShardSpec configuration,
runs the zero-fill kernel, reads the result back to host, and verifies that
every element is zero and that the tensor topology matches the reference
produced by ShardTensor2dMesh.
"""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

KVPE_DIM = 576


def _build_reference_kv_tensor(submesh, num_users, max_seq_len):
    """Create a KV cache via the original from_torch + ShardTensor2dMesh path."""
    mesh_rows = submesh.shape[0]
    mesh_cols = submesh.shape[1]
    per_device_seq = max_seq_len // mesh_rows

    program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128, exp_approx_mode=False)
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, KVPE_DIM],
        grid=program_config.grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    kv_cache_2d_mesh_mapper = ttnn.ShardTensor2dMesh(submesh, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None))

    torch_kv = torch.zeros((num_users, 1, max_seq_len, KVPE_DIM), dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_kv,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem,
        mesh_mapper=kv_cache_2d_mesh_mapper,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("max_seq_len", [1024 * 32, 1024 * 64, 1024 * 128])
@pytest.mark.parametrize("num_users", [1, 32, 64])
@pytest.mark.requires_grid_size((12, 10))
def test_dram_zero_fill(bh_2d_mesh_device, num_users: int, max_seq_len: int) -> None:
    """Zero-fill a KV-cache-shaped DRAM tensor and verify all zeros."""
    if is_slow_dispatch() and (num_users > 1 or max_seq_len > 1024 * 32):
        pytest.skip("Host readback (ttnn.to_torch) for this shape is too slow in slow dispatch mode")

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires 8 devices (4x2 mesh)")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    mesh_rows = submesh.shape[0]
    mesh_cols = submesh.shape[1]

    # Poison DRAM with random data so that a no-op kernel can't pass by luck.
    per_device_seq = max_seq_len // mesh_rows
    poison = ttnn.from_torch(
        torch.randn(num_users, 1, per_device_seq, KVPE_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn.deallocate(poison, force=True)

    # Warmup (includes kernel compilation)
    t0 = time.perf_counter()
    warmup_tensor = DRAMZeroFill.allocate_kv_cache_on_device(
        submesh,
        num_users=num_users,
        max_seq_len=max_seq_len,
        kvpe_dim=KVPE_DIM,
        mesh_shape=(mesh_rows, mesh_cols),
    )
    ttnn.synchronize_device(submesh)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    ttnn.deallocate(warmup_tensor, force=True)

    # Real run (cached kernel)
    t0 = time.perf_counter()
    output_tensor = DRAMZeroFill.allocate_kv_cache_on_device(
        submesh,
        num_users=num_users,
        max_seq_len=max_seq_len,
        kvpe_dim=KVPE_DIM,
        mesh_shape=(mesh_rows, mesh_cols),
    )
    ttnn.synchronize_device(submesh)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"\nDRAMZeroFill  num_users={num_users}  max_seq_len={max_seq_len}  shape={list(output_tensor.shape)}"
        f"  warmup={warmup_ms:.2f} ms  run={elapsed_ms:.2f} ms"
    )

    # Verify data is all zeros
    result = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    expected = torch.zeros_like(result)
    assert torch.equal(result, expected), (
        f"Non-zero values found: max abs = {result.abs().max().item()}, "
        f"non-zero count = {result.count_nonzero().item()} / {result.numel()}"
    )

    # Verify topology matches reference from_torch + ShardTensor2dMesh path
    ref_tensor = _build_reference_kv_tensor(submesh, num_users, max_seq_len)

    ref_topo = ref_tensor.tensor_topology()
    out_topo = output_tensor.tensor_topology()

    assert (
        out_topo.distribution_shape() == ref_topo.distribution_shape()
    ), f"distribution_shape mismatch: {out_topo.distribution_shape()} != {ref_topo.distribution_shape()}"
    out_placements = [str(p) for p in out_topo.placements()]
    ref_placements = [str(p) for p in ref_topo.placements()]
    assert out_placements == ref_placements, f"placements mismatch: {out_placements} != {ref_placements}"

    out_coords = [str(c) for c in out_topo.mesh_coords()]
    ref_coords = [str(c) for c in ref_topo.mesh_coords()]
    assert out_coords == ref_coords, f"mesh_coords mismatch: {out_coords} != {ref_coords}"

    # Verify memory config matches
    assert (
        output_tensor.memory_config() == ref_tensor.memory_config()
    ), f"memory_config mismatch:\n  got:  {output_tensor.memory_config()}\n  want: {ref_tensor.memory_config()}"

    ttnn.deallocate(ref_tensor, force=True)
