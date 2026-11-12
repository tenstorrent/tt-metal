# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("k,n", [(1024, 1280), (8192, 1024)])
def test_dram_sharded_matmul_preserves_nonparticipant_l1(per_core_mesh_device, k, n):
    device = per_core_mesh_device
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Blackhole DRAM worker placement regression")
    storage = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    workers = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = max(7, *(c.x for c in workers)), max(0, *(c.y for c in workers))
    quiet = next(((x, y) for y in range(grid.y) for x in range(grid.x) if x > max_x or y > max_y), None)
    if quiet is None:
        pytest.skip("The program rectangle covers the compute grid")
    quiet_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*quiet), ttnn.CoreCoord(*quiet))})
    dram = device.dram_grid_size()
    banks = dram.x * dram.y
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    torch.manual_seed(0)
    a = torch.randn((1, 1, 32, k), dtype=torch.bfloat16)
    b = torch.randn((1, 1, k, n), dtype=torch.bfloat16)

    def memory(layout, buffer_type, cores, shape):
        return ttnn.MemoryConfig(layout, buffer_type, ttnn.ShardSpec(cores, shape, ttnn.ShardOrientation.ROW_MAJOR))

    in0 = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, storage, [32, k // 8]),
    )
    in1 = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            dram_grid,
            [k, (n // 32 + banks - 1) // banks * 32],
        ),
    )
    out_memory = memory(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, storage, [32, n // 8])
    out = ttnn.from_torch(
        torch.zeros((1, 1, 32, n), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_memory,
    )
    live = []
    sentinel = 0x5A5A5A5A
    try:
        for words in (8192, 256, 16):
            config = memory(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, quiet_grid, [1, words])
            config.experimental_set_per_core_allocation(True)
            while True:
                try:
                    live.append(
                        ttnn.from_torch(
                            torch.full((1, words), sentinel, dtype=torch.int32),
                            dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device,
                            memory_config=config,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                        )
                    )
                except RuntimeError:
                    break
        assert live, "No sentinel was allocated on the nonparticipant core"
        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k // 8 // 32,
            per_core_M=1,
            per_core_N=n // 8 // 32,
            fused_activation=None,
        )
        compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        result = ttnn.matmul(
            in0,
            in1,
            program_config=program,
            memory_config=out_memory,
            optional_output_tensor=out,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )
        got = ttnn.to_torch(result).float()
        expected = a.float() @ b.float()
        assert (got - expected).norm() / expected.norm() < 0.02
        changed = sum(int((ttnn.to_torch(tensor) != sentinel).sum()) for tensor in live)
        assert changed == 0, f"Matmul changed {changed} words on nonparticipant core {quiet}"
    finally:
        for tensor in live:
            ttnn.deallocate(tensor)
