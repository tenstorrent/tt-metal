# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mcast_in0", [True, False])
@pytest.mark.parametrize("work_cores", [2, 3])
def test_matmul_1d_multicast_preserves_live_tail(per_core_mesh_device, mcast_in0, work_cores):
    device = per_core_mesh_device
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Blackhole multicast storage regression")
    grid = device.compute_with_storage_grid_size()
    if grid.x < 3 or grid.y < 2:
        pytest.skip("Requires a 3 by 2 core region")
    m, n = (32, work_cores * 64) if mcast_in0 else (work_cores * 32, 64)
    torch.manual_seed(7)
    a = torch.randn(m, 256).bfloat16()
    b = (torch.randn(256, n) * 0.02).bfloat16()
    x = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        in0_block_w=4,
        per_core_M=1,
        per_core_N=2,
        out_block_h=1,
        out_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        fuse_batch=False,
        mcast_in0=mcast_in0,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    # Keep a real persistent prefetch ring alive on the unused multicast tail and
    # an outside control. These cores are not inputs or outputs of this matmul.
    idle, outside = ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)
    one_core = lambda core: ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=one_core(outside),
        receiver_domain=one_core(idle),
        ring_size=65536,
        max_receivers_per_pipe=1,
    )
    pipe = space.create_pipe(outside, one_core(idle))
    sentinel = bytes([0x5A]) * pipe.ring_size()
    physical = [device.worker_core_from_logical_core(core) for core in (idle, outside)]
    chip = device.get_device_ids()[0]
    for core in physical:
        ttnn.cluster.write_to_core(chip, core.x, core.y, pipe.buffer_address(), sentinel)
    result = ttnn.matmul(
        x,
        y,
        program_config=program,
        dtype=ttnn.float32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute,
    )
    got = ttnn.to_torch(result)
    expected = a.float() @ b.float()
    assert (got - expected).norm() / expected.norm() < 0.01
    for logical, core in zip((idle, outside), physical):
        actual = ttnn.cluster.read_from_core(chip, core.x, core.y, pipe.buffer_address(), pipe.ring_size())
        changed = sum(actual[i : i + 4] != sentinel[i : i + 4] for i in range(0, len(sentinel), 4))
        assert changed == 0, f"Matmul changed {changed} words in the live prefetch ring on {logical}"
