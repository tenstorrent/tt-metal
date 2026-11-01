# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole llk_pack_block through the metal API (pack_block), output checked against the input as golden.

A compute kernel copies tiles to DEST and packs them in blocks of N tiles with pack_block. The cells cover the forms llk_pack_block takes: the first block of an output
(plain block, or the per-tile pack for Float32 up to 7 tiles and 4 tile block-float blocks), the following blocks (W offset), with and without a pack init before every block
(fused kernels between ops), half and full DEST sync, 16 and 32 bit DEST.
"""

import pytest
import torch

import ttnn

COMPUTE_SOURCE = """
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t in_cb = 0;
    constexpr uint32_t out_cb = 16;
    constexpr uint32_t ntiles = get_compile_time_arg_val(0);
    constexpr uint32_t nblocks = get_compile_time_arg_val(1);
    constexpr uint32_t reinit = get_compile_time_arg_val(2);
    constexpr uint32_t total = ntiles * nblocks;

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    compute_kernel_hw_startup(in_cb, out_cb);
    copy_init(in_cb);
    cb_in.wait_front(total);
    cb_out.reserve_back(total);
    for (uint32_t b = 0; b < nblocks; b++) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < ntiles; i++) {
            copy_tile(in_cb, b * ntiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (reinit) {
            PACK((llk_pack_init(out_cb)));
        }
        pack_block(0, out_cb, ntiles);
        tile_regs_release();
    }
    cb_out.push_back(total);
    cb_in.pop_front(total);
}
"""

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t total = get_compile_time_arg_val(1);
    cb_reserve_back(in_cb, total);
    cb_push_back(in_cb, total);
}
"""

TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}

# (input dtype, output dtype, fp32 DEST, full DEST sync)
CONFIGS = [
    (ttnn.bfloat16, ttnn.bfloat16, False, False),
    (ttnn.bfloat16, ttnn.bfloat16, False, True),
    (ttnn.float32, ttnn.float32, True, True),
    (ttnn.bfloat16, ttnn.float32, False, False),
    (ttnn.bfloat16, ttnn.float32, False, True),
    (ttnn.bfloat16, ttnn.bfloat8_b, False, False),
    (ttnn.bfloat16, ttnn.bfloat8_b, False, True),
    (ttnn.float32, ttnn.bfloat8_b, True, True),
    (ttnn.float32, ttnn.float32, True, False),
    (ttnn.int32, ttnn.int32, True, True),
    (ttnn.int32, ttnn.int32, True, False),
]


NTILES = [4, 5, 6, 7, 8, 10, 12]


def fits(fp32_dest, full_sync, ntiles):
    """DEST tiles of the section: 8 half sync and 16 full sync for a 16 bit DEST, half of that for a 32 bit DEST."""
    return ntiles <= ((8 if full_sync else 4) if fp32_dest else (16 if full_sync else 8))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_case(device, in_dt, out_dt, fp32_dest, full_sync, ntiles, nblocks, reinit):
    total = ntiles * nblocks
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    torch.manual_seed(ntiles * 7 + nblocks)
    if in_dt == ttnn.int32:
        x = torch.randint(-100000, 100000, (1, 1, 32, 32 * total), dtype=torch.int32)
    else:
        x = torch.rand(1, 1, 32, 32 * total) * 2 - 1

    def mem(dt):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(core, (32, 32 * total), ttnn.ShardOrientation.ROW_MAJOR)
        )

    t_in = ttnn.from_torch(x, dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(in_dt))
    t_out = ttnn.from_torch(torch.zeros_like(x), dtype=out_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(out_dt))
    in_cb = ttnn.cb_descriptor_from_sharded_tensor(0, t_in, core_ranges=core)
    out_cb = ttnn.cb_descriptor_from_sharded_tensor(16, t_out, core_ranges=core)
    dm = ttnn.KernelDescriptor(
        kernel_source=DM_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[0, total],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default),
    )
    cc = ttnn.ComputeConfigDescriptor()
    cc.fp32_dest_acc_en = fp32_dest
    cc.dst_full_sync_en = full_sync
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    if in_dt in (ttnn.float32, ttnn.int32):
        cc.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    comp = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[ntiles, nblocks, int(reinit)],
        config=cc,
    )
    pd = ttnn.ProgramDescriptor(cbs=[in_cb, out_cb], kernels=[dm, comp])
    mesh_pd = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
    ttnn.generic_op([t_in, t_out], mesh_pd)
    ttnn.synchronize_device(device)
    golden = ttnn.to_torch(t_in).double()
    got = ttnn.to_torch(t_out).double()
    return golden, got


@pytest.mark.parametrize("device", [(1,)], indirect=True)
@pytest.mark.parametrize("in_dt,out_dt,fp32_dest,full_sync", CONFIGS)
@pytest.mark.parametrize("ntiles", NTILES)
@pytest.mark.parametrize("reinit", [False, True])
def test_pack_block_api(device, in_dt, out_dt, fp32_dest, full_sync, ntiles, reinit):
    if not fits(fp32_dest, full_sync, ntiles):
        pytest.skip("the block does not fit the DEST section")
    golden, got = run_case(device, in_dt, out_dt, fp32_dest, full_sync, ntiles, 3, reinit)
    if in_dt == ttnn.int32:
        assert torch.equal(golden, got)
    elif out_dt == ttnn.bfloat8_b:
        assert pcc(golden, got) > 0.999
    elif out_dt == ttnn.float32 and in_dt == ttnn.bfloat16:
        assert torch.equal(golden, got)
    else:
        assert torch.equal(golden, got) if in_dt == out_dt == ttnn.bfloat16 else pcc(golden, got) > 0.99999
