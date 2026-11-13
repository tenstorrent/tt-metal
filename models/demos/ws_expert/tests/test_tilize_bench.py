# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tilize cost on one core: compute fast_tilize (with / without the matmul mode switch) vs data-movement tilize
(32 B face rows by local NoC reads or RISC copies). Checks the tiles against ttnn's tilize."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, cb, crs, kpath, read_stamps, stamps_tensor, to_dev
from models.demos.ws_expert.ws_expert import noc, sharded_l1


def tile_order(x):
    """[32, W*32] -> tiles (face layout) flattened, as the device stores them."""
    W = x.shape[1] // 32
    t = x.reshape(2, 16, W, 2, 16).permute(2, 0, 3, 1, 4)  # tile, face row, face col, r, c
    return t.reshape(W, 1024)


@pytest.mark.parametrize("W", [4, 8, 12])
@pytest.mark.parametrize("switch", [0, 1])
def test_compute_tilize(device, W, switch):
    REPS = 64
    grid = crs([(0, 0)])
    x = torch.randn(32, W * 32)
    # row-major block as W "tiles" worth of pages: pass it as a ROW_MAJOR tensor read page by page into CB 0
    xt = ttnn.from_torch(x.reshape(W, 1024), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ot = to_dev(torch.zeros(32, W * 32), ttnn.bfloat16, device)
    st = stamps_tensor(device, [(0, 0)])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [xt.buffer_address(), xt.buffer_address(), W, 0]
    wr[0][0] = [ot.buffer_address(), REPS * W, W, W]
    cp[0][0] = [st.buffer_address()]
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_reader.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(xt).get_compile_time_args()) * 2,
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_writer.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(ot).get_compile_time_args()),
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("tilize_bench_compute.cpp"),
                core_ranges=grid,
                compile_time_args=[W, REPS, switch],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
            ),
        ],
        semaphores=[],
        cbs=[cb(0, ttnn.bfloat16, W, grid), cb(1, ttnn.bfloat4_b, 4, grid), cb(16, ttnn.bfloat16, 2 * W, grid)],
    )
    ttnn.generic_op([xt, ot, st], prog)
    s = read_stamps(st, 1)[0]
    cyc = int((s[1] - s[0]) & 0xFFFFFFFF)
    out = ttnn.to_torch(ot).float()
    ok = torch.allclose(out, x.to(torch.bfloat16).float())
    logger.info(f"TILIZE compute W {W} switch {switch}: {cyc / (REPS * W):.1f} cyc/tile ({cyc / REPS:.0f} a block) ok {ok}")


@pytest.mark.parametrize("W", [8])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_dm_tilize(device, W, mode):
    REPS = 16
    core = (0, 0)
    x = torch.randn(32, W * 32)
    src = ttnn.from_torch(
        x.reshape(32 * W, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs([core]), [32 * W, 32], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    dst = sharded_l1(device, [core], [32 * W, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    st = stamps_tensor(device, [core])
    nx, ny = noc(device, *core)
    k = ttnn.KernelDescriptor(
        kernel_source=kpath("dm_tilize_bench.cpp"),
        core_ranges=crs([core]),
        compile_time_args=[W, REPS, src.buffer_address(), dst.buffer_address(), st.buffer_address(), mode, nx, ny],
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    ttnn.generic_op([src, dst, st], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    s = read_stamps(st, 1)[0]
    cyc = int((s[1] - s[0]) & 0xFFFFFFFF)
    out = ttnn.to_torch(dst).float().reshape(W * 32 * 32)
    want = tile_order(x.to(torch.bfloat16).float()).flatten()
    # dst as a TILE tensor converts back to row-major: compare against x itself
    ok = torch.allclose(ttnn.to_torch(dst).float(), torch.cat([x[:, t * 32 : t * 32 + 32] for t in range(W)], 0).to(torch.bfloat16).float())
    logger.info(f"TILIZE dm mode {mode} W {W}: {cyc / (REPS * W):.0f} cyc/tile ok {ok}")
