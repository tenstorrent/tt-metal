# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The 2D-blocked matmul on device: [M, K] @ [K, N] for shapes that do not fit L1 whole.

unified_kernels/matmul_blocked.cpp walks the output in [mt, nt] blocks and K in blocks of kt,
gathering every operand with a custom load. This is what a layer's four large matmuls all
are -- the output projection at K = N = d_model, gate and up at N = ffn, and down at K = ffn.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_matmul_blocked.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import core_block, make_cb, split_evenly, unified_program

KERNEL = "unified_kernels/matmul_blocked.cpp"
CB_A, CB_B, CB_OUT, CB_ACC = 0, 1, 16, 24
TILE = 32


def run(device, mtot, ktot, ntot, mt, kt, nt, cores=1, seed=0, fidelity=None, acc="l1", a=None, b=None):
    """Extents and block widths all in TILES. a/b let a caller supply the operands."""
    assert mtot % mt == 0 and ktot % kt == 0 and ntot % nt == 0
    torch.manual_seed(seed)
    if a is None:
        a = (torch.rand([mtot * TILE, ktot * TILE]) - 0.5).to(torch.bfloat16)
    if b is None:
        b = ((torch.rand([ktot * TILE, ntot * TILE]) - 0.5) / (ktot * TILE) ** 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )

    ta, tb = to_dev(a), to_dev(b)
    # NaN-filled: the allocator reuses addresses, so an output block nothing writes would
    # otherwise hold a previous run's values and a missed block would pass.
    tout = to_dev(torch.full([mtot * TILE, ntot * TILE], float("nan")))

    # M-blocks across cores: rows of the output, independent, no reduction needed.
    nblocks = mtot // mt
    ncores = min(cores, nblocks)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(nblocks, ncores)

    ct_args = [mt, ktot, ntot, kt, nt]
    for t in (ta, tb, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    addrs = [t.buffer_address() for t in (ta, tb, tout)]
    rt_args = {c: addrs + [begin, count] for c, (begin, count) in zip(core_list, shares)}

    cbs = [
        make_cb(CB_A, core_ranges, num_pages=mt * kt),
        make_cb(CB_B, core_ranges, num_pages=kt * nt),
        make_cb(CB_OUT, core_ranges, num_pages=mt * nt),
    ] + ([make_cb(CB_ACC, core_ranges, num_pages=mt * nt)] if kt != ktot else [])

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=core_list,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        defines=[("MMB_ACC_DST", "1")] if acc == "dst" else None,
        **(fidelity or {}),
    )
    logger.info(
        f"blocked matmul [{mtot}x{ktot}]@[{ktot}x{ntot}]t  mt={mt} kt={kt} nt={nt} "
        f"(mb={nblocks} kb={ktot // kt} nb={ntot // nt}) cores={ncores}"
    )
    out = ttnn.generic_op([ta, tb, tout], program)
    for t in (ta, tb):
        ttnn.deallocate(t)
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]
    return got, a.to(torch.float32) @ b.to(torch.float32)


def pcc(got, want):
    return torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pcc", type=float, default=0.99)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        # Small and square, every blocking of the same matmul. All must agree: the blocking
        # is a decomposition, not an approximation.
        for mt, kt, nt in ((4, 8, 8), (2, 4, 8), (4, 2, 2), (1, 8, 8), (2, 8, 1), (1, 1, 1)):
            got, want = run(device, 4, 8, 8, mt, kt, nt)
            v = pcc(got, want)
            ok = v >= args.pcc
            logger.info(f"  4x8 @ 8x8  mt={mt} kt={kt} nt={nt}: pcc={v:.6f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"small-{mt}-{kt}-{nt}")

        # Non-square, which is what the FFN needs and what the square projection could not
        # express: K and N differ.
        for mtot, ktot, ntot, mt, kt, nt in (
            (4, 8, 16, 2, 4, 8),
            (4, 16, 8, 2, 8, 4),
            (8, 4, 12, 4, 2, 6),
        ):
            got, want = run(device, mtot, ktot, ntot, mt, kt, nt)
            v = pcc(got, want)
            ok = v >= args.pcc
            logger.info(
                f"  {mtot}x{ktot} @ {ktot}x{ntot}  mt={mt} kt={kt} nt={nt}: pcc={v:.6f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"rect-{mtot}-{ktot}-{ntot}")

        # The real llama-3.2-1B FFN shapes at S=512: d_model 2048, ffn 8192. B is 16384 tiles
        # (32MB) for the gate and up matrices, so these exist only because both dimensions
        # are blocked.
        for name, ktot, ntot in (("gate/up", 64, 256), ("down", 256, 64)):
            got, want = run(device, 16, ktot, ntot, 8, 8, 16)
            v = pcc(got, want)
            ok = v >= args.pcc
            logger.info(
                f"  FFN {name}: [512x{ktot * 32}]@[{ktot * 32}x{ntot * 32}]  pcc={v:.6f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"ffn-{name}")

        # Partitioned across cores.
        for ncores in (2, 4, 8):
            got, want = run(device, 8, 8, 8, 1, 8, 8, cores=ncores)
            v = pcc(got, want)
            ok = v >= args.pcc
            logger.info(f"  multicore cores={ncores}: pcc={v:.6f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"mc-{ncores}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
