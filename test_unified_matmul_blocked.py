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


def run(
    device,
    mtot,
    ktot,
    ntot,
    mt,
    kt,
    nt,
    cores=1,
    seed=0,
    fidelity=None,
    acc="l1",
    a=None,
    b=None,
    mcast=False,
    in0_thread=1,
    in1_thread=0,
    depth=None,
    hoist=False,
    zones=False,
    wdtype=None,
    adtype=None,
    odtype=None,
    skew=0,
    share_pair=False,
):
    """Extents and block widths all in TILES. a/b let a caller supply the operands."""
    assert mtot % mt == 0 and ktot % kt == 0 and ntot % nt == 0
    torch.manual_seed(seed)
    if a is None:
        a = (torch.rand([mtot * TILE, ktot * TILE]) - 0.5).to(torch.bfloat16)
    if b is None:
        b = ((torch.rand([ktot * TILE, ntot * TILE]) - 0.5) / (ktot * TILE) ** 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape).to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )

    # The WEIGHT's format is the interesting one: B is much the larger operand, so halving
    # its bytes halves most of the traffic. A, the accumulator and the output stay bfloat16.
    wdtype = wdtype or ttnn.bfloat16
    # Activations are the harder call than weights: A is what a previous stage produced and
    # the output is what the next one consumes, so the format is a pipeline-wide decision
    # rather than a per-matmul one. The ACCUMULATOR stays bfloat16 regardless -- it carries
    # the running partial across k-blocks, which is exactly where narrowing compounds.
    adtype = adtype or ttnn.bfloat16
    odtype = odtype or ttnn.bfloat16
    ta, tb = to_dev(a, adtype), to_dev(b, wdtype)
    # NaN-filled: the allocator reuses addresses, so an output block nothing writes would
    # otherwise hold a previous run's values and a missed block would pass.
    tout = to_dev(torch.full([mtot * TILE, ntot * TILE], float("nan")), odtype)

    # The unit of work is one OUTPUT BLOCK, an (m, n) tile, indexed flat as m*nb + n. Both
    # dimensions are split across cores and neither needs a reduction -- different m or
    # different n write disjoint output. Splitting M alone made mt fight the core count.
    mb, nb = mtot // mt, ntot // nt
    nunits = mb * nb
    if mcast:
        # Multicast needs the strict mapping: core (r, c) owns block (r, c), so the grid IS
        # mb x nb and every core holds exactly one block. A broadcast is collective -- every
        # core in a row or column must make the same calls in the same order -- which a
        # split_evenly range cannot promise.
        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, mb - 1))])
        core_list = [ttnn.CoreCoord(x, y) for y in range(mb) for x in range(nb)]
        shares = [(0, 1)] * len(core_list)  # unused by the multicast path, but read
        ncores = len(core_list)
    else:
        ncores = min(cores, nunits)
        core_ranges, core_list = core_block(ncores)
        shares = split_evenly(nunits, ncores)

    ct_args = []
    named_ct_args = [("mt", mt), ("ktot", ktot), ("ntot", ntot), ("kt", kt), ("nt", nt)]
    for t in (ta, tb, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    addrs = [t.buffer_address() for t in (ta, tb, tout)]
    rt_args = {c: addrs + [begin, count] for c, (begin, count) in zip(core_list, shares)}

    # Default 2 with multicast and 1 without, which is measured rather than tidy. Depth pays
    # only when the read path is LATENCY bound: with multicast just 16 of 64 cores touch DRAM
    # and there is headroom for another read in flight (out-projection 350.2 -> 307.6us, FFN
    # gate/up 1482.7 -> 1329.5). Without it every core reads for itself, DRAM is already
    # saturated, and the extra requests only add contention -- 16 cores measured 807.8us at
    # depth 1, 830.1 at 2 and 842.9 at 3, i.e. steadily worse.
    depth = (2 if mcast else 1) if depth is None else depth

    # `depth` blocks per operand CB. The kernel reserves ONE block at a time -- num_pages
    # comes from the Shape, not the CB -- so a deeper CB is purely permission for the data
    # movement thread to run ahead: with one block it cannot reserve k-block b+1 until
    # compute has popped b, so its reads and the compute serialise. The reads within a block
    # are already deep (every page is issued before the single barrier); this is depth
    # ACROSS blocks. Same idea as the flash kernel's stream_buffering.
    cbs = [
        make_cb(CB_A, core_ranges, dtype=adtype, num_pages=depth * mt * kt),
        make_cb(CB_B, core_ranges, dtype=wdtype, num_pages=depth * kt * nt),
        make_cb(CB_OUT, core_ranges, dtype=odtype, num_pages=mt * nt),
        # ALWAYS, even at kb == 1 where the kernel never accumulates. matmul_blocked
        # declares Storage<Out> acc_storage(kCbAcc) unconditionally, and a Storage must not
        # name a buffer the host did not create -- an unallocated CB reports zero pages, so
        # the capacity assert in Storage's constructor fires and the kernel hangs. Skipping
        # it here saved mt*nt pages and broke that contract; hazard 20 in
        # unified_api_hazards.md, found by the assert added for hazard 1.
        make_cb(CB_ACC, core_ranges, num_pages=mt * nt),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=core_list,
        cbs=cbs,
        compile_time_args=ct_args,
        named_compile_time_args=named_ct_args,
        runtime_args=rt_args,
        defines=(
            ([("MMB_ACC_DST", "1")] if acc == "dst" else [])
            + [("MMB_IN0_THREAD", str(in0_thread)), ("MMB_IN1_THREAD", str(in1_thread))]
            + ([("MMB_MCAST", "1"), ("MMB_GRID_H", str(mb)), ("MMB_GRID_W", str(nb))] if mcast else [])
            + ([("MMB_ABL_HOIST", "1")] if hoist else [])
            + ([("MMB_SKEW", str(skew))] if skew else [])
            + ([("MMB_SHARE_PAIR", "1")] if share_pair else [])
            + ([("TT_UNIFIED_MCAST_ZONES", "1")] if zones else [])
        )
        or None,
        **(fidelity or {}),
    )
    logger.info(
        f"blocked matmul [{mtot}x{ktot}]@[{ktot}x{ntot}]t  mt={mt} kt={kt} nt={nt} "
        f"(mb={mb} kb={ktot // kt} nb={nb} = {nunits} blocks) cores={ncores}{' mcast' if mcast else ''}"
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

        # Multicast: each operand tile read from DRAM once and broadcast to the cores that
        # share it -- A along a grid row (same m), B down a column (same n). The grid is
        # exactly mb x nb, so these also check that the strict mapping lines up.
        for mtot, ktot, ntot, mt, kt, nt in (
            (4, 8, 8, 2, 8, 4),
            (4, 8, 8, 1, 4, 2),
            (8, 8, 16, 2, 8, 4),
            (4, 8, 8, 4, 8, 8),  # 1x1 grid: the degenerate case, nobody to broadcast to
        ):
            got, want = run(device, mtot, ktot, ntot, mt, kt, nt, mcast=True)
            v = pcc(got, want)
            ok = v >= args.pcc
            logger.info(
                f"  mcast {mtot}x{ktot} @ {ktot}x{ntot} mt={mt} nt={nt} "
                f"(grid {mtot // mt}x{ntot // nt}): pcc={v:.6f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"mcast-{mtot}-{ktot}-{ntot}-{mt}-{nt}")

        # bfloat8_b weights: B's circular buffer carries a different data format, and so a
        # different page size (1088, not 2048 -- it is a block format with an exponent
        # section), from A's and the output's. matmul_block_init reconfigures the unpacker
        # from the CB, and pages.page_bytes comes from the CB too, so nothing in the kernel
        # names a format; if either of those were not true this would be quietly wrong
        # rather than broken, which is why it is checked against the same threshold.
        # The OUTPUT's format is the one that had a bug: the packer is programmed once, for
        # one buffer, and a blocked matmul packs to two. kt=4 here gives kb=2 so the
        # accumulator is live and both destinations are exercised; kt=8 gives kb=1 so the
        # output is the only one. Both matter -- the first is where the packer has to switch
        # back and forth, and it is the case that was wrong.
        B8 = ttnn.bfloat8_b
        for mtot, ktot, ntot, mt, kt, nt, mc in (
            (4, 8, 8, 2, 4, 8, False),
            (4, 8, 8, 4, 8, 8, False),
            (4, 16, 8, 2, 8, 4, False),
            (4, 8, 8, 2, 8, 4, True),
            (8, 8, 16, 2, 8, 4, True),
        ):
            for tag, ad, wd, od in (
                ("W       ", None, B8, None),
                ("W+out   ", None, B8, B8),
                ("W+A+out ", B8, B8, B8),
            ):
                got, want = run(device, mtot, ktot, ntot, mt, kt, nt, mcast=mc, adtype=ad, wdtype=wd, odtype=od)
                v = pcc(got, want)
                ok = v >= args.pcc
                logger.info(
                    f"  bfloat8_b {tag} {mtot}x{ktot} @ {ktot}x{ntot} mt={mt} kt={kt} nt={nt}"
                    f"{' mcast' if mc else ''}: pcc={v:.6f}  {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append(f"bf8-{tag.strip()}-{mtot}-{ktot}-{ntot}-{mt}-{kt}-{nt}")

        # THE PAIR PARAMETER IS LOAD-BEARING, and this is the proof rather than the claim.
        #
        # Two collectives on ONE data-movement thread sharing one handshake pair corrupts,
        # silently, on hardware. Three conditions have to coincide and all three are needed:
        #
        #   * both collectives on the same thread (in0_thread == in1_thread), so the
        #     thread-derived default would give them the same pair,
        #   * a receiver holding a LIVE buffer while the next round's ready counting runs --
        #     that is what MMB_SKEW does, and where it sits matters: before the load it lands
        #     after the previous pop, the buffer is already free, and nothing corrupts,
        #   * depth 1, so there is no spare slot for an early broadcast to land in
        #     harmlessly. At the default depth of 2 this passes, which is why it stayed
        #     latent.
        #
        # The mechanism: (0, 0) is row 0's sender and waits for EQUALITY on its ready
        # counter. Lower rows finish their own row broadcast and increment (0, 0) for the
        # COLUMN collective while it is still counting ROW receivers, so the wait is
        # satisfied by the wrong increments and the broadcast goes out into a buffer a
        # receiver has not freed.
        #
        # Asserted as a DIFFERENCE, not an absolute: the control must be correct and the
        # shared-pair build must be measurably worse. If shared pairs ever become safe this
        # fails loudly, which is the right outcome -- it would mean the parameter can go.
        skewed = dict(mcast=True, depth=1, in0_thread=0, in1_thread=0, skew=5000)
        good, want = run(device, 16, 64, 64, 2, 8, 8, share_pair=False, **skewed)
        bad, _ = run(device, 16, 64, 64, 2, 8, 8, share_pair=True, **skewed)
        good_pcc, bad_pcc = pcc(good, want), pcc(bad, want)
        ok = good_pcc >= args.pcc and bad_pcc < 0.99
        logger.info(
            f"  shared handshake pair corrupts: distinct={good_pcc:.6f}, shared={bad_pcc:.6f} "
            f"({'corrupted as expected' if bad_pcc < 0.99 else 'NO LONGER CORRUPTS'})  {'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failed.append("shared-pair")

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
