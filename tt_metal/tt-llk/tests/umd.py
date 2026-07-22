# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
- make payload out of (process, iteration)
- MMIO write
- DMA read same address
"""

import argparse
import multiprocessing as mp
import struct
import sys

SCRATCH_ADDR = 0x10000 # L1
NBYTES = 256 # 4b-aligned


def _tag_word(worker_id: int, i: int) -> int:
    return ((worker_id & 0xFF) << 24) | (i & 0x00FFFFFF)


def worker(worker_id: int, iters: int, start, result_q):
    try:
        from ttexalens import tt_exalens_init, check_context
        import tt_umd

        tt_exalens_init.init_ttexalens()
        umd = check_context().devices[0]._umd_device
        ttdev = getattr(umd, "_UmdDevice__device")  # tt_umd.TTDevice
        cores = umd.soc_descriptor.get_cores(tt_umd.CoreType.TENSIX, tt_umd.CoordSystem.TRANSLATED)
        core = cores[worker_id % len(cores)]
        x, y = core.x, core.y
        words = NBYTES // 4

        # start all workers together
        start.wait()

        stale = 0  # our tag, expired
        foreign = 0  # not our tag
        first_foreign = None  # these happen very rarely
        for i in range(iters):
            payload = struct.pack("<I", _tag_word(worker_id, i)) * words
            ttdev.noc_write(x, y, SCRATCH_ADDR, payload)  # MMIO
            #ttdev.noc_read(x, y, SCRATCH_ADDR, NBYTES)  # wait for the write to commit
            got = ttdev.dma_read_from_device(x, y, SCRATCH_ADDR, NBYTES)  # DMA
            if got == payload:
                continue
            w = struct.unpack_from("<I", got, 0)[0]
            if ((w >> 24) & 0xFF) == worker_id:
                stale += 1
            else:
                foreign += 1
                if first_foreign is None:
                    first_foreign = (i, (w >> 24) & 0xFF, f"0x{w:08x}")
        result_q.put((worker_id, f"{x},{y}", iters, stale, foreign, first_foreign, None))
    except Exception as e:
        result_q.put((worker_id, "?", -1, 0, 0, None, f"{type(e).__name__}: {str(e).splitlines()[0]}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=12, help="concurrent processes / cores")
    ap.add_argument("--iterations", type=int, default=5000, help="write+read rounds per process")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    start = ctx.Barrier(args.cores)
    procs = [ctx.Process(target=worker, args=(i, args.iterations, start, q)) for i in range(args.cores)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    total_stale = total_foreign = errors = 0
    for wid, loc, iters, stale, foreign, first, err in sorted(results):
        if err is not None:
            errors += 1
            print(f"worker {wid:2} core={loc:>5} ERROR {err}", flush=True)
            continue
        total_stale += stale
        total_foreign += foreign
        tag = "FOREIGN" if foreign else ("stale-only" if stale else "clean")
        extra = f"  first foreign: iter{first[0]}: got worker {first[1]}'s tag ({first[2]})" if first else ""
        print(f"worker {wid:2} core={loc:>5} iters={iters} stale={stale} foreign={foreign}:  {tag}{extra}", flush=True)

    print(
        f"\n{total_stale} stale + {total_foreign} foreign "
        f"({errors} errored)",
        flush=True,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
