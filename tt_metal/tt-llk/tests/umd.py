# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Init device -> MMIO write -> DMA read back
Each write encodes the host process, NOC location and iteration
Stale: worker x read an old value it wrote, not the new one (not relevant in this repro, we reset the link)
Foreign: worker x read a value worker y wrote to a different address

python umd.py --cores 16 --iterations 300
"""

import argparse
import gc
import multiprocessing as mp
import struct
import sys

SCRATCH_ADDR = 0x10000
NBYTES = 256
WORDS = NBYTES // 4


def encode(worker_id: int, noc_x: int, noc_y: int, i: int) -> int:
    """worker(4b) | noc_x(6b) | noc_y(6b) | iteration(16b)"""
    return ((worker_id & 0xF) << 28) | ((noc_x & 0x3F) << 22) | ((noc_y & 0x3F) << 16) | (i & 0xFFFF)


def decode(word: int):
    return (word >> 28) & 0xF, (word >> 22) & 0x3F, (word >> 16) & 0x3F, word & 0xFFFF


def worker(worker_id: int, iters: int, start, result_q):
    try:
        from ttexalens import tt_exalens_init
        import tt_umd

        # decide on a target NOC location for this worker
        ctx = tt_exalens_init.init_ttexalens()
        umd = ctx.devices[0]._umd_device
        cores = umd.soc_descriptor.get_cores(tt_umd.CoreType.TENSIX, tt_umd.CoordSystem.TRANSLATED)
        core = cores[worker_id % len(cores)]
        x, y = core.x, core.y

        start.wait()  # release all workers together

        stale = 0
        foreign = 0
        samples = []
        for i in range(iters):
            ctx = tt_exalens_init.init_ttexalens() # reinit link
            ttdev = getattr(ctx.devices[0]._umd_device, "_UmdDevice__device")
            gc.collect() # necessary, else we run out of TLBs

            payload = struct.pack("<I", encode(worker_id, x, y, i)) * WORDS
            ttdev.noc_write(x, y, SCRATCH_ADDR, payload)
            got = ttdev.dma_read_from_device(x, y, SCRATCH_ADDR, NBYTES)
            if got == payload:
                continue

            sw, sx, sy, si = decode(struct.unpack_from("<I", got, 0)[0])
            if (sw, sx, sy) == (worker_id, x, y):
                stale += 1
            else:
                foreign += 1
                if len(samples) < 8:
                    samples.append((i, sw, sx, sy, si))
        result_q.put((worker_id, f"{x},{y}", iters, stale, foreign, samples, None))
    except Exception as e:
        result_q.put((worker_id, "?", -1, 0, 0, [], f"{type(e).__name__}: {str(e).splitlines()[0]}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=300)
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

    total_stale = total_foreign = total_inits = errors = 0
    for wid, loc, iters, stale, foreign, samples, err in sorted(results):
        if err is not None:
            errors += 1
            print(f"worker {wid:2} core={loc:>5} ERROR {err}", flush=True)
            continue
        total_stale += stale
        total_foreign += foreign
        total_inits += iters
        tag = "FOREIGN" if foreign else ("stale-only" if stale else "clean")
        print(f"worker {wid:2} core={loc:>5} inits={iters} stale={stale} foreign={foreign}:  {tag}", flush=True)
        for (i, sw, sx, sy, si) in samples:
            print(
                f"        iter {i}: DMA read returned worker {sw}'s data from core ({sx},{sy}) iter {si}"
                f"  (expected worker {wid} core {loc})",
                flush=True,
            )

    rate = f" (~1 per {total_inits // total_foreign:,} inits)" if total_foreign else ""
    print(f"\n{total_stale} stale + {total_foreign} foreign over {total_inits:,} link inits{rate} ({errors} errored)",
          flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
