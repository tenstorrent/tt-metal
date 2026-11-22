# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU microbenchmark (no device): the engram host job of one 5120-token chunk, numba (tt/engram_bfp8.py) vs C++
(tt/engram_host_cpp.py), at 1 / 8 / 16 / 24 threads: the hash, the gather (+ TILE bytes into 32 chips' shares, q = 160)
and HOST_HIDE's whole host job of a layer (TtEngramRows.fill_tp without the upload: numba = today's _valid, hash,
gather and next_tail; C++ = engram_host_cpp.fill_job, one call), each alone and while another Python
thread spins (pure Python: it holds the GIL whenever it may), at the default GIL switch interval (5 ms) and at
HOST_HIDE_SWITCH_MS (0.5 ms). Median ms of ``REPS`` calls ``GAP_S`` apart (the pools' threads asleep at each call,
as in the model: two calls a chunk), measured in the calling thread (GIL re-take included).

Inputs: the canonical prompt's chunk 2 and the real hasher (spec galaxy_8x4-full) when this host has them, else
synthetic; the rows are layer 1's real row ids in a table of the real table's size (384M rows x 272 B: a sparse
anonymous mapping, only the rows the chunk reads are written), so the gather's page / TLB pattern is the real one.

    python_env/bin/python models/demos/deepseek_v41_flash_d_p/tests/perf/bench_engram_host_cpp.py
"""

from __future__ import annotations

import mmap
import statistics
import sys
import threading
import time

import numpy as np
import torch

from models.demos.deepseek_v41_flash_d_p.tt import engram_bfp8 as nb
from models.demos.deepseek_v41_flash_d_p.tt import engram_host_cpp as cpp

CHUNK, CHIPS, REPS, THREADS, GAP_S = 5120, 32, 30, (1, 8, 16, 24), 0.003


def inputs():
    from models.demos.deepseek_v41_flash_d_p.tests.test_engram_host_cpp import CHUNK as C
    from models.demos.deepseek_v41_flash_d_p.tests.test_engram_host_cpp import real, synthetic_hasher

    r = real()
    if r is not None:
        hs, prompt, _ = r
        ids, tail, n_rows = prompt[C : 2 * C], hs.token_map[prompt[C - 3 : C]], 384_006_168
    else:
        hs = synthetic_hasher(0)
        ids = torch.randint(0, 129_280, (C,))
        tail, n_rows = hs.token_map[:3].clone(), int(hs.offsets[0, -1] + hs.primes[0, -1, -1])
    layer = hs.layer_ids[0]
    h, _ = nb.ngram_hashes(hs, ids, layer, tail, 8)
    buf = mmap.mmap(
        -1, n_rows * 272, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_NORESERVE", 0x4000)
    )
    packed = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, 272)
    u = torch.unique(h).numpy()
    packed[u] = np.random.default_rng(0).integers(0, 256, (u.size, 272), dtype=np.uint8)
    outs = tuple(np.empty(CHUNK // CHIPS * 6528, dtype=np.uint8) for _ in range(CHIPS))
    return hs, layer, ids, tail, packed, outs, buf, r is not None


class Spinner:
    """Another Python thread that wants the GIL all the time (the main thread's dispatch, at its worst)."""

    def __enter__(self):
        self.stop, self.n = False, 0

        def spin():
            while not self.stop:
                self.n += 1

        self.t = threading.Thread(target=spin, daemon=True)
        self.t.start()
        time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def median_ms(fn) -> float:
    fn()
    ts = []
    for _ in range(REPS):
        time.sleep(GAP_S)
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def run(print_fn=print):
    hs, layer, ids, tail, packed, outs, buf, is_real = inputs()
    q = CHUNK // CHIPS
    print_fn(
        f"inputs: {'real prompt + hasher, layer ' + str(layer) + ' row ids' if is_real else 'synthetic'}; "
        f"table {packed.shape[0]} x 272 B; numba threads max {__import__('numba').config.NUMBA_NUM_THREADS}"
    )

    def numba_job(th):
        vocab = hs.token_map.numel()
        x = ids.reshape(-1).long()
        bad = ((x < 0) | (x >= vocab)).nonzero()
        valid = int(bad[0]) if bad.numel() else x.numel()
        h, new_tail = nb.ngram_hashes(hs, x[:valid], layer, tail, th)
        assert valid == 0 or (int(h.min()) >= 0 and int(h.max()) < packed.shape[0])
        nb.tile_rows_chips(packed, h, valid, outs, q, th)
        last = x[max(0, valid - 3) : valid]
        return torch.cat([tail.long(), hs.token_map[last]])[-3:].clone()

    want_tail = numba_job(8)

    def jobs(th):
        h, _ = nb.ngram_hashes(hs, ids, layer, tail, th)
        return {
            "numba hash": lambda: nb.ngram_hashes(hs, ids, layer, tail, th),
            "C++ hash": lambda: cpp.ngram_hashes(hs, ids, layer, tail, th),
            "numba gather": lambda: nb.tile_rows_chips(packed, h, CHUNK, outs, q, th),
            "C++ gather": lambda: cpp.tile_rows_chips(packed, h, CHUNK, outs, q, th),
            "numba job": lambda: numba_job(th),
            "C++ job (1 call)": lambda: cpp.fill_job(hs, layer, ids, tail, packed, outs, q, th, want_tail=want_tail),
        }

    rows = {}
    old = sys.getswitchinterval()
    try:
        for cond in ("alone", "spin 5 ms", "spin 0.5 ms"):
            sys.setswitchinterval(0.0005 if cond == "spin 0.5 ms" else 0.005)
            for th in THREADS:
                for name, fn in jobs(th).items():
                    if cond == "alone":
                        rows[(name, th, cond)] = median_ms(fn)
                    else:
                        with Spinner():
                            rows[(name, th, cond)] = median_ms(fn)
    finally:
        sys.setswitchinterval(old)
    names = list(jobs(1))
    for cond in ("alone", "spin 5 ms", "spin 0.5 ms"):
        print_fn(f"\n{cond}: median ms of {REPS} (threads {' / '.join(map(str, THREADS))})")
        for name in names:
            print_fn(f"  {name:16s} " + "  ".join(f"{rows[(name, th, cond)]:7.2f}" for th in THREADS))
    del packed
    buf.close()
    return rows


def test_bench_engram_host_cpp():
    run()


if __name__ == "__main__":
    run()
