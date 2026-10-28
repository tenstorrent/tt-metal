# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU (no device): the engram host path leaves torch's intra-op thread pool as it found it (FIX.10, DS41-133).

numba's omp threading layer sets the OpenMP max-threads ICV to NUMBA_NUM_THREADS (every logical CPU) when it launches
its pool, and torch takes its pool size from the same ICV: on the Galaxy host (64 logical CPUs) torch went 32 -> 64
threads after the first engram pack, and the fused head's per-layer host layout then ran ~112 s a layer instead of
~6 s. tt/engram_bfp8._run restores the count around every kernel launch.

    scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tt/perf_engram_host.py
"""

import threading

import torch

from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import pack_into, pack_rows, tile_rows


def _rows(n: int):
    g = torch.Generator().manual_seed(0)
    w = torch.randint(0, 0x7F, (n, 256), dtype=torch.uint8, generator=g)
    s = torch.randint(120, 134, (n, 8), dtype=torch.uint8, generator=g)
    return w, s


def test_numba_keeps_torch_threads():
    before = torch.get_num_threads()
    w, s = _rows(4096)
    for threads in (16, 8, None):
        exps, mags = pack_rows(w, s, threads=threads)
        assert torch.get_num_threads() == before, f"pack_rows(threads={threads}) changed torch's pool"
    packed = torch.empty(4096, 272, dtype=torch.uint8).numpy()
    pack_into(w, s, packed, 16)
    assert torch.get_num_threads() == before, "pack_into changed torch's pool"
    ids = torch.randint(0, 4096, (64, 24), generator=torch.Generator().manual_seed(1))
    tile_rows(packed, ids, 64, 8)
    assert torch.get_num_threads() == before, "tile_rows changed torch's pool"
    # a launch from another thread (the gather pool, ENGRAM_ASYNC) leaves the caller's pool alone too
    seen = {}

    def other():
        pack_rows(w, s, threads=16)
        seen["n"] = torch.get_num_threads()

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert torch.get_num_threads() == before and seen["n"] == before, (before, seen)
    # (and the guarded calls compute the same bytes: packing is independent of the thread count)
    e1, m1 = pack_rows(w, s, threads=1)
    assert torch.equal(e1, exps) and torch.equal(m1, mags)
