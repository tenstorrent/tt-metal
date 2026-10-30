"""Deterministic test of the cross-chip DRAM aliasing hazard of a barrier-less, non-persistent all_gather_async
(the Sol pre-pass gather configuration since commit 1a1423cf877), written 2026-10-29 after the job 162394 hang analysis.

Mechanism under test: the gather's writer on chip A puts its slice straight into the output's DRAM address on chip B. The
output is allocated fresh per call, so it can land on an address the host freed moments before, while chip B's device
is still running an earlier program that reads that buffer. Here that earlier program is a one-core spin-reader
(sol_alias_reader.cpp) that re-reads a buffer M for a chip-dependent number of iterations: one chip per ring lags
(SOL_ALIAS_SLOW_ITERS), the others finish at once (SOL_ALIAS_FAST_ITERS) and run the gather. The host frees M right
after enqueueing the spin-reader (the model does the same with its per-pass mask slice) so the gather output takes M's
address; the test asserts that it did. If the fast chips' slices land in M while the slow chip is still reading it, the
slow chip reports changed words.

Variants (SOL_ALIAS_VARIANTS): nobar = the current default ({"num_workers_per_link": 1, "barrier": False});
barrier = same with the ring-wide barrier semaphore; persistent = the CCL manager's ping-pong output buffer (no barrier).
The nobar result is reported (HAZARD REPRODUCED / NOT REPRODUCED); barrier and persistent must be clean (asserted).
Every variant also checks the gathered data against the host reference.
v4 (jobs 163796 / 163835: the gather output never took M's address). The DRAM allocator (FreeListOpt, size-segregated
free lists, BEST policy) hands out an EXACT-size free block first, in the order the blocks were freed, before any larger
block; a block freed at the top of the heap merges with the top free region and is no longer an exact fit. So after the
warm-up, same-size fillers are allocated (and kept) until two consecutive fillers land above every address seen so
far: every exact-size hole is then plugged, M is the top buffer, and once M is freed the only free block of that size
or larger is the top region starting at M's address, which the gather output must take. After
the host frees M a same-size probe is allocated and freed again to record whether the free took effect at once
(ttnn.deallocate with its default force=False skips buffers that are still shared); the variant nobar_force frees M
with force=True. A nobar run that never aliases although M was freed at once is reported as such; a run where the free
did not take effect shows that in-flight ops keep their buffers alive, which would make the hazard moot.
Env: SOL_ALIAS_SLOW_ITERS (200), SOL_ALIAS_FAST_ITERS (2), SOL_ALIAS_SLOW_POS (5: ring position of the lagging chip),
SOL_ALIAS_REPS (3 nobar repetitions), SOL_ALIAS_VARIANTS (nobar,barrier,persistent,nobar_force,nobar),
SOL_ALIAS_MAX_FILLERS (24)."""
from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.tensor import from_torch
from .common import GALAXY_RING

KDIR = os.path.dirname(os.path.abspath(__file__))
TILE_BYTES = 2048
ROW_BYTES = 128
NONE = 0xFFFFFFFF


def _gather(ccl, t, sp_axis, num_links, topology, variant):
    pb = variant == "persistent"
    barrier = variant == "barrier"  # nobar and nobar_force: no barrier, fresh output
    return ttnn.experimental.all_gather_async(
        t,
        persistent_output_buffer=(
            ccl.get_ag_ping_pong_buffer(t.shape, 2, sp_axis, dtype=ttnn.bfloat16) if pb else None
        ),
        dim=2,
        multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=sp_axis,
        barrier_semaphore=ccl.get_barrier_semaphore(sp_axis) if barrier else None,
        num_workers_per_link=1,
    )


def _rows(t: ttnn.Tensor) -> list[list[int]]:
    """Per-device uint32 rows of a [rows, cols, 1, 32] tensor sharded on both mesh axes."""
    out = []
    for d in ttnn.get_device_tensors(t):
        out.append([int(v) for v in ttnn.to_torch(d).reshape(-1).tolist()])
    return out


@GALAXY_RING
def test_sol_gather_alias(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    S = (rows, cols)[sp_axis]
    assert S == 8, "the Sol pre-pass gathers over an 8-chip ring"
    H, NBP, D = 14, 96, 128  # pooled K at the 5 s geometry: [1, H, nb_pad, D] per chip, gathered to [1, H, S * nb_pad, D]
    PAGES = H * S * NBP * D * 2 // TILE_BYTES
    slow_iters = int(os.environ.get("SOL_ALIAS_SLOW_ITERS", "200"))
    fast_iters = int(os.environ.get("SOL_ALIAS_FAST_ITERS", "2"))
    slow_pos = int(os.environ.get("SOL_ALIAS_SLOW_POS", "5"))
    reps = int(os.environ.get("SOL_ALIAS_REPS", "3"))
    variants = os.environ.get("SOL_ALIAS_VARIANTS", "nobar,barrier,persistent,nobar_force,nobar").split(",")
    max_fillers = int(os.environ.get("SOL_ALIAS_MAX_FILLERS", "24"))
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    def new_m():
        return from_torch(torch.ones(1, H, S * NBP, D).bfloat16(), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None] * 4, layout=ttnn.TILE_LAYOUT)

    seen = []  # every address an output-sized buffer has had in this test (holes the allocator may hand out again)

    # Per-device cfg row (word 0 iterations, word 1 expected pattern = bf16 1.0 pair) and result row.
    cfg_t = torch.zeros(rows, cols, 1, 32, dtype=torch.int32)
    for r in range(rows):
        for c in range(cols):
            cfg_t[r, c, 0, 0] = slow_iters if (r, c)[sp_axis] == slow_pos else fast_iters
            cfg_t[r, c, 0, 1] = 0x3F803F80
    both = [0, 1, None, None]
    cfg = from_torch(cfg_t, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=both)
    res = from_torch(torch.zeros(rows, cols, 1, 32, dtype=torch.int32), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=both)
    # Gather input: per chip [1, H, NBP, D], distinct per ring position.
    tx_t = torch.randn(S, H, NBP, D).bfloat16()
    tx = from_torch(tx_t, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT)
    expected = torch.cat([tx_t[s] for s in range(S)], dim=1).unsqueeze(0).float()  # [1, H, S * NBP, D]

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx, page, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=page, core_ranges=core, format_descriptors=[fmt])

    def program(M):
        ct = [PAGES]
        for t in (M, cfg, res):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "sol_alias_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=ct,
            runtime_args=[],
            common_runtime_args=[M.buffer_address(), cfg.buffer_address(), res.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[k], semaphores=[], cbs=[cb(0, TILE_BYTES, ttnn.bfloat16), cb(1, ROW_BYTES, ttnn.uint32), cb(2, ROW_BYTES, ttnn.uint32)]
        )

    # Warm-up: compile the spin-reader and the gather kernels once (fast iterations everywhere would hide nothing, but a
    # first-call compile inside the timed sequence would let the slow chip finish before the gather even starts).
    M0 = new_m()
    seen.append(M0.buffer_address())
    ttnn.generic_op([M0, cfg, res], program(M0))
    ttnn.synchronize_device(mesh_device)
    for v in sorted(set(variants)):
        g = _gather(ccl, tx, sp_axis, num_links, topology, v)
        seen.append(g.buffer_address())
        ttnn.synchronize_device(mesh_device)
        if v != "persistent":
            ttnn.deallocate(g)
    ttnn.deallocate(M0)
    ttnn.synchronize_device(mesh_device)

    # Plug every exact-size hole: fillers (kept alive) until two consecutive ones land above everything seen so far.
    fillers, top, rising = [], max(seen), 0
    for _ in range(max_fillers):
        f = new_m()
        a = f.buffer_address()
        fillers.append(f)
        rising = rising + 1 if a > top else 0
        top = max(top, a)
        if rising >= 2:
            break
    print(f"[alias] {len(fillers)} fillers x 2.75 MB after the warm-up, top 0x{top:x}, exact-size holes plugged: {rising >= 2}", flush=True)

    summary = []
    nobar_hits = 0
    nobar_runs = 0
    seq = []
    for v in variants:
        seq += [v] * (reps if v == "nobar" else 1)
    for i, v in enumerate(seq):
        M = new_m()
        addr_m = M.buffer_address()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.generic_op([M, cfg, res], program(M))  # the slow chip reads M for ~slow_iters x 1344 pages
        # host-side free while the device still reads it: the model's per-pass mask pattern (default force=False)
        ttnn.deallocate(M, force=(v == "nobar_force"))
        probe = new_m()  # same size: takes M's slot iff the free took effect at once
        freed_now = probe.buffer_address() == addr_m
        ttnn.deallocate(probe)
        g = _gather(ccl, tx, sp_axis, num_links, topology, v)
        addr_g = g.buffer_address()
        ttnn.synchronize_device(mesh_device)
        wall = time.perf_counter() - t0
        got = _rows(res)
        aliased = addr_g == addr_m
        slow = [r for r in got if r[3] == slow_iters]
        fast = [r for r in got if r[3] == fast_iters]
        assert all(r[6] == 0xA11A5 for r in got), "spin-reader did not write every device's result row"
        bad_slow = [r[0] for r in slow]
        bad_fast = [r[0] for r in fast]
        first = [(r[1], r[2]) for r in slow if r[0]]
        # gathered data check (all devices)
        gerr = 0.0
        for d in ttnn.get_device_tensors(g):
            gt = ttnn.to_torch(d).float()
            gerr = max(gerr, float((gt - expected).abs().max()))
        if v != "persistent":
            ttnn.deallocate(g)
        line = (
            f"[alias] {i:2d} {v:11s} freed_at_once={freed_now} aliased={aliased} (M 0x{addr_m:x}, gather out 0x{addr_g:x}) wall {wall*1e3:.0f} ms; "
            f"slow chips {len(slow)}: bad words {bad_slow} (first bad iter/page {first}); fast chips {len(fast)}: bad {sum(bad_fast)}; "
            f"gather max abs err {gerr:.3g}"
        )
        logger.info(line)
        print(line, flush=True)
        summary.append((v, aliased, bad_slow, bad_fast, gerr, freed_now))
        if v.startswith("nobar"):
            nobar_runs += 1
            nobar_hits += int(any(bad_slow))
    verdict = []
    for v, aliased, bad_slow, bad_fast, gerr, freed_now in summary:
        verdict.append(f"{v}: freed_at_once={freed_now} aliased={aliased} slow_bad={sum(bad_slow)} fast_bad={sum(bad_fast)} gather_err={gerr:.3g}")
    print("[alias] SUMMARY " + " | ".join(verdict), flush=True)
    if nobar_runs:
        print(
            f"[alias] NOBAR HAZARD {'REPRODUCED' if nobar_hits else 'NOT REPRODUCED'}: {nobar_hits}/{nobar_runs} runs saw the lagging chip's buffer "
            f"change under its reader",
            flush=True,
        )
    nobar_freed = [fr for v, _, _, _, _, fr in summary if v.startswith("nobar")]
    nobar_aliased = [al for v, al, _, _, _, _ in summary if v.startswith("nobar")]
    if nobar_freed and not any(nobar_freed):
        print("[alias] VERDICT: ttnn.deallocate did not free M while the spin-reader held it (buffer shared by the in-flight op): the address cannot be reused before the op completes, so this aliasing path does not exist", flush=True)
    elif nobar_aliased and not any(nobar_aliased):
        print("[alias] VERDICT: M was freed at once but the gather output never took its address: INCONCLUSIVE (allocator placement)", flush=True)
    elif nobar_hits:
        print("[alias] VERDICT: HAZARD CONFIRMED on hardware: a chip's gather overwrote a buffer another chip was still reading", flush=True)
    else:
        print("[alias] VERDICT: output aliased the freed buffer and nothing changed under the lagging reader: the fast chips' slices did not reach M's slot during the window (timing) or the writes landed elsewhere", flush=True)
    # The fixes must be clean and the gather correct everywhere; the nobar outcome is reported, not asserted.
    for v, aliased, bad_slow, bad_fast, gerr, freed_now in summary:
        assert gerr == 0.0, f"{v}: gathered data differs from the reference (max abs err {gerr})"
        if v in ("barrier", "persistent"):
            assert sum(bad_slow) == 0 and sum(bad_fast) == 0, f"{v}: buffer changed under the reader ({bad_slow}, {bad_fast})"
