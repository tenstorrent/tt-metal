# Full-mesh ring_mla for a TP-deduped KV cache: what it costs and why

Measurements for the shard-major plan in `RING_MLA_FULL_MESH_SHARD_MAJOR_PLAN.md`. Everything here is
an 8x4 Blackhole Galaxy on `FABRIC_2D_TORUS_XY`, kimi50k shapes: a 5120-row Q chunk against a 56320
prefix (11 chunks), `q_chunk=32`, bf16 Q / bfp8 KV, `d_q=d_k=576`, `d_v=512`, 16 Q heads.

## Summary

| design | device time | overhead vs no dedup |
| --- | ---: | ---: |
| no dedup: ring_mla alone, SP-sharded cache | 5.7614 ms | — |
| **shipped: high_bw_all_gather + 8-ring ring_mla** | **5.9232 ms** | **+0.162 ms** |
| full-mesh 32-ring, best tuning | 6.7219 ms | +0.959 ms |
| full-mesh 32-ring, inherited k=640 | 7.0604 ms | +1.298 ms |

The shipped TP-AG design costs 5.9x less than the best full-mesh alternative. Everything that was
suspected of causing that gap has been measured and eliminated: k tuning is flat, padding does not
add time, transport is fine, unit count is at its floor, and ring width costs 2.1 us/step. What
remains -- 0.806 ms at k=640 -- is **not yet attributed**, and the last section says why the obvious
explanation does not hold and what instrument would settle it.

## How this was measured, and one retraction

Device program duration from the in-process realtime profiler
(`tests/ttnn/profiling/realtime_profiler_utils.profile_realtime_program`). Each program contributes
its max duration across chips and the programs are summed, so a two-op arm counts both. 10 timed
iterations after 2 warmups; the reported figure is the minimum.

Spread within a run is ~0.2%, and the same configuration reproduces to **0.04%** across separate
processes and across two worktrees 194 commits apart (8-ring k=640: 5.7625 / 5.7614 / 5.7639 /
5.7602 ms).

**An earlier round of this work used `time.perf_counter()` around the dispatch and every number it
produced is discarded.** That measures host enqueue plus device plus sync, untraced, and host
overhead was 1.2-2.3 ms -- roughly 20% -- varying run to run. Within-run spread was 16-19% and the
same setting drifted 5% between runs, which is larger than most of the differences being compared.
It inflated the 32-ring penalty (1.263x wall clock vs 1.221x device) because host cost scales with
call duration and so weighs on the slower arm. `KV_DEDUP_UNSTRIPE_FINDINGS.md` had already recorded
that untraced dispatch is host-bound; the harness was built into that trap anyway.

## What was implemented

### A K chunk may now span any number of cache regions

Each device's slab stores that rank's region of every global chunk back to back, and adjacent regions
are non-adjacent in global K. A K chunk wider than one region therefore sees global K jump at each
boundary it crosses. `compute_streaming.hpp` derived exactly one such jump, which is correct only up
to two regions. Against a 5-tile region a 20-tile chunk crosses three.

The boundaries are evenly spaced, so the decode is arithmetic rather than a table --
`straddled_k_tile()` in `chunked_prefill_utils.hpp`, beside the `chunked_kv_global_tile_for_local`
mapping it decodes column by column:

    crossings = col < straddle_col ? 0 : 1 + (col - straddle_col) / straddle_period
    k_pos     = k_start_tile + col + crossings * straddle_jump

`straddle_period == 0` reproduces the old single-jump behaviour, so anything spanning at most two
regions is bit-identical. Both per-column stamps use it; `compute_common.hpp` and
`compute_streaming.hpp` had separate copies of the same single-jump logic.

### Straddling chunks stamp by run, not by column

The fix above routed every wide chunk through the per-column path, which issues one single-tile op
per column where the contiguous path issues one range op per row. `straddle_run()` walks the runs
between boundaries -- global K is contiguous inside a run, so each stamps exactly like the fast path.
A 20-column chunk over a 5-tile region issues 4 range stamps per row instead of 20 single stamps. A
chunk that straddles nothing yields one run covering every column, i.e. the original fast path.

### Accuracy coverage against the cache region

`test_ring_mla_full_mesh_tp_striped_kv_accuracy` ran one `k_chunk` (32 = a single tile), which sits
inside the 2-tile region and never straddles. Now bracketed:

| k_chunk | tiles | regions spanned | boundaries | before | after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 1 | 0 | 0.9993317485649372 | unchanged |
| 128 | 4 | 2 | 1 | 0.9996549557871974 | unchanged |
| 256 | 8 | 4 | 3 | 0.9729822628603274 | **0.9996846607641989** |

k=256 was wrong output, not slow output -- ATOL 20.9, not a rounding difference. It was committed as
`xfail(strict)` before the fix and flipped to a pass after. Run stamping left all three bit-identical,
so it is pure restructuring.

Wider chunks also read slightly **more** accurately (k128 0.99965 vs k32 0.99933): fewer
online-softmax rescales, less accumulated rounding. Unlocking wide units costs no precision.

## Experiments

### 1. Unit count on a fixed ring (8-ring, 11 chunks)

Ring width and total K math held constant; only `k_chunk` varies, so only the work-unit count moves.

| k_chunk | units | per step | device |
| ---: | ---: | ---: | ---: |
| 640 | 88 | 11 | 5.7625 ms |
| 320 | 176 | 22 | 6.1679 ms |
| 160 | 352 | 44 | 7.1509 ms |

Per-unit cost **5.26 us**. The largest fixed component is the online-softmax accumulator rescale over
`q_local x head_dim_v` = 640 x 512, which is paid per unit regardless of unit width.

### 2. Unit size on the 32-ring (11 chunks)

| k_chunk | units | per step | pad | device | vs 8-ring 88u |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 352 | 11 | 0.0% | 7.2017 ms | 1.250x |
| 256 | 224 | 7 | 1.8% | 6.7502 ms | 1.171x |
| 352 | 160 | 5 | 0.0% | 6.7219 ms | 1.166x |
| 640 | 96 | 3 | 9.1% | 7.0604 ms | 1.225x |
| 896 | 64 | 2 | 1.8% | L1 FAIL | 2049664 B > 1572864 B |
| 1760 | 32 | 1 | 0.0% | L1 FAIL | 3691264 B > 1572864 B |

**L1 caps the unit width at roughly 20-22 tiles.** The plan's suggestion to sweep 1280 is infeasible,
and the wide-unit regime where amortization would be strongest is unreachable on this hardware.

At matched 352 units the 8-ring costs 7.1509 and the 32-ring 7.2017: **ring width costs 0.7%**, or
2.1 us per extra step, when there is enough compute per step to hide it.

### 3. Unit size with padding pinned at zero (12 chunks)

11 chunks gives a 1760-row shard, which only 160 / 352 / 1760 divide -- so the tuned k=640 pads 9.1%
and the comparison above confounds padding with unit width. At 12 chunks the shard is 1920 rows (60
tiles) and every width below divides it exactly.

Baseline: 8-ring k=640 at 12 chunks, 96 units, **6.2748 ms**. (Device time scales with work almost
exactly: 6.2748 / 5.7639 = 1.089 against a 12/11 = 1.091 work ratio.)

| k_chunk | units | per step | device | overhead | unit cost | residual | us/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 320 | 10 | 7.6167 ms | 1.342 | 1.178 | 0.164 | 5.1 |
| 320 | 192 | 6 | 7.1639 ms | 0.889 | 0.505 | 0.384 | 12.0 |
| 384 | 160 | 5 | 7.1080 ms | 0.833 | 0.337 | 0.497 | 15.5 |
| 480 | 128 | 4 | 7.0386 ms | 0.764 | 0.168 | 0.595 | 18.6 |
| 640 | 96 | 3 | 7.0805 ms | 0.806 | 0.000 | 0.806 | 25.2 |

`unit cost` is `(units - 96) x 5.26 us` from experiment 1; `residual` is what is left over. The
residual is a subtraction, not a measurement of any particular stall -- see the last section.

**Overhead is flat at 0.76-0.89 ms across k=320..640.** The two terms trade off and cancel: fewer
units cuts per-unit cost and starves each ring step of the compute that hides its arrival. There is
no k that escapes it. k=640 is 42 us from the best, inside the 2.8% within-arm spread.

### 4. The shipped TP-AG path (current main, `d15123f921d`)

`high_bw_all_gather` over the TP axis rebuilding one SP rank's slab rank-major, then the ordinary
8-deep ring_mla with `kv_block_cyclic_cache_tp_sharded` so its reader decodes. Both programs counted.

| arm | device | programs | overhead |
| --- | ---: | ---: | ---: |
| sp_only, ring_mla alone | 5.7614 ms | 1 | — |
| tp_ag, gather + ring_mla | 5.9232 ms | 2 | **+0.162 ms (2.8%)** |

The baselines of this worktree and the striped one agree to 0.02% (5.7614 vs 5.7625), so these
numbers sit on the same scale as everything above despite the 194-commit gap between the bases.

### 5. Transport: 8 hops versus 32

Standalone `high_bw_all_gather` delivering the identical 32.44 MB to every chip, split either 8 ways
(7040 rows/shard, SP axis) or 32 ways (1760 rows, full mesh).

| hops | rows/shard | device | effective | per hop |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 7040 | 0.4359 ms | 74.4 GB/s | 54.5 us |
| 32 | 1760 | 0.4793 ms | 67.7 GB/s | 15.0 us |

Splitting into 32 smaller shards costs **43 us**, a 10% bandwidth penalty on 4x smaller payloads.

## Conclusions

**Padding does not add time; it wastes time already being spent.** At k=640 the 32-ring processes
1920 rows per shard whether the cache holds 11 chunks (160 of them padding) or 12 (none): 7.0434 vs
7.0805 ms, the same cost for 9% more useful work. So padding cannot be tuned away by choosing k --
only the widths that divide the shard avoid it, and those are not free either. Eliminating it at the
tuned width needs the op to process a **short final unit** instead of padding to full width, which is
worth ~9% of K work at depths where the shard is not a multiple of k.

**Transport is not the bottleneck.** 32 hops cost 43 us more than 8 -- 19x smaller than the 806 us of
residual, and the entire transfer (479 us) is less than the residual it was blamed for. The
small-payload penalty is real but irrelevant at this scale.

**Unit count is real but already at its floor.** 5.26 us per unit, and the 32-ring cannot go below 96
units at k=640 (3 per shard) without exceeding L1.

**Ring width itself is nearly free** -- 2.1 us per step when hidden, 0.068 ms over 32 steps.

**The remaining 0.806 ms is unattributed, and it is NOT waiting for data.** That was the working
hypothesis and experiment 5 refutes it: the whole 32-hop gather lands in 0.479 ms while compute runs
6.275 ms, and shards are consumed nearest-first, so arrivals run about 13x ahead of consumption.
Three further candidates are ruled out by the matched-352 comparison, where 32 ring steps cost only
51 us more than 8 (2.1 us/step): per-step loop overhead, the per-iteration Q re-read
(`need_q_read = (q_per_core > 1) || !q_pushed`), and the per-slice worker barrier would each scale
with step count and would have shown up there. Extra straddle runs are too small to matter -- about
288 additional run iterations across the op, each a divide and a compare.

The one variable that differs between the two comparisons is units per ring step: 11 where the cost
vanishes, 3 where it is 25.2 us/step. Why low units-per-step costs that much is not answerable from
whole-op timing.

**What would settle it.** The realtime profiler reports one duration per program, and a fused op is
one program spanning CCL and compute cores, so it cannot show where inside the op the time goes.
Kernel-zone profiling (Tracy device profiler) timestamps zones per core and would show directly
whether compute cores stall on `out_ready_sem`, stall on CB credits, or are simply busy. That
distinction decides whether an all-gather change is the right target at all.

**What a fix would be worth, if the residual turns out to be recoverable.** At 96 units the 32-ring
would land at 5.7625 + (96-88) x 5.26 us = **5.805 ms, +0.043 ms** over no dedup, beating the shipped
TP-AG path's +0.162 ms. That is the upper bound on the prize; whether any of it is reachable depends
on what the residual actually is.

## Status against the shard-major plan

| phase | state |
| --- | --- |
| 1, physical-to-logical mapper | already exists: `chunked_kv_global_tile_for_local`, striped-aware via `kv_rank_stride_Nt`, and host and device call the same function so the drift the plan's Risk 3 fears is structurally impossible. Missing only an exhaustive host test. |
| 2, shard-major planning | already is: `num_local_k_chunks = div_up(kv_local_padded_N, k_chunk_size)` spans stripes. |
| 3, reader | no work needed: the K slice is a contiguous physical range, and the physical slab is contiguous. The reader's region references are all in the skip decision. |
| 4, logical-position masks | **done** -- the two changes above. |
| 5, overlap and balance | already is: `RingIdSequencer` starts at the local shard with `wait_min(0)` and alternates backward/forward arrival waves. Only the lane-imbalance cyclic shift is absent, and the plan makes that conditional. |
| 6, hot path | unit sweep done (flat); mask coalescing done; the unit-size sweep to 1280 is infeasible on L1. |

The plan's premise was that k-chunk rounding was the barrier. It is not: rounding costs nothing in
math and 9% in wasted capacity. What the plan never names is the cost that appears at low
units-per-step, which is also the only thing left unexplained here.

## Reproducing

    # 32-ring unit size and chunk depth, both rings
    pytest tests/nightly/blackhole/sdpa/test_ring_mla_ring_width_perf.py -q -s

    # transport at 8 vs 32 hops
    pytest tests/nightly/blackhole/sdpa/test_ag_hop_count_perf.py -q -s

    # accuracy across the cache region
    pytest "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_full_mesh_tp_striped_kv_accuracy" -q -k torus_xy

The shipped TP-AG arm lives on `ipotkonjak/kimi-tp-shard-kv-dedup` as
`tests/nightly/blackhole/sdpa/test_tp_ag_ring_mla_perf.py`; it needs that branch's
`kv_block_cyclic_cache_tp_sharded` and cannot run here.
