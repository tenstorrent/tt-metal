# `vsa_sdpa` streaming kernel — design notes

Companion to `VSA_SCOPE.md` (requirements) and `VSA_PLAN.md` (journal). This is the distilled
design of the fine-stage op as it ships (v17 + hardening), what bounds it, and what the block
looks like around it. Op: `ttnn.transformer.vsa_sdpa(q, k, v, indices, block_counts, streaming=True)`,
code under `ttnn/cpp/ttnn/operations/transformer/sdpa/` (`vsa_sdpa*`, `device/vsa_sdpa_stream_*`,
`device/kernels/{dataflow,compute}/vsa_sdpa_stream_*`).

## 1. Problem shape

Per device (SP shard) at 15 s / 768p: 14 heads, `S_local = 14464` query tokens = 226 q-rows of 64,
`n_kv_blocks = 1808` gathered K/V blocks of 64 tokens (the full sequence), each q-row lists ~197
blocks (sparsity 0.9) or all blocks if it is an exempt (text/audio) row. Head dim 128, bf16, HiFi2.
Listed math is ~9% of dense attention, but the naive per-row gather (v1) re-read every listed block
from DRAM per row: ~20 GB of scattered 2 KB tile reads per layer per device, 178 ms — 2.4x slower
than the dense ring SDPA it replaces.

## 2. Architecture (v17)

**One head group per core group.** The 120 compute cores are split into per-head groups. In each
group one core is the *leader*; the rest are *workers*. Every core is *resident* for a set of q-rows
(chunk-cyclic: 4-row chunks dealt round-robin over the group's cores; a *pass* processes up to
`rmax = 15` rows per core).

**Leader streams K/V once per pass.** The leader walks the head's blocks in ascending block id and
DMA-reads each block's K and V (16 KB each) from DRAM into a ring of `stream_depth = 12` L1 slots
(pairs of blocks per pump). Slot reuse is gated on the minimum *posted progress* of all consumers
(workers' ackbox words in the leader's L1, plus the leader's own compute, since the leader also
works its own rows — "leader-as-worker"). Because the DRAM read of a block happens once per pass
per head, total DRAM traffic is `passes x n_kv_blocks x 32 KB` per head — the design's key property.

**Publish by multicast log, consume by pull.** For each fetched block the leader multicasts a 16 B
log entry `{block, slot, seq}` into a ring in every worker's L1 (2 entries per multicast, sequence
words make the protocol barrier- and semaphore-free: a consumer spins until `seq == n+1`). Workers
build per-row membership bitmaps for the pass (from the row's index list) and, for each published
block, list which resident rows need it. Listed blocks are *pulled* (NoC read) from the leader's
slot into the worker's own slot of the same index: K by the writer RISC (NOC_1, tagged with
per-half transaction ids), V by the reader RISC (NOC_0). The pull model — rather than pushing K/V
into workers — is what decouples delivery from compute: the leader runs ahead by `stream_depth`
blocks regardless of any worker's compute.

**Windows and halves.** A worker's slot ring is two *halves* of 6. Arrivals fill the open half;
the window closes when the half is full or a starvation check trips, then a *window* message goes
to the compute: the list of `(slot, count, needs_mask)` per row visited. Windows are closed
lazily (v10): the K-marker to the writer and the emission wait for "landed" are non-blocking
checks polled while the next half fills; K acks come back in order per half; progress is posted
to the leader only up to `post_limit` (the first unconfirmed pull), so the leader can never recycle
a slot a worker is still pulling from.

**Compute: group-major window engine (v11/v12/v14/v15).** The compute kernel buffers all visits of
a window and runs them phase-major in chunks: (1) QK for every visit into a per-window `cb_qk`
region (cross-visit DEST batching), masks stamped for ragged/partial blocks; (2) running row max,
four visits per DEST acquire, first visit seeded with -inf; (3) `corr = exp(old_max - new_max)` for
non-first visits, four per acquire; (4) O and sum rescale by corr (blocked packs); (5) probs
`exp(qk - max)` and row sums; (6) PV via the MOP (`matmul_block` with `kt = probs row stride`),
L1-accumulate pack into the resident O slot (overwrite on a row's first visit); credits (freed
slots) go back to the reader as `cb_free` pages; at pass end each row is normalised by `1/sum`
and written out. One pack->unpack sync per phase per chunk instead of per visit.

**Role-split binaries.** Leader and worker bodies together overflow the kernel-config buffer, so
the reader/writer are built twice with `VSA_IS_LEADER` and pushed as separate kernels on the
leader/worker core sets; `override_runtime_arguments` resolves them by *name*, not index.

## 3. Correctness invariants (the ones that bit)

- **Nothing in L1 may be read before this run writes it.** The op runs after other ops and on
  program-cache hits; whatever a CB page held before is arbitrary. The kreq window-marker bug
  (writer read word 2 of a page the reader had only written words 0-1 of) was benign with zero or
  same-op leftovers and a chip-wide hang with anything else. Regression:
  `tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa_trace.py::test_vsa_sdpa_cache_hit_loop`
  poisons L1 (height-sharded L1 tensor of 1000.0 / NaN, or a matmul) between a cache-miss and
  cache-hit invocation. The `fill1000_<n>k` variant poisons only the top n KiB (buffers allocate
  top-down) — the bisection tool that found the page.
- **Reader-side `invalidate_l1_cache()`** wherever a RISC reads L1 that NoC or another RISC wrote
  (rows landed by `noc_async_read`, pages produced by the other RISC). Blackhole's data cache is
  off by default; the fence is the documented contract regardless.
- **Slot reuse is gated on compute, not on delivery.** Progress is posted only after the compute
  released a window's slots (credits), so the leader's DRAM refill of a slot can never race a PV
  that still reads it.
- **Leader exit barrier**: the leader's multicasts are non-posted; `async_write_barrier()` before
  exit so a late log entry cannot land in the next program's L1.
- **Every protocol word the leader zeroes is re-posted by the worker until observed** (READY
  magic in the ackbox flag words), so the handshake does not depend on launch skew or prior state.
- **Protocol traps**: an out-of-range block/slot in a log entry, a posted count beyond anything
  published, or a malformed control page parks the RISC in a *named* infinite loop
  (`vsa_trap_*`) so a hang's triage names the corruption rather than the waiter.

Test surface: `test_vsa_sdpa.py` (shapes, ragged rows, m), `test_vsa_sdpa_trace.py` (untraced
repeat, trace replay on 1 and 32 devices with fresh addresses, cache-hit loops), model tests
`test_vsa_block_minimax_h3.py` (traced/untraced, `VSA_REPEAT`, `VSA_KERNEL=v1`, `VSA_PLACEMENT`),
`test_vsa_transformer_minimax_h3.py` (sparsity 0 == dense; placements == identity),
`test_vsa_pipeline_minimax_h3.py` (real weights, 15 s / 768p).

## 3b. How this differs from the standard (compute-streaming) SDPA

The dense path (`ring_joint_sdpa`, the `sdpa` compute kernel family) is a flash-attention loop:
each core owns a *q chunk* (256 rows at 15 s), sweeps the keys in *k chunks* of 512, and per
k chunk does one 256x512x128 QK matmul, one running-max/exp/sum pass over the 256x512 scores, one
rescale of the 256x128 O accumulator, and one 256x512x128 PV matmul. K/V chunks arrive from DRAM
(and, in the ring variant, from the neighbouring device) into a double-buffered CB; every core is
independent. Everything is dense and regular, so the reader can prefetch blindly and the compute
kernel is a fixed schedule.

`vsa_sdpa` keeps the same online-softmax math but the work is *sparse and irregular*:

| | dense ring SDPA | `vsa_sdpa` streaming |
|---|---|---|
| unit of K/V work | k chunk: 512 keys, used by all 256 rows of the q chunk | block: 64 keys, used only by the resident rows that *listed* it (~11% of rows at sparsity 0.9) |
| K/V delivery | each core streams its own K/V from DRAM / ring neighbour | one leader per head streams from DRAM once per pass; workers pull only the blocks their rows list from the leader's L1 |
| who decides what to compute | static schedule | per-row index lists -> per-block visit lists built at run time (bitmaps), windows closed dynamically |
| softmax bookkeeping (max, corr = exp(dm), rescale O, sum) | once per 512 keys per row | once per *visit*; a visit is a row's listed blocks inside a 6-block window -> mostly 1 block (64 keys): ~8x more bookkeeping per key |
| matmul shapes | QK 256x512x128, PV 256x512x128 | QK 64x(64..384)x128, PV 64x(64..384)x128 per visit; cross-visit DEST batching recovers some issue efficiency |
| sync | CB push/pop between the core's own RISCs | multicast log ring, pull acks, progress posts, credits -- a protocol across ~9 cores per head |
| exp | `exp_approx_mode=False` (model config) | op default `math_approx_mode=True` (see note below) |

**Where the utilization goes (15 s, 16 ms, listed-math peak = 60% would be ~6.6 ms):** the dense
kernel reaches ~70% because its exp/reduce work per key is amortised over 512 keys and every
matmul is large; here, per 64-key visit the FPU work (~1 k cycles of HiFi2 tile matmuls) is matched
by ~1.1 k cycles of SFPU exp *plus* ~0.9 k cycles of max/corr/rescale/sum reductions that dense pays
once per 512 keys, and on top of that the delivery protocol floor (consume every visit with no math)
is 7-11 ms of the 16. Measured phase timers put PACK at 93% busy (exp + packs), MATH 89%, UNPACK 87%:
the three TRISCs are saturated on different things and alternate waits. Batching more blocks per visit
would fix the ratio, but a row lists ~1 block in 9, so a 12-slot window holds ~1.3 of them; a window
wide enough to average 6 selected blocks per row (~50 slots) does not fit L1 next to resident rows
(the rows-for-depth sweep is monotonically worse). That is the structural reason this design plateaus
at ~25%: block-sparse selection at 64 tokens buys a 9x reduction in math but makes every remaining
unit of work 8x less amortised.

**Exp mode.** `vsa_sdpa`'s compute-kernel default is now `math_approx_mode=False` (exact SFPU exp for
both the probabilities and `corr = exp(dmax)`), matching the dense path's `exp_approx_mode=False`. It
was inherited as `True` from the op this was forked from; the earlier floor measurements used the
approximate exp. Cost of the exact exp: see the standalone numbers below (the first measurement of it ran a stale
library and is superseded).

## 3c. Determinism

The dense block is run-to-run deterministic, and so is `vsa_sdpa` since the arrival-bin change: a
worker's (and the leader-as-worker's) windows close on fixed bins of `half_slots` arrivals, never on
a starvation check, so the partition of a row's blocks into visits -- and with it the bf16 order in
which the online softmax combines partial results -- is a pure function of the inputs. The bins are
deadlock-free against the leader's slot gate because fetch lag (4) + bin width (6) < stream depth
(12): the leader always publishes past a bin boundary before it can be gated on that worker's
progress. Verified by `test_vsa_sdpa_trace_replay`: untraced repeat and traced replay are bit-exact
(PCC 1.000000; before: 0.99998, and 0.999 before the kreq-marker fix). Cost, standalone medians vs
timing-driven windows (both with exact exp): 15 s 16.7 -> 17.1 ms topk / 16.5 -> 16.0 ms model,
10 s 7.4 -> 7.9 ms, 5 s 2.4 -> 2.5 ms -- neutral to +6%. The coarse stage is deterministic as well.

## 4. Performance and its ceiling

Standalone, 15 s heaviest shard, median over runs (approximate exp, timing-driven windows, the
configuration the levers were measured in): **16.2 ms topk / 16.0 ms model order, 24-25% of HiFi2
peak on the listed math** (v1: 81.9 ms, 4.8%). As shipped (exact exp, deterministic windows):
17.1 / 16.0 ms, 23-25%. Delivery floor (probe: consume visits, no
math) 10.7 / 7.1 ms; math floor (QK+PV, no softmax) ~10.7 ms; per-TRISC busy after the last
levers: PACK 93%, MATH 89%, UNPACK 87%.

Why the 60% target is out of reach for this design, losslessly:
- Per 64-key block per q-row the FPU does 2 x 2x4x2 tiles of HiFi2 matmul (~1024 cycles) while the
  SFPU exp of the 2x2 probs tiles costs ~1.1k cycles on the pack thread; head dim 128 fixes this
  ratio. The exp is the floor: ~6.5 ms of the 16 ms, and any faster exp is a fidelity change.
- The three TRISCs alternate waits (MATH-heavy QK vs PACK-heavy exp); the deferred-PV region is a
  DEST half-sync handoff, not issue work. Fusions that looked free (v9, v16) cost more than they
  saved because they broke the 3-thread overlap.
- The levers that remained were measured and closed: rows-for-depth trade (monotonically worse),
  MOP PV (neutral, kept), conditional rescale (40% skip rate, neutral), larger K batching *is* the
  window mechanism already (a visit spans many blocks per row). A coarser VSA block (256 tokens)
  would amortise every per-visit cost 4x but changes the model's selection granularity.
- Practical ceiling of this design: ~26-28%. With every remaining pack/unpack trim, ~30%.

## 5. In the block (tracy, one transformer block period, 768p, sparsity 0.9, interleaved placement)

Device 0 unless noted; "max" is the slowest device (the block waits for it). Measured as the ops
between two consecutive attention ops (an exact block period). The 15 s row is with exact exp and
deterministic windows; 5/10 s with the earlier approx-exp / timing-window kernel (standalone deltas
+3% and -3..+6%).

| duration | dense block (dev0 / max) | VSA block (dev0 / max) | dense attention | `vsa_sdpa` | VSA-only ops |
|---|---|---|---|---|---|
| 5 s  | 15.8 / 17.5 ms | 19.0 / 20.3 ms | 7.4 ms  | 3.3 ms  | 6.2 ms  |
| 10 s | 39.5 / 41.2 ms | 37.4 / 39.0 ms | 24.7 ms | 9.0 ms  | 11.7 ms |
| 15 s | 73.4 / 75.4 ms | 58.6 / 59.2 ms | 51.4 ms | 19.6 ms | 15.3 ms |

Component breakdown (ms, device 0):

| component | 5 s dense | 5 s VSA | 10 s dense | 10 s VSA | 15 s dense | 15 s VSA |
|---|---|---|---|---|---|---|
| attention core (ring SDPA / `vsa_sdpa`) | 7.44 | 3.26 | 24.70 | 9.01 | 51.43 | 20.10 |
| full K/V all-gather (fine-stage input) | - | 3.02 | - | 5.43 | - | 8.11 |
| coarse pooling q/k/v | - | 0.33 | - | 0.59 | - | 1.96 (matmul 0.48 -> 0.34 with a full-grid program config; the 3 transposes, 0.9 ms, are DRAM-bound) |
| pooled K/V gather + assembly | - | 0.40 | - | 1.03 | - | ~0.5 (was 1.63; now two aligned all-gathers) |
| coarse scores + mask + softmax | - | 0.46 | - | 0.79 | - | 0.28 |
| coarse output o_c (probs@V, tile->token) | - | 0.40 | - | 0.72 | - | 1.11 |
| top-k selection + index assembly | - | 0.43 | - | 1.08 | - | 0.45 (was 2.92 with host-side assembly) |
| gate branch (gate proj, heads, blend) | - | 1.16 | - | 2.03 | - | 2.93 |
| shared ops (norms, projections, MLP, adaLN) | 8.34 | 9.52 | 14.78 | 16.74 | 22.00 | 23.75 |

The shared ops are ~8% dearer under VSA because the sequence is padded to whole 64-token tiles
(14464 vs 13632 rows per device at 15 s). The K/V all-gather is link-bound: each device receives 7 x 51.8 MB = 363 MB per tensor at 15 s in
4.16 ms = 87 GB/s over 2 ring links (~87% of 2 x 50 GB/s); the CCL knobs, persistent vs barrier
semaphores and Linear vs Ring were swept (`test_vsa_kv_gather_perf.py`): Ring is 1.9x Linear, the
generic `ttnn.all_gather` is 6% faster than the async op, nothing else moves it. Dense ring
attention moves the same bytes under its compute. The remaining structural lever is that gather:
overlap it on a second command queue, or stream remote blocks inside the kernel over the fabric.

**Load balance.** Under the identity placement the SP-rank-0 devices hold every exempt (dense-list)
row: `vsa_sdpa` 24.3 ms there vs 15.9 ms median, and the block waits. `striped` spreads them over
shards but parks them at the front of each shard, moving the imbalance inside the kernel (first
pass / first workers). `interleaved` (default) also spaces them evenly within the shard:
min/median/max 17.0/17.5/20.2 ms at 15 s (identity: 15.4/15.9/24.3), 8.9/9.1/9.4 at 10 s (identity
7.6 median / 13.5 max), 3.1/3.2/3.3 at 5 s (identity 2.4 / 5.8). The residual spread at 15 s is the
+-1 exempt tile per shard (18 exempt tiles over 8 shards).

### 5a. K/V all-gather: link-bound (2026-09-03)

`test_vsa_kv_gather_perf.py` gathers the model K shard ([1,14,S_local,128] bf16, 51.8 MB at 15 s) along
the 8-wide SP axis with every all_gather_async configuration (Ring/Linear, persistent vs barrier
semaphores, chunks_per_sync / workers_per_link / buffers, the generic `ttnn.all_gather`):

| config | 15 s ms | GB/s received per device | 10 s ms |
|---|---|---|---|
| Ring, persistent buffer (model path) | 4.16 | 87 | 2.64 |
| Ring, tuned hyperparams (16/3/2) | 4.16 | 87 | 2.66 |
| Ring, other knob settings | 4.2-4.9 | 74-85 | 2.7-3.1 |
| Linear (any) | 7.7-9.0 | 40-47 | 4.9-5.5 |
| `ttnn.all_gather` (generic) | 3.91 | 93 | 2.50 |
| 4 links | n/a: the axis has 2 ethernet channels | | |

A device receives 7 x 51.8 MB = 363 MB per gather; 87-93 GB/s over 2 links is ~90% of 2 x 50 GB/s.
The serial time cannot be cut by tuning; only fewer bytes (excluded) or overlap (a second command
queue during the coarse stage, ~8 ms hidden at 15 s) remain.

### 5c. Coarse-stage cost reduction (2026-09-03)

**Device-side index assembly (shipped, default).** The streaming kernel takes the coarse stage's top-k
rows as they are (`list_len`, `exempt_ids`, per-device `dense_row_mask`), building the exempt prefix,
dense-list rows and sentinel handling into its per-row bitmaps; the host graph loses the concat /
tilize / int32 blend / typecast / untilize chain (~2.9 ms per block at 15 s). Output is bit-identical to
the host-assembled path (`test_vsa_sdpa_raw_selection_matches_assembled`).

**Padded pooled gathers (opt-in `MiniMaxH3VSAConfig.padded_pooling`).** The pooled K^T / V gathers were
falling into all_gather's composite path (broadcast + concat, ~1.6 ms at 15 s) because 226 tiles per shard
is not tile-aligned. With padding to 256 slots per shard the gathers are plain ring all-gathers; scores
and top-k run in the padded per-shard numbering and the kernel maps ids back
(`coarse_slots_shift`/`coarse_real_per_shard`). Block-level A/B at 15 s: 60.0 / 60.5 -> 58.9 / 59.6 ms
(the two small aligned gathers cost ~0.5 ms where the composite cost 1.56); now the default.

With device-side assembly and padded pooling the 15 s block is 58.9 ms on device 0 / 59.6 ms on the
slowest device (dense 73.4 / 75.4): VSA is 21% faster than dense at 15 s, up from 15% this morning.

## 5b. Planned kernel work (not started)

**Distributed group window (v18 candidate).** Today a worker's window is its own 12-slot ring, so a
row at 11% density sees ~1.3 of its blocks per window and visits are single-block: the softmax
bookkeeping (max, corr, rescale, sum) is paid per 64 keys, ~8x more often than dense's 512-key
chunks. Plan: every core in the head group reads a disjoint slice of the block sequence from DRAM
(no single leader), publishes availability, and the group's aggregate L1 holds ~48-96 consecutive
blocks. A core then gathers, per row, all of that row's listed blocks in the window (~5-11) from
peer L1 into its local ring and runs ONE QK/max/exp/rescale/PV over them -- dense-sized chunks.
Consequences: pulls become row-stationary (NoC traffic ~3x: ~164 vs ~55 MB per core per sweep,
~3 ms at 50 GB/s, acceptable but shared); all of a core's rows stay resident for one sweep
(accumulators ~500 KB) so a head needs one sweep instead of two passes (halves DRAM reads and
publish traffic); double-buffered 12-block visits need ~768 KB of slots -- tight at depth 12,
comfortable with 6-block slices; a window may slide only after every peer finished gathering from
it (group barrier per window, ~20-40 per sweep); dense rows chunk their visits to the ring size.
Expected 25% -> 35-45% util (exp per key unchanged; bookkeeping ~30% -> ~4% of visit cycles, better
MOP efficiency). Risk: many-to-many peer-L1 traffic (v8b's push model regressed for related
reasons) and the window barrier letting a slow core stall the group. First step: host-side check of
the visit-size distribution the real selections give under 48- and 96-block group windows.

**Intra-group load balance.** Rows are dealt chunk-cyclic with no cost weighting; an exempt
(dense-list) row costs ~3x a sparse row (9x the blocks, ~3x cheaper per block thanks to full
windows), and the leader's slot gate runs the whole group at the slowest core's pace, so one dense
row is ~+12-15% on its core and therefore on the head; under identity placement shard-0 cores held
2-3 of them. Plan: (1) per-core busy-time distribution from the probe-9 timers on an exempt-carrying
shard; (2) cost-aware dealing on the host (listed-block counts are known from the coarse stage;
dense rows weighted ~3x), which needs an explicit per-core row list in the reader/writer runtime
args instead of `row_start/row_stride`; (3) splitting a dense row's blocks across cores with a
partial-softmax merge belongs with the distributed-window redesign, not standalone.

## 6. Knobs and tools

- `TT_VSA_RMAX`, `TT_VSA_DEPTH`: resident rows per pass / stream depth (defaults 15 / 12; 14 fits
  an empty L1 only). A depth-18 configuration once hung (cb_corr sizing, fixed); treat non-default
  knob values as experimental.
- `TT_VSA_PROBE`: 1 delivery floor, 2 math floor, 3 protocol-only floor, 7 print CB layout, 9
  per-TRISC phase timers (DPRINT `VSAC ...`). Output is garbage in probe modes.
- `run_suite_then_bench.sh`, `run_sweep_rmax_depth.sh`, `run_vsa_perf_sweep.sh` (`MODES`/`DURS`),
  `run_h3_safe.sh` (galaxy reset + safe pytest), `run_debug_watcher.sh`.
- Post-mortem: `scripts/run_safe_pytest.sh` runs tt-triage on a dispatch timeout; a
  `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` hook can additionally dump NIU counters, stream
  registers (CB received/acked) and halt/step RISCs with ttexalens before the reset.
