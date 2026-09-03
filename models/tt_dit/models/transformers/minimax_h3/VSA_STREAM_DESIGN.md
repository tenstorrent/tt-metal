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

**Exp mode note.** `vsa_sdpa`'s compute-kernel default is `math_approx_mode=True` (inherited from the
op it was forked from) and the model does not override it, whereas the dense path passes
`exp_approx_mode=False`. The floor measurements above were taken with the default. Switching to the
exact exp is a one-line config change; it costs SFPU time on the pack thread (the current
bottleneck) and has not been measured yet.

## 3c. Determinism

The dense block is run-to-run deterministic. `vsa_sdpa` is **not bit-deterministic**: a worker's
window boundaries are closed by a starvation check as well as by the half filling, so the
partition of a row's blocks into visits -- and therefore the order in which bf16 partial results
are combined by the online softmax -- depends on arrival timing. Repeats agree to PCC 0.99998 (it
was 0.999 before the kreq-marker fix, which was the larger source). A drafted change closes windows
on fixed arrival bins instead (deadlock-free because fetch lag + bin width < stream depth), which
makes the partition, and so the result, a pure function of the inputs; it is not applied. The
coarse stage (pooling, scores, top-k) is deterministic.

## 4. Performance and its ceiling

Standalone, 15 s heaviest shard, median over runs: **16.2 ms topk / 16.0 ms model order, 24-25%
of HiFi2 peak on the listed math** (v1: 81.9 ms, 4.8%). Delivery floor (probe: consume visits, no
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
between two consecutive attention ops (an exact block period).

| duration | dense block (dev0 / max) | VSA block (dev0 / max) | dense attention | `vsa_sdpa` | VSA-only ops |
|---|---|---|---|---|---|
| 5 s  | 15.8 / 17.5 ms | 19.0 / 20.3 ms | 7.4 ms  | 3.3 ms  | 6.2 ms  |
| 10 s | 39.5 / 41.2 ms | 37.4 / 39.0 ms | 24.7 ms | 9.0 ms  | 11.7 ms |
| 15 s | 73.4 / 75.4 ms | 63.1 / 63.7 ms | 51.4 ms | 20.1 ms | 19.3 ms |

Component breakdown (ms, device 0):

| component | 5 s dense | 5 s VSA | 10 s dense | 10 s VSA | 15 s dense | 15 s VSA |
|---|---|---|---|---|---|---|
| attention core (ring SDPA / `vsa_sdpa`) | 7.44 | 3.26 | 24.70 | 9.01 | 51.43 | 20.10 |
| full K/V all-gather (fine-stage input) | - | 3.02 | - | 5.43 | - | 8.11 |
| coarse pooling q/k/v | - | 0.33 | - | 0.59 | - | 2.30 |
| pooled K/V gather + assembly | - | 0.40 | - | 1.03 | - | 1.63 |
| coarse scores + mask + softmax | - | 0.46 | - | 0.79 | - | 0.28 |
| coarse output o_c (probs@V, tile->token) | - | 0.40 | - | 0.72 | - | 1.11 |
| top-k selection + index assembly | - | 0.43 | - | 1.08 | - | 2.92 |
| gate branch (gate proj, heads, blend) | - | 1.16 | - | 2.03 | - | 2.93 |
| shared ops (norms, projections, MLP, adaLN) | 8.34 | 9.52 | 14.78 | 16.74 | 22.00 | 23.75 |

The shared ops are ~8% dearer under VSA because the sequence is padded to whole 64-token tiles
(14464 vs 13632 rows per device at 15 s). The K/V all-gather runs on 20 cores at ~9 GB/s; dense ring
attention moves the same bytes under its compute. The remaining structural lever is that gather:
overlap it on a second command queue, or stream remote blocks inside the kernel over the fabric.

**Load balance.** Under the identity placement the SP-rank-0 devices hold every exempt (dense-list)
row: `vsa_sdpa` 24.3 ms there vs 15.9 ms median, and the block waits. `striped` spreads them over
shards but parks them at the front of each shard, moving the imbalance inside the kernel (first
pass / first workers). `interleaved` (default) also spaces them evenly within the shard:
min/median/max 17.0/17.5/20.2 ms at 15 s (identity: 15.4/15.9/24.3), 8.9/9.1/9.4 at 10 s (identity
7.6 median / 13.5 max), 3.1/3.2/3.3 at 5 s (identity 2.4 / 5.8). The residual spread at 15 s is the
+-1 exempt tile per shard (18 exempt tiles over 8 shards).

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
