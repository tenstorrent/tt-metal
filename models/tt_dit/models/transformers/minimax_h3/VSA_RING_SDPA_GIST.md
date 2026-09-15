# VSA fine-stage attention on Blackhole: the single-chip kernel, and how the K/V ring all-gather is fused into it

MiniMax-H3 (FastH3) video generation on a 4x8 Blackhole galaxy, TP=4 x SP=8. This note explains `vsa_sdpa`
(the block-sparse fine stage of Video Sparse Attention) as implemented, then `vsa_ring_sdpa` (the same kernel
with the sequence-parallel K/V all-gather fused into it), with measured numbers throughout. Everything below is
lossless: bf16 in/out, HiFi2 matmuls, exact SFPU exp for every correction, exact row sums; outputs are bit-identical
run to run and match the dense reference at the bf16 floor.

Branch: `cglagovich/fast_h3_vsa`. Design records: `VSA_STREAM_DESIGN.md` (kernel), `VSA_RING_SDPA_SPEC.md` (fusion).

---

## 1. The problem shape

Per device at 15 s / 768p: 14 heads, `S_local = 14400` query tokens (225 q-rows of 64), the full sequence of
1800 K/V blocks of 64 tokens (T = 115200 after the SP gather), head dim 128. The coarse stage has already picked,
for every q-row, the top-k = 179 blocks (sparsity 0.9) plus a few *exempt* blocks (text/audio tiles) that every row
attends; rows of exempt tokens are *dense* (they attend every block). Listed math is ~10 % of dense attention.

| clip | blocks per shard | tile rows per shard | k | dense rows per shard |
|---|---|---|---|---|
| 5 s | 84 | 168 | 66 | 1-2 |
| 10 s | 151 | 302 | 119 | 1-2 |
| 15 s | 225 | 450 | 179 | 2-3 |

The naive kernel (one gather of the listed blocks per q-row) re-read every listed block from DRAM once per row:
~20 GB of 2 KB tile reads per layer per device, 178 ms, 2.4x slower than the dense ring SDPA it replaces. The whole
design below exists to read each K/V block from DRAM once and share it.

---

## 2. Compute pattern: one streaming leader per head, many resident workers

**Core groups.** The 120 compute cores (108 when the ring gather takes a row, see section 6) are split into 14 head
groups of 7-8 cores. One core per group is the **leader**, the rest are **workers**; the leader also computes its own
rows ("leader-as-worker"). Every core is *resident* for a fixed set of q-rows for the duration of a **pass**; L1
holds at most `rmax` rows (18 in ring mode, 10 standalone), so a head with ~30 rows per core runs 2-3 passes.

**The leader streams K/V once per pass.** It walks the head's blocks in order and DMA-reads each block's K and V
(16 KB + 16 KB) from DRAM into a ring of `stream_depth` L1 slots (10 in ring mode, 22 standalone). Slot reuse is
gated on the minimum *posted progress* of all consumers. Total DRAM traffic is therefore
`passes x n_blocks x 32 KB` per head, independent of how many rows list a block.

**Publish by multicast log, consume by pull.** For every fetched block the leader multicasts a 16-byte log entry
`{block, slot, seq}` into a ring in each worker's L1. A worker spins on the sequence word (`seq == n+1`), looks the
block up in its per-row membership **bitmaps** (built once per pass from the rows' index lists) and, if any resident
row lists it, *pulls* K and V from the leader's slot into its own slot of the same index: K on the writer RISC
(NoC 1), V on the reader RISC (NoC 0), each pull tagged with a transaction id. Pulling rather than pushing is what
lets the leader run `stream_depth` blocks ahead regardless of any worker's compute.

**Windows and halves.** A worker's slot ring is two *halves* of `stream_depth/2` arrivals. Arrivals fill the open
half; when it is full the window closes and the reader emits one **VISIT** message per resident row that listed
anything in it (the list of `(slot, count, mask)` entries) plus a **WINDOW** message. Windows close on fixed arrival
bins, never on timing, so the partition of a row's blocks into visits, and with it the bf16 order in which the online
softmax combines them, is a pure function of the inputs. That is where run-to-run determinism comes from.

**The compute kernel: a group-major window engine.** It buffers all visits of a window and runs them phase-major in
chunks, with one pack-to-unpack sync per phase per chunk instead of per visit:

1. QK for every visit into a per-window scratch region (`cb_qk`, cross-visit DEST batching), masks stamped for
   ragged/partial blocks.
2. Running row max: four visits per DEST acquire, the first visit of a row seeded with -inf.
3. `corr = exp(old_max - new_max)` only for rows whose anchor moved (lazy rescale, section 3).
4. O and sum rescale by `corr` (blocked packs).
5. Probabilities `exp(qk - reference)` and per-visit partial row sums.
6. PV through the matmul MOP with `kt` = the probs row stride, L1-accumulated into the row's resident O slot
   (overwritten on a row's first visit). Freed slots go back to the reader as **credits** (`cb_free` pages); at
   pass end each row is normalised by `1/sum` and written out.

Per 64-key block per q-row the FPU does 2 x 2x4x2 tiles of HiFi2 matmul (~1 k cycles) while the SFPU exp of the
2x2 probs tiles costs ~1.1 k cycles on the pack thread and the max/corr/rescale/sum reductions another ~0.9 k. The
dense kernel pays those reductions once per 512 keys; here every *visit* pays them, and a visit averages 1.3 blocks
on random lists and 2.4 on real ones. That fixed ratio is the structural ceiling of this design (~25-28 % of HiFi2
peak on the listed math).

**Role-split binaries.** Leader and worker bodies together overflow the kernel-config buffer, so the reader and
writer are compiled twice (`VSA_IS_LEADER`) as separate kernels on the leader and worker core sets.

---

## 3. Data reuse and exact numerics

**What is resident per row (L1, bf16 32x32 tiles of 2 KB):** Q (8 tiles, 16 KB), the O accumulator (8 tiles,
16 KB), anchor + threshold max tiles (4 tiles, 8 KB), a corr/candidate slot (2 tiles, 4 KB), plus bitmap and
fixed-point row-sum words: ~45 KB per row. Each stream slot costs K (16 KB) + V (16 KB) + its share of the QK
scratch and masks: ~43 KB. Fixed buffers (index row, counts, output double-buffer, row-sum FIFOs, control rings)
~100 KB. Blackhole L1 is 1.5 MB, the traced block leaves ~1.5 MB minus a 64 KB `l1_small` region: 18 rows x 10
slots fits, 20 x 10 and 22 x 8 do not. Rows and slots trade directly, and both matter: rows set the pass count and
the DRAM re-streaming, slots set how many blocks a visit can batch.

**Exact row sums off the FPU.** The 16-bit DEST cannot accumulate a row sum exactly over hundreds of blocks (at
858 listed blocks the bf16 running sum was 70 % off). Per visit the compute forms the partial sums with a block
matmul of the bf16 probs against a ones column (fp32 inside the FPU, one bf16 rounding per visit) and streams them
to the **writer RISC** through a 16 B header FIFO and a tile FIFO (`PARTIAL`, `CORR`, `FIRST`, `FLUSH`). The
writer accumulates in 64-bit fixed point (scale 2^16), applies the same corr the compute applied to O, and answers
`FLUSH` with bf16 sums the compute reciprocates. It serves one 16-row slice per loop so the K service is never
starved.

**Lazy anchor rescale (FlashAttention-3 style).** Each row keeps an ANCHOR max and a THRESHOLD = anchor + 2 logits.
Per visit the FPU reduce forms the candidate max and packs it; the UNPACK and MATH RISCs split the rows and decide
"moved" by comparing bf16 bit patterns as order-preserving integers (the threshold keys are cached contiguously per
row slot), then exchange masks by mailbox so all three threads branch alike. Only moved visits (~4 % on real
selections) pay the exact `exp`, the O rescale and a `CORR` message. Probabilities are taken against the threshold,
so the exp only ever sees non-positive inputs. Result: PCC 0.99969 against fp32 at 1024 blocks, every row within
the bf16 floor, gain 0.998-1.001.

---

## 4. Sync, without semaphores in the hot path

- **Leader -> workers**: the multicast log ring; the sequence word in each 16 B entry is the barrier (a consumer
  spins until `seq == n+1`). Two entries per multicast.
- **Workers -> leader**: each worker posts its progress (the highest slot it has finished pulling, up to the first
  unconfirmed pull) into an *ackbox* word in the leader's L1; the leader recycles slot `s` only when the minimum
  posted progress, including its own compute's credits, has passed `s`. So a DRAM refill can never race a PV that
  still reads the slot.
- **Within a worker**: reader -> writer K requests (`cb_kreq`), writer -> reader K acks in order per half
  (`cb_kack`); reader -> compute visits (`cb_ctrl`); compute -> reader credits (`cb_free`); compute -> writer
  row-sum FIFOs. Window closes are lazy: the K marker and the "landed" check are polled while the next half fills.
- **Invariants that bit** (all now regression-tested): nothing in L1 is read before this run writes it (a stale
  control word from a previous op hung the chip); `invalidate_l1_cache()` wherever a RISC reads what NoC or another
  RISC wrote; every protocol word the leader zeroes is re-posted by the worker until observed; malformed protocol
  state parks the RISC in a *named* trap loop so a hang's triage names the corruption.

**Row dealing (host).** Sparse rows are dealt to consumers in 4-row chunks round-robin (adjacent rows share ~55 %
of their lists, which the multi-row batching relies on) and passes are filled in order. Dense rows cost ~7 sparse
rows and are placed one per (pass, consumer) bin on the least-loaded bin; in ring mode they go into pass 0 (see
section 7). Every consumer of a group runs the same number of passes in lockstep with the leader's sweep, so it is
the per-pass cost that must balance.

---

## 5. Where the single-chip time goes

Standalone `vsa_sdpa`, 15 s per-device shape, 120 cores, random lists with the model's k and 4 dense rows,
slowest device:

| measurement | ms |
|---|---|
| full kernel | 17.35 |
| delivery floor: consume every visit, no math (`TT_VSA_PROBE=1`) | 9.67 |
| the two K/V all-gathers it needs (separate ops) | 8.4 |

So 56 % of the kernel is the delivery protocol and the leader's streaming; the math sits on top. Timers from the
lever pass (`VSA_STREAM_DESIGN.md` sections 10-11, real device-5 indices) put the three compute TRISCs busy
~55-60 % and idle 40-45 % waiting for the next window, with MATH's busy time split QK 13 %, candidate max +
decision 17 %, deferred PV + partial sums 15 %, exp 7 %, flush 3 %. The leader's reader spends 30 % waiting on its
own compute's credits, 17 % on the workers' progress, 20 % issuing V reads, 13 % publishing; worker readers spin
62-87 % of the time for the next log entry.

The event timeline (probe 10) named the idle: a **convoy**. Real selections are spatially clustered, so per-window
compute varies 2-3x between cores, and an 18-20 arrival ring (two windows) cannot average it out. The kernel
alternates between bursts, where the slowest consumer gates the leader and everyone else idles, and sparse
stretches, where every consumer waits for the leader's cadence. Things measured and rejected on the way: a
compute-free leader with a 40-deep ring (worse: fewer consumers, a fourth pass), a third ring stage, moving V
fetches to the other RISC, deeper prefetch, per-bank address generation, dropping the workers' pulls entirely (the
leader's egress is not the limit), two passes instead of three (DRAM bandwidth is not the limit). What did help:
the cheap lazy-max decision (-2.5 ms), 2-stage rings of 9-10 (-0.5 ms), and blocked-stride stream orders (-9.5 %
standalone, parked behind a leader-protocol race).

In the transformer block (tracy, per-op device time on the slowest device, two-op path):

| op | calls | ms |
|---|---|---|
| all_gather_async K and V (fine stage) | 2 | 8.7 |
| vsa_sdpa | 1 | 18.3 |
| all-gather + matmul (QKV / out projections) | 4 | 15.3 |
| matmul + reduce-scatter | 1 | 3.2 |
| embeddings, tilize/untilize, typecasts, norms, small matmuls | ~90 | ~22 |
| block period (traced, 20 replays) | | 62.3 |

The fine stage's gather + attention is 27 of the 62 ms. The gather is pure latency the attention could hide.

---

## 6. Fusing the ring all-gather

**Idea.** Forward K/V shards around the SP ring while the attention consumes the shards that have landed, carrying the
online-softmax state across shards. One program per device: the gather's sender cores take the first row of the
compute grid, the VSA engine the remaining 108 cores; the leaders read their own shard from the local K/V tensors and
every other shard from the persistent gathered buffers as it lands. Pass 0 is the only pass that can overlap (later
passes reuse the row slots), so the game is to make pass 0 long enough to cover the gather and to let it consume each
shard as early as possible.

**Two gather backends** are selectable (`gather=`):

- `ring_attention` (default, the merge candidate): the stock `ring_attention_all_gather_async` helper, unmodified.
  One worker per link per direction on direct fabric connections, K and V as its two inputs, store-and-forward
  through the gathered buffer one *slice* (device shard) per hop, one signal to the fused op per landed slice, the
  diametric shard of an even ring split across both directions. The leaders gate **per shard** through
  `RingSDPAOpReceiver` (it implements the split second-half wait).
- `fused_kv`: the op's own gather, forked from `all_gather_async`'s multi-worker kernels: two workers per link per
  direction behind a fabric MUX, plain head-split K and V in, a **token-major page walk with a stride between heads**
  (per tile row: K of every head, then V of every head) so every head's blocks land progressively, per-`chunks_per_sync`
  landed counters. The leaders gate **per block** by polling those counters over the NoC, with the per-shard signal as
  the always-arriving fallback.

Ordering is fixed by the schedule in both modes (gating only delays), so both are bit-identical run to run and equal
`vsa_sdpa` on the gathered K/V up to bf16 rounding order.

**Three compute-side changes that mattered as much as the gather.**

1. *Resident rows 10 -> 18 in ring mode* (host cap lifted from 16 to 32; the kernel arrays were already sized 32). Two
   passes instead of three, and a pass 0 long enough to matter.
2. *Dense rows into pass 0.* A dense row costs ~7 sparse rows of work for one L1 slot. With pass 0 full at `rmax`,
   the dealer used to drop the block's 4 dense rows into pass-1 bins that already held their share, so pass 1 ran 19
   cost units against a 16 average and pass 0 stayed shorter than the gather. Now a dense row evicts one sparse row
   from a pass-0 bin into the lightest later bin, and later passes are cost-balanced across consumers. Pass 0 becomes
   19 sparse + 1 dense = 26 units on four consumers, pass 1 shrinks to 12-13 rows everywhere.
3. *Landing order in runs.* Pass 0 streams the own shard first, then one shard per direction at a time; with the
   per-block gate the blocks are interleaved across the gather workers' quarters in runs of 32 consecutive blocks,
   because real selections list neighbouring blocks and the windows batch on that locality (runs of 1: 24.5 ms,
   8: 23.8, 32: 23.7 in the block).

---

## 7. What the ring version measures

**Op level** (15 s per-device shape, 4 dense rows, slowest device, ms):

| | two-op (gather x2 + vsa_sdpa) | ring, stock gather | ring, fused gather |
|---|---|---|---|
| default | 25.44 | 22.70 | 19.89 |
| per-shard gate on the fused gather | | | 20.07 |
| gate held open (compute racing the gather, timing only) | | | 20.71 |
| serialized (gather, then compute) | | | 29.2 |
| no math (`TT_VSA_PROBE=1`, delivery floor) | 17.9 | 18.78 | |

**In the block** (tracy device time of the fused op on the slowest device; sender cores = the gather's kernels,
VSA cores = the 108 attention cores; ms from the first kernel start):

| variant | gather end | VSA cores end min / median / max | op |
|---|---|---|---|
| stock gather, per-shard gate (default) | 13.8 | 21.4 / 23.3 / 25.0 | 26.4 |
| stock gather, gate held open | 13.9 | 18.1 / 18.8 / 20.1 | 21.25 |
| stock gather, serialized | 13.7 | 31.5 / 32.2 / 33.6 | 34.6 |
| fused gather, per-shard gate | 11.1 | 18.6 / 20.5 / 21.8 | 23.4 |
| fused gather, per-block gate | 11.2 | 18.5 / 20.2 / 21.8 | 23.3 |
| fused gather, serialized | 10.6 | 29.8 / 30.4 / 32.7 | 32.8 |

**Block period and end to end** (traced block, 20 replays; denoise per step over 8 steps with real weights, vsa
baselines measured the same day):

| clip | block: two-op / ring (fused) | denoise: vsa / ring stock / ring fused |
|---|---|---|
| 5 s | 20.50 / 18.94 | 0.859 / 0.850 (-1.0 %) / 0.816 (-5.0 %) |
| 10 s | 39.78 / 37.32 | 1.701 / 1.640 (-3.6 %) / 1.568 (-7.8 %) |
| 15 s | 62.30 / 59.48 | 2.727 / 2.617 (-4.0 %) / 2.569 (-5.8 %) |

Correctness gates for both backends: 8 op-level variants (1 and 2 links, eager and traced replay with the alternate
semaphore set, dense rows, an odd block count that splits the gather's rows mid-block) bit-exact and PCC 0.9997 vs
torch; the traced block at PCC 100 % against the untraced one at 5, 10 and 15 s.

---

## 8. What is preventing better overlap, in order

1. **Gather bandwidth.** The stock helper's single worker per link moves K+V in 13.8 ms inside the program; the
   MUX'd two-worker gather in 11.1 ms. The per-core data settles what that costs: under per-shard gating every head
   group's pass 0 stretches to *last arrival + one shard of work* (~13.8 + 1.9 ms) however fast the group is, then
   its pass 1 follows; the groups with 7 consumers (32 rows per consumer, 14 in pass 1) finish last, 3.6 ms after
   the 8-consumer groups. Every millisecond off the gather comes straight off the block. Note this is *not*
   contention: the senders finish at 13.8 ms whether or not the VSA is running.
2. **Pass 0 is bounded by L1.** With 18 rows and the dense rows, pass 0 is ~15 ms of compute at 15 s, enough to
   cover either gather; but pass 1 (~6-8 ms) never overlaps anything. Only more L1 per core (or fewer rows per core,
   i.e. more cores) shrinks it, and rows cost 45 KB each.
3. **The 12 (or 4) sender cores** cost the attention 10 % of its cores, ~1.8 ms of compute at 15 s.
4. **Gate granularity**, once the gather is fast and signals promptly, is worth ~0.1-0.2 ms in the block and 0.2 ms
   standalone. It was worth 0.7-3 ms only while the gather's signal lagged its landing.
5. **Per-shard tail with random selections.** The block test's random-data selection is uniform over shards; real
   selections are front-loaded on the near shards, which is why the stock path gains 4 % end to end while the block
   test shows none. Ordering within pass 0 cannot fix a tail that lands last by construction.
6. **The kernel itself** remains the floor: 21-22 ms on 108 cores under the gather's DRAM traffic, ~40 % of TRISC
   time idle in the convoy described in section 5. That is the compute problem, not the overlap problem.

**Recommendation for the merge.** The stock-gather path is correct at every shape and worth 1-4 % on the denoise
step with no CCL changes. Adding MUX multi-worker support to `ring_attention_all_gather_async` behind a default-off
`num_workers_per_link` (one worker = today's kernels, byte for byte) would recover most of the fused gather's 5-8 %
with per-shard signalling unchanged; per-chunk counters and the token-major walk are worth the last 0.2 ms only.

---

## Appendix: how the numbers were taken

- Op level: `tests/ttnn/unit_tests/operations/sdpa/test_vsa_ring_sdpa_perf.py -k 15s` (shapes `5s|10s|15s`; env
  `VSA_RING_GATHER`, `TT_VSA_RING_COARSE`, `TT_VSA_RING_GATE_OPEN`, `TT_VSA_RING_WAIT_ALL`, `TT_VSA_PROBE`).
- In the block: `python -m tracy -r -p -m pytest models/tt_dit/tests/models/minimax_h3/test_vsa_block_minimax_h3.py -k untraced`
  with `VSA_RING_BLOCK=1` (`VSA_RING_GATHER`, `VSA_BLOCK_SECONDS`); per-op durations from `ops_perf_results*.csv`,
  per-core kernel end times from `profile_log_device.csv` (`*-KERNEL` zones per core and RISC for the op's
  `GLOBAL CALL COUNT`, relative to the earliest kernel start on the device).
- Block period: the traced block test with `VSA_BLOCK_PERF_ITERS=20`. End to end: `test_vsa_e2e_perf_minimax_h3.py`
  with `VSA_E2E_MODE=vsa|ring`, `VSA_E2E_SECONDS`, 8 steps, real weights.
