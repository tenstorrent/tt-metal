# Implementation plan: DRAM-optimized matmul for small Mt (≤8), regime A (M≪N)

Consolidates the experiment learnings (`forward_experiments_findings.md`, Exp 1-7) into the
DRAM-optimized matmul. Scope: **regime A (N>M), Mt = 1..8 tiles.** Large Mt (2D partition) deferred.

## Target & vehicles
- **Prototype (fast iteration, correctness by constant-input):** `tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/`.
  Already does reader==consumer + bank-adjacent multi-reader and WINS Mt=1 (1.09-1.27×); LOSES Mt≥2
  (all-NOC0 readers contend → only 8 effective cores → compute-bound).
- **Production:** a new program-factory path in `ttnn/.../experimental/minimal_matmul/device/`
  (regime-A variant), selected by host when M≪N + low AI + in1 DRAM-width-sharded.

## Final architecture (all learnings applied)
1. **Split-NOC multi-reader (Exp5) — the core fix.** 8·P reader/compute cores, alternating **NOC0/NOC1**,
   each placed near its per-NoC optimal core (`get_optimal_dram_bank_to_logical_worker_assignment(NOC0)`
   and `(NOC1)`). Sustains ~500 GB/s (98-99% of cap) up to 96 cores — vs all-NOC0's 390-428. This gives
   Mt≥2 the extra compute cores it needs while holding the read at peak.
2. **reader==consumer, no forwarding (Exp2/4).** Each core reads its OWN unique contiguous in1 N-sub-band
   `[n_sub, K]` (peak: `one_packet_with_state` + triple-TRID, 16KB bursts) and computes it. No mcast/scatter
   of in1 (funnels to ~150 GB/s).
3. **in1 layout:** DRAM-width-sharded along N, laid out **N-major per bank ([N_band, K])** so each reader's
   N-sub-band is *contiguous* (the peak pattern). P readers/bank each take a contiguous 1/P sub-band.
4. **in0 = dedicated loader(s) + K-block-streamed broadcast (Exp6).** 1-2 separate loader cores read the
   tiny in0 `[M,K]` interleaved (K-block streamed → bounded L1), and **multicast** each K-block to the
   compute cores. NOC choice irrelevant (mc0≡mc1); do NOT split by idle-NoC (Exp6). Do NOT have each core
   read in0 from DRAM (8P× redundant — the prototype's current bug).
   - L=1 for Mt≤2, L=2 for Mt≥4 (Exp6: L=2 halves the small dent).
   - Pipeline the loader's read (one DM RISC) ∥ mcast (other DM RISC).
5. **Per-core RISC assignment (must accommodate split-NOC):**
   - NOC0-reader core: in1 read = BRISC/NOC0; output write = NCRISC/NOC1.
   - NOC1-reader core: in1 read = NCRISC/NOC1; output write = BRISC/NOC0.
   - in0 received passively into L1 (mcast, no RISC); compute on TRISC.
6. **output:** interleaved, each core writes its own `[Mt, n_sub]` on its free DM RISC.
7. **Partition heuristic (small Mt):** pick num_cores = 8·P and L from (M,K,N):
   - P grows with Mt (compute-cores-needed): Mt=1→P=1 (8c), Mt=2→P=2 (16c), Mt=4→P=4 (32c), Mt=8→P=8 (64c),
     capped so N_tiles % num_cores handled (round-down active cores) and each core's n_sub ≥ 1.
   - **Guard the in0-broadcast burden 8P·Mt ≲ N_tiles (Exp7).** If violated (small-N shapes, e.g. …×768),
     cap P (fewer cores, accept some compute-bound) — do NOT let broadcast dominate.
   - even NOC0/NOC1 split.

## Phased implementation

**Phase 0 — Baseline.** Build `regime_a_mm`, reproduce prototype numbers (Mt=1 win, Mt≥2 loss) as the
bar. Record per-shape branch µs from `bh_skinny_results.md` as the comparison target.

**Phase 1 — Split-NOC multi-reader (the core change).** In `regime_a_mm` sharded mode:
- Place P readers/bank alternating NOC0/NOC1 near opt0[b]/opt1[b]; store per-core NoC.
- Create the reader kernel TWICE (RISCV_0/NOC0 set, RISCV_1/NOC1 set); output writer on the opposite RISC
  per core. Make `reader_sharded.cpp`/`reader_subband.cpp` NoC-agnostic (contiguous N-sub-band read).
- Switch in1 to N-major contiguous sub-bands.
- **Milestone:** Mt=2,4,8 flip from loss to win; in1 read ~500; constant-input correctness (out==K).

**Phase 2 — in0 dedicated loader + broadcast.** Replace per-core in0 (`in0_writer.cpp` phase 1) with:
- L loader cores: `loader_read.cpp` (interleaved, K-block, pipelined) ∥ `loader_mcast.cpp` (broadcast to
  the compute rectangle). Reuse the kernels from `sp_forward/` (proven in Exp6).
- Compute cores receive in0 into a `cb_in0` (mcast dst, uniform L1 offset across all targets).
- Handle the **BH mcast rectangle gotcha**: compute cores are bank-scattered → broadcast to the full
  compute-grid in TWO column-bands (logical cols 0-6 and 7-10, contiguous phys rects), NOC1 corner-swap
  (start=max,end=min). Output writer now owns NCRISC/BRISC alone (in0 off the compute cores' RISCs).
- **Milestone:** in1 read stays ~500 with in0 concurrent; correctness holds; aggregate ≈ Exp6 (~480-490).

**Phase 3 — Partition heuristic + edge cases.** Host picks (P, L) per §7 above; handle N_tiles not divisible
by 8P (active-row round-down), small-N burden cap, shallow-K / L1-budget edges (the prototype's FAIL/L1:
per-shape CB sizing; K-block streaming already bounds in0). **Milestone:** all 20 regime-A shapes run,
correct, and meet/beat the projected speedups in `forward_experiments_findings.md`.

**Phase 4 — Productionize into `minimal_matmul`.** New program-factory variant:
- Host regime-A detector (M≪N, AI<threshold, in1 width-sharded) → dispatch to this factory.
- Port validated kernels; real PCC (not constant-input); **program-cache correctness in
  `override_runtime_arguments`** — carefully index output/in0/loader addrs (recall the BH fused-Kpar cache
  bug where the output-addr index missed extra args). ttnn pytest with the 20 shapes, PCC≥0.99.

## Test plan
- **Correctness:** prototype constant-input (out==K, all K); ttnn PCC≥0.99 fresh + program-cache replay
  (fresh-vs-cached must match — the cache bug wrote all-zeros on replay). Always verify PCC explicitly
  (empty grep ≠ pass — the auto-block silent-corruption lesson); check `N_sub % subblock_w` validator.
- **Perf:** device-profiler kernel-time BW-util (`parse_kernel_bw.py`) + wall-clock speedup vs branch, on
  all 20 regime-A shapes. Track in1-read BW, aggregate (in0+in1), and vs `bh_skinny_results.md` branch µs.
- **Sweep:** per-shape (P, L, kb) around the heuristic to confirm the picker; ablate split-NOC vs all-NOC0
  (should reproduce Exp5's 500 vs 390).
- **Hang-safety:** timeout+auto-`tt-smi -r` harness (device hangs are non-deterministic under load); run
  each config in its own device session so `ReadDeviceProfilerResults` flushes (mm_sweep tooling).

## Key pitfalls to bake in (from experiments)
- NOC1 multicast corner-swap (start=max,end=min) or multi-core mcast HANGS (single-core masks it).
- BH grid x=8-9 gap: mcast only to contiguous valid worker rects (two column-bands).
- L1 budget: deep in1 CB (triple-TRID) + fp32 output block; per-shape CB sizing; in0 K-block-streamed.
- Split-NOC needs per-NoC optimal placement (opt0 vs opt1) — wrong-NoC center degrades (Exp5).
- Burden 8P·Mt ≲ N_tiles or in0 broadcast dents/collapses (Exp7) — the heuristic must cap P on small N.
- Program-cache override arg indexing (BH fused-Kpar cache bug).

## EXECUTION STATUS (in `regime_a_mm`)
- **Phase 0 (baseline):** reproduced — Mt=1/8c = 96% util; Mt=2/24c all-NOC0 = 70%; Mt=4/24c all-NOC0 = 36%.
- **Phase 1 (split-NOC): DONE + correct.** `--sharded` now places P readers/bank alternating NOC0/NOC1 near
  opt0/opt1; in1 read on the reader's NoC, in0+output on the other. `--nosplit` forces all-NOC0 for A/B.
- **Ablation (validates the whole plan):** with in0 read made free (`--skipin0`): Mt=4 36%→73% (removing
  redundant per-core in0 read), then +split-NOC 73%→**90.6%**; Mt=2 70%→77%→**93.4%**. Both levers needed
  and compound; ceiling ~90-93% util (vs branch ~74%). Confirmed the dominant Mt≥2 cost is the **redundant
  per-core in0 DRAM read** (at Mt=4/24c it is 36MB vs 27MB in1 — exceeds in1), then read-contention.
- **Phase 2 (in0 broadcast): implemented + correct, NOT yet beating branch.**
  - Up-front full-in0 broadcast: correct but REGRESSES (serial prologue — compute waits for all in0).
  - Streamed-into-full-cb0 (per-K-block mcast + monotonic valid, no backpressure since cb0 holds full in0):
    correct; 64×6144×4608 69.5%→72.2% (approaching branch 74.3%) but 64×6144×1536 49.7%→**26.6%** (regress).
  - **Loader efficiency (DONE, big gains).** Iterated the loader: (1) drop per-block mcast barrier via
    `noc_semaphore_inc_multicast`; (2) multi-block mcast chunks (chunk=8) to amortize per-mcast cost;
    (3) mcast to the compute-core bounding box, not the whole grid; (4) per-chunk read∥mcast overlap
    (read chunk c while compute consumes c-1; only chunk-0 read is a small prologue).
    Progression on the hard small-N shape 64×6144×1536: **34% → 64% → 70.5%** (ceiling skipin0=96%).
    64×6144×4608: 80.8% → **88.0% (1.18× branch 74.3)**. Correct throughout.
  - **L=2 loaders (implemented, measured): NOT the lever for small-N.** Two loaders (contiguous K-split,
    2 valid semaphores) marginally help large-N (4608: 88.3%→88.8%) but HURT small-N (1536: 70.7%→63.1%).
    Root cause reframed: small-N is limited by the **per-core in0/in1 ratio = Mt/N_block**, not loader
    throughput. At 1536 P3, N_block=2 → each compute core receives as much in0 (768KB) as it reads in1
    (768KB), so the broadcast L1 writes contend with the in1 read; adding a 2nd loader adds write pressure.
    The lever is the **partition** (pick P to balance compute-boundedness vs in0-burden), not more loaders:
    1536 sweep — P1/8c compute-bound 54.7%, P3 in0-heavy 70%, **P2/16c sweet spot 71.7%** (branch 76.8, 0.93×).
  - Status: large/mid-N Mt=2 clearly beat branch (4608 = 88%, 1.18×). Smallest-N (1536) tops at ~72% (P2),
    a genuinely hard small shape (N=1536,K=6144,M=64) — bcast still beats non-bcast there (72 vs 58) but not
    the branch's 76.8. Next: Phase 3 partition heuristic to auto-pick P (8P·Mt ≲ N), and DRAM-shard in0 for
    contiguous loader reads. Default nloaders=1 (L=2 behind `--nloaders 2`).
  - L1: full-in0 cb0 only fits Mt≤2 with shallow K (in0_full = M_block·Kt ≲ ~700 tiles). Deep-K (e.g.
    Kt=480) and Mt≥4 need a depth-limited (backpressured) streamed cb0.
- Kept behind `--bcast` so the default path is unaffected. Mt=1 → use non-bcast (per-core in0 read = 96%).

### Next steps to finish Phase 2 (ordered)
1. **Cut the loader's mcast cost:** pipeline it (issue several K-blocks before a barrier, triple-buffer like
   the in1 reader) and stop per-block valid barriers; mcast to a tight bounding box of the compute cores,
   not the whole grid; run the loader read∥mcast on its two RISCs.
2. **L=2 loaders** (Exp6) split by K-block interleave to halve the loader's per-block work.
3. **Depth-limited backpressured cb0** for Mt≥4-8 (remote-CB handshake) so it fits L1 and streams.
4. Re-measure across the 20 regime-A shapes; target the ablation ceiling (~90%).

## NEXT STEP (recommended): K-partitioning for small-N (split-K instead of N-split)
Small-N is capped by the per-core in0/in1 ratio. Two ways to add cores:
- **N-split (current):** 8P cores by splitting each bank's N_band into P → in0/in1 = **Mt·P/N_band** (grows
  with cores). At 64×6144×1536 P3: ratio 1.0 → ~71%.
- **K-split (recommended):** 8·Pk cores by splitting each bank's K into Pk slices; each core keeps the FULL
  N_band and reads only in0[:, k-slice] (Mt·Kt/Pk) → in0/in1 = **Mt/N_band**, *independent of Pk*. At 1536:
  ratio 0.33 (same as the large-N shapes already at ~88%). Total in0 broadcast = 8·Mt·Kt (vs N-split
  8P·Mt·Kt = P× more). Bonus: a K-slice is a contiguous row-range of a K-major bank shard → contiguous
  in1 reads (better than strided N-sub-bands); composes with split-NOC (the Pk cores/bank alternate NoCs).
- **Cost:** a partial-sum reduction across the Pk cores of each bank ([Mt, N_band] partials summed). Cheap
  for small Mt·N_band (exactly small-N/small-Mt). This is the **split-K (Pk) lever** — minimal_matmul already
  has a fused on-device K reduction (TT_MM_K_FUSED, won 4.32× on output-starved shapes); the main branch's
  (S,Pk) heuristic already picks Pk for these shapes.
- **Prediction (from existing data):** util tracks N_block/N_band; K-split gives 1536 an effective N_block =
  N_band = 6 (like 4608), so ~80-88% expected (up from 71.7%, beating branch 76.8) minus the small reduction
  cost. Decisive test = prototype the reduction in regime_a_mm.
- **Heuristic:** N-split while N_band/P ≳ Mt (large N); K-split when N is small; or joint (S,Pk).

## K-SPLIT IMPLEMENTATION PLAN (regime_a_mm `--ksplit Pk`)
Goal: fill cores via K-slices (not N-slices) so each core keeps the full N_band → in0/in1 ratio = Mt/N_band
(independent of core count). Reuse compute.cpp's built-in `REDUCE_K` column reduction; add chain-forwarding.
- **Cores:** 8 banks × Pk slices; bank-adjacent, split-NOC (alternate NOC0/NOC1 across the Pk).
- **in1 read:** each core reads its bank's contiguous K-slice [Kt/Pk, N_band] (reader_sharded + a byte
  offset k_start·N_band·tb). N_block = N_band (full bank width). K_num_blocks = (Kt/Pk)/kb.
- **in0:** each core reads only in0[:, k-slice] = Mt·(Kt/Pk) from DRAM (per-core; 8× redundant across banks
  but Pk× smaller per core — no broadcast machinery for v1; per-k-slice broadcast is a later optimization).
- **compute:** compute.cpp with `-DREDUCE_K`; `is_reduce_bottom` = (slice==0). Bottom copies its partial to
  out_cb; others wait cb_reduce (c_7) and `reduce_add_block(intermediate, cb_reduce)→out_cb`.
- **reduction chain (new DM `in0_reduce_writer.cpp`, on the core's non-in1 RISC):** phase1 = read in0
  k-slice into cb0; phase2 = if !bottom: reserve cb_reduce, wait recv_sem, push cb_reduce (feeds compute);
  then cb_wait_front(out_cb); if !top: NoC-write out_cb → next slice's cb_reduce base + inc its recv_sem;
  if top: write out_cb → DRAM [Mt, N_band]. One [Mt,N_band] block per link (matmuls overlap; only the
  add+forward tail serializes — cheap for small Mt·N_band).
- **Test:** constant-input `out==K`; then measure 64×6144×1536 (expect >71.7%, target ~78-88%).

### K-split EXECUTION STATUS (`--ksplit Pk`) — implemented + correct, but needs broadcast to win
- **Implemented + CORRECT:** reader_sharded reads a contiguous K-slice (byte offset); `in0_reduce_writer`
  does phase-1 in0 k-slice read + phase-2 chain reduction; compute.cpp reused with `-DREDUCE_K`. Fixed a
  real bug: a middle core forwarded to `get_write_ptr(cb_reduce)` which drifts after it receives a block →
  captured the uniform cb_reduce base once at kernel start. Pk=2,3,4 all PASS (out==K).
- **Perf finding (contradicts the optimistic prediction):** 64×6144×1536 K-split = **~64% flat across
  Pk 2/3/4** — BELOW N-split-broadcast (71.7%). Root cause: this v1 reads in0 **per-core** (no broadcast),
  and each k-slice is read by all 8 banks' slice-j cores → **8× redundant in0 DRAM read** (= 8·Mt·Kt total),
  which competes with in1 at the DRAM cap. K-split's *L1-burden* win (ratio Mt/N_band) is real but is offset
  by this redundant *DRAM read*. (K-split per-core still beats N-split per-core 58% — consistent with less
  total in0 read than N-split's 8P·Mt·Kt — but loses to broadcast which eliminates the in0 read entirely.)
- **Conclusion:** K-split needs **per-k-slice in0 broadcast** (each k-slice read once, mcast to its 8
  slice-cores) to get BOTH low L1 burden (Mt/N_band=0.33) AND low DRAM read.

### K-split broadcast — ATTEMPTED, blocked by a fundamental placement conflict
- **Ceiling confirmed:** K-split with in0 read made FREE (`--skipin0`) = **94.9%** at Pk=2 (vs 62.6%
  per-core-read). So IF in0 could be delivered per-k-slice cheaply, K-split would reach ~90% and beat branch.
- **The blocker:** per-k-slice broadcast requires the 8 slice-j cores to form an addressable rect (so the
  broadcast for slice j hits exactly them, disjoint from other slices). But those 8 cores are one-per-bank,
  and the in1 read REQUIRES bank-adjacency — which scatters them across the grid. Grouping them into a Pk×8
  rect (rows=slices for broadcast, cols=banks) breaks adjacency. Measured (`--rect`, per-core read): the
  non-bank-adjacent in1 read **collapses to 28.6%** (vs 62.6% bank-adjacent) — reader_sharded's triple-TRID
  can't hide the longer latency to a far bank. So rect (good broadcast, dead read) and bank-adjacent (good
  read, un-groupable broadcast) are **mutually exclusive** on this HW.
- **Net:** neither K-split variant beats N-split-broadcast (71.7%) for small-N: per-core-read is DRAM-read
  bound (62%), rect is read-collapse bound (28%). **N-split-broadcast remains the best small-N option
  (~72%, 0.93× branch).** The K-split reduction machinery is implemented + correct and remains valuable for
  output-starved/deep-K regimes (classic split-K), just not for the small-N in0-burden problem.
### ✅ K-split in0 STORE-AND-FORWARD (`--fwd --chain {bank|nn}`) — SOLVES small-N, matches branch
Insight (user): mcast needs a rect (breaks bank-adjacency); per-core read is 8× redundant. Instead, per
k-slice group (the 8 same-slice cores, one per bank) do a **unicast store-and-forward chain**: one injector
reads in0[:,k-slice] from DRAM ONCE, uses it locally, and forwards each block to the next core in the chain;
each core stores in cb0 (its matmul) and forwards on. No rect (unicast to a specific next core → bank-
adjacency preserved), and in0 read from DRAM once (Mt·Kt total, not 8×). cb0 holds the full k-slice (no ring
backpressure). Orthogonal to the reduction chain (which runs along banks). New kernel `in0_fwd_reduce_writer.cpp`.
- **Results (64×6144×1536, branch 76.8; ceiling/skipin0 94.9):** per-core-read K-split 62% →
  **store-and-forward Pk=3 = 76.0% (nn) / 75.7% (bank) — 0.99× branch**, and beats N-split-broadcast (71.7).
  Pk=2 = 72.8, Pk=4 = 70.2 (Pk=3 is the sweet spot). Correct (out==K) throughout.
- **Pipelining matters:** dropping the per-block forward barrier (data-then-inc are NoC-ordered) lifted
  Pk=3 from 74.0% → 76.0%.
- **Chain order:** nearest-neighbor (nn) is marginally better than bank-index (0-7) order (~+1-2pt) — NoC
  distance/congestion has a small effect because the forwarded in0 volume is tiny. So order helps a little,
  not decisively.
- **Verdict:** store-and-forward is the small-N answer. It keeps bank-adjacent reads, avoids redundant in0
  reads, needs no rect, and brings the hardest small-N shape to branch parity (0.99×). Combined with the
  earlier wins (large/mid-N Mt=2 at ~88% = 1.18×), the regime-A small-Mt story now wins or ties everywhere.
  Remaining gap to the 94.9% ceiling = store-and-forward NoC/chain overhead + reduction tail (future tuning:
  deeper forward pipeline, overlap reduction).

## NEXT IDEA: in0 RING ALL-GATHER (every core is an injector)
Store-and-forward uses ONE injector per k-slice group (8 same-slice cores) to read the whole k-slice and
forward it down a chain — the injector is a serial read + the chain has fill latency (8 hops). Ring idea:
**every core in the group is an injector.** Each core reads a distinct shard of the k-slice's in0 from DRAM,
computes it locally, and rotates shards around the ring so every core eventually computes every block. Same
total in0 DRAM read (Mt·Kt once) but spread across G=8 cores (G× more injectors, parallel read, no chain
fill), and a tight L1→L1 neighbor forward with a small ring buffer (not the full k-slice in cb0).

### Mechanics (per k-slice group, G=8 cores in a ring, Kb = Kt_local/kb blocks)
- Split the Kb blocks into G **contiguous shards** (sizes ceil/floor for the Kb%G remainder). Ring position
  p reads shard p from DRAM (parallel across the 8 cores).
- Step r = 0..G-1: core p holds shard (p−r) mod G; it matmul-**accumulates** that shard (in0 ⊗ in1) into its
  partial, forwards the shard to p+1, receives the next from p−1. After G steps every core computed all Kb
  blocks. Ring buffer = 2 shards per core (double-buffer), not the full k-slice → big L1 win (enables deeper
  K / larger Mt than store-and-forward's full-k-slice cb0).
- **The subtlety — in1 order.** The matmul sum is commutative, so a core may compute blocks in any order,
  BUT each block needs its MATCHING in1 block (in0[k] ⊗ in1[k]) at the same step. So the in1 read order must
  follow the shard-arrival order: core p reads its N-band's in1 shards in order p, p−1, … (rotated). Each
  shard is contiguous; the rotated order = G contiguous segments with ~G row-buffer seeks over the whole
  in1 read. For large shards (Kb≫G) this is ~free; for tiny Kb it costs some in1 BW — the key thing to
  measure. (Fallback "Ring-B": read in1 sequentially at peak and buffer the whole k-slice's in1, matching
  in0 as it arrives — avoids rotation but costs L1 = Kt_local·N_band; only for small k-slices.)
- **Remainder (Kb%G≠0):** variable contiguous shard sizes; a core whose shard is empty (Kb<G) skips
  its read and just relays. Every core still runs G steps (some no-op compute/forward).
- Orthogonal to the split-K reduction (still down the bank column) and split-NOC (unchanged).

### RISC placement for the in0 shard read (make it a knob, measure all)
Each core does: in1 read (big), in0 shard read (small, once), ring forward/recv (L1↔L1), reduction, output.
Options to A/B:
  (a) in0 shard read on the SAME RISC/NoC as in1, BEFORE the in1 stream (small prologue, one NoC to DRAM);
  (b) same RISC, interleaved/after in1;
  (c) in0 read on the OTHER RISC (as today) — parallel with in1 but both hit DRAM.
The ring forward/recv goes on whichever DM RISC is free at that phase.

### Plan
1. Kernel `in0_ring_writer.cpp`: shard-read + G-step ring (accumulate via compute over rotated shards) +
   reduction + output. cb0 = ring buffer (2 shards). Reuse compute REDUCE_K for the column reduction.
2. Reader: in1 read in rotated shard order (reader_sharded with a per-shard base offset loop), or Ring-B.
3. Host: `--ring [--ringorder bank|nn] [--in0risc same-before|same-after|other]`; build the ring per
   k-slice group (order = bank or nn); shard the Kb blocks (contiguous, remainder-aware); wire ring
   next/prev coords + shard offsets/sizes + rotated in1 order.
4. Test: correctness (out==K) incl. Kb%G≠0 and Kb<G; util on 64×6144×1536 vs store-and-forward (76%) and
   ceiling (94.9%); sweep ring order + in0-RISC placement; watch in1 BW for the rotation cost.
Expected: the parallel read + no chain-fill + small L1 could close some of the 76%→94.9% gap; main risk is
the rotated in1 read BW (measure) and ring sync overhead. If Ring-A's in1 rotation hurts, fall back to
Ring-B for small k-slices.

### ✅✅ in0 RING ALL-GATHER (`--ring [--chain bank|nn]`) — BEST small-N result, beats branch
Implemented (`reader_ring.cpp` = in1 in rotated shard order; `in0_ring_writer.cpp` = own-shard DRAM read +
symmetric cyclic forward/recv + REDUCE_K reduction + output). Every core reads its own shard (parallel), the
shards rotate around the ring; compute is unchanged (cb0 in0 and cb1 in1 fed in the SAME rotated order so
in0[k]⊗in1[k] matches — sum commutative). cb0 = full k-slice (G slots). Fixed the same cb_reduce base-drift
bug (Pk≥3). v1 requires Kb=K_num_blocks_eff divisible by 8.
- **Results (64×6144×1536; store-and-forward 76; N-split-bcast 71.7; branch 76.8; ceiling 94.9):**
  **Pk=3 = 88.5% (bank) / 87.4% (nn)** — **1.15× branch**, 93% of the in0-free ceiling. Pk=2 = 87.3% (nn),
  Pk=6 = 82.0%. Correct throughout. Pk=3 (24 cores) is the sweet spot.
- **Why it wins over store-and-forward:** in0 DRAM read is parallelized across all 8 group cores (8 injectors
  vs 1), and there's no chain-fill latency (each core starts on its own shard immediately).
- **The in1 rotated-read risk did NOT materialize** — 88.5% util means the G≈8 row-buffer seeks over the
  in1 read cost little; no need for the Ring-B fallback.
- **Chain order: minor, config-dependent** (~1pt). store-and-forward preferred nn; ring slightly preferred
  bank at Pk=3. in0's small volume makes NoC distance/congestion a secondary factor.

### in0-read RISC placement A/B (`--in0risc other|same`) — config-dependent, ~±4pt
Tested reading the injector's/ring's own in0 shard on the OTHER RISC (2nd DM RISC, parallel with in1) vs
the SAME RISC/NoC as in1 (read the shard first, then the in1 stream; `reader_ring` does it, signals the
writer via shared L1). Results on 64×6144×1536:
| | in0risc=other | in0risc=same |
|---|---|---|
| Pk=2 (W=3, more in0/core) | 87.2% | **91.6% (bank) / 90.9% (nn)** |
| Pk=3 (W=2, less in0/core) | **88.4% (bank) / 87.7%** | 85.9% / 86.1% |
- **Low Pk (more in0 per core) → SAME RISC wins**: keeping the (larger) in0 read on the in1 NoC path cuts
  cross-subchannel DRAM contention, outweighing the small serial prologue.
- **High Pk (less in0 per core) → OTHER RISC wins**: the in0 read is tiny, so parallelism with in1 beats
  the prologue cost.
- **Best overall: Pk=2, in0risc=same, bank order = 91.6%** — 1.19× branch, 97% of the 94.9% in0-free ceiling.
  Heuristic: pick same-RISC when in0/core is large (low Pk / high W), other-RISC when small.

### same-RISC read ORDER A/B (`--in0order before|after|interleave`) — order is FORCED to "before"
- **before (in0 shard, then in1): 91.9% ✓** (best; reproduces).
- **after (in1, then in0 shard): DEADLOCK/hang.** Compute processes shard 0 first, so it needs in0 slot 0
  before it can drain cb1; reading in0 after in1 fills cb1 → reader blocks on cb_reserve(cb1) → deadlock.
- **interleave (in0 blocks spread through in1, signal after last): DEADLOCK too** — the in0-ready signal
  (whole slot 0) still lands after cb1 backpressures.
- **Conclusion:** on the same RISC, in0 MUST be read BEFORE the in1 stream (compute needs in0 slot 0 before
  in1 backpressures). Order is not a free knob; "before" is both the only viable and the best (91.9%). A
  deadlock-free fine interleave would require per-block ring forwarding (bigger restructure) for negligible
  gain — before is already 97% of the ceiling.

### ring vs dedicated-reader (store-and-forward): when to use which
Direct A/B on 64×6144×1536:
| | ring all-gather | store-and-forward |
|---|---|---|
| Pk=2 (W=3) | **91.2%** | 73.1% |
| Pk=6 (W=1, ring in1 read fully strided) | **81.8%** | 67.6% |
- **Ring wins in every measured regime** — including W=1, disproving the hypothesis that the strided in1
  read would favor dedicated readers. Parallel read (8 injectors) + no chain-fill beats the rotated-in1 seek
  cost even when every in1 block is a seek.
- **Use dedicated readers (store-and-forward) only where the ring v1 does NOT apply:** Kb=K_num_blocks_eff
  not divisible by G=8, or Kb<G (very shallow k-slice / high Pk) — the ring can't shard cleanly there, but a
  single injector reads any Kb. (Lifting ring's Kb%8 constraint via variable-size shards would make the ring
  universal and retire store-and-forward.)
- **Mt=1 shapes that don't need K-split**: use plain reader==consumer (no in0 sharing at all) — neither ring
  nor forward.

### Small-N progression (64×6144×1536), final
N-split per-core 58% → N-split broadcast 71.7% → K-split per-core-read 62% → K-split store-and-forward 76% →
K-split ring all-gather 88.5% → **+ in0 read on the in1 RISC/NoC (Pk=2) = 91.6% (1.19× branch, 97% of ceiling).**
The ring is the answer: bank-adjacent reads preserved (no
rect), in0 read once and parallelized, tight L1↔L1 rotation. Regime-A small-Mt now WINS everywhere
(large/mid-N Mt=2 ~88% = 1.18×; hardest small-N 88.5% = 1.15×). Remaining ~6pt to ceiling = ring/reduction
sync overhead (future: overlap reduction tail, deeper forward pipeline, small ring buffer instead of full cb0).

## Expected outcome
Regime-A geomean ~1.25× over branch (up to ~1.9× on low-branch-BW shapes), all shapes win/tie except the
Mt=16 outlier (out of scope). Mt=2/4/8 losses in the current prototype become wins via split-NOC.

---

## FULL REGIME-A BENCHMARK (best config per shape) — BH p150b, HiFi2, bf16
Measured 2026-07-07 with `regime_a_mm`. `util%` = achieved / 500 GB/s on minimal traffic 2·(MK+KN+MN)
(same metric as branch "br BW%"), so **speedup = util / branch_BW%**. Constant-input correctness (out==K)
verified for every winning config. Two sweeps merged (best of): ring all-gather K-split, and N-split±bcast.
Raw per-config logs: `tools/mm_sweep/regime_a_bench_ring.txt`, `regime_a_bench_nsplit.txt`.

### Mt=1 (M=32) — core GEMV-like skinny, WIN
| shape M×K×N | branch BW% | best util% | speedup | winning codepath |
|---|---:|---:|---:|---|
| 32×2048×512   | 48.1 | 71.7 | **1.49×** | ring, K-split Pk2 (16c), in0=other-RISC |
| 32×2048×1536  | 69.4 | 83.1 | **1.20×** | ring, Pk2 (16c), in0=other |
| 32×2048×2048  | 74.7 | 85.2 | **1.14×** | ring, Pk2 (16c), in0=other |
| 32×6144×1536  | 74.6 | 95.0 | **1.27×** | ring, Pk2 (16c), in0=other |
| 32×6144×2304  | 73.7 | 92.6 | **1.26×** | ring, Pk3 (24c), in0=same |
| 32×6144×3072  | 72.7 | 96.6 | **1.33×** | ring, Pk2 (16c), in0=other |
| 32×6144×6144  | 76.4 | 97.4 | **1.27×** | ring, Pk2 (16c), in0=same |
| 32×6144×9216  | 77.9 | 97.6 | **1.25×** | N-split P9+bcast (72c) |
| 32×256×6144 (Kt=8) | 63.2 | 85.1 | **1.35×** | N-split P12+bcast (96c) — ring N/A (K too shallow) |

### Mt=2 (M=64) — WIN
| shape | branch BW% | best util% | speedup | winning codepath |
|---|---:|---:|---:|---|
| 64×6144×1536  | 76.8 | 91.5 | **1.19×** | ring, Pk2 (16c), in0=same |
| 64×15360×1536 | 71.8 | 90.4 | **1.26×** | ring, Pk3 (24c), in0=same |
| 64×4608×6144  | 75.4 | 90.8 | **1.20×** | N-split P6+bcast (48c) |
| 64×6144×4608  | 74.3 | 89.2 | **1.20×** | ring, Pk3 (24c), in0=other |
| 64×6144×9216  | 77.8 | 94.8 | **1.22×** | N-split P9+bcast (72c) — ring OOMs (large N_band) |

### Mt=4 (M=128) — boundary of regime A (AI 108-124), CURRENT GAP (loses)
| shape | branch BW% | best util% | speedup | codepath |
|---|---:|---:|---:|---|
| 128×2304×6144 | 71.8 | 63.3 | 0.88× | N-split P8+bcast (64c) |
| 128×6144×768  | 60.5 | 46.8 | 0.77× | ring, Pk3 (24c), in0=same |
| 128×6144×2304 | 73.9 | 41.4 | 0.56× | N-split P3 (24c), per-core in0 |
| 128×6144×4608 | 76.4 | 44.4 | 0.58× | N-split P3 (24c), per-core in0 |
| 128×15360×768 | 64.0 | 19.2 | 0.30× | N-split P3 (24c), per-core in0 |

### Mt=16 (M=512) — out of regime A, loses
| shape | branch BW% | best util% | speedup | codepath |
|---|---:|---:|---:|---|
| 512×6144×1536 | 61.5 | 17.9 | 0.29× | N-split P2 (16c), per-core in0 |

### Extra shapes added (no branch reference)
| shape | Mt | best util% | codepath | note |
|---|---:|---:|---|---|
| 32×6144×12288 | 1 | 94.9 | N-split P6+bcast (48c) | very large N (ring OOMs) |
| 64×6144×3072  | 2 | 93.4 | ring, Pk2 (16c), in0=other | mid-N Mt=2 |
| 32×6144×256   | 1 | 62.3 | ring, Pk3 (24c), in0=same | extreme-skinny N (Nt=8); N-split N/A |
| 128×6144×9216 | 4 | 52.6 | N-split P6 (48c) | Mt=4 large-N — same gap |
| 256×6144×6144 | 8 | 37.1 | N-split P4 (32c) | Mt=8 boundary — gap |

### Conclusions
- **Mt=1 & Mt=2 (the DRAM-bound core of regime A): win comprehensively, geomean ≈1.26× over branch,
  util 71-98%** (90-98% of DRAM cap on the deep-K shapes).
- **Codepath split by N-per-bank:** ring all-gather (K-split) wins at small/mid N (full-N_band output block
  fits L1); N-split+in0-broadcast wins at large N (9216/12288 — ring OOMs there because cb0=full-k-slice and
  the fp32 output block scale with N_band). in0-read RISC (same vs other) flips with in0-per-core.
- **Known gap Mt≥4 (M≥128):** loses (0.3-0.9×). Causes: (1) ring OOMs (full-k-slice cb0 + fp32 out block =
  Mt·N_band) → falls back to N-split which underperforms; (2) M=128 wants ~27 compute cores + in0/reduction
  overhead grows; (3) these have higher AI (less DRAM-bound), where the branch's compute levers do better.
  **Fix (deferred):** small ring buffer instead of full-k-slice cb0 + N-sub-block the compute → lets the ring
  apply uniformly to Mt≥4 and large-N.

---

## REAL-MODEL SHAPE CLASSIFICATION (FLUX.2 1024px SP4/TP8, LTX)
Roofline: BH ridge = 304.1 TFLOP/s / 500 GB/s = **608 FLOP/byte**. Memory-bound iff
AI = MKN/(MK+KN+MN) < 608. Regime A = M < N (in1 is the big read). Classified 2026-07-07.

**Memory-bound AND regime A (what this op targets):** FLUX 11/18, LTX 4/17.

| model | shape M×K×N | AI | Mt | status |
|---|---|---:|---:|---|
| FLUX | 32×256×6144 | 28 | 1 | ✅ win (85%, 1.35×) |
| FLUX | 32×6144×1536 | 31 | 1 | ✅ win (95%) |
| FLUX | 32×6144×2304 | 31 | 1 | ✅ win (93%) |
| FLUX | 32×6144×4608 | 32 | 1 | ✅ win (~95%) |
| FLUX | 32×6144×6144 | 32 | 1 | ✅ win (97%) |
| FLUX | 512×6144×768 | 293 | 16 | ⚠️ GAP (large-Mt) |
| FLUX | 512×15360×768 | 301 | 16 | ⚠️ GAP |
| FLUX | 512×6144×2304 | 392 | 16 | ⚠️ GAP |
| FLUX | 512×2304×6144 | 392 | 16 | ⚠️ GAP |
| FLUX | 512×3072×6144 | 410 | 16 | ⚠️ GAP |
| FLUX | 512×6144×4608 | 429 | 16 | ⚠️ GAP |
| LTX  | 32×2048×512 | 30 | 1 | ✅ win (72%, 1.49×) |
| LTX  | 32×2048×1536 | 31 | 1 | ✅ win (83%) |
| LTX  | 32×2048×2048 | 31 | 1 | ✅ win (85%) |
| LTX  | 256×2048×1024 | 186 | 8 | ⚠️ GAP (large-Mt) |

**Key:** the M=32 (Mt=1) subset we already win; **the DRAM-bound regime-A work in FLUX is dominated by
M=512 (Mt=16) — 6 of its 11 — exactly the current large-Mt gap.** LTX adds one M=256 (Mt=8). So the
large-Mt ring fix is the top lever for real FLUX/LTX acceleration.

Compute-bound (AI>608, DRAM opt won't help): FLUX 1024×6144×{2304(636),4608(737)}, 1024×2304×6144(636),
1024×3072×6144(683); LTX 1216×4096×{3072,4096}, 4864×4096×{1024,3072,4096}.
Regime B (M>N, memory-bound — needs a transposed/in0-big variant, not this op): FLUX 1024×128×768,
1024×6144×768, 1024×6144×128; LTX 1216×4096×{32,512,1024}, 1216×2048×1024, 4864×4096×{32,512},
4864×2048×1024, 32×2048×32.

---

## NEXT (HIGH PRIORITY): large-Mt ring fix (Mt=8..16 regime A — FLUX M=512, LTX M=256)
**Root cause of the gap (L1 OOM):** the ring holds the full k-slice in cb0 (= Mt·Kt_local) AND the compute's
fp32 output accumulator is Mt·N_band. Both scale with Mt (and N_band), so Mt=16 / large-N OOM (e.g.
512×2304×6144 wants a 16·24=384-tile fp32 accumulator ≈ 4.6 MB double-buffered).

**Fix = N-sub-block the output + higher Pk (keep in0 resident & reused, shrink cb0):**
1. **N-sub-block the compute:** set `N_block_tiles = Nsb` (small, e.g. 2-3) and `N_blocks_per_core =
   N_band/Nsb`. compute.cpp ALREADY supports this with `reuse_in0_block` — it holds in0 in cb0 and reuses it
   across the N-sub-blocks (only pops after the last), so **in0 is delivered ONCE (ring) and re-read from L1,
   no extra DRAM.** The fp32 accumulator shrinks to Mt·Nsb.
2. **Higher Pk (K-split):** cb0 = Mt·Kt_local = Mt·(Kt/Pk); raise Pk so it fits (also gives the extra
   compute cores large Mt needs). e.g. 512×2304×6144: Pk=12 → Kt_local=16 → cb0 = 16·16 = 256 tiles (512 KB);
   Nsb=3 → out CBs ≈ 16·3·(2bf16+2fp32+2reduce) ≈ 0.5 MB; in1 CB small → total < 1.5 MB. ✓
3. **Auto (Pk, Nsb) picker (host):** choose Pk (≤ ~12, cores ≤ grid) and Nsb (subblock-multiple) from an L1
   budget: cb0(Mt·Kt/Pk) + out(≈6·Mt·Nsb) + in1(≈4·kb·Nsb) ≤ L1; prefer larger Nsb (fewer N-passes) and
   smallest Pk that fits.

**Kernel work:**
- `reader_ring`: deliver in1 as **n-sub-block outer × rotated-K-shard inner** (loop N_band/Nsb sub-bands, each
  reading its Nsb columns across the rotated shard order) — matches the compute's (n_block, k_block) order and
  the in0 rotation.
- `in0_ring_writer`: the ring itself is unchanged (fills cb0 = full k-slice once); the **reduction phase loops
  the N_blocks_per_core output blocks** (reduce each Mt·Nsb block down the bank column) instead of one block.
- host: set N_block_tiles=Nsb, N_blocks_per_core, wire the (Pk,Nsb) picker; extend the sub-band offsets.
- Fallback if cb0 still doesn't fit at max Pk (huge Mt): true small-ring-buffer cb0 with per-shard
  backpressure + **re-ring in0 per N-sub-band** (costs extra in0 traffic) — only if needed.

**Validation:** correctness (out==K) incl. N_blocks_per_core>1 and Pk high; util on the FLUX M=512 set
(512×6144×768, ×2304, ×4608; 512×2304×6144; 512×3072×6144; 512×15360×768) and LTX 256×2048×1024; target
converting today's 0.3-0.9× losses to ≥1.0× (ideally ~80-90% util like the Mt=1/2 shapes).

### ✅ IMPLEMENTED + CORRECT — but these large-Mt shapes are COMPUTE-BOUND (2026-07-07)
**Implementation (done, all PASS out==K):**
- `compute.cpp`: new `IN0_KSLICE_RESIDENT` compile flag. When set, cb0 holds the full k-slice (block-major,
  = K_num_blocks in0 blocks) and is `cb_wait_front`'d ONCE, addressed per k-block via a new `in0_base` tile
  offset in `matmul_blocks` (no pop until the end) → in0 delivered ONCE by the ring, reused across all
  N_blocks_per_core sub-blocks with ZERO extra DRAM. (Stock `reuse_in0_block` only preserves the LAST k-block,
  so it can't be used when K_num_blocks>1 — hence this flag. Off by default; main path unchanged.)
- `reader_ring.cpp`: N_bpc>1 branch reads each N-sub-band as [kb,Nsb] blocks in rotated shard order (strided:
  Nsb tiles/k-row, stride N_band). Plus a `--nsbcontig` diagnostic (constant-input only) that reads each
  sub-band contiguously to measure the layout-optimal ceiling.
- `in0_ring_writer.cpp`: reduction phase loops the N_bpc output blocks; cross-core backpressure via `red_recv`
  (fwd, prev→us "block nb sent") + `redfree` (reverse, us→prev "cb_reduce slot free"), cb_reduce = 2 slots
  (nb%2). red_prev coords added.
- host: `--nsb <Nsb>` flag; N_sub/N_bpc; compute cct N_block=Nsb / N_blocks_per_core=N_bpc; cb3 intermediate
  made **single-buffered** (matches factory — the 2× fp32 accumulator was a needless L1 hog); redfree sem.

**Results (util = achieved/500 GB/s on 2(MK+KN+MN); best feasible Pk; strided reads):**
| shape | Pk | cores | nsb | util | vs old gap |
|---|---:|---:|---:|---:|---|
| 512×6144×768  | 12 | 96 | 1 | 30% | was ~18-30% (OOM/N-split fallback) |
| 512×6144×2304 | 12 | 96 | 3 | 34% | |
| 512×6144×4608 | 12 | 96 | 3 | 41% | |
| 512×2304×6144 |  9 | 72 | 3 | 33% | |
| 512×3072×6144 | 12 | 96 | 3 | 38% | |
| 256×2048×1024 (LTX) | 8 | 64 | 2 | 32% | |

**These shapes are COMPUTE-BOUND, not read-bound — the DRAM-optimal thesis doesn't apply here:**
1. **util scales ~linearly with core count** (512×2304×6144: Pk3/24c=22% → Pk9/72c=33%; 512×3072×6144:
   Pk6/48c=26% → Pk12/96c=38%). If DRAM-bound, util would be flat in cores.
2. **strided vs contiguous read = IDENTICAL** (512×2304×6144: both 33.5% via `--nsbcontig`) → the strided
   sub-band read is NOT the bottleneck; the layout-optimal ceiling is the same.
3. All 5 RISCs (reader/writer/3×compute) have equal kernel duration (~290K cyc) → pipeline-balanced, waiting
   on the compute critical path.
4. Roofline: these have AI 293-429 (ridge=608, nominally memory-bound) but at 96 cores (the 8·Pk cap) with
   split-K reduction overhead + HiFi2 fp32-dest + tiny Nsb subblocks (2×2/2×3), compute runs at only ~29% of
   peak (512×6144×4608: 330µs vs the 135µs DRAM-optimal / 95µs compute-optimal). The branch (~61% on
   512×6144×1536) beats us by using all 110 cores with compute-optimized blocking and NO reduction overhead.

**Net:** the fix removes the OOM blocker (all FLUX M=512 / LTX M=256 now RUN + are correct) and ~doubles util
vs the old gap, but does NOT beat the branch on these compute-bound shapes. **Closing the gap is a
COMPUTE-efficiency problem, not a DRAM one:** need (a) >96 cores (hybrid split-K × N-split to reach 110),
(b) larger compute subblocks / bf16 accumulation, (c) lower split-K reduction overhead. Reconfirms the
original scoping: the DRAM-optimal op wins the low-AI Mt=1/2 core of regime A; large-Mt high-AI shapes belong
to the compute-optimized matmul. Non-nsb ring path verified un-regressed (64×6144×1536 Pk2 = 445 GB/s, 89%).

### 🎯 ABLATION (2026-07-07): the large-Mt limiter is in0 DELIVERY, NOT compute or in1 (CORRECTS the above)
The "compute-bound" read above was WRONG — "util scales with cores" is ambiguous because compute AND in1-read
AND redundant-in0-read all scale with cores. Decoupled with two ablation flags (`--skipin0` in0-read free,
new `--skipin1` in1-read free; both feed compute empty CBs — timing-only, out≠K expected). Ran the 2×2 matrix
on the **N-slice** path (`--sharded --preaders P`: 8·P cores, each owns a contiguous N-sub-band, reads FULL K,
**no reduction**).

**512×6144×4608, N-slice P9 (72 cores)** [in0-free DRAM-bound = 123µs; full = 135µs]:
| config | µs | reading |
|---|--:|---|
| compute-only (skipin0+skipin1) | 174 | pure compute |
| in1+compute (skipin0) | 179 | **in1 delivery adds ~5µs → FREE, fully overlapped** |
| in0+compute (skipin1) | 1083 | **in0 delivery adds ~900µs → the killer** |
| baseline (all) | 1267 | |

**512×2304×6144, N-slice P12 (96 cores)** [in0-free = 69µs; in0-read-once = 4.7µs]: compute-only 82,
in1+compute 83 (in1 free again), in0+compute 560.

**Findings:**
1. **in1 delivery is SOLVED / free.** N-slice contiguous sub-bands (reader==consumer, no reduction) overlap
   compute completely: in1+compute ≈ compute-only (179 vs 174; 83 vs 82). The user's ask — "what in1-delivery
   config hits DRAM-bound with in0 free" — is answered: **N-slice, ~110 cores.**
2. **Compute is NOT the bottleneck at full grid.** compute-only scales inversely with cores (4608: 48c=255,
   72c=174 → ~12.4k core-µs → **113µs at 110c < 123µs in0-free target**; 6144: 96c=82 → ~72µs at 110c ≈ 69µs
   target). At the 8·Pk≤96/72-core caps we were mildly compute-limited; at 110 cores compute clears the target.
3. **in0 delivery is THE bottleneck, via REDUNDANT per-core reads.** In N-slice every core needs the full
   in0[M,K] (shared across all N), so per-core in0 read makes DRAM in0 traffic = cores × M·K — it gets WORSE
   with more cores (4608: 48c +432µs, 72c +909µs = exactly 72×6.3MB÷500). This is why N-slice baseline is
   catastrophic (1267µs) and why the ring (K-slice, shares in0 but adds reduction) landed at 330µs.

**Reframed problem (work backwards on in0):** in0 is IDENTICAL for all cores and tiny — read it ONCE
(M·K·2 = 4.7µs @6144K / 13µs @6144-M×K) and BROADCAST to all N-slice cores, overlapped with the ~70-123µs
compute. Then total ≈ max(compute, in1-free, in0-once) = the in0-free DRAM-bound (~69-123µs). That would beat
the current ring (223-330µs) by ~2.5-3× and hit the roofline. Candidate in0-delivery mechanisms: (a) mcast
broadcast from 1-few loader cores (existing `--bcast`; challenge = BH grid col-gaps + overlap); (b) ring
all-gather DECOUPLED from K-split (each core reads 1/G of in0, rotates to full — the current ring already does
this but is welded to the reduction; split it so N-slice keeps in1-free + no-reduction while sharing in0);
(c) DRAM-sharded in0. **Next: prototype in0-once-broadcast on top of the N-slice path and re-run the matrix.**

### ❌ (a) in0 BROADCAST prototyped — mcast-BW-bound (~13 GB/s), does NOT work (2026-07-07)
Built a STREAMING broadcast (`--bstream`, `in0_bstream_loader.cpp` + `in0_writer.cpp` bstream receiver): small
D-deep ring cb0 (bounds L1 for large Mt, unlike the old full-in0 `--bcast`), loader(s) read in0 + mcast
block-by-block into every N-slice core's cb0 with a credit handshake (receiver `ready`+=1 on reserve, loader
mcasts + `valid`+=1). Extended to L loaders (K-blocks interleaved k%L, parallel injection). **Correct at 96
cores** (needed `noc_async_write_multicast_loopback_src`; non-loopback mcast hung at scale).
Results (512×2304×6144, P12/96c; compute+in1 floor 83µs, ring 223µs, DRAM-bound 74µs):
| config | in0-delivery (loader µs) | total µs |
|---|--:|--:|
| 1 loader, per-tile read | 202 | 242 |
| 1 loader, contig read | 197 | 221 |
| 2 loaders | ~200 | 212 |
| 48 receivers (P6) | 212 | — |
Mt=2 cross-check (64×6144×1536, ring=91%≈44µs): bstream loader=61µs, total=63µs (63% util) — WORSE than the
ring even at small Mt. The mcast rate is ~13 GB/s in EVERY case — independent of #loaders (2 don't help → not
source-injection-bound), #receivers (48==96), reads (contig barely helps), and shape. Broadcasting in0 to ~100
cores from few sources is tree/contention-bound at ~13 GB/s, so in0 delivery (~200µs for Mt=16) dominates and
never hides under the 83µs compute. **(a) is REJECTED.**

**Why the ring (b) should beat it:** the ring all-gather distributes BOTH the in0 read (each core reads 1/G from
DRAM in parallel = high aggregate BW) AND the forwarding (nearest-neighbor short hops, no centralized mcast
tree) — exactly why the K-split ring already hit 88-91% on Mt=1/2. **(b) is necessary: decouple the ring
all-gather from the K-split reduction so the N-slice path (in1-free, no reduction) shares in0 via the ring.**
New flags: `--bstream [--nloaders L] [--bdepth D] [--bcontig]`; `--skipin1` (compute/in1 ablation).

### ⚠️ (b) N-SLICE + in0 RING (per-group) prototyped — CORRECT but P-REDUNDANT → loses (2026-07-07)
Built `--nsring P`: N-slice (P N-sub-bands/bank, in1-free reader==consumer, NO reduction) + in0 delivered by a
ring all-gather. Reused the ring plumbing with full-K/no-reduction (is_bottom=is_top=1, N-slice output offsets).
Two kernels: reader_ring strided sub-band read (added `force_strided` + `n_base` offset); **new
`in0_nsring_writer.cpp` = STREAMING ring all-gather** into a D-slot RECYCLING cb0 (credits: recv fwd +
slotfree reverse) so it fits L1 at full K (the full-cb0 ring OOMs: cb0=Mt·Kt). Correct at 96 cores; streaming
cb0=2 shards runs the real 512×2304×6144 (Kt=72) that OOM'd before.
Results (512×2304×6144; in0-free floor 83µs, K-split ring 223µs):
| config | cores | in0-redundancy | µs |
|---|--:|--:|--:|
| nsring P6 | 48 | 6× | 286 |
| nsring P12 | 96 | 12× | **322** |
**More cores = SLOWER** (322>286) → the limiter is the **P-fold redundant in0 read**: each of the P per-bank
rings reads the FULL in0 (2.36MB × P = 28MB at P=12), contending with in1 on the 8 banks. Worse than even the
broadcast. **Root cause:** N-slice makes every core need full in0; a *per-group* ring (P independent 8-rings)
reads in0 once *per group* = P×. **The per-group ring is REJECTED.**

**The true (b) = a GLOBAL read-once ring:** ONE ring over all 8P cores, each reads 1/(8P) of in0 ONCE and the
shards rotate around the whole 8P-core ring so every core ends with full in0 — in0 read **once** (2.36MB),
forwarding distributed (nearest-neighbor along a Hamiltonian order). That removes the redundancy and, per the
ablation (compute+in1 floor 83µs, in0-once-read 4.7µs), should land near ~83-100µs (vs K-split ring 223). It's a
bigger kernel (cross-bank 1/(8P) in0 partition + 8P-step recycling ring + Hamiltonian core order), but it's the
only in0-delivery scheme left that reads once AND distributes forwarding. **NEXT: global read-once ring.**
Flags: `--nsring P [--nsdepth D]`; kernel `in0_nsring_writer.cpp`.

### ❌ (b) GLOBAL read-once ring implemented — CORRECT but divisibility → compute-bound (2026-07-07)
Built `--gring P`: ONE ring over ALL 8P cores (greedy-NN Hamiltonian order), each reads 1/(8P) of in0 ONCE
(shard = Nblk/(8P) k-blocks) and shards rotate around the whole ring → full in0 per core, **no P-redundancy**.
Reuses the streaming `in0_nsring_writer` + reader_ring with `ring_G = 8P` (kernels already parameterize G).
Correct (out==K).
Results: 512×2304×6144 gring P3 (read-once, 24c) = **408µs**; 512×6144×4608 gring P6 (48c) = **642µs** — WORSE
than the K-split ring (~223µs) and the per-group nsring (322µs).
**Root cause = divisibility caps the core count → compute-bound.** The global ring needs P|N_band (N-slice) AND
8P|Nblk where Nblk=Kt/kb (integer shard). For 512×2304×6144 (N_band=24, Nblk≤72): the intersection is P∈{1,3}
→ **max 24 cores**; for 512×6144×4608 → max 48. Mt=16 needs ~96 compute cores, so gring is starved (24-48c) and
the matmul is compute-bound long before in0 delivery matters. Read-once removed the redundancy but can't beat the
core-count wall. (Lifting it needs fractional/M-split shards — uneven ring, much more complex.) **gring REJECTED.**

### 🔑 OVERALL CONCLUSION for large-Mt regime A (in0 delivery is the wall)
Every scheme that gives an N-slice grid the FULL in0 it needs was tried and loses to the plain K-split ring:
| in0 delivery to ~all cores | result | why |
|---|---|---|
| broadcast (`--bstream`) | ~200µs+ | mcast tree-bound ~13 GB/s |
| per-group ring (`--nsring`) | 322µs | P× redundant in0 read |
| global read-once ring (`--gring`) | 408-642µs | divisibility → 24-48 cores → compute-bound |
| **K-split ring (`--ksplit --ring`)** | **~223µs** | each core needs only a K-SLICE of in0 (1/Pk) → cheap in0 |
**The ablation's "in0-free floor 83µs" is unreachable for N-slice: delivering full in0 to ~96 cores costs
≥~140-200µs by any method.** N-slice trades free in1 for expensive full-in0; K-split trades cheap (sliced) in0
for a reduction. For Mt=16/high-AI shapes the K-split ring wins because in0 delivery, not in1, is the true cost.
**Recommendation: keep the K-split ring for large-Mt; the N-slice/in1-free path is a dead end here.** All modes
remain env-gated (`--nsring`, `--gring`, `--bstream`) for reference; kernels `in0_nsring_writer.cpp`,
`in0_bstream_loader.cpp`.

---

## PLAN: 2D (M×N) matmul for large Mt — M-split × (K-split | N-slice) with SMALL-fanout sharing (2026-07-07)

**Why:** every large-Mt scheme so far made each core own ALL M rows, so it needed the full in0[M,K]; delivering
that to ~96 cores is the wall (broadcast 13GB/s, per-group ring P-redundant, global ring divisibility-starved).
**Fix (user's idea): go 2D — each core owns an M-BLOCK, not all M.** Then (1) in0 per core shrinks by the
M-split factor Sm, (2) at a fixed core budget we trade K-split depth for M-parallelism → shallower reduction,
(3) the cores that share the same in1 (an M-column of Sm cores) receive it via a SMALL mcast (fanout Sm≈2-4) —
mcast is viable again precisely because the fanout is tiny, unlike the failed 96-wide in0 broadcast. All sharing
fanouts stay small: 8-wide ring for in0, Sm-wide mcast for in1. No 96-core delivery anywhere.

### Decomposition (primary: K-split × M-split)
Grid = 8 banks × Pk K-slices × Sm M-blocks  (cores = 8·Pk·Sm ≤ 110).
Core (b, k, m) computes the partial C[M-block m, N_band b] over k-slice k = in0[M-blk m, k-slice k] @ in1[k-slice k, N_band b].
- **in1[k-slice k, N_band b]** is IDENTICAL for the Sm M-slaves (fixed b,k, varying m). ONE reader per (b,k)
  reads it from bank b (reader==consumer, contiguous), **mcasts to the Sm M-slaves** (fanout Sm). ⇒ in1 read ONCE.
- **in0[M-blk m, k-slice k]** is shared across the 8 banks (fixed k,m, varying b) ⇒ **8-wide ring all-gather**
  (each bank reads 1/8, rotate) — the EXISTING K-split ring machinery, but the read is restricted to M-rows
  [m·Mt/Sm : (m+1)·Mt/Sm]. ⇒ in0 read ONCE (no redundancy). Per-core cb0 = [Mt/Sm, Kt/Pk] (streamed, tiny).
- **Reduction** over Pk k-slices, per (b,m). Depth Pk (SHALLOWER than pure K-split at the same core count).
- fp32 accumulator = [Mt/Sm, N_band] — shrunk in M by Sm; compose with the existing N-sub-block
  (`IN0_KSLICE_RESIDENT`) path if N_band is still too big.

**Key trade at fixed 96 cores:** Pk=12,Sm=1 (today) → 12-deep reduction, in0/core=[16,6].  vs  Pk=6,Sm=2 →
6-deep reduction, in0/core=[8,12], in1 mcast to 2.  vs  Pk=4,Sm=3 → 4-deep, etc. Hypothesis: the K-split ring's
223µs (vs 82µs compute floor) is dominated by reduction depth + in0-ring; halving reduction depth via Sm should
cut it. Sweep (Pk,Sm) at fixed cores.

### Alternative: N-slice × M-split
Grid = 8 banks × P N-sub-bands × Sm M-blocks. Keeps in1 FREE (reader==consumer peak read of the full-K
sub-band) + mcast to Sm; **no reduction**. But in0[M-blk m,:] is shared across 8P cores (wide ring). M-split
shrinks it to in0/Sm per core but the ring is still 8P-wide. Use only if the K-split×M-split reduction proves
costlier than the wide in0 ring. K-split×M-split is PRIMARY (all fanouts small).

### Implementation steps (prototype `regime_a_mm`)
1. **Grid/placement → 3D** (bank, k-slice, m-block). Core-index map ci = ((b·Pk + k)·Sm + m); bank-adjacent
   placement per (b) with the Sm·Pk cores of a bank co-located (the Sm M-slaves must be NoC-adjacent for a cheap
   in1 mcast; the Pk k-slices form the reduction column as today).
2. **in1 small mcast** (NEW): the (b,k) reader reads its in1 k-slice into cb1 and mcasts to the Sm slaves'
   cb1 (fanout Sm, adjacent) + a valid sem. Reuse the credit pattern from `in0_bstream_loader` but Sm-wide and
   local. Slaves consume from cb1 as usual. (in1 read stays once; mcast is on-chip L1→L1, cheap at Sm≈2-4.)
3. **in0 8-ring, M-restricted:** reuse `in0_ring_writer` phase-1 but the shard read covers only M-rows of
   M-block m: tile index (m·Mt/Sm + mm)·Kt + (k_start + ...). cb0 = [Mt/Sm, Kt/Pk] via the 8-ring.
4. **Compute:** M_block_tiles = Mt/Sm, N_block=N_band (or N_sub), REDUCE_K over Pk; m0 = m·Mt/Sm, n0 = b·N_band.
5. **Reduction chains** per (b,m) down the Pk column (reuse the K-split reduction).
6. **Correctness** (out==K) across (Pk,Sm), then **sweep (Pk,Sm) at fixed cores** on the FLUX M=512 set + LTX
   256; target: beat the K-split ring (223µs) toward the 82µs compute floor.

### L1 (512×2304×6144, Pk=6, Sm=2, 96 cores): cb0=[8,12]=192KB (streamed less), cb1=few [kb,24] blocks,
fp32 acc=[8,24]=768KB (N-sub-block if needed), out=[8,24]. Fits with N-sub-blocking.

### Risks: (a) 3D placement on the gapped BH grid (co-locate Sm slaves for cheap mcast); (b) in1 mcast + in0
ring + reduction all active — sync complexity; (c) if reduction depth WASN'T the K-split bottleneck, Sm won't
help (then fall back to N-slice×M-split). **First milestone: K-split×M-split correct at Pk=6/Sm=2, then A/B vs
Pk=12/Sm=1 to isolate the reduction-depth effect.**

### ✅ MILESTONE 1 (K-split × M-split) DONE — reduction-depth hypothesis CONFIRMED (2026-07-08)
Implemented `--msplit Sm` on the K-split ring path: grid = 8 banks × Pk k-slices × Sm m-blocks (preaders=Pk·Sm),
core (b,k,m) computes C[M-block m, N_band b] over k-slice k. M_block=Mt/Sm; m0=m·M_block; reduction over k
(stride Sm in core index, depth Pk); in0 8-ring reads in0[M-block m, k-slice k] (M-restricted, **read once** —
each 8-ring shard covers only the M-block); composes with nsb (N-sub-block) for the fp32 accumulator. Correct
(out==K) at Pk6/Sm2 and Pk3/Sm4. **Milestone 1 reads in1 REDUNDANTLY (Sm×)** — mcast is milestone 2.
`--skipin1` added to reader_ring to isolate compute+reduction+in0-ring.

**A/B at fixed 96 cores, 512×3072×6144, nsb=3 (in1 FREE via --skipin1 → isolates reduction depth):**
| Pk / Sm | reduction depth | µs (in1-free) | µs (real in1) |
|---|--:|--:|--:|
| 12 / 1 | 12 | 241 | 246 |
| 6 / 2  | 6  | **183 (1.32×)** | 215 |
| 3 / 4  | 4  | **171 (1.41×)** | 344 (4× in1 dominates) |
- **Reduction depth was a real cost: halving it (Sm=2) gives 1.32×, quartering (Sm=4) 1.41×** on the
  compute+reduction+in0 path. Diminishing returns past Sm=2.
- Even with 2× redundant in1, Pk6/Sm2 (215) already BEATS Pk12/Sm1 (246) and the plain K-split ring (~223).
- Real-in1 at Sm=4 (344) is in1-redundancy-bound → **milestone 2 (in1 mcast, fanout Sm) is needed** to realize
  the shallow-reduction win at high Sm (would bring Pk6/Sm2 ≈183-200, Pk3/Sm4 ≈180 by removing the Sm× in1).
- Compute floor ~82µs; remaining gap = reduction (still 4-6 deep) + in0-ring + HiFi2/nsb-subblock compute.

**NEXT (milestone 2): in1 mcast to the Sm M-slaves (fanout Sm, adjacent cores) → in1 read once**, then re-A/B.
Flags: `--ksplit Pk --msplit Sm --ring --nsb Ns [--skipin1]`.

### ✅ MILESTONE 2 (in1 mcast/forward across M-slaves) DONE — 2D M-split WINS (2026-07-08)
The m==0 reader of each (b,k) group reads in1 from DRAM ONCE and forwards it to the Sm-1 M-slaves (unicast to
each, fanout Sm-1) with a valid/ready credit handshake (in reader_ring; `mrole` runtime arg: 0=slave recv,
1=reader read+fwd, 2=solo). in1 DRAM read is now ONCE (no Sm× redundancy). Correct (out==K).

**A/B at fixed 96 cores, nsb=3, milestone 2 (in1 read once):**
| shape | Pk12/Sm1 (baseline) | Pk6/Sm2 (2D) | speedup |
|---|--:|--:|--:|
| 512×3072×6144 | 247µs | **196µs** | **1.26×** |
| 512×6144×4608 | 333µs | **288µs** | **1.16×** |
- Both beat the prior K-split ring (~223/330). Sharing in1 dropped Pk6/Sm2 from 215 (m1, 2× in1) to 196 (m2,
  1× in1) — the forward adds only ~13µs over the 183µs skipin1 floor.
- **Sm=2 is the sweet spot.** Sm=4 (Pk3/Sm4) = 342µs even with shared in1: the SINGLE forwarding reader is
  overloaded (it reads a deep k-slice [Kt/Pk,N_band] AND does Sm-1=3 unicasts) → load-imbalanced. Refinement:
  replace the Sm-1 unicasts with a small mcast (fanout Sm-1) or a forward-tree so higher Sm scales.

**Net:** the 2D (M-split × K-split) decomposition WINS on large-Mt regime A — 1.16-1.26× over the best prior
K-split ring, by trading K-split depth for M-parallelism (shallower reduction) while sharing in1 via a small
(fanout Sm) forward. This is the first scheme to beat the plain K-split ring on Mt=16. Remaining gap to the
82µs compute floor = reduction (still Pk-deep) + in0-ring + HiFi2/nsb-subblock compute.
**NEXT: small mcast (not Sm-1 unicasts) to unlock higher Sm; auto (Pk,Sm,nsb) picker; sweep full FLUX/LTX set.**
Flags: `--ksplit Pk --msplit Sm --ring --nsb Ns [--skipin1]`.

### ✅ in1 MCAST (fanout Sm-1) beats unicast; in0 read is NEGLIGIBLE (2026-07-08)
**(1) in1 forward: mcast > unicast.** Added `--in1mcast` (one `noc_async_write_multicast` to the Sm-1 slave
strip + `inc_multicast` valid, NoC-corner-oriented) with a contiguous vertical-strip placement for the Sm cores
(`find_strip`; gated to the mcast path so unicast keeps the better find_near/bank-adjacent placement). A/B at
Sm=4/64c (512×3072×6144): **mcast 218µs vs unicast 335µs = 1.54×** (both correct). The single reader no longer
serializes Sm-1 writes → unlocks higher Sm. Caveat: vertical Sm-strips don't pack 24 groups in the 11×10 grid
at Sm=4/96c (max 22) → for 96c/Sm4 need a 2×2-block placement + loopback mcast (future).

**(2) in0 read is already optimal — do NOT batch.** Isolated via `--skipin0` on the ring writer: Pk6/Sm2 full
= 204µs vs skipin0 (in0 DRAM read free) = 203µs → **in0 DRAM read costs ~1µs, fully overlapped.** in0 is read
ONCE (8-ring, per-tile page reads but tiny per core) and reused across N-sub-blocks (IN0_KSLICE_RESIDENT). The
per-tile-vs-batched question is moot: the read isn't on the critical path. The remaining gap to the 82µs compute
floor is **ring forwarding + reduction depth**, not the in0 read.
**NEXT: 2×2-block placement to run mcast Sm=4 at 96c; auto (Pk,Sm,nsb) picker; full FLUX/LTX sweep.**
Flags added: `--in1mcast`, `--skipin0` (ring writer). (Note: `--skipin1` deadlocks with M-split forward — use
`--skipin0` alone for ablation.)

### ✅ #1 in0-DELIVERY experiments (2D M-split, Pk6/Sm2, 512×3072×6144 @96c) — RING WINS (2026-07-08)
Tested the in0 delivery variants for the 2D path:
| in0 delivery | µs | note |
|---|--:|---|
| **ring all-gather (current)** | **209** | each core reads 1/8 shard + short 8-hop forward; in0 read ONCE |
| direct per-core (`--in0direct`) | 233 | each core reads FULL [M-block,k-slice] (8× redundant), NO forward |
| skip read (`--skipin0`) | 203 | ring forward kept; in0 read removed |
**Conclusions:** (a) in0 READ is ~6µs (209→203, mostly overlapped). (b) DIRECT is WORSE (+24µs): the 8×
redundant read contends with in1 on the banks — the ring's read-once + cheap short forwards beats it. (c) So
in0 FORWARDING is NOT the residual cost; the ring is already the right in0 delivery. mcast-in0 (1 core reads
full group-in0 + fanout-7 mcast) not implemented: it concentrates the read on 1 core and uses the known-slow
large-fanout mcast (~13 GB/s) → expected worse than the ring; skipped.
**⇒ in0 is solved (ring, ~6µs read, cheap forward). The ~100µs residual above the 94µs DRAM roofline is
REDUCTION + compute efficiency, NOT in0.** Next: #2 (minimize reduction) and #3 (compute efficiency).
Flags added: `--in0direct`. `--skipin0` now also works on the ring writer.

### ⚠️ CORRECTION to #1 (2026-07-08): in0 FORWARDING is ~20-27µs, NOT negligible
The earlier "#1 in0 solved / forwarding not the cost" claim was WRONG — `--skipin0` only skips the READ, not the
ring FORWARD. Added `--skipfwd` (skip the ring forward + recv waits, push garbage) and ran the full ablation
(Pk6/Sm2, 512×3072×6144, 96c):
| ablation | µs | isolates |
|---|--:|---|
| full ring | 203 | read + forward + compute + reduction + in1 |
| skip read (`--skipin0`) | 201 | forward + compute + reduction + in1 |
| skip forward (`--skipfwd`) | 183 | read + compute + reduction + in1 |
| skip read + forward | 174 | compute + reduction + in1 (no in0 delivery at all) |
**Decomposition:** in0 READ ≈ 2-9µs (cheap ✓), in0 FORWARD ≈ **20-27µs** (201-174 / 203-183) — real, ~25% of
the 109µs residual over the 94µs DRAM roofline. Ring still BEATS direct (209 vs 233: direct's 8× redundant read
+24µs contention > ring's forward ~25µs), so ring is the best AVAILABLE in0 delivery — but forwarding is a
genuine secondary cost. The DOMINANT residual is compute+reduction (skip-both floor = 174µs vs 94 roofline; in1
~13µs, so compute+reduction ≈ 160µs — reduction depth + HiFi2 inefficiency). **Lever priority: reduction+compute
(#2/#3) first (~160µs), then in0 forwarding (~25µs, e.g. mcast-in0 or fewer hops).** Flags: `--skipfwd`.

### ✅ #2 REDUCTION ablation — reduction is DEPTH-bound; compute prefers BIG M-blocks (2026-07-08)
`--noreduce` (is_bottom=1 all → compute copies its partial; only the REAL top writes DRAM → 1 write/output, no
redundant-write confound; non-top discards) cleanly isolates reduction. 512×3072×6144 @96c nsb3:
| config | M-block | compute (noreduce) | reduction (full−noreduce) | full |
|---|--:|--:|--:|--:|
| Pk12/Sm1 | 16 | 181 | **+68** | 249 |
| Pk6/Sm2  | 8  | 201 | **+3**  | 204 |
| Pk3/Sm4  | 4  | 313 | +13 | 326 |
**Findings:** (1) reduction cost is DEPTH-driven serial latency — Pk12's 12-deep chain = +68µs (doesn't overlap
compute), Pk6's 6-deep = +3µs. Proven. (2) compute efficiency strongly prefers BIG M-blocks: M16=181, M8=201,
M4=313. M-split shrinks the M-block → hurts compute. So Pk6/Sm2 (204) is the M-split optimum (trades +20 compute
for −65 reduction). (3) **At the optimum, reduction is only 3µs — the DOMINANT residual is COMPUTE EFFICIENCY:
noreduce=201 vs the 64µs compute roofline ≈ 3×** (subtract in0-ring ~30 + in1 ~13 → pure compute ~140-160 vs 64).

**Answer to "optimal N-slice vs K-split":** the ideal keeps a BIG M-block (compute) AND shallow reduction. M-split
gives shallow reduction but shrinks M-block (compute hit). **N-SLICE (separate cores per N-sub-band + K-split)
would give cores WITHOUT shrinking the M-block AND with shallow Pk** — e.g. Pk3 × Ns4 (96c): M-block=16 (good
compute like the 181 config), depth-3 reduction (~cheap), N/core=N_band/4. Predicted to beat Pk6/Sm2. NOT yet
implemented (current framework has K-split×M-split×nsb-single-core-loop, but not K-split×N-SLICE-as-cores).
**⇒ Next levers, reordered:** (a) implement N-slice × shallow-K-split (keep M16, add cores via N, shallow Pk);
(b) compute efficiency (#3) — the 3× gap is the real wall now: subblocks are 2×1 (sbw=1 because N_sub=3 not even);
pick N_sub even (nsb=4→sbw=2→2×2 full-DST) and/or bf16 accum. Flags: `--noreduce`, `--skipfwd`.

### ✅ #a N-slice + #b compute — pure compute is UNPACK-BOUND at bf16, ABOVE the DRAM roofline (2026-07-08)
**(a) N-slice × K-split (`--nslice Ns`)** implemented + correct (each core: full M, N_band/Ns cols, k-slice; reduce
over k; no in1 mcast since each owns its sub-band; in0 8-ring per (k,n) → Ns× redundant). A/B @96c 512×3072×6144
nsb3: Pk6/Ns2 (M16) = 225µs vs Pk6/Sm2 (M8) = 208µs — **N-slice is SLOWER**: the Ns× redundant + bigger (M16) in0
ring delivery outweighs the M16 compute benefit. So M-split (Pk6/Sm2 ≈ 204-208) stays the best config. BUT
N-slice removed the in1-mcast deadlock, enabling clean PURE-compute measurement (skipin0+skipfwd+skipin1).

**(b) Pure compute is UNPACK-BOUND, not fidelity/DST-bound.** Pk6/Ns2 M16 pure compute:
| variant | pure compute µs |
|---|--:|
| nsb3 (2×1 subblock) | 174 |
| nsb4 (2×2, full fp32 DST) | 167 |
| nsb4 sbh4 (4×1) | 167 |
| nsb4 **LoFi** (vs HiFi2) | 167 |
**Invariant to subblock shape AND to MathFidelity (LoFi==HiFi2==167).** ⇒ the matmul is UNPACK-bandwidth-bound
(reading in0/in1 tiles from L1 CBs), NOT math/fidelity/DST-bound (confirms the earlier "compute floor is
unpack-bound" note). Levers that DON'T work: bigger subblocks, lower fidelity. The only unpack lever = FEWER
INPUT BYTES (bf8 → ~½ unpack, ~85% util per prior note) — an input-precision change (analogous to the disallowed
bf16-accum).

**🔑 CONCLUSION: pure compute (~167µs bf16) is ABOVE the DRAM roofline (94µs) → this shape is COMPUTE(unpack)-BOUND
at bf16.** At the real unpack-bound compute rate (~2.6× the LoFi peak), the roofline ridge drops from 608 to ~234
FLOP/byte; the FLUX M=512 shapes (AI 293-429) all exceed that → all compute-bound. The best config (M-split
Pk6/Sm2 ≈ 204-208) is already NEAR the unpack floor (167 compute + ~40 delivery). **"Pure compute below the DRAM
roofline is NOT achievable at bf16"** — it requires bf8 inputs (halve unpack). Remaining bf16 headroom is small
(~204 → ~167 floor, if delivery were free). Flags added: `--nslice`, `--lofi`, `--sbh`.

### ⚠️⚠️ MAJOR CORRECTION (2026-07-08): NOT unpack-bound — I was using kb=1 (shallow-K). Deep-K is the lever.
User flagged: SP2_compute_floor_findings.md shows the compute ceiling (~2.4 TF/core, 90%) needs DEEP K-blocks
(kb≥16-32); shallow kb collapses efficiency. **I used kb=1 in ALL large-Mt experiments.** Re-tested (512×3072×6144
@96c M16 N-slice):
| kb | PURE compute µs | FULL (Pk6/Sm2) µs |
|--:|--:|--:|
| 1 | 167 | 205 |
| 2 | **128 (1.30×)** | **179 (1.15×)** |
**kb 1→2 alone = 1.30× pure compute / 1.15× full.** The 167µs "unpack-bound, fidelity/subblock-invariant"
plateau was NOT a fundamental unpack wall — it was **shallow-K per-output-block pack+reconfig overhead**, which is
exactly fidelity/subblock-invariant (SP2 explicitly overturned the "HiFi2 unpack-bound ~50%" belief as
dataflow-confounded). **My earlier "compute-bound at bf16, can't beat the DRAM roofline" conclusion was WRONG.**

**Corrected picture:** with deep kb (16-32), pure compute → SP2's ~84µs (90% of 2.4 TF/core × 96) which is BELOW
the 94µs DRAM roofline ⇒ **the shape becomes DRAM-BOUND, not compute-bound.** The whole DRAM-optimal thesis DOES
apply to large-Mt — once compute is efficient.

**BLOCKER (why we can't just set kb=16):** the ring architecture caps kb. (1) the 8-bank ring needs Keff =
Kt_local/kb ≥ 8 (8 shards) ⇒ **kb ≤ Kt_local/8**; (2) deep kb needs large Kt_local ⇒ small Pk ⇒ cb0 (full k-slice
= Mt·Kt/Pk) OOMs. So at Pk6, kb≤2; reaching kb≥16 needs Pk=1 (cb0=3MB OOM). The ring's block-major cb0 also
forces compute-kb == ring-kb. **The ring structurally prevents deep-K.**

**FIX (next):** decouple compute-K from the ring — read in0 **contiguously** (direct read into [M,Kt_local] cb0;
redundant but cheap ~24µs) + read in1 **contiguously** (no rotated shards) so compute can use kb=Kt_local (deep).
The earlier in0-direct=233 was with kb=1 (shallow compute); with deep-K the compute win (~167→~84) should dwarf
the redundant-read cost. This is THE path to DRAM-bound large-Mt. Immediate partial win available now: use kb=2
(ring max at Pk6) → 179µs. Flags: `--kb`, `--in0direct` (needs in1-contiguous + Keff%8 FATAL relaxed for deep-K).

### ⭐ DEEP-K path implemented (`--in0direct` + kb=Kt_local) — compute win REAL, but contiguous delivery is the wall (2026-07-08)
Built the contiguous deep-K path: `--in0direct` makes the writer read the full [M-block, k-slice] contiguously
into cb0 (kb=Kt_local, K_num_blocks=1) and reader_ring reads in1 as one contiguous [Kt_local,N_sub] block/nb
(natural K order, no rotated shards). Correct (out==K).
| config (512×3072×6144 @96c) | PURE compute | FULL |
|---|--:|--:|
| ring kb=1 | 167 | 205 |
| ring kb=2 (ring max @Pk6) | 128 | 178 |
| **in0direct kb=16 (DEEP-K)** | **108 (1.5× vs kb1)** | 265 |
**Deep-K compute win is REAL and large (167→108, 1.5×)** — confirms SP2. BUT `in0direct`'s full config = 265
(worse than ring 178): contiguous cb0 requires each core to read its whole [M-block,k-slice] → **8× redundant
DRAM read**, which contends with in1 and dominates.

**The core tension (unresolved):**
- Ring = read-once in0, but block-major cb0 caps compute-kb at **Kt_local/8** (8-bank shards) → shallow-K (kb≤2
  at Pk6). Deep-K via the ring needs Kt_local≥128 (Keff≥8 at kb≥16) ⇒ small Pk ⇒ full-k-slice cb0 (Mt·Kt/Pk)
  OOMs (M16). Impossible for Kt<128 (e.g. this shape Kt=96).
- Direct = contiguous cb0 (deep-K), but 8× redundant read.
Neither gives read-once + deep-K. Best end-to-end stays ring kb=2 = 178µs.

**Paths to read-once + deep-K (next):** (1) ring all-gather that writes shards to their k-OFFSET in a contiguous
[M,Kt_local] cb0 (then compute kb=Kt_local) — an L1-layout change to the ring writer; (2) exploit high-Kt shapes
(Kt≥128, e.g. 512×6144×N) with Pk=1 + M-split (M2/M4) so a k-slice is deep AND cb0 fits — deep-K WITH read-once
(no ring needed at Pk1: each core reads its M-block's full-K contiguously, but that's Sm-redundant not 8×...).
Flags: `--in0direct` (deep-K), `--kb Kt_local`.

### ⭐ M-SHARD RING (read-once + contiguous + deep-K) implemented — but serialized, loses to overlapped ring (2026-07-08)
Built `--mshard`: shard in0 by M (not K). Each of 8 banks reads M-rows [ring_pos·Mw : +Mw] (Mw=M_block/8), full
K, into cb0 at its M-OFFSET — m-major ⇒ CONTIGUOUS [M_block,Kt_local] cb0 (contiguous forwards, same M-offset on
next). Read-ONCE (no redundancy), deep-K (kb=Kt_local, K_num_blocks=1). Correct (out==K).
| config (512×3072×6144 @96c) | FULL µs |
|---|--:|
| ring kb=2 (shallow, overlapped) | **178** |
| mshard kb16 (read-once+contiguous+deep-K) | 207 |
| mshard kb16 + in1mcast | 204 |
| in0direct kb16 (redundant+deep-K) | 265 |
**mshard BEATS in0direct (207 vs 265: read-once helps) but LOSES to ring kb=2 (178).** Root cause: mshard pushes
cb0 **once** (full [M,Kt_local]) so compute can't start until the whole 8-step gather finishes — it **loses the
ring↔compute overlap** the K-ring has (K-ring pushes per-shard; compute accumulates over K incrementally). The
serialization (~30µs) exceeds the deep-K compute win (~20µs: 128→108). Net loss.

**The fundamental conflict:** deep-K needs the full k-slice contiguous *before* the matmul (no incremental-K
overlap); the K-ring overlaps *because* it's incremental-K (shallow). Can't have both with a single deep matmul.

**Fix to realize deep-K (next):** push each M-shard [Mw,Kt_local] as it arrives and have compute do a deep-K
matmul on those Mw M-rows as an M-BLOCK (M_blocks_per_core = G) — overlaps the M-shard ring with compute while
staying deep-K. Caveat: Mw is small (M_block/8; =1 for M-split Sm2) → thin M-blocks; better with N-slice (M16→Mw=2)
or larger M per core. **Current practical best remains ring kb=2 = 178µs** (the kb=1→2 fix, 1.15× over 205).
Flags: `--mshard`, `--in0direct`, `--kb Kt_local`.

### ⚠️ OVERLAPPED deep-K (`--moverlap`) attempted — hangs + OOMs; deep-K + overlap + fit are mutually exclusive (2026-07-08)
Built the full overlapped-deep-K machinery: (1) `compute.cpp` IN1_RESIDENT (in1 held resident, reused across
M-blocks; new `in1_base` in matmul_blocks); (2) `in0_mshard_overlap_writer.cpp` = STREAMING M-shard ring
(push each shard as it arrives, D-slot recycling, slotfree/recv credits) so compute overlaps the ring; (3) host
wiring (M_blocks_per_core=G, IN1_RESIDENT, Pk1). Two blockers:
- **OOM on target shapes:** deep-K at Pk1 holds the FULL k-slice resident for BOTH in0 [M,Kt] AND in1 [Kt,N_band]
  per core. At Kt=96, in1-resident alone = 96·24 = 4.6 MB. To shrink needs K-split (Pk>1) → reduction (the
  complex extension not yet built).
- **Hang:** the streaming M-shard ring credit protocol (slotfree/recv) deadlocks with the in1 forward + compute
  consumption (M-split m-slaves share in1). Not debugged.

**🔑 FUNDAMENTAL CONCLUSION (deep-K for large-Mt):** deep-K, ring-overlap, L1-fit, and read-once are MUTUALLY
EXCLUSIVE with the current architecture:
- ring overlaps ONLY by sharding K (the commutative reduction dim) ⇒ shallow-K (kb≤Kt_local/8).
- deep-K needs the full k-slice contiguous before the matmul ⇒ serial (mshard, no overlap) OR redundant (direct).
- deep-K at Pk1 (to avoid reduction) ⇒ full-K in0+in1 resident ⇒ OOM (Kt≥~64).
For 512×3072×6144 (Kt=96, AI 410): deep-K compute floor (kb16) = 108µs > the 94µs DRAM roofline; kb≥32 (needed for
~84) requires Kt_local≥256 (impossible at Kt=96) or Pk1 (OOM). **So this shape is COMPUTE(deep-K)-BOUND near the
DRAM-bound crossover, and DRAM-bound (94) is NOT reachable here.** The banked win is the kb=1→2 fix: **ring kb=2 =
178µs** (from 205, 1.15×), which is near the practical shallow-K overlapped floor.

**Paths to actually hit DRAM-bound (future):** (a) higher-Kt FLUX shapes (K=6144, Kt=192) have more deep-K
headroom AND better fit (Kt_local can be ≥128 at Pk2-3 with M-split) — the deep-K + K-split + fit may line up;
(b) bf8 inputs (halve unpack ⇒ deep-K compute ~½ ⇒ well below DRAM roofline); (c) reduction-aware overlapped
M-shard (Pk>1) to shrink residents — the remaining big build. Flags: `--moverlap`, `--modepth`, `--mshard`.

### ✅ DEEP-K on REAL FLUX shapes (K=6144, Kt=192) — helps ~1.2×, but ring caps kb ⇒ ~2× off roofline (2026-07-08)
Tested deep-K (kb = ring max = Kt_local/8) vs shallow (kb=1) on the real FLUX M=512 set (M-split Pk6/Sm2, 96c,
unicast bank-adjacent):
| shape | Kt | best kb | deep-K µs (util) | kb=1 µs (util) | speedup | roofline |
|---|--:|--:|--:|--:|--:|--:|
| 512×6144×4608 | 192 | 4 | 246 (55%) | 303 (45%) | 1.23× | 135 |
| 512×6144×2304 | 192 | 4 | 169 (44%) | 196 (38%) | 1.16× | 74 |
| 512×3072×6144 | 96  | 2 | 178 (53%) | 205 (46%) | 1.15× | 94 |
| 512×15360×768 | 480 | — | (OOM deep) | 317 (25%) | — | 80 |
**Deep-K helps ~1.15-1.23× on real FLUX (K=6144 deeper than the K=3072 test), util 44-55%.** BUT still ~2× off
roofline. Cause: the 8-bank ring caps kb ≤ Kt_local/8 (= 4 at Pk6/Kt192), which is only SEMI-deep (SP2 needs
kb≥16 for 90%). Going deeper needs Sm4 (M4, thin-M ⇒ compute penalty, 369µs) or Pk1 (OOM/few-cores) or Sm4-strip
(FATAL). So the ring architecturally caps kb → compute stays ~semi-deep → 44-55% util. 512×15360×768 (Kt=480):
deep-K OOMs (cb0=full k-slice=Mt·Kt/Pk huge for deep Kt) and it's thin-N (768) so low-util (25%) anyway.

**Verdict on "deep-K reaches DRAM-bound on real FLUX": NO, but it's the biggest single lever found** — ~1.2×
over shallow-K, lifting the large-Mt shapes from ~38-46% to ~44-55% util. To close the last ~2× to roofline still
needs kb≥16 (blocked by the 8-bank ring's kb≤Kt_local/8 cap) — i.e. the non-ring deep-K delivery (mshard/direct),
which serializes/OOMs. **Recommended default for large-Mt: M-split Pk6/Sm2 with kb = Kt_local/8 (ring-max deep-K),
unicast bank-adjacent.** Remaining upside (roofline) requires either bf8 inputs or a deep-K-capable delivery
redesign (decouple compute-kb from the 8-shard ring without OOM).

---

## UNIFIED (Ns, Pk, Sm) FACTORIZER + first-class padding (2026-07-09)

Implemented `--unified`: a single ring-all-gather grid parameterized by the three orthogonal factors at once —
N-slice Ns (`--nslice`), K-split Pk (`--ksplit`), M-split Sm (`--msplit`). Grid = 8 banks x Pk x Ns x Sm cores
(so Pk*Ns*Sm <= 13 on the 110-core BH). `--unified` forces `ring=true` and `ksplit>=1` (ring all-gather in0 is
the DEFAULT delivery). Explicit sweep points are selectable: `--unified --ksplit P --nslice N --msplit S --kb K --nsb B`.

**g-ordering:** group g = k*(Ns*Sm) + n*Sm + m (m innermost => M-slaves adjacent for the in1 forward; reduction
is over k with stride mfac=Ns*Sm). Per-core: kk=slice/mfac, sub=slice%mfac, mm=sub%Sm, nn=sub/Sm.

**First-class padding (key deliverable):** any arbitrary (Ns,Pk,Sm,kb,nsb) is runnable regardless of shape
divisibility. Padded dims computed in the `if (unified)` block: Kt_local=roundup(cdiv(Kt,Pk), kb*8) (=> Keff mult
of 8 for the 8-shard ring), Kt_s=Pk*Kt_local; M_block=cdiv(Mt,Sm), Mt_s=Sm*M_block; N_band=cdiv(Nt,8),
N_own=cdiv(N_band,Ns), N_sub=nsb?nsb:N_own, N_bpc=cdiv(N_own,N_sub), N_own_s=N_bpc*N_sub, N_band_s=Ns*N_own_s,
Nt_s=8*N_band_s. Buffers allocated at padded strides; in0 filled real[m<Mt,k<Kt]=1 / pad=0 so pad-K products
vanish => out[real M,N]==K exactly, independent of layout. Correctness check restricted to the real [Mt,Nt] tile
subregion (stride Nt_s). Padded strides (Kt_s/Nt_s/N_band_s/N_own_s/Mt_s) threaded through reader_ring (Kt, N_band),
in0_ring_writer (Kt, Nt), and runtime args (nn_off, bank_n0, base_off). Padding is IDENTITY when divisible, so all
legacy paths are byte-for-byte unchanged (guarded by `!unified`; legacy `--ksplit --ring --nsb` reconfirmed PASS).

**Bug fixed during bringup:** compute runtime arg N_end_tile must span ALL N_bpc sub-blocks (= N_bpc*N_sub), not
N_block (=N_sub). Passing N_sub clamps sub-blocks 1..N_bpc-1 to empty -> compute emits 1 block while reader/writer
expect N_bpc -> deadlock. Legacy happened to pass N_band(=N_bpc*N_sub) so never hit it.

**Validated (all PASS, max_rel_err 0.0, bad 0):**
- Ns/Pk/Sm combos, divisible 512x3072x6144: Pk6/Sm2 (96c), Pk6/Ns2 (96c), Pk3/Sm2/Ns2 (96c, all three factors).
- Non-divisible (padding): odd K (Kt95), odd N (Nt190), odd M+Sm2 (Mt15), and ALL-odd 480x3040x6080 kb3 Pk5/Sm2
  nsb5 (Mt15/Kt95/Nt190, heavy pad) — all correct.
- Infeasible points fail CLEANLY with L1-OOM TT_THROW (not corruption/hang): Pk1 (full-K resident cb0), nsb0
  (huge out/reduce/intermediate blocks), Pk4/kb3/nsb7. These just need more K-split or smaller nsb.

**PROCESS LESSON (re-confirmed):** `kill -9` on a hung test binary mid-device-op WEDGES the board — after which
even known-good configs (legacy, u3) hang, giving false "deterministic hang" signals. The G/H/I "hangs" in the
first batch were entirely this, not padding bugs; all three PASS in isolation on a freshly-reset healthy device.
Run one at a time with a SIGTERM `timeout`, let it finish, reset only between.
