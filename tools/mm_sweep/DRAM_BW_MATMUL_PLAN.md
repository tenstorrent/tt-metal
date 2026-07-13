# DRAM-BW-optimal minimal matmul — plan

**Status:** planning / investigation. No kernels written yet. This document restates the ask,
makes the implicit constraints explicit for review, states the central design principle, and
breaks the work into microbenchmark-first subproblems with milestones.

---

## 1. Restated goal (simplified)

Build a **new variant of `minimal_matmul`** whose *only* job is to run **low-arithmetic-intensity**
matmuls (one outer dim much smaller than the other: `M << N` or `N << M`) at **near-peak DRAM read
bandwidth (~450–500 GB/s on BH p150b)**. It is a separate program-factory path with its own kernels,
selected by host C++, opaque to the user except for the in1 memory layout.

The one-sentence framing that drives every design decision:

> These shapes are **DRAM-bound on the *big* operand's read**. The op is fast if and only if that
> read runs at peak bandwidth. Everything else (compute, the small operand, the output write) is
> cheap and must be arranged so it *never* steals bandwidth or serializes the read.

Why "low AI" ⇒ "big-operand-read-bound": for a matmul, arithmetic intensity ≈ **min(M, N) rows**
(K and N — or K and M — cancel between FLOPs and bytes). So the small outer dim *is* the AI, and the
large outer dim's operand is a nearly-unreused stream from DRAM. Concretely on the two ends:

| regime | small operand (broadcast) | **big operand (the read to optimize)** | output |
|---|---|---|---|
| **A: M ≪ N** | in0 `[M,K]` (tiny) | **in1 `[K,N]`** — weights | `[M,N]` (small, M tiny) |
| **B: N ≪ M** | in1 `[K,N]` (tiny) | **in0 `[M,K]`** — activations | `[M,N]` (moderate) |

Both regimes appear in the FLUX/LTX results: Regime A = the `32×K×N` / `64×K×N` rows; Regime B =
the `{2048,4096,8192,16384}×6144×128` rows. Today they sit at **48–79 % of 500 GB/s**; the goal is
to close that to ~90 %+.

---

## 2. The central design principle: **reader == consumer**

The entire prior body of work (see the memory notes) converges on one root cause: **peak DRAM BW is
only reached when the core that reads a chunk of the big operand is also the core that consumes it in
compute.** The moment read data must cross the NoC to reach a different compute core (scatter or
mcast delivery), you pay a per-core egress funnel and a per-block handshake, and effective BW
collapses (measured 71–180 GB/s in the dedicated-reader/scatter prototypes) — *not* because the DRAM
read is slow (dedicated readers hit ~450–509 GB/s in isolation), but because the **delivery hop** is
the ceiling.

The existing `matmul_multicore_reuse_mcast_dram_sharded` factory already exploits reader==consumer
(each of ~8 cores reads its own DRAM-sharded in1 band and computes it) and that is exactly why it
reaches ~450–473 GB/s. But it is unusable for us as-is because it **hard-requires in0 to be
L1-sharded** (`cb_in0_sharded`, `a.shard_spec().has_value()` TT_FATALs) and is effectively
**decode-only** (`per_core_M == 1` for multi-block shards). We keep its *idea* and its in1-reader
mechanics; we discard its in0 handling and its M-limit.

**Design rule for the new op:** partition the *output* `[M,N]` across cores (output-stationary). Each
core reads directly from DRAM the slice of the **big** operand it needs, in **large contiguous
transfers**, and never forwards it. The **small** operand is broadcast to all cores over the NoC
(cheap, because it is small). This confines all cross-core NoC traffic to the small operand + the
output write, leaving the big read as pure per-core DRAM→L1.

---

## 3. Proposed architecture

One op, one output-stationary skeleton, two mirrored dataflow configs chosen by host based on which
operand is big.

### 3.1 Regime A (M ≪ N) — in1 is big
- **in1**: DRAM-**width-sharded along N** (allowed lever). Each core owns a contiguous N-column band
  and streams `[K, n_band]` from its shard with large sequential reads → reader==consumer, peak BW.
- **in0** `[M,K]`: **DRAM-interleaved (required)**. It is tiny (M = 1–2 tiles). Read once by a small
  set of cores and **multicast** to all compute cores, or (simplest) each core reads the whole tiny
  in0 itself. Broadcast cost is negligible relative to the in1 stream.
- **output** `[M,N]`: interleaved, partitioned by N; each core writes its own `[M, n_band]`.
- Compute: standard tiled matmul per core over its N-band × full K, fp32 accum, HiFi2.

### 3.2 Regime B (N ≪ M) — in0 is big (the hard one)
- **in0** `[M,K]`: **DRAM-interleaved (required — cannot shard).** Partition output by **M-rows**;
  each core owns a row band and reads `[m_band, K]` interleaved, reader==consumer.
  - Open question (drives Subproblem 5): can an interleaved big-operand read, issued as
    reader==consumer with enough outstanding requests and large per-core row bands, reach peak? The
    ubench note ("1 reader/bank, tile-granular, ~500 GB/s") says interleaved *can* saturate DRAM;
    today's matmul only gets 290–400 because its NoC is *also* doing mcast/forward/output. Removing
    the forwarding may be enough. **This must be proven before committing to Regime B.**
- **in1** `[K,N]`: small; DRAM-sharded or interleaved, broadcast to all cores.
- **output** `[M,N]`: interleaved, partitioned by M.

### 3.3 The in1 DRAM-shard spec (must depend on in1 only)
Per your constraint, the shard spec is a pure function of in1's `[K,N]` and dtype — **not** of M, not
of the matmul blocking, not of in0. Proposed canonical spec:

- **Width-shard N across a fixed shard grid** (natural choice: the 8 DRAM banks, or a fixed core
  column set), K contiguous within each shard: shard shape `[K, ceil(N / n_shards) padded]`.
- The **compute partition adapts to this fixed layout** at op time: if there are more N-compute-cores
  than shards, several cores read disjoint contiguous sub-ranges of the same shard (still large,
  still contiguous). This is the decoupling you asked for: choose the shard once when the weight is
  created, before any input shape is known; the matmul figures out the core partition later.
- A tiny host helper `dram_bw_matmul_create_in1_config(in1_shape, dtype) -> MemoryConfig` gives model
  authors the canonical shard spec without knowing anything about activations.

---

## 4. Constraints — explicit (including the unstated ones, for your review)

**Given (from you):**
1. in0 stays **DRAM-interleaved** — no L1/DRAM sharding of activations.
2. **bf16** in/out, **HiFi2**, **fp32 accumulation**.
3. Graceful internal padding for shapes that don't divide grid/blocking (as minimal_matmul does).
4. Simple user API; in1 memory layout is the only thing the user must care about.
5. in1 may be **DRAM-sharded**; its shard spec depends on in1 alone.
6. Separate program-factory path / kernels, decoupled from existing minimal_matmul.

**Unstated constraints I am adopting — please confirm or correct:**
7. **BH p150b is the primary/only target** for tuning (11×10 grid, 8 DRAM channels, ~500 GB/s
   assumed peak, 304 TFLOP/s HiFi2). WH is best-effort, not tuned. *(All prior levers are grid-fit;
   I won't try to make one config serve both.)*
8. **Non-batched (B=1), 2D matmul, no transpose/bias/activation fusion** in v1. Fusions come later,
   after the BW target is hit, to avoid confounding the bandwidth study.
9. **Output is DRAM-interleaved** (mirrors in0's contract; keeps the op composable in tt_dit).
10. **Correctness target PCC ≥ 0.999** vs torch fp32 reference, verified fresh *and* on program-cache
    replay (the split-K cache bug taught us to always check replay).
11. **The op auto-selects the regime** (A vs B) and the core partition from shapes internally; the
    user does not pass tuning knobs. Env overrides exist for sweeping only.
12. **"Near peak" = ≥ 90 % of the measured dedicated-reader ceiling** for the shape's dominant read,
    not a fixed 500. If the honest silicon ceiling for a layout is 450, 90 % of *that* is success;
    500 is aspirational. We measure the ceiling first (Subproblem 1) so the target is grounded.
13. **Scope is genuinely low-AI only.** When both M and N are large (AI near/above the ridge, 608
    F/B), this op should *fall back* / defer to the existing reuse matmul — it is not trying to be a
    general matmul. Host picks this variant only when `min(M,N)` is small.
14. **K is the full contraction each core does** (no split-K in v1). Split-K re-reads inputs and adds
    a reduction hop — the opposite of what a BW-optimal op wants. It only helped before as a
    *core-filling* trick, which reader==consumer + right core count makes unnecessary here.

If any of 7–14 is wrong (especially 8, 13, 14), it changes the plan materially — flag it.

---

## 5. Decomposition into subproblems (microbenchmark-first)

Bottom-up, each with a concrete pass/fail number. Do **not** write the fused op until 1–5 have
answers; the op is just their composition.

**SP1 — DRAM read ceiling & amortization curve.** For both layouts (interleaved, width-sharded),
measure achievable read BW vs (transfer size, # reader cores, readers-per-channel) using a pure-read
ubench. Deliverable: the honest peak per layout and the min per-core transfer size to reach ≥90 % of
it. *(Partly known: ~450–509 GB/s, BW climbs 171@8MB→414@128MB, 2 readers/ch doesn't aggregate.
Re-confirm on the current build and pin the amortization knee.)* Tools to reuse:
`ubench 8b_dram_interleaved_2reader`, the `tools/mm_sweep` read benches.

**SP2 — Compute-core floor.** For each target shape, compute the min # cores so compute time <
read time (i.e. not compute-bound), using the HiFi2 unpack-bound floor (compute floor is
unpack-bandwidth-bound, not math-bound). Deliverable: `cores_needed(shape)`. Expectation: for M≤2
tiles this is *tiny* (compute is ~5 % of read time), so few reader==consumer cores saturate DRAM;
this is what makes the 8-core dram-sharded result possible and generalizable.

**SP3 — Small-operand broadcast cost.** Measure mcast BW of the small operand (in0 for A, in1 for B)
to N compute cores, and confirm it overlaps the big read without stealing DRAM/NoC BW. Deliverable:
broadcast is < a few % of runtime and fully hidden. Decide read-it-everywhere vs mcast-from-few.

**SP4 — Output write.** Measure interleaved output write BW and confirm it overlaps reads. Regime B
output `[big M, small N]` is the one to watch. Deliverable: write never on the critical path.

**SP5 — Reader==consumer on the *interleaved* big operand (Regime B, the critical unknown).** Build
a ubench: output-stationary M-partition, each core reads its interleaved `[m_band, K]` and computes,
no forwarding. Does it hit ≥90 % of SP1's interleaved ceiling? **Go/no-go for Regime B.** If no,
fall back options: (a) more outstanding requests / bigger row bands, (b) accept a lower Regime-B
target, (c) a bounded 2-hop delivery only if it beats direct interleaved.

**SP6 — Block size / L1 budget.** With SP1–SP5 fixed, tune per-core block (K-block depth, N/M block,
subblock) for double-buffered reads that keep compute fed and the DRAM read continuously outstanding.
Reuse the auto-block-sizer discipline (subblock-multiple blocks, L1 budget, K-padding) — and its
lesson: **always verify PCC, empty grep ≠ pass.**

**SP7 — Assembly + regime auto-select.** Compose into the new factory + kernels. Host heuristic picks
A/B and core partition from `(M,N,K,grid,ceiling)`. Verify PCC fresh + cached. Sweep the FLUX/LTX
skinny set; compare BW-util to `bh_skinny_results.md`.

---

## 6. Execution order / milestones

- **M0 (now):** this doc reviewed; constraints 7–14 confirmed. Pick primary shapes: a Regime-A shape
  (`32×6144×9216`, currently 77.9 %) and a Regime-B shape (`16384×6144×128`, currently 78.9 %).
- **M1:** SP1 + SP2 + SP5 answered (the three that decide feasibility). Gate: Regime-A path is
  clearly viable to ~90 %; Regime-B go/no-go decided.
- **M2:** Regime A end-to-end (new factory, in1 width-shard reader adapted from
  `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`, interleaved-broadcast in0, output write).
  PCC-correct, hits ≥90 % of ceiling on the Regime-A shape. Ship this first — it's the tractable,
  allowed-lever half and covers most skinny FLUX/LTX shapes.
- **M3:** Regime B (or its fallback per SP5), interleaved in0 reader==consumer.
- **M4:** in1 canonical shard-spec helper + host auto-select + graceful padding hardening; full
  FLUX/LTX skinny re-sweep vs baseline; update `bh_skinny_results.md`.
- **M5 (later):** fusions / batching / WH port, only after BW target is met.

---

## 7. Risks & open questions

- **Regime B interleaved read (SP5)** is the biggest unknown. If interleaved reader==consumer can't
  exceed today's ~79 %, Regime B may be inherently capped by the in0-interleaved constraint, and the
  honest deliverable there is "matched, documented ceiling," not 500.
- **in1 shard-spec vs core-partition mismatch:** a fixed shard (say 8-bank) with many N-compute-cores
  means several cores per shard. Must confirm sub-range reads within one shard stay contiguous/large
  (SP1 covers this).
- **Padding × sharding:** N not divisible by n_shards, K not tile-multiple → padded shards; the
  reader must skip pad without corrupting BW or PCC. minimal_matmul's padding discipline is the model.
- **Grid geometry:** 11×10, gy=10 not pow2 — the classic BH slicing footgun. The new partition math
  must be divisor-based / round-down from the start (don't inherit pow2 assumptions).
- **Measurement hygiene:** device-profiler kernel-time, not wall-clock (the wall-clock timing bug
  cost us a wrong "regress" conclusion once). PCC on fresh + cached. tracy with
  `--disable-device-data-push-to-tracy` to avoid the large-trace host wedge.

## 8. Success criteria (targets grounded in measured SP1/SP5 ceilings)

Measured on this BH p150b (kernel-time, dispatch-excluded): contiguous per-bank read ceiling
**~510 GB/s** (SP1); interleaved reader==consumer ceiling **~419 GB/s** (SP5, 82 % of contiguous).

1. New op, separate factory/kernels, opaque API except in1 layout helper.
2. in0 interleaved, bf16 in/out, HiFi2, fp32 accum, graceful padding — all honored.
3. **Regime A** (M≪N, in1 width-sharded → 16 KB bursts): ≥90 % of ~510 ⇒ **~460 GB/s** on skinny
   FLUX/LTX shapes.
4. **Regime B** (N≪M, in0 interleaved): ≥90 % of ~419 ⇒ **~375–400 GB/s**. 500 is NOT reachable with
   interleaved in0 — this is the documented honest ceiling, not a shortfall. Regime B needs ~all cores
   as readers (8 cores interleaved = only 205; ~96–110 cores = 419).
5. PCC ≥ 0.999 fresh and cached; no regressions on the existing skinny set.

## 8b. SP2 result — compute ceiling & core floor (measured)
Per-core HiFi2/bf16 compute ceiling **R ≈ 2.4–2.5 TFLOP/s (~90 % of the 2.76 TF/core peak ⇒
~275 TFLOP/s aggregate)**, measured by a **compute-only microbench** (minimal_matmul's real
compute.cpp on 1 core, DRAM/NoC stubbed). Lever = **deep K-block (kb ≥ 32)** — kb16→64 lifts
79 %→91 %. (`ttnn.matmul`'s default 150 TFLOP/s/49 % was dataflow-limited, NOT the compute ceiling.)
`cores_needed = min(M,N) × BW / R`; op budget = max(compute_cores, read_cores≈16 A / 32 B).
Every FLUX/LTX skinny shape needs **≤ 32 cores** and is **read-bound** (nothing compute-bound — even
512×6144×1536 fits at 109 cores). ⇒ **compute is essentially never the limiter; right-size the grid
per shape (16–32 cores), leaving ~80–95 free.** Compute kernel must use deep-K blocks to hit ceiling.
Note: `minimal_matmul` currently hangs on this tree, but that's irrelevant here — this is a
functionally separate op with its own kernels/factory. SP2's ceiling came from a compute-only harness
reusing only `compute.cpp`. (bf16 in/out, HiFi2, fp32 accum are fixed; bf8/LoFi are out of scope.)

## 8c. SP3/SP4 results — broadcast & output-write are hidden (contention checks)
**SP4 (output write, MEASURED):** interleaved output write on the 2nd RISC, concurrent with the big
read, degrades read BW by **<1.5 %** (Regime B 4–6 MB out → 0.7–1.3 %; Regime A 0.5 MB → 0.4 %). No
GDDR R/W-turnaround pathology. ⇒ write output interleaved on NCRISC, no special path.
**SP3 (small-operand broadcast):** must **mcast once + reuse**, NOT redundant per-core read
(redundant = +25 % DRAM on Regime B). Mcast volume (1.5–6 MB B / <1 MB A) ≈ SP4's write ⇒ expect
~1–3 % (B) / <1 % (A), overlapped. (Faithful mcast microbench hangs on a BH mcast-rectangle issue —
non-contiguous worker grid; integration detail, fix with device worker-set + semaphores.)
**Composition:** big read ≈99 % of cost; compute (~90 % util, hidden), broadcast (~1–3 %), write
(<1.5 %) all overlap ⇒ **the op is read-bound at ~419 GB/s (B) / ~510 (A)** — "memory is the only
limiter" holds. Final validation = a full read+mcast+write+compute run at the read ceiling.

## 8d. Regime-A prototype — BUILT & VALIDATED
Standalone tt_metal matmul (`tests/.../perf_microbenchmark/regime_a_mm/`, target `test_regime_a_mm`),
reader==consumer, reuses real `compute.cpp`, independent of the (hanging) op. On 32×6144×9216, 8
cores: PCC exact; in1 read **494 GB/s (97 % of 510)**; composite (read+in0+compute+write) **482 GB/s
(94 %)**; **1.28× faster than the existing branch** (235 µs vs 293 µs). **Key lever = bank-adjacent
core placement** (`get_optimal_dram_bank_to_logical_worker_assignment`): clustered rect capped at 343,
adjacent → 494; pipeline depth didn't matter. **Regime A needs no in0 mcast** (in0 tiny, 8-core
redundant read ~2.6 %). Next: wrap as ttnn op (factory + in1 shard-spec helper); then Regime B.

## 9. What SP1/SP5 changed vs the original plan
- The DRAM read ceiling is **~510, not ~445** — the sibling branch's number was wall-clock.
- There is **no DRAM amortization knee**; the real lever is **contiguous burst size** (16 KB→510,
  2 KB→100). This *confirms* the width-shard-in1 design (K-deep contiguous columns = big bursts).
- Regime B is **viable but capped at ~82 % of peak**; earlier the interleaved read was an open
  go/no-go and a feared ~100 GB/s floor. It is a GO, with honest per-regime targets above.
