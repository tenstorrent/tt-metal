# (Cgt,Ceq) Exact-Count Engine — ISA Verification and Microbench Spec (`cgtceq_perf.cpp`)

**Scope.** Verifies the RADIX_BUCKET_GPU.md §6.1 item 7a / IMPL-3 claim (a complete
instruction sequence for exact threshold counting with RISC readback via memory-mapped
Dst at `0xFFBD_8000`), step by step, against `tt-isa-documentation/BlackholeA0` (checkout
at `/home/nachiket/tt-isa-documentation`), and specs the microbench the audit's Gate-2
deliverable requires (RADIX_BUCKET_GPU.md:662, correction #8). Read-only analysis; no
device runs performed.

**Headline verdict.** The IMPL-3 sequence is **real, instruction-for-instruction
verifiable, and largely already scaffolded in-tree** — with two corrections:
(1) the ordering primitive must NOT be a bare `STALLWAIT` (it gates Tensix-coprocessor
instructions only; a RISCV `lw` from `0xFFBD_8000` never passes through the Wait Gate,
so `STALLWAIT` cannot order it) — the safe menu is `tensix_sync()` (the measured ≥25.1-cyc
PassSync floor), a Tensix-semaphore post gated on `WAIT_SFPU` polled by the RISC via the
PC-buffer, or a polled sentinel word in Dst itself; and
(2) the helper names the task attributes to `ckernel_sfpu_topk.h` (`CountD1`,
`HistMacro`, `HistSum`) are **not LLK library helpers** — they are perf-test *arms*
inside `tt_metal/tt-llk/tests/sources/sfpu_count_above_perf.cpp`. `ckernel_sfpu_topk.h`
contains only the bitonic machinery. The bench therefore extends the count_above
scaffold rather than calling any library function.

---

## 1. Step-by-step verification of the IMPL-3 sequence

The claimed sequence (RADIX_BUCKET_GPU.md:636-639, :778): per-vector
`SFPGT(SET_VD)` + `SFPIADD` accumulate → cross-lane fold (1× `SFPTRANSP` + 3× `SFPIADD`,
then 7× (`SFPSHFT2 SUBVEC_SHFLROR1` + `SFPIADD`)) → `SFPSTORE` to a Dst row →
ordering → TRISC reads the scalar via memory-mapped Dst.

### 1.1 `SFPGT` SET_VD (writes -1/0 mask) — VERIFIED, with a gating footnote

`BlackholeA0/TensixTile/TensixCoprocessor/SFPGT.md`:

- Line 3: compares "on 32-bit sign-magnitude integers … or FP32 values, in which case it
  uses the total order where -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN". This is
  exactly the XOR-map/radix-on-IEEE-bits order of RADIX_BUCKET_GPU.md §4.1, so bisection
  runs on **raw bf16/fp32 bit patterns with no premap**.
- Lines 26-29: `IsVcSmaller = SignMagIsSmaller(LReg[VC], LReg[VD])`; with
  `SFPGT_MOD1_SET_VD` (bit 8, line 53), `LReg[VD].i32 = IsVcSmaller ? -1 : 0`.
  With VC = threshold and VD = data register, the mask is **strict `x > T`** — i.e.
  `SFPGT` natively computes the Cgt predicate.
- **Gating footnote confirmed (audit correction #9):** line 25 gates the whole lanewise
  body on `VD < 12 || LaneConfig.DISABLE_BACKDOOR_LOAD`, and line 29 writes VD **only if
  `VD < 8 || VD == 16`** — SET_VD into LReg 8-15 is a silent no-op. The bench must keep
  the mask target in LReg0-7 (the count_above scaffold already ping-pongs L_A/L_B there).

### 1.2 `SFPIADD` accumulate — VERIFIED, with a representation trap already documented in-tree

`SFPIADD.md` (BH page defers to WH; behavior identical): plain lanewise
`LReg[VD].u32 = LReg[VC].u32 + LReg[VB=VD].u32` with `CC_NONE` leaving flags alone
(WormholeB0 SFPIADD.md:22-45). Adding a stream of -1/0 masks makes the accumulator hold
**-(count) in two's complement**. Two consequences the bench must honor (both already
written down in `sfpu_count_above_perf.cpp:1235-1241`):

- Negate once at the end (`SFPIADD` 2SCOMP mode against `LCONST_0`, i.e.
  `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`, WH SFPIADD.md:28-29) so the stored value is
  non-negative and hence bit-identical under two's complement and sign-magnitude.
- Do **not** rely on `InstrModLoadStore::INT32_2S_COMP` on store: SFPSTORE conversion
  from two's complement is a **no-op on Blackhole** (SFPSTORE.md deprecation notes for
  `MOD0_FMT_INT32_SM`; `ckernel_sfpu_add_int.h:28-29` per the count_above comment).
- 32-bit wrap is a non-issue at count granularity (max count per pass = N ≤ 2^18), but
  the mask-accumulate chain wraps silently — the HistNibble drain-every-15-vectors issue
  (SORTING.md:1288-1292) does not apply here because each lane accumulates at most one
  unit per vector.

### 1.3 Cross-lane fold: `SFPTRANSP` + `SFPSHFT2 SUBVEC_SHFLROR1` — VERIFIED, ~25-30 issue cycles, once per decision

- `SFPTRANSP` (WH-shared page, functional model lines 38-49): swaps
  `LReg[Base+i][j*8+c] ↔ LReg[Base+j][i*8+c]` for `Base ∈ {0,4}`. With the accumulator
  in LReg0 and LReg1-3 zeroed, post-transpose `LReg[i]` row 0 holds accumulator row `i`;
  3× `SFPIADD` (LReg0 += LReg1..3) collapses the 4 rows so row 0 (lanes 0-7) carries the
  8 per-column partials.
- `SFPSHFT2.md` mode `SFPSHFT2_MOD1_SUBVEC_SHFLROR1` (lines 96-107, define =3 at :160):
  rotates lanes right by one **within each 8-lane group**. 7× (ROR1 + `SFPIADD`) reduces
  the 8 partials so every lane of the group holds the full 32-lane sum.
- **Scheduling caveat (SFPSHFT2.md:166-171):** the cycle after a ROR1 the SFPU accepts
  only `SFPNOP`; hardware auto-stalls software-issued sequences (so each ROR1 is
  effectively 2 cycles), but inside an `SFPLOADMACRO` sequence the stall is NOT automatic
  and a violation is UB. The fold must be software-issued, not macro-hosted (consistent
  with SORTING.md:1246-1256: macros cannot host reductions; `LReg[16]` is ALU-opaque).
- Fold cost: 1 TRANSP + 3 IADD + 7×(ROR1@2cyc + IADD) + negate ≈ **26 issue cycles**,
  paid once per data-dependent decision (accumulators persist across all tiles of a
  pass). The IMPL-3 "~18-20 instructions" is the instruction count; ~25-30 cycles is the
  honest issue-cycle figure after ROR1 bubbles.
- **Dual-accumulator bonus:** `SFPTRANSP` transposes `LReg[0:4]` and `LReg[4:8]`
  independently, so a (Cgt, Cge) pair — one accumulator in LReg0-group, one in
  LReg4-group — folds under **one** TRANSP; the ROR1 chain must run per accumulator
  (or be skipped entirely, §2.3 arm R2/R3).

### 1.4 `SFPSTORE` to a Dst row — VERIFIED

`SFPSTORE.md` (BH): moves 32 datums from an LReg to four consecutive Dst rows.
`MOD0_FMT_INT32` stores sign-magnitude/FP32 bits as Dst Integer "32" with no conversion.
Since the folded count is made non-negative first (§1.2), representation is unambiguous.
The exact (row, column) where lane 0 lands should be pinned at bring-up by a marker
pattern (the pack_exp_histogram "prove by construction" method) rather than derived from
the cross-lane diagram; the RISC then reads that one word (or the 8/32-word window,
§2.3). Note `dprint_tensix.h:153-157` addresses Dst rows as `addr[i + (row << 4)]` — 16
32-bit words per row.

### 1.5 RISC-side Dst mapping — VERIFIED VERBATIM, and already wrapped in-tree

- `Dst.md:103` states, verbatim: *"RISCV T0 / T1 / T2 have `Dst` mapped into their
  address space, starting at address `0xFFBD_8000`"*, controlled per thread by
  `RISC_DEST_ACCESS_CTRL_SEC[].{no_swizzle, unsigned_int, fmt}`. `fmt=1` maps
  `int32_t Dst32b[512][16]` with sign-magnitude→two's-complement conversion on load
  (Dst.md:105-113, 132-141); `fmt=0` maps FP32/uint32 with the bit-layout unswizzle
  (Dst.md:128-131). Non-negative counts are identical under both.
- `Dst.md:115`: T0/T1 must use single-element loads (`lw` for 32-bit fmt); T2 may load
  wider. IMPL-3's "TRISC1 reads the scalar" is the MATH thread's RISCV using a plain
  `lw` — SEC1 is the MATH-thread section (`ckernel_dest.h:58-61`).
- **In-tree helper exists:** `tt_llk_blackhole/common/inc/ckernel_dest.h` —
  `RISCV_DEST_START_ADDR 0xFFBD8000` (line 14), `configure_dest_access<MathThreadId>(DataFormat::Int32)`
  (lines 133-140) does the one-time fmt/swizzle/signedness setup the audit flagged.
  In-tree read precedent: `tt_metal/hw/inc/api/debug/dprint_tensix.h:153,187`
  (BH direct Dst reads).
- One-time setup cost: 3 `cfg_reg_rmw_tensix` RMWs — outside the timed loop.

### 1.6 Ordering primitive — the one real correction: bare `STALLWAIT` is NOT safe

`STALLWAIT.md` (BH): the Wait Gate blocks **Tensix-coprocessor instructions selected by
the block mask** (B0-B8) until conditions hold. A RISCV load from `0xFFBD_8000` is not a
Tensix instruction and never consults the Wait Gate; meanwhile the RISCV frontend runs
arbitrarily far ahead of the backend (the count_above file states this is exactly why
every timed region ends with `PROFILER_SYNC`, `sfpu_count_above_perf.cpp:1244-1249`).
So `SFPSTORE; STALLWAIT; lw` can read stale Dst. The verified-safe menu, in expected
cost order:

1. **`tensix_sync()`** (drain; what PassSync measured). Baseline control. Measured
   composite floor: **≥25.1 cyc per data-dependent restart** (SORTING.md:1220), and that
   is a lower bound — it excludes the RISC's Dst read and the scalar sum
   (`sfpu_count_above_perf.cpp:1184-1189`).
2. **Tensix semaphore, SFPU-gated:** math thread issues
   `t6_semaphore_post<p_stall::WAIT_SFPU>(sem)` (`ckernel.h:291-300`) after the
   `SFPSTORE`; the in-stream `STALLWAIT(STALL_SYNC, WAIT_SFPU)` orders the SEMPOST after
   SFPU completion, and the RISC polls `semaphore_read(sem)` via the PC-buffer MMIO
   (`ckernel.h:262-266`). RISC-visible, no full pipeline drain. (SyncUnit throughput is
   1 sync-instr/cycle globally — irrelevant at per-decision granularity,
   SORTING.md:1247-1252.)
3. **Polled sentinel in Dst:** RISC pre-writes a sentinel (e.g. `0xFFFFFFFF`, impossible
   for a non-negative count) into the target Dst word via the same MMIO window, issues
   the pass, then polls that word until it changes. No sync instruction at all; waits for
   the actual store, not the whole pipe. Cheapest hypothesis; race-free because the
   sentinel and count occupy the same word.

The bench must measure all three (§2.3); the honest prior for the full rendezvous stays
**25-100 cyc/decision** (audit debate, RADIX_BUCKET_GPU.md:784) until arm data exists.

### 1.7 Ceq — no equality compare needed

`SFPGT` gives strict `>` only (and BH also has `SFPLE`, SFPGT.md:8). Exact Ceq comes
from a **dual count in one pass**: `Cge(T) = Cgt(pred(T))` where `pred(T)` is T's
predecessor in the sign-magnitude key order (bit-pattern arithmetic on the raw key), and
`Ceq = Cge − Cgt`. Two thresholds → two loads + two macro-scheduled SFPGTs + two IADDs
per vector ≈ **4 cyc/vec** (IMPL-1's "dual count", RADIX_BUCKET_GPU.md:695). A
single-count pass is 2.0 cyc/vec (CountD1, architectural floor — single shared SFPU
issue port across all three Tensix threads, SORTING.md:1217-1256). The bisection loop
needs the dual form only at the final certification step; interior probes need Cgt alone.

---

## 2. `cgtceq_perf.cpp` microbench spec

### 2.1 Harness pattern (read from the existing scaffolds)

From `sfpu_count_above_perf.cpp` (the direct template), `pack_exp_histogram_{test,perf}.cpp`,
and `topk_negfilter_perf.cpp`:

- **Structure:** one C++ source with `#ifdef LLK_TRISC_UNPACK / _MATH / _PACK` sections;
  all three threads MUST declare the same `START_PERF_MEASURE` zones in the same order —
  under `--enable-perf-counters` the zones form a three-thread semaphore barrier that
  deadlocks on a mismatch (`sfpu_count_above_perf.cpp:1255-1257`). Zones: `INIT`,
  `TILE_LOOP`; every timed region ends `PROFILER_SYNC()` (RISC runs ahead of backend).
- **Variants:** arms are compile-time `#define`s (`COUNT_ARM`, `ITER_COUNT`, `THR_BITS`)
  guarded by `#ifndef` — they enter the variant hash and sweep like parameters
  (`helpers/test_variant_parameters.py:316-355`, `helpers/llk_params.py` enum). Python
  driver modeled on `perf_sfpu_count_above.py`; two-phase pytest
  (`--compile-producer` / `--compile-consumer`), consumer under `flock`; ttsim cannot
  substitute (no SFPLOADMACRO, not cycle-accurate).
- **Measurement discipline** (`perf_sfpu_count_above.py` docstring): cycles/vector from a
  **two-point slope** across ITER_COUNT (e.g. 512 vs 2048) to cancel the ~30-cycle marker
  pair and setup; run `test_profiler_overhead.py` first; validate controls
  (`ReplayLoad` ≈ 1.0, `ReplaySwap` ≈ 2.0) before reading any real arm.
- **Rendezvous timing** (from `ARM_PASS_SYNC`, `sfpu_count_above_perf.cpp:1133-1200`):
  restart cost = `(arm_slope − CountD1_slope) × VECTORS_PER_SEGMENT` — the segmented-
  restart method is the right instrument for per-decision costs.
- **Thread-work split** (from `pack_exp_histogram_perf.cpp`): keep non-participating
  threads' TILE_LOOP empty; for the L1-streamed arms, use the `PerfRunType`
  L1_TO_L1 / MATH_ISOLATE split as in `topk_negfilter_perf.cpp:349-399`.

### 2.2 Deliverable (i): count-pass cost additive to unpack, N=32768

Arms (bf16 row of 32 tiles, 1024 vectors):

| arm | what it measures | expected (prior) |
|---|---|---|
| `C0` stream floor | `unpack_to_dest` only (UNPACK_ISOLATE / stock LLK path) | 3.855-3.94 cyc/vec (disputed band, SORTING.md §0a-ter) |
| `C1` resident single count | CountD1 on Dst-resident data (MATH_ISOLATE) | 1.997-2.0 cyc/vec |
| `C2` resident dual count | (Cgt,Cge) two-threshold loop, resident | ~4.0 cyc/vec |
| `C3` streamed single count | unpack + CountD1, L1_TO_L1 | ~5.9 (3.94+2.0) — **additive**, SORTING.md:1638-1645: SFPU serializes with `unpack_to_dest` (shared Dest), PACK does not |
| `C4` streamed dual count | unpack + dual count, L1_TO_L1 | ~7.9 |

Pass/fail question for Gate 2 economics: is `C3 − C0` ≈ `C1` (fully additive, stock
path) — and record it as the per-pass tax the bisection multiplies. (The §0a-bis
split-Dest concurrency escape at ~1.26-1.5 cyc/vec is out of scope here; note it as the
unbuilt mitigation.)

### 2.3 Deliverable (ii): fold + store + RISC-read rendezvous cost per decision

Segmented-restart method (VECTORS_PER_SEGMENT = 64, as PassSync), arms crossed over
{ordering primitive} × {fold depth}:

Ordering: `S0` = tensix_sync (control; must reproduce ≥25.1); `S1` =
`t6_semaphore_post<WAIT_SFPU>` + pc_buf poll; `S2` = Dst sentinel poll.

Fold depth: `R0` full SFPU fold (TRANSP + 3 IADD + 7×(ROR1+IADD) + negate; RISC reads 1
word); `R1` partial fold (TRANSP + 3 IADD; RISC reads 8 words, sums scalar); `R2` no fold
(SFPSTORE raw accumulator; RISC reads 32 words). `R1`/`R2` trade ~14-25 SFPU cycles for
8/32 MMIO `lw`s of unknown latency — the arm data decides; MMIO load latency from
`0xFFBD_8000` is itself an unmeasured constant worth reporting separately.

Report per arm: cycles/decision (slope delta × segment size), broken into fold, store,
rendezvous, read, and branch. Honest prior: **25-100 cyc**; the go/no-go input for
RADIX_BUCKET_GPU.md correction #8.

### 2.4 Deliverable (iii): full bisection p50/p95 to the K-th threshold

Driver: new TRISC1-side state machine (no in-tree precedent — IMPL-4 notes no compute
kernel today runs a cross-thread data-dependent loop count). For a *resident* variant
(N ≤ Dst capacity per pass; iterate over Dst-resident tiles) the loop is MATH-thread
only, which is the right first bench; the streamed variant (unpack re-streaming per pass,
CB credits data-dependent) is a separate, later bench — it is a new kernel architecture
(hang surface) and should not block the constant-measuring bench.

Loop (bf16, 16-bit sign-magnitude key space, ≤16 decisions):
`lo=0x0000, hi=0xFFFF` in XOR-mapped key order → probe T = midpoint (as raw bf16
pattern) → count pass (Cgt; final step dual (Cgt,Cge)) → fold/store/rendezvous/read →
branch: stop when `Cgt < K ≤ Cgt+Ceq` (certified K-th value) or `Cgt == K` (valid set),
else halve. Instrument: decisions-per-row and cycles-per-row, distributions over ≥100
seeded rows each of:

- **random** uniform bf16 (expect ~10-14 decisions; p50/p95 of decisions and cycles),
- **clustered** (all values within one binade, RadiK's adversarial case; bisection on raw
  keys still makes 1 bit/decision progress — this is the case that distinguishes
  bisection from value-space bucketing),
- **all-equal** (expect ≤2 decisions: first dual count returns Cgt=0, Ceq=N),
- plus K∈{31,32,33}-straddle, ties-exactly-at-threshold, all-negative, ±0/Inf/NaN
  specials (Gate-3 list, RADIX_BUCKET_GPU.md:441-444).

Model check: cycles/row ≈ decisions × (N/16·(2 or 4) + rendezvous). At N=32768 the count
pass is ~2048-4096 cyc, so even 100-cyc rendezvous is ~2-5% — the p95 *decision count*
(distribution-dependent) dominates row cost; at N=256 the rendezvous dominates and the
bench should show the crossover the sorting networks already own (SORTING.md:1366-1371).

### 2.5 Correctness companion (`cgtceq_test.cpp`)

Marker-pattern rows with host-known (Cgt,Ceq) per threshold; exact all-above case (the
silent-undercount hazard of cross-thread Simple collisions, SFPLOADMACRO.md:149 per
SORTING.md:1253-1255); prologue/epilogue the 1-deep software pipeline (the perf arm
tolerates a stated 1-vector bias; the correctness arm must not —
`sfpu_count_above_perf.cpp:1225-1233`); lane-0 Dst word location proven by construction.

---

## 3. What exists vs what must be written

**Exists (verified locations):**
- `SFPGT`/`SFPLE`, `SFPIADD`, `SFPTRANSP`, `SFPSHFT2` ROR1, `SFPSTORE` — ISA-verified
  above; TTI macros in `tt_llk_blackhole/common/inc/ckernel_ops.h`.
- **CountD1 / HistMacro / HistSum / PassSync / MultiPass / MaskStore / HistNibble** —
  perf-test arms in `tt_metal/tt-llk/tests/sources/sfpu_count_above_perf.cpp:222-231`
  (constants `ARM_COUNT_D1=2 … ARM_HIST_SUM=10`), driven by
  `tests/python_tests/perf_sfpu_count_above.py` with `CountArm` in `helpers/llk_params.py`
  and `COUNT_ARM`/`COUNT_ITER_COUNT` TemplateParameters in
  `helpers/test_variant_parameters.py:329-355`. **They are NOT in
  `ckernel_sfpu_topk.h`** — that header holds only the bitonic machinery
  (`bitonic_topk_load8/16`, `bitonic_topk_ph0_st1_to_1 … ph3_st4_to_1`,
  `bitonic_topk_step_N`, uint16 strip/prepare helpers; `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h:124-561`).
- RISC Dst access config: `tt_llk_blackhole/common/inc/ckernel_dest.h`
  (`RISCV_DEST_START_ADDR`:14, `configure_dest_access<thread>`:133-140).
- RISC-visible semaphores: `ckernel.h:262` (`semaphore_read` via pc_buf), `:291`
  (`t6_semaphore_post<WaitRes>` with `p_stall::WAIT_SFPU = 0x800`,
  `ckernel_instr_params.h:292`).
- Dst MMIO read precedent: `tt_metal/hw/inc/api/debug/dprint_tensix.h:153,187`.
- Harness: zones/PROFILER_SYNC/replay/MOP idioms all in `sfpu_count_above_perf.cpp`;
  histogram readback idiom (SETDMAREG modes 6/7 + poison + `tensix_sync` + regfile) in
  `pack_exp_histogram_test.cpp:136-180` (not needed by this bench but the readback-proof
  style transfers).

**New code required:**
1. `tests/sources/cgtceq_perf.cpp` — dual-threshold count loop (2 accumulators), the
   §1.3 fold, negate+`SFPSTORE`, the three ordering arms, RISC readback + scalar branch.
2. `tests/sources/cgtceq_test.cpp` — correctness companion (§2.5).
3. TRISC1 bisection driver (resident variant) — genuinely new; no in-tree
   data-dependent-loop compute kernel exists (IMPL-4).
4. Python: `perf_cgtceq.py` + `CgtCeqArm` enum + TemplateParameters (`CGTCEQ_ARM`,
   `FOLD_DEPTH`, `SYNC_PRIM`, `ITER_COUNT`, `THR_BITS`, `K`, `DIST_SEED`) following the
   `#define`-guard convention (constexpr silently breaks variant sweeps).
5. Sentinel-poll and pc_buf-poll rendezvous helpers (a few lines each; no library home
   needed for a test).

---

## 4. Corrections to the audit text (feed back into RADIX_BUCKET_GPU.md §4.5 rewrite)

1. **"SFPTRANSP + 3x SFPIADD … then 7x (SFPSHFT2+SFPIADD) ≈ 18-20 instructions"** — right
   instruction count; add the ROR1 SFPNOP bubble (fold ≈ 25-30 issue cycles) and the
   macro-hosting prohibition (UB inside SFPLOADMACRO sequences, SFPSHFT2.md:170-171).
2. **"SFPSTORE … STALLWAIT; then TRISC1 reads"** — replace STALLWAIT with one of the
   §1.6 primitives; STALLWAIT cannot order a RISCV load (it gates Tensix instructions
   only). The Expert debate already demanded this (RADIX_BUCKET_GPU.md:782); this report
   confirms it from STALLWAIT.md semantics and supplies the two cheap candidates.
3. **"cost … tens of cycles, not hundreds"** — keep the Critic's 25-100 prior; the
   `tensix_sync` control is already ≥25.1 before the read/sum/branch.
4. The `(Cgt,Ceq)` invariant needs no equality compare: dual strict-count at
   predecessor-key (~4 cyc/vec) and only at the certification step.
5. Task-premise fix: `CountD1`/`HistMacro`/`HistSum` live in
   `sfpu_count_above_perf.cpp`, not `ckernel_sfpu_topk.h`; SORTING.md's tables
   (:1217-1220, :1328-1342) are measurements of those arms.

## 5. Context guardrail

Per the audit's own consensus (IMPL-2, RADIX_BUCKET_GPU.md:609-615): this bench prices a
**correctness oracle**. Even a perfect 25-cyc rendezvous does not create a win region
before Gate 4 (candidate materialization) — at N=32768 the bisection costs
~decisions×2-4k cycles on top of the unpack floor while the incumbent bitonic path pays
zero threshold-search cost. The bench's value is (a) closing dependency-map open dep #1
with measured constants, and (b) giving the dual-RISC BF16 histogram alternative
(RADIX_BUCKET_GPU.md §4.2, :385) an honest SFPU-side comparator for the Gate-3 shootout.
