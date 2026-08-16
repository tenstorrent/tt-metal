# Top-K / sorting work — handoff

Branch `nkapre/sorting`. Everything below was measured on the Blackhole in this
machine. Nothing is pushed. `SORTING.md` has the long form; this file is the state
of play and the traps.

---

## FINAL RESULTS (2026-08-16 campaign — supersedes the "Bottom line" below)

Everything in this table is committed on the branch, default-on, and measured at
the level stated. Canonical harness: `_canonical_topk_sweep.py` (per-cell Tracy,
3 arms, noise-gated speedups); LLK layer: paired-arm drivers, CtrlSwap/CtrlLoad
tripwire at 2.000 on every quoted run.

| deliverable | commit | level | result |
| :--- | :--- | :--- | :--- |
| Replay default-on (`ckernel_sfpu_topk.h`) | `d64993c` | **op, Tracy** | `ttnn.topk` single-core **1.154x–1.202x** (N∈[4k,128k] × k∈[8,512]; k=512@65536: 158.6→135.3 ms); multi-core topk and all sort factories ~1.00x |
| Column-parallel multi-core `topk_large_indices` | `ffb2b3c` | **op, Tracy** | single-row k=512/1024/2048 × W=32k/64k/100k: **24–89 µs** (5.8–9.3x vs per-row single-core; ~3000–4500x vs the single-core `ttnn.topk` this workload is stranded on today) |
| Unfused SFPLOADMACRO merge/rebuild (`ckernel_sfpu_topk_xl.h`) | `7257487` | MATH_ISOLATE + op | merge 1.195/1.221/1.235x, **step 1.053/1.074/1.090x** (K=512/1024/2048) on the path the op actually calls; op-level ~1.5–2% (34.71→34.18 µs @ k512/65536) |
| SFPLOADMACRO×index-tracking probe | `c8c5ca2` | silicon fact | macro SFPSWAP ≡ software under index tracking (sensitivity-proven); SFPU lane map = 4 rows × even cols, +0/+2 passes — SFPSWAP ISA docs exonerated |
| Canonical sweep harness | `9d131ab` | infra | the table generator; survives hangs (per-cell subprocess) |

Adjudications from fresh silicon: the K=512 fused step is **1.136x** (the 1.174x
variant does not reproduce); `topk_local_sort` at op level is **1.154–1.202x**
with STORE (both old isolate numbers under-predicted). Correctness gates all
green with defaults ON: topk/sort/sampling/moe 347 passed; deepseek gate /
moe_grouped_topk / router nightlies; `test_topk_xl.py` 71/71; new macro suite
12/12 incl. chained `num_chunks=4` and mutation controls; `topk_large_indices`
120/122 (2 = IOMMU-only perf-runner env gate); `stable=True` spot-check exact.

Open items: the LOAD-arm monolithic-battery hang (2/2 under Tracy full battery,
0 in ~3,040 isolated calls and 0 in the STORE battery) is unexplained — STORE
ships on its own clean record; treat monolithic multi-config Tracy batteries of
the LOAD-only arm as suspect. `ttnn.topk` routing for k>64 onto the new
multi-core op (PR 2 in the plan) is designed but not built.

---


### Tree-era final numbers (2026-08-16, competition2, deterministic, 24/24 clean)

Anchor top-2048 of 65,536 (all measured, device-kernel time): stock ttnn.topk
631,507 µs → ttnn.topk with our routing **70.8 µs (8,921x)**; stock
topk_large_indices (row-parallel, 1 core/row) 356.6 µs → with our multi-core
log-tree **41.9 µs on 26 cores (15,083x vs stock ttnn.topk, 8.5x vs the stock
op)**. blaze fused SDPA+topk cell: 24.5 µs (32 cores, measured on this box).
1M-context CSA shape (top-512 of 262,144): **32.0 µs** at P=64, still
descending at the cap. Gap to the llm_perf roofline: 13–23x everywhere — and
roofline-v2 (llm_perf branch nkapre/topk-roofline-v2) proves the roofline sits
below the bitonic comparator critical-path floor; the streaming-selection
family (THRESHOLD_SELECT_DESIGN.md, shelved) is the algorithmic path to it.
Full table: TOPK_LEDGER.html (artifact source, committed) and
scratchpad competition2/competition_table.{csv,md}; rerun with
`_canonical_topk_sweep.py --competition --with-blaze --allow-header-edit`.

## Bottom line

| kernel | called by | shipping | ours | speedup | basis |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **`ttnn.topk`** (whole op) | — | **171 µs** (k=32, 65 cores) | **NOT BUILT** | — | Tracy, end-to-end |
| `topk_local_sort` | **`ttnn.topk`** | 4877 cyc/call | 4397 | **1.109x** [DISPUTED — being re-measured 2026-08-16, see canonical sweep; SORTING.md measures -9.062 cyc/vec = -11.90%, i.e. 4877→4297 = 1.135x] | MATH_ISOLATE |
| `topk_xl` merge | `topk_large_indices` only | 91 | 46 | 1.978x | MATH_ISOLATE |
| `topk_xl` rebuild | `topk_large_indices` only | 374 | 350 | 1.069x | MATH_ISOLATE |
| `topk_xl` **step** (merge+rebuild) | `topk_large_indices` only | 459 | 404 | 1.136x [DISPUTED — being re-measured 2026-08-16, see canonical sweep; SORTING.md §0a-quater gives 465→396 = 1.174x for the same K=512 step] | MATH_ISOLATE |
| MoE gate `bitonic_top8` | `generalized_moe_gate` | 11.0 cyc/vec | **not beaten** | — | MATH_ISOLATE |

At larger K the `topk_xl` step improves: **1.208x at K=1024, 1.255x at K=2048**
(merge alone 2.19x / 2.33x — its fixed ~11-cycle envelope amortises).

**Read these two caveats before quoting any number above:**

1. **`MATH_ISOLATE` is math-thread time only.** It excludes unpack, pack and
   dispatch, and it measures with operands already resident in Dest. It is *not*
   end-to-end.
2. **`ttnn.topk` does not call `topk_xl`.** It only calls `topk_local_sort`
   (`ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp`).
   `topk_xl` is used exclusively by
   `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/`. So the biggest
   speedups above are on a kernel `ttnn.topk` never touches.

**The only improvement `ttnn.topk` would actually see is `topk_local_sort` at
~1.11x, math-thread only, and it has never been measured at the op level.**

---

## The single biggest gap — CLOSED 2026-08-16

**Was: nothing integrated into a shipping op.** Now: replay is default-on and
measured at op level (see FINAL RESULTS), the topk_xl macro work reaches the
shipping unfused path, and large-k single-row shapes have a multi-core factory.
The original close-the-gap steps, kept for history:

1. Apply `TOPK_REPLAY_STEP_LOAD` / `TOPK_REPLAY_STEP_STORE` (already in
   `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h`, behind `#ifdef`, default
   build verified byte-identical) as the default — **for `ttnn.topk` only until the
   `ttnn.sort` full-battery hang is root-caused; see "Post-audit findings" below.**
2. Rebuild.
3. Re-run Tracy on `ttnn.topk` and diff against the 171 µs baseline.

That is a few hours and it converts "a kernel got faster" into "the op got faster",
which is the only claim that survives review. **Do this before any further kernel
work.**

---

## What is genuinely new vs re-plumbing

Be precise about this in any write-up.

**New (ours):**
- **Scheduling `SFPSWAP` into an `SFPLOADMACRO` Simple slot with the store riding
  the load's address.** Only `mul_int.h` and `where.h` used `SFPLOADMACRO` before,
  and neither scheduled an `SFPSWAP`. This is the merge and rebuild win.
- A four-sub-unit macro (Load+Simple+MAD+Round+Store) at 1.000 cyc/vector.
- An `SFPGT`+`SFPAND` value-preserving filter. (Correction 2026-08-16: `SFPGT` is
  NOT "used by zero files in the tree" — shipping call sites exist in
  `ckernel_sfpu_rounding_ops.h:57,:69` and `ckernel_sfpu_softmax_k.h:76`. It is
  unused by any sort/topk kernel; the filter *construction* is ours, but the only
  claim of first use that survives is the `SFPLOADMACRO`-scheduled `SFPSWAP` above.)

**Re-plumbing an existing in-file idiom:** the `topk_local_sort` replay change.
Phases 0/1/2 already used replay; we extended it to phases >= 4. Legitimate
optimisation, not an invention.

**Discoveries, not inventions:** the packer exponent histogram, zero-compression,
and the unpacker-floor diagnosis are all existing Tenstorrent hardware that nobody
had switched on or measured.

**A diagnosis, not a win:** the "3.32x streaming floor" is a measurement plus a
proposal. **No kernel was built that removes the LLK handshake.**

---

## Traps that cost real time here

- **`build_metal.sh` cannot re-enable Tracy on an existing build dir.** It reads
  `ENABLE_TRACY` from `CMakeCache.txt` and honours a cached `OFF` (lines 295-305),
  and it only ever *passes* `-DENABLE_TRACY=OFF`. Fix: edit the cache or configure
  fresh. Cost: one 45-minute no-op rebuild.
- **Timing cannot distinguish "free" from "silently not executed".** A
  misconfigured macro `Sequence` degenerates `SFPLOADMACRO` into a plain `SFPLOAD`
  and measures identically. **Every macro result needs a mutation control.**
- **A faster kernel can be a wrong kernel.** One rebuild version measured *faster*
  than the correct one and passed every single-merge test, because the rebuild only
  permutes its K survivors. **Only `num_chunks=4` (chained merge+rebuild) caught
  it.** Any correctness suite here must include it.
- **`--collect-only` opens the device** unless you add `--compile-producer`.
- **`ckernel_unpack_template::run(count)`** takes a `uint8_t` but feeds a 7-bit
  field: `count <= 128`, or it truncates to 0 and the loop silently runs zero times.
- **`PROFILER_SYNC()` at the end of every timed zone**, or you measure the RISC-V
  push rate (~1.0 for everything, including the `SFPSWAP` control).
- **Run producer and consumer inside one `flock /tmp/tt-device.lock`** — a
  concurrent producer wipes `/tmp/tt-llk-build` between phases.
- **Do not use `scripts/run_safe_pytest.sh` for tt-llk tests.** It `cd`s to the
  tt-metal root and activates the wrong venv.
- Controls are the tripwire: **`SFPSWAP` must measure exactly 2.00x `SFPLOAD`**.
  Both are documented constants. If it doesn't, discard the run.

---

## Files

**Modified shipping LLK** (behind `#ifdef`, default byte-identical, ttnn suite
191 passed):
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h`

**New, all under `tt_metal/tt-llk/tests/`** — perf drivers and correctness suites
for: the macro merge (71/71), the rebuild (76/76 incl. `num_chunks=4`), the
negative-threshold filter (9 passed), packer zero-compression (35/35), the packer
exponent histogram (38/38), the unpack ceiling, and the pipeline.

**The side-by-side table** comes from
`tests/python_tests/perf_topk_rebuild_xl.py`, which carries baseline and ours as
paired arms in one kernel and one run.

---

## Competition sweep

The ad-hoc K×W "competition table" loops (scratchpad `layers_grid.sh` /
`kw_grid.sh`) are folded into `_canonical_topk_sweep.py --competition` as a
deterministic, rerunnable mode: K∈{512,1024,2048} × W∈{2048…262144}, four
measured layers in fixed order (`op` = topk_large_indices alone, `routed` =
ttnn.topk largest=True composite, `stocknow` = largest=False on the committed
header, `prebranch` = same with `TOPK_DISABLE_REPLAY_STEP` temporarily armed —
header edit gated on `--allow-header-edit`, git-asserted, restored in
`finally`) plus the llm_perf roofline model (PR 671+676 — aspirational, no
such kernel exists) as a constants row with gap columns. Per-cell fresh Tracy
subprocess, seed derived from (k,W,layer), correctness gates timing (WRONG
cells never enter the table), and every record is stamped with HEAD sha +
`git diff --stat` md5 + `_ttnn.so` mtime/md5 so a mid-run rebuild is visible
in the output instead of silently corrupting the A/B. Output:
`competition_table.{csv,md}` in `--out`, exec-summary anchors at k=512/2048 @
W=65536.

```bash
python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
    --competition --allow-header-edit --out generated/canonical_sweep/competition
```

---

## Post-audit findings (2026-08-16)

From an adversarial audit of this file and `SORTING.md` against the tree, plus the
first canonical-sweep runs:

- **The fused-only gap.** Every `topk_xl` macro win above was measured in **fused**
  mode, but the one shipping consumer
  (`ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp`)
  calls `topk_xl_merge<K, false>` / `topk_xl_rebuild<K, false>` — **unfused**. In
  unfused mode `bitonic_sort_len_k` is 2 `SFPSWAP`, not 4 (header census,
  `ckernel_sfpu_topk_xl.h:131-132`), so the fused numbers do not transfer as-is.
  **The fused macro wins currently have no shipping consumer.** An unfused probe has
  been built; the canonical sweep measures it.
- **Intermittent full-battery hang — do NOT default the replay define on for
  `ttnn.sort` until root-caused.** The `replay_load` arm hung `ttnn.sort` @ W=8192
  (cross-core factory) in **2/2 full-battery runs**, but **0/10 isolated
  reproduction attempts** (including under Tracy and under Watcher), across ~3040
  clean isolated calls. Order-/state-dependent, unreproduced in isolation, cause
  unknown. This is on top of the already-flagged STABLE_SORT replay-window overlap.
- **`tt_metal/tt-llk` is vendored, not a submodule** (no `.gitmodules` entry, no
  `.git` inside it). LLK edits ship in the same tt-metal commit — there is no
  separate pin to bump, and no "submodule drift" explanation for any diff there.
- **The canonical harness is
  `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py`** — a
  per-cell-subprocess sweep (each (arm, shape, k) cell in its own process, so one
  hang cannot poison the battery). It supersedes ad-hoc invocations of the tt-llk
  perf drivers for cross-arm comparisons; every number tagged
  `[DISPUTED — being re-measured 2026-08-16]` in `SORTING.md` and this file is being
  re-established by it.

**Deferred header fix — apply to
`tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h` after the
sweep completes** (the sweep orchestrator concurrently rewrites that file's arm
block, so it must not be edited mid-sweep; leave the SWEEP_ARM marker state alone):
the comment on the `TOPK_REPLAY_STEP_LOAD`-only branch claims slots `[21, 29)` are
"unclaimed and needs no coordination with any other recording at all". That is
**false under STABLE_SORT**: the phase-3 lattice recorded by
`bitonic_topk_ph3_st4_to_1` is `replay_count = STABLE_SORT ? 9 : 5` slots from 16,
i.e. slots 16-24, overlapping the load window at 21-24. Replace the comment with:

```
// Load only: 0-7 hold load16(4, 8), 8-15 store16(4, 8); the phase-3 compare
// lattice occupies 16-20 (or 16-24 under STABLE_SORT, replay_count = 9). The
// buffer is REPLAY_BUF_SIZE = 32 deep. Without STABLE_SORT, [21, 29) does not
// overlap anything. Under STABLE_SORT it shares slots 21-24 with the lattice,
// which is safe for the same reason as the STORE window below: every consumer
// of slots >= 16 re-records before it replays — the step loop re-records at
// the top of EVERY step (`init_step_load` is step-scoped), and the phase >= 4
// "steps 4 to 1" tail re-records the lattice itself, because `init_phase` is
// reset to true at the top of every phase per (face, col) pass.
#define TOPK_STEP_LOAD_REPLAY_START 21
```

## Next, in priority order

1. **Integrate `topk_local_sort` and get the real `ttnn.topk` delta.** Everything
   else is speculation until this exists.
2. **The multi-core cliff is enormous and dwarfs every kernel win — but the old
   "128x" compared mismatched shapes.** The 171 µs (k=32, 65 cores) Tracy row is
   **W=32768**; the 21.9 ms single-core row is **W=131072** — a 4x-wider shape, so
   dividing them is not a speedup. Matched-honest framing from the same report:
   k=32 single-core at **W=65536 is 10.95 ms** vs 171 µs multi-core at W=32768 —
   ~64x wall clock on 2x the elements, i.e. **~32x per element**. k=512
   (W=65536, single core) is **158 ms**. [Numbers to be superseded by the canonical
   sweep re-measure, 2026-08-16.] `k <= 64` is a hard gate for the multi-core path
   (`topk_device_operation.cpp:75`), so all of k=128/256/512 falls off it. Threshold
   selection has **no k constraint** — k enters the kernel only as an integer
   compared against a count, so k=5/17/100/1000 should be bit-identical kernels
   (structural claim; no k-sweep test exercises it yet). Getting large-k onto
   multiple cores is worth far more than any of the above.
3. Upstream the `topk_xl` merge/rebuild macro work — it is drop-in with an
   identical signature and `ckernel_sfpu_topk_xl.h` is untouched.
4. Report the doc bugs in `SORTING.md` §0b upstream (BH zero-run counter counts
   *preceding* zeroes; `MIN_THRESHOLD_RELU` uses sign-magnitude order not IEEE;
   `ENABLE_ACC_STATS` is 45 on BH vs 46 on WH; `ROW_2_MAX`/`ROW_3_MAX` are wrong on
   both WH and BH).
