# Top-K / sorting work — handoff

Branch `nkapre/sorting`. Everything below was measured on the Blackhole in this
machine. Nothing is pushed. `SORTING.md` has the long form; this file is the state
of play and the traps.

---

## Bottom line

| kernel | called by | shipping | ours | speedup | basis |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **`ttnn.topk`** (whole op) | — | **171 µs** (k=32, 65 cores) | **NOT BUILT** | — | Tracy, end-to-end |
| `topk_local_sort` | **`ttnn.topk`** | 4877 cyc/call | 4397 | **1.109x** | MATH_ISOLATE |
| `topk_xl` merge | `topk_large_indices` only | 91 | 46 | 1.978x | MATH_ISOLATE |
| `topk_xl` rebuild | `topk_large_indices` only | 374 | 350 | 1.069x | MATH_ISOLATE |
| `topk_xl` **step** (merge+rebuild) | `topk_large_indices` only | 459 | 404 | 1.136x | MATH_ISOLATE |
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

## The single biggest gap

**Nothing has been integrated into a shipping op, so there is no end-to-end
number for anything we did.** To close it:

1. Apply `TOPK_REPLAY_STEP_LOAD` / `TOPK_REPLAY_STEP_STORE` (already in
   `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h`, behind `#ifdef`, default
   build verified byte-identical) as the default.
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
- An `SFPGT`+`SFPAND` value-preserving filter (`SFPGT` is used by zero files in
  the tree).

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

## Next, in priority order

1. **Integrate `topk_local_sort` and get the real `ttnn.topk` delta.** Everything
   else is speculation until this exists.
2. **The multi-core cliff is worth 128x and dwarfs every kernel win.** Measured:
   `ttnn.topk` at k=32 is 171 µs on 65 cores and **21.9 ms on one**. k=512 is
   **158 ms**. `k <= 64` is a hard gate for the multi-core path
   (`topk_device_operation.cpp:75`), so all of k=128/256/512 falls off it. Threshold
   selection has **no k constraint** — k=5/17/100/1000 are bit-identical kernels.
   Getting large-k onto multiple cores is worth far more than any of the above.
3. Upstream the `topk_xl` merge/rebuild macro work — it is drop-in with an
   identical signature and `ckernel_sfpu_topk_xl.h` is untouched.
4. Report the doc bugs in `SORTING.md` §0b upstream (BH zero-run counter counts
   *preceding* zeroes; `MIN_THRESHOLD_RELU` uses sign-magnitude order not IEEE;
   `ENABLE_ACC_STATS` is 45 on BH vs 46 on WH; `ROW_2_MAX`/`ROW_3_MAX` are wrong on
   both WH and BH).
