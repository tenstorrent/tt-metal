# `llk-build-quasar` compile-time investigation — summary for feedback

**Status:** ongoing. This is a coworker-facing summary written to solicit
feedback and advice. It is deliberately concise; the full detail, measurements,
and code live in PR
[#48767](https://github.com/tenstorrent/tt-metal/pull/48767) (branch
`nsextonTT/llk-quasar-pch`) — please read the PR description and commit messages
for the numbers behind each claim.

Feedback especially wanted on: the two open threads at the end, and whether the
memory-scaling extrapolation mistakes below have a cleaner methodology.

## The original problem

The `llk-build-quasar` PR-gate job compiles **~400k kernel variants**
(413,234 test items in the measured run) with
`pytest --compile-producer -m quasar`. Each variant fans out to **4**
`riscv-tt-elf-g++` subprocesses (one per Quasar TRISC role:
unpack / math / pack / sfpu). The job takes **~20 min**, and — the puzzle that
started this — **CPU did not scale** when moving from a 16-core to a 40-core
runner. Adding cores did not make it faster, which meant the binding constraint
was not raw compute.

## The sequence of findings (including the dead ends)

Roughly chronological. Several promising theories were overturned by real
measurement; those are kept here on purpose — the value of this investigation has
been in *not* shipping the plausible-but-wrong fixes.

1. **PCH (precompiled headers).** Every one of the ~4× compiles re-parses the
   same ~2600 lines of stable headers (`ckernel.h` alone pulls in ~1256 lines of
   `constexpr` op wrappers). Added GCC PCH support for the variant-independent
   header prefix + per-role common headers. **Verified the mechanism end-to-end
   with a real compiler** (matching flags reuse the `.gch`; mismatched flags
   silently fall back to parsing from source with an identical correct result, so
   no miscompile risk). **But: the PCH speedup is not yet realized on a runner** —
   see the regression below, and the JUnit median compile time (~8 ms) did *not*
   move, which is at least consistent with PCH not engaging on most variants.
   Diagnosing *why* generation was failing needs a real runner with the toolchain
   (not available in the investigation sandbox).

2. **Regression: PCH fail-open without memoization (found on a live run, fixed).**
   After the branch landed, a real run showed the job using **<half the CPU
   (~2 cores)**, same memory, wall-clock **past the 20-min baseline**. Root cause:
   `build_pch()` failed *open* but did **not** memoize the failure, so on the
   per-variant hot path (~100k+ calls) every subsequent variant re-acquired the
   process-global PCH `FileLock` and re-ran the doomed compiles, serializing all
   xdist workers behind one lock → CPU collapse. Fixed by gating generation to at
   most one attempt per worker and writing a `.pch_failed` marker. **Falsified
   along the way:** the `shell=True` → `shlex.split()` change was suspected but
   proven *not* the cause (tokenization is byte-identical). Mitigation if it
   recurs: `TT_LLK_DISABLE_PCH=1`.

3. **Memory / worker-count tuning — a string of measured reversals.** This is the
   part most worth reading critically.
   - `-n auto` (40 workers on the 40-core xlarge): **OOM, exit 137**, even with
     `-Os` and serial inner builds.
   - `-n 30`: measured **92% mean / 100% peak RAM, ~46% mean swap** — RAM fully
     saturated and swapping for much of the run. That is *why* more cores didn't
     help: the job was memory/swap-bound, not compute-bound.
   - **Inner-parallel g++ (`LLK_BUILD_PARALLELISM=1`, fewer heavyweight Python
     workers).** Each xdist worker is a full Python process importing numpy +
     torch (~0.6–1.2 GiB RSS) — a *fixed* tax that scales with `-n`, whereas the
     compiler memory that drives swap scales with *total g++*. Holding total g++
     constant while cutting worker count pays that tax fewer times. This is the
     lever that made the box memory-safe.
   - `-n 30 → 18`: derived from a per-worker RAM estimate targeting ~80% RAM,
     zero swap.
   - `-n 10 → 18` (again): a two-point linear memory fit said ~1.55 GiB/worker,
     projecting -n 18 to ~69% RAM with headroom.
   - **`-n 18` reverted to `-n 10` — confirmed net negative.** JUnit A/B (same
     413k tests) showed no extra useful compiles, but the **cache-hit (no-compile)
     items inflated ~29%** (median 7 ms → 9 ms) and **+22% total wasted per-item
     CPU**, RAM 53% → 86% at the swap cliff. Mechanism: `-n 18 × 4 = up to 72
     concurrent g++` on 40 cores; oversubscription during g++ bursts deschedules
     the cheap single-threaded Python bookkeeping on other workers. **The
     ~1.55 GiB/worker linear model was wrong** — actual was ~3.2–3.6 GiB/worker
     once measured at matched config, ~2× off. That model class is now flagged as
     unreliable for extrapolation on this job.

4. **The realized-parallelism cap: the pytest-xdist coordinator.** With memory no
   longer binding, CPU was *still* low (~25% mean) — workers were **starving
   (~48–65% of wall-clock idle in both `-n 10` and `-n 30`)**, not computing. The
   single xdist master, throttled by `--maxschedchunk=10` (in the shared
   `pytest.ini`, never touched by this branch), can't feed ~413k ultra-fast
   (~8 ms median) items fast enough: ~41k dispatch round-trips + 413k inbound
   reports, all serial on one event loop. **More workers made it worse, not
   better** — a direct A/B that points at coordinator throughput. Tried
   `--maxschedchunk=2000` (helped only ~23 s), then switched to
   **`--dist=worksteal`** (removes per-top-up master round-trips; ignores
   `maxschedchunk`). The mechanism is *strongly inferred* from xdist source + the
   "more workers = worse" A/B, but **not directly measured** (the per-second
   sidecar samples that would clinch it aren't retrievable via `gh`).

5. **Small, safe per-item wins.** `__slots__` on `TestOutcome`; dropping `-O3` →
   `-Os` for the compile-only job; and gating the autouse `_seed_torch_rng`
   fixture to skip in producer mode (it fires ~400k times and buys nothing when
   no ELF is ever run and stimuli are excluded from the compile hash).

## Where things landed (as of now)

- **Best confirmed config:** `num-workers=10`, `LLK_BUILD_PARALLELISM=1`
  (4 inner g++ per worker), `--dist=worksteal`, `-Os`. At `-n 10`: **~61% peak
  RAM, zero swap, tied-fastest wall-clock** of all tested configs, no wasted CPU.
  This is the current default in `.github/actions/llk-compile-quasar/action.yml`.
- The starvation finding says the remaining ceiling is **per-item framework
  overhead + coordinator throughput**, not compute or memory — which is what the
  two open threads target.

## Two open threads

1. **Producer-mode stimuli/golden guard.** In producer mode the test body's
   `pytest.skip()` fires only at the *end* (inside `run()`), so stimuli + golden
   are computed per item and thrown away. Golden is already short-circuited to a
   cheap dummy; this branch additionally made that guard explicit and
   order-independent (defense-in-depth). The bigger prize — skipping the
   torch-based per-item **stimuli** fill — was investigated and **deliberately not
   implemented**: `generate_stimuli()` is one generic function called from 25 of
   26 quasar files, several test bodies transform the stimuli *values* before the
   skip point, and there is no runner in the sandbox to prove all-zeros is
   crash/hang-safe across every transform. Flagged as requiring real-runner
   validation.

2. **Test-item deduplication by `variant_id`.** The ~413k items collapse onto
   ~2.5–3.2k distinct compile variants (~120–160:1). Deduplicating them at
   *collection* time is the order-of-magnitude lever, but the dedup key must
   exactly mirror `generate_variant_hash()` or consumer mode fails with a hard
   missing-ELF. Full design, options, and the belt-and-suspenders drift-detection
   proposal are in
   [`quasar_compile_dedup_design.md`](quasar_compile_dedup_design.md).

## Caveats for readers

- Almost everything quantitative here was measured on **real CI runs** (sidecar
  memory/CPU + JUnit artifacts). The two things that are **inferred, not directly
  measured**, are (a) the coordinator-starvation *mechanism* (strongly supported
  by xdist source + A/B, but the clinching per-second samples aren't retrievable)
  and (b) the dedup savings *magnitude* (order-of-magnitude is an upper bound from
  the redundancy ratio, not a profiled number — no Quasar toolchain in the
  sandbox).
- The memory-scaling extrapolations were wrong twice (once ~2× off). Treat any
  future "project RAM at `-n X`" claim on this job with suspicion until measured
  at matched config.
