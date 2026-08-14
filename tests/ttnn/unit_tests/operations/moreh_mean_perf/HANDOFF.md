# Handoff: perf comparison of `malimpic/reduce-partial-scaler` vs its merge-base

**Session date:** 2026-08-13/14
**Box used:** Wormhole **n300** (2 chips), 64-core compute grid, AICLK 1000 MHz (1 cycle = 1 ns).
Host had 8 cores — builds are the slow part.

**Everything in this directory travels with git. Nothing else from the session does.**
See [Carrying this to another machine](#9-carrying-this-to-another-machine) — read that first if
you are setting up fresh.

---

## 1. What was established

Two commits compared:

| | SHA | |
|---|---|---|
| head | `9751f4fd9f5` | `moreh: reduce a whole bias-grad column in one reduce call` |
| base | `fda1e45f96f` | merge-base with `main` |

`git merge-base HEAD main` — re-derive it rather than trusting the SHA if the branch was rebased.

**Whole-op result (metric: `DEVICE KERNEL DURATION [ns]`, medians of 3 runs/commit):**
11 of 32 suite cases faster, 20 unchanged, 1 unresolved, 4/4 controls held.

Three distinct mechanisms, not one effect:

1. **Ragged-penalty removal** — `moreh_mean`/`moreh_sum` along H. `sum_h_ragged` −31.2%,
   `moreh_mean` ragged −6.9% to −51.6% depending on `Ht`. Aligned shapes unchanged.
   Absolute saving is **constant** (~1.67 µs per column-per-core) — the percentage only shrinks
   because the denominator grows.
2. **Reduce-call consolidation** — softmax family. `softmax_backward` `_small` −16% to −26.7%;
   forward `_small` −4.0% to −6.1%. Aligned gains **as much as or more than** ragged here, so
   this is a different mechanism from (1).
3. **No effect** — all `_large` softmax variants, `layernorm_large`, `bias_bw`,
   `topk_router_gpt` (a pure refactor onto the helper, no partial scaler at all).

**Zone-scope result (`moreh_mean` H, Ht=4, math thread, ns/core over 7 columns):**
the ragged penalty *inside the rewritten region* went **12,299 → 341 ns (−97%)**, and that
saving (11,958 ns) reconciles with the whole-op saving (12,215 ns) to **2.1%** — so essentially
all of the whole-op win lives in that compute region. `mask_phase` = 14,386 ns at base, gone at
head; 13,240 of it was `mask_work` (two `copy_tile`s + `mask_tile`).

Published writeups:

- Whole-op audit — https://claude.ai/code/artifact/b3c64e0f-1a3b-4c32-a864-e01faacdb146
- Zone attribution — https://claude.ai/code/artifact/351166f0-7bb7-4531-bd02-3b939a8a363c

---

## 2. Corrections to `PERF_SETUP_HANDOFF.md`

That file (from the prior session, at the repo's parent dir) is accurate on the `.claude`
symlink setup but **wrong or silent on three things that cost real time**:

1. **Submodules are not initialised.** The first build dies instantly at CMake configure with
   `Missing submodules`. All four must be initialised (§3).
2. **`--build-tests` does NOT give you `perf-ceiling-dm`.** On this branch `noc_estimator` is
   only an `add_library(... OBJECT)`; the `noc_estimate` CLI the skill's wrapper looks for is
   declared **only** on `origin/llk_helper_library`
   (`add_executable(noc_estimate ...)` in `tests/tt_metal/tt_metal/data_movement/noc_estimator/CMakeLists.txt`).
   `--build-tests` does build `unit_tests_data_movement`, `generate_noc_latencies` and
   `test_noc_estimator`. **So the theoretical-ceiling half of the perf skills is unavailable
   here** unless you cherry-pick that target. The measured comparison does not need it.
3. **`build_metal.sh` does not enable ccache by default** — it needs `-c`. This matters a lot:
   with ccache warm, switching between the two commits is a ~540-step incremental build instead
   of the ~1,864-step cold build.

Also still true from that file: `scripts/tt-probe.sh` does not exist on this branch (only on
`origin/llk_helper_library`). It is not needed for any of the perf work — everything here drives
the device through `scripts/run_safe_pytest.sh`.

---

## 3. Setup on a fresh machine

```bash
# 0. Skills. Claude Code discovers skills at .claude/skills; without this symlink NONE load.
#    Requires a tt_ops_code_gen checkout next to the tt-metal clone.
cd <tt-metal>
ln -sfn /path/to/tt_ops_code_gen tt_metal/third_party/tt_ops_code_gen
ln -sfn tt_metal/third_party/tt_ops_code_gen .claude
echo tt_metal/third_party/tt_ops_code_gen >> .git/info/exclude   # local only
#    The perf-ceiling-dm scripts compute the repo root as ../../.. from their own dir, so they
#    must be invoked via the .claude/... path, not the real submodule path.

# 1. Submodules — REQUIRED, the build fails immediately without them.
git submodule update --init --recursive

# 2. Build. -c enables ccache; point it somewhere big and set a real max size.
export CCACHE_DIR=/path/on/big/disk/ccache CCACHE_MAXSIZE=60G
./build_metal.sh --build-tests -c        # profiler (Tracy) is ON by default

# 3. venv — must come AFTER the build (it does `pip install -e .` against build/).
./create_venv.sh
source python_env/bin/activate
python3 -c "import ttnn; import tracy.serve_wasm"   # both must succeed for --profile
```

`--build-tests` is optional for everything in this directory. Keep it only if you want
`unit_tests_data_movement` for NoC-latency calibration.

**Commit switching, once ccache is warm:**

```bash
git checkout <sha>                       # untracked files here survive this
./build_metal.sh --build-tests -c        # ~540 steps, minutes not hours
```

Submodule pins are **identical** between the two commits (verified), so no submodule churn when
switching — the only difference is the branch's own code.

---

## 4. What is in this directory

```
bench_moreh_mean_h.py         whole-op bench, moreh_mean along H, Ht sweep 1/4/16/32
                              + moreh_mean-along-W contamination control
bench_reduce_partial_suite.py whole-op bench, 32 cases across 7 op families,
                              forced softmax strategies, 4 untouched-path controls
bench_topk_router_gpt.py      separate file: needs dispatch_core_axis=ROW device fixture
bench_zones_moreh_mean.py     zone bench, Ht in {1,4} ONLY (marker budget, see §6)
check_ragged_correctness.py   torch-comparison probe for the ragged paths, runs on both commits

tools/extract_perf.py         parses ops CSV for bench_moreh_mean_h.py; --diff two JSONs
tools/extract_suite.py        parses ops CSV for the 32-case suite and the topk bench; --diff
tools/zones.py                pairs DeviceZoneScopedN records, normalises per core,
                              CHECKS THE MARKER CAP
tools/zones_head.patch        zone instrumentation for the head kernels
tools/zones_base.patch        zone instrumentation for the base kernels (different structure!)

results/*.json                this session's measurements, for diffing against future runs
```

Note: the extractors write their output JSON **next to the script** (i.e. into `tools/`).
`results/` holds the session snapshots; `cp` new runs in or pass explicit paths when diffing.

All of it is **untracked**, deliberately: a `git checkout` of either commit leaves it in place,
so the same harness measures both sides.

---

## 5. Reproducing the whole-op comparison

```bash
source python_env/bin/activate
D=tests/ttnn/unit_tests/operations/moreh_mean_perf

# correctness first — establishes both commits are correct on the paths being timed
scripts/run_safe_pytest.sh --run-all $D/check_ragged_correctness.py

# measure (repeat at each commit)
unset TT_METAL_DPRINT_CORES
rm -rf ~/.cache/tt-metal-cache          # fresh kernel build; else you profile the old binary
scripts/run_safe_pytest.sh --profile --run-all $D/bench_reduce_partial_suite.py
# prints: SAFE_PYTEST: PROFILER CSV: <path>

python3 $D/tools/extract_suite.py <label> $D/bench_reduce_partial_suite.py <csv>
python3 $D/tools/extract_suite.py --diff <base>.json <head>.json
```

`BENCH_CHECK=1` adds numeric checks to the benches — leave it **off** when measuring, it adds
`to_torch` readbacks that appear as extra CSV rows.

`--profile` masks pytest's exit code (the tracy wrapper exits 0 regardless), so confirm
correctness with a separate plain run.

---

## 6. Reproducing the zone comparison

```bash
git apply $D/tools/zones_head.patch     # at head; zones_base.patch at the merge-base
unset TT_METAL_DPRINT_CORES             # zones share SRAM with DPRINT/Watcher
rm -rf ~/.cache/tt-metal-cache          # kernels are JIT — no host rebuild needed for a patch
scripts/run_safe_pytest.sh --profile --run-all $D/bench_zones_moreh_mean.py
python3 $D/tools/zones.py <label>       # reads generated/profiler/.logs/profile_log_device.csv
git checkout -- ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl \
                ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp
```

Two patches exist because the two kernels differ structurally: base has the mask phase, head
does not.

**Marker budget is the binding constraint and it fails silently.** 250 optional markers per RISC
per dispatch; a zone costs 2; the count is *executions*, not names. The per-tile wait/math pair
burns 4 markers per input tile:

| shape | tiles/core | markers | |
|---|---|---|---|
| Ht=1 | 7 | ~42 (46 measured) | ok |
| Ht=4 | 28 | ~126 (130–186 measured) | ok |
| Ht=16 | 112 | ~462 | **truncates silently** |
| Ht=32 | 224 | ~910 | **truncates silently** |

Past the cap zones are **absent**, not truncated — starts/ends stay balanced and nothing is
logged, so the profile looks complete while describing only the start of the kernel.
`tools/zones.py` prints the peak marker count per `(core, RISC)`; check it every run. To zone the
Ht=16/32 shapes you must first hoist the zones out of the per-tile loop.

---

## 7. Traps already hit — do not re-discover these

1. **The ops CSV has no test-name column.** An extractor keyed on a pytest node id labels every
   row `None` and silently collapses all cases into one. Match on `OP CODE` +
   `COMPUTE KERNEL SOURCE` + reduce-dim length instead (that also proves *which program factory
   ran*) and abort rather than guess. Both extractors do this.
2. **`normalization/softmax` W needs rank ≥ 5.** At rank 4 with `dim == rank-1` the op routes to
   the **attention-optimized** factory (`softmax.cpp`), not `general_w_*`. ttnn normalizes rank-3
   input up to 4D, so rank 3 does not work either. Caught only because the extractor asserts the
   kernel path.
3. **`softmax_backward` has its own strategy enum** — `SoftmaxBackwardOpParallelizationStrategy`,
   not `SoftmaxOpParallelizationStrategy`. Passing the forward one is a `TypeError`.
4. **`moreh_linear_backward` requires a pre-allocated `bias_grad`** — it `TT_FATAL`s on
   `bias_grad.has_value()` rather than allocating.
5. **Inner zone numbers are not comparable across the two commits.** `rdc_math` reads 4,876 ns
   (base) vs 10,078 ns (head) on the *aligned* shape — an apparent 2× regression that is an
   artifact: base's accumulator reload sits in `reload_accumulator_if_needed`, outside the zoned
   `reduce_tile`, so base hides work in the un-zoned residual. **Only `col_total` brackets the
   same work on both sides.** Ranking off the inner zone invents a regression.
6. **A flat ±3% noise band is wrong on this box.** Three of the four controls carry 4–6%
   run-to-run spread of their own; `sum_w_control` at +3.6% briefly tripped a false
   "controls failed". Judge each case against **its own measured spread**, floor 3%. The win
   cases all sit at 0.2–3.1% spread, which is why their deltas are believable.
7. **Per-RISC whole-kernel columns give no attribution.** All five threads are resident for the
   whole kernel, so every one drops by the same ~31%. Do not read the reader's −30.9% as "the
   reader got faster". Use zones.

---

## 8. Open threads

Ranked by how much they'd change the picture:

1. **`bias_bw` — the headline commit shows no measurable win** (+0.5% / +0.1%). Base reduced one
   tile at a time (24 calls/column); head collapses that to 1 (aligned) or 6 (ragged). Going
   24 → 1 changed nothing measurable, so this op is not bound by reduce-call overhead at this
   shape. Separately, **ragged costs 2.6× aligned in both commits** (45.8 vs 17.8 µs) at
   identical padded tile counts. Needs **ablation** (`/perf-measure` §0) to attribute — no cause
   claimed yet.
2. **`layernorm` non-large is ~2–4% slower and unresolved.** `aligned +3.6%` / `ragged +2.3%`
   against 2.9–3.5% own spread. Consistent in direction, too close to call. A first single run
   read +4.5%; the 3-run median pulled it to +2.3%. More trials will not settle it — needs a
   bigger lever or an ablation.
3. **Zone the `softmax_backward` `_small` path.** It is the biggest winner (−16% to −26.7%) and
   its mechanism is reduce-call consolidation, not mask removal, so the `moreh_mean` breakdown
   does **not** transfer. Expect a different decomposition.
4. **`moreh_mean` zone coverage stops at Ht=4.** The headline whole-op deltas run to Ht=32, which
   cannot be zoned without hoisting zones out of the per-tile loop (§6).
5. **`moreh_mean` whole-op numbers are single-run**, unlike the other seven families (median of
   3). Its constant-absolute-saving pattern and `moreh_sum`'s independent agreement are what
   carry it.
6. **After the change this path is reader-bound.** Unpack's `rdc_wait` went 7,298 → 16,208 ns, so
   further compute-side work on the reduce buys little. The lever is the reader.

Not started: folding the zone section into the whole-op artifact; cherry-picking `noc_estimate`
to enable `perf-ceiling-dm`.

---

## 9. Carrying this to another machine

This directory is untracked, so `git push` will not take it. Commit it explicitly on a scratch
branch (verified: the path is **not** covered by any ignore rule, so plain `git add` is enough):

```bash
git checkout -b malimpic/reduce-partial-scaler-perf-notes
git add tests/ttnn/unit_tests/operations/moreh_mean_perf/
git commit -m "perf: bench harness, tooling and measurements for reduce-partial-scaler"
git push -u origin HEAD
```

Then on the new machine: clone, `git checkout` that branch, and work through §3. Do **not**
expect `build*/`, `python_env/` or the ccache to come with it — those are machine-local and §3
rebuilds them from scratch.

If you would rather not push a branch, `git diff` is no use for untracked files — use
`tar czf perf-notes.tgz tests/ttnn/unit_tests/operations/moreh_mean_perf/` and copy that.

**Do not commit this directory onto `malimpic/reduce-partial-scaler` itself** unless you mean to
ship the benches — the branch's tracked tree was clean at the end of this session and the perf
work deliberately left it that way.
