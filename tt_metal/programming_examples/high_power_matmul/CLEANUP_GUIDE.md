# tt-ember / high_power_matmul cleanup guide

This document is for someone picking up the power-measurement work with little or no prior
context. It explains where everything currently lives, which code produced which result in the
paper, and what the cleaned-up end state should look like: **one workload binary in tt-metal
that covers every grid size and every workload variant, and one driver script in tt-ember that
runs any of them.**

Nothing here needs to be done in one go. Section 6 breaks it into small, independently
reviewable PRs.

---

## 1. Background in five paragraphs

**tt-ember** (<https://github.com/tenstorrent/tt-ember>, private at the time of writing) is a
Python pipeline that measures power on Tenstorrent boards. It starts tt-umd's `telemetry`
binary in the background (samples VCORE, TDC current and TDP power at 50 Hz), runs a workload in
the foreground, and then joins the two logs on wall-clock time so every telemetry sample is
attributed to the workload interval that was running. From that it computes dynamic current
(raw minus the idle floor measured between intervals), charge, energy, energy-delay product and
pJ/FLOP, and draws charts.

**The workload** is `metal_example_high_power_matmul`, a TT-Metalium programming example that
lives in *this* directory (`tt_metal/programming_examples/high_power_matmul/`). It runs
`C = A x B` repeatedly on a sequence of core grids (3x2, 3x3, ... up to the device maximum),
pausing 5 s between grids, and prints the start/end timestamp of every grid's run. tt-ember
reads those timestamps. The example is **not on tt-metal `main`**; it only exists on side
branches, which is the main thing this cleanup fixes.

**The kernel has three parts** — a *reader* (NoC reads from DRAM into L1), a *compute* kernel
(the FPU work) and a *writer* (NoC writes back to DRAM). Power experiments switch each one into
an "idle" mode that still does the circular-buffer handshake (so nothing deadlocks) but skips
the real work. Comparing a run with one part idled against the fully active run isolates that
part's share of dynamic power. These are the `POWER_CASE` scenarios.

**The paper** that uses all of this is *"Energy Efficiency Across Tenstorrent Accelerator
Generations: Wormhole versus Blackhole"* (Turkmanović, Cvetković). It compares a Wormhole
**n300** against a Blackhole **p100a**. The PDF is not in git; ask Nikola or Haris for
`wormhole_vs_blackhole_energy_paper_3.pdf` (draft 3, 2026-09-10), which is the version this
guide's section numbers refer to.

**People.** Nikola Cvetković (Tenstorrent, `ncvetkovic`) ran the Blackhole side on a p100a
dev box. Haris Turkmanović (University of Belgrade, GitHub `turkmanovic`) ran the Wormhole
side and wrote most of the cross-op analysis code. Haris also has a fork of tt-metal, `https://github.com/turkmanovic/tt-metal`, which is the only place
some of the tt-metal commits below exist.

---

## 2. The workload's knobs

Every experiment is a combination of these. The goal of the cleanup is for a single build of
the example to accept all of them.

| Knob | How it is passed | Values | What it does | Where it exists today |
|---|---|---|---|---|
| `M N K iters` | positional args 1–4 | e.g. `1024 2048 2048 160` | matmul shape and repeat count | everywhere |
| `fixed_tiles_per_core` | positional arg 5 | `0` (default) or N | `0` = *split mode*: the same total work is divided across the active cores, so every grid does identical FLOPs. N > 0 = every core does exactly N tiles, so work grows with the grid. | `1dca802` and later |
| `POWER_CASE` | env var | `0`–`5` | shorthand for the idle combinations below | `v0-powercase`, `ember-p100-build` (0–4); case 5 only in the uncommitted v0 changes |
| `HIGH_POWER_DISABLE_READER` / `_COMPUTE` / `_WRITER` | env vars, forwarded as JIT defines | set/unset | idle one kernel | same as `POWER_CASE` |
| `HIGH_POWER_WRITE_AMPLIFICATION_PCT` | env var | `0`–`100` | extra write traffic, so the write path is stressed like the read path | same as `POWER_CASE` |
| `HIGH_POWER_BLOCK_M` / `_N` | env vars | e.g. `2` / `4` | L1 output blocking with tile reuse: each core accumulates an M x N patch of output tiles, cutting DRAM reads per multiply from 2.00 to 0.75 | `v0-blocked-matmul` only |
| `HIGH_POWER_OP` | env var, forwarded as a JIT define | `matmul add silu exp sigmoid gelu recip` | which per-tile instruction the compute kernel runs; the reader/writer traffic is byte-identical, so only the math differs | uncommitted v0 changes only |
| grid list | built in `build_test_grids()` | — | see below | differs per branch |

`POWER_CASE` meanings (the names are what tt-ember uses as output directory names):

| Case | tt-ember name | Reader | Compute | Writer | Write amp |
|---|---|---|---|---|---|
| 0 | `regular` | real | real | real | off |
| 1 | `writer_amp` | real | real | real | 100% |
| 2 | `compute_idle` | real | **idle** | real | 100% |
| 3 | `reader_compute_idle` | **idle** | **idle** | real | 100% |
| 4 | `reader_idle2` | **idle** | real | real | 100% |
| 5 | `writer_idle` | real | real | **idle** | 100% |

Case 1 is the "everything active" reference; 2, 4 and 5 each idle exactly one kernel against it.
Before case 5 existed, case 0 was used as a stand-in for "writer idle" — the published paper's
Figure 1 does this — but it answers a different question (cost of write amplification, not of
the writer as a whole).

**Grid lists.** Three versions exist:

* **Hard-coded, 15 grids** (`1dca802`): 3x2 … 7x7, then the device maximum appended. Fine on
  Wormhole (max 8x7), but on Blackhole it jumps from 7x7 (49 cores) straight to 11x10 (110).
* **Hard-coded, 18 grids** (`v0-powercase`, `ember-p100-build`): adds 8x7, 9x8, 10x9, which are
  filtered out on smaller devices. This is what the published paper's Blackhole data used.
* **Generated, 24 grids on Blackhole** (uncommitted v0 changes): for each width `x` from 3, emit
  `x x (x-1)`, `x x x`, `x x (x+1)` clipped to the device. On Wormhole this is exactly the same 15
  grids as before; on Blackhole it also fills 7x8, 8x8, 8x9, 9x9, 9x10, 10x10. **Keep this one.**

---

## 3. Where everything lives

### 3.1 tt-metal

| Branch / commit | Where | Base | Contents | Status |
|---|---|---|---|---|
| `ncvetkovic/high_power_usage_workload` @ `fbd63da` | `tenstorrent/tt-metal` | old main | first versions of the example + core-count sweep | superseded |
| `ncvetkovic/high_power_usage_workload` @ `1dca802` ("Latest") | `turkmanovic/tt-metal` only | same | adds `fixed_tiles_per_core`; pins tt-umd `38577fd` | **the build tt-ember `main` pairs with** |
| `ncvetkovic/blackhole_grid_sweep` @ `c4597f1` | `turkmanovic/tt-metal` | `1dca802` | June p150b experiment, grid up to 11x11 | obsolete |
| `ncvetkovic/v0-powercase` @ `eefc9e9` | local only; identical to `tt-metal-power-case.patch` in tt-ember | `1dca802` | `POWER_CASE` 0–4, `HIGH_POWER_DISABLE_*`, write amp, 18 grids | produced the paper's data |
| `ncvetkovic/v0-blocked-matmul` @ `c67d239` | local only; identical to `tt-metal-blocked-matmul.patch` in tt-ember | `eefc9e9` | `HIGH_POWER_BLOCK_M/N` | produced paper Fig. 2 |
| uncommitted changes on top of `c67d239` | local only; identical to `tt-metal-high-power-op.patch` in tt-ember (that patch is against `eefc9e9`, so it includes blocking) | `c67d239` | `HIGH_POWER_OP`, `POWER_CASE=5`, generated 24-grid list | produced the cross-op sweep and the 1 GHz Fig. 1 |
| `ncvetkovic/ember-p100-build` @ `21c938c` | `tenstorrent/tt-metal` | `main` @ `e0c2360` (2026-09-02) | the example ported to current main: API drift fixed (`mm_init` → `matmul_init` + `compute_kernel_hw_startup`, new `DPRINT` syntax), `POWER_CASE` 0–4, 18 grids | **the PR base**; no blocking, no `HIGH_POWER_OP`, no case 5 |
| `ncvetkovic/ember-cleanup-guide` | this branch | `ember-p100-build` | this document | — |

The three `.patch` files are committed in tt-ember (branch `ncvetkovic/p100a-crossop-iso-clock`)
at:

* `docs/results/old/p100a_power_cases/patches/tt-metal-power-case.patch`
* `docs/results/old/p100a_power_cases/patches/tt-metal-blocked-matmul.patch`
* `docs/results/p100a_crossop/patches/tt-metal-high-power-op.patch`

They are the durable copy of the v0 work. The first two are `git format-patch` output of the
commits above; the third is a plain diff against `eefc9e9`.

**Important trap:** the v0 line sits on a June tt-metal. Porting its changes onto current main
hits the same API drift `ember-p100-build` already fixed. Both breakages are invisible at build
time, because kernels are JIT-compiled when the program launches:

* `mm_init(in0, in1, out)` is gone. `matmul_init(in0, in1)` alone does **not** configure the
  packer; without `compute_kernel_hw_startup<SrcOrder::Reverse>()` `pack_tile()` produces
  nothing, the writer blocks forever, and the device hangs at ~65 W until the firmware watchdog
  trips.
* `DPRINT << x` is a hard `static_assert` now; use `DPRINT("fmt", args)`.

### 3.2 tt-ember

Remote `origin` = `https://github.com/tenstorrent/tt-ember`. `upstream` =
`https://github.com/turkmanovic/tt-energyprofiler` (Haris's pre-open-source repo).

```
main 7fce2fe
 ├─ ncvetkovic/fix-telemetry-tdp-regex ─┬─ ncvetkovic/fix-first-interval-baseline ─ ncvetkovic/fix-baseline-drift
 │                                      └─ ncvetkovic/ttnn-op-power-breakdown
 ├─ ncvetkovic/settle-trim
 └─ haris/compare-runs2-power-cases
      └─ ncvetkovic/p100a-power-case-results
           └─ haris/n300-p100a-power-case-results
                └─ ncvetkovic/p100a-crossop-iso-clock
051cd49 (pre-open-source history) ─ ncvetkovic/p150b-power-sweep-results
upstream/master (tt-energyprofiler, pre-open-source)
```

| Branch | What it adds | Notes |
|---|---|---|
| `ncvetkovic/fix-telemetry-tdp-regex` | 3-line `parser.py` fix | Newer tt-umd prints `TDP 16/150 W` instead of `TDP: 16 W`. `main`'s parser misreads it. Needed for any tt-metal newer than `1dca802`. |
| `ncvetkovic/fix-first-interval-baseline` | `parser.py` | Stops reporting a fabricated zero when an interval's idle baseline is unusable (typically the first grid). |
| `ncvetkovic/fix-baseline-drift` | `parser.py` | The idle floor drifts upward as the board warms (+0.5 to +3.7 A over a sweep). Interpolates the baseline over time instead of averaging it. |
| `ncvetkovic/settle-trim` | `parser.py`, `--settle-max-ms` | Drops the startup inrush at the head of each interval (one 8 ms spike tripled a reported peak). Same change as commit `f05dc30` in the results chain — **it exists twice**. |
| `haris/compare-runs2-power-cases` | `compare_runs2.py`, `run_power_cases.py` (284 lines), docs | The `POWER_CASE` sweep driver: resets the board and calls `auto.py` once per case. |
| `ncvetkovic/p100a-power-case-results` | p100a results (`docs/results/p100a_power_cases/`), `make_pj_per_flop*.py`, the two tt-metal patches, analysis write-up | Paper Fig. 2. |
| `haris/n300-p100a-power-case-results` | n300 results, n300-vs-p100a analysis, `make_pj_per_flop_by_engine.py`, `run_power_cases.py` grows to 1034 lines (`--ops`, case 5, per-engine and cross-op plots), cross-op n300 data in `docs/results/new/` | Paper Fig. 1, Tables 2–3. Moves p100a results into `docs/results/old/`. Accidentally commits `generated/inspector/` and `generated/watcher/` (tt-metal runtime output — delete). |
| `ncvetkovic/p100a-crossop-iso-clock` | p100a cross-op sweep at 1.35 GHz and 1.0 GHz, 1 GHz regeneration of Fig. 1, AICLK-pinning tools, `tt-metal-high-power-op.patch` | Not in the paper yet. |
| `ncvetkovic/ttnn-op-power-breakdown` | `tools/op_power_breakdown/`: `ttnn_ops_workload.py`, `preview_op_breakdown.py`, p100a results | A *different* workload: the 12 ops of a Llama-style decoder block, run through ttnn. Paper Figs. 3–4, Table 4. |
| `ncvetkovic/p150b-power-sweep-results` | `auto.py` fixes for Blackhole (telemetry freq units, venv path, `TT_METAL_HOME`), p150b raw data | June; not in the paper. The `auto.py` fixes are worth salvaging, the ~130k lines of raw telemetry are not. |
| `upstream/master` | Haris's pre-open-source repo, ~3.7M lines of raw logs | Do not merge. |

---

## 4. Paper → code map

"Data" paths are in tt-ember on the branch named in the second column.

| Paper item | tt-ember branch | Data / script | tt-metal build | Board |
|---|---|---|---|---|
| Table 2, §V-A/B numbers (crossover 36–42 cores, 26.2 vs 26.8, 21.1 pJ/FLOP) | `haris/n300-p100a-power-case-results` | `docs/results/old/n300_power_cases/power_comparison_analysis.md`, CSVs in `docs/results/old/*/data/` | `1dca802` + power-case patch (= `v0-powercase`) | n300 + p100a |
| Table 3 (ablation fractions) | same | same | same | both |
| Fig. 1 (pJ/FLOP per kernel stage) | same, commit `6125d7a` | `docs/results/old/n300_power_cases/make_pj_per_flop_by_engine.py`, p100a at 1.35 GHz, 18 grids | same | both |
| Fig. 1 at 1 GHz (candidate replacement) | `ncvetkovic/p100a-crossop-iso-clock` | `docs/results/p100a_crossop/paper_figures_1ghz/` (uses real `writer_idle`) | v0 + high-power-op patch | p100a |
| Fig. 2 (naive vs 2x4 blocked) | `ncvetkovic/p100a-power-case-results` | `make_pj_per_flop_naive_vs_blocked.py`, `regular.csv` vs `regular_blocked_2x4.csv` | `v0-blocked-matmul`, `HIGH_POWER_BLOCK_M=2 HIGH_POWER_BLOCK_N=4` | p100a only |
| Table 4, Figs. 3–4 (decoder block) | `ncvetkovic/ttnn-op-power-breakdown` | `tools/op_power_breakdown/` | stock tt-metal `main` @ `e0c2360` (ttnn only, no example needed) | p100a committed; **n300 data is not in any branch — get it from Haris** |
| Cross-op sweep (not in paper) | `haris/n300-…` (n300, `docs/results/new/`) and `ncvetkovic/p100a-crossop-iso-clock` (p100a) | `run_power_cases.py --ops …`, `docs/results/p100a_crossop/tools/crossop_corrected_flops.py` | v0 + high-power-op patch (n300 build not recorded) | both |

Workload arguments used in the paper: `1024 2048 2048 160` (M N K iters, split mode), telemetry
`-f 50`, `--slot-ms 1`, `--trim-ms 1.0`. The cross-op sweep used `K=4096`
(`1024 2048 4096 160`), so its absolute pJ/FLOP is not comparable with the paper's.

Things that are known to be missing or fragile:

1. **The newest tt-metal work is not pushed as commits.** `HIGH_POWER_OP`, case 5 and the
   generated grid list exist as a `.patch` file in tt-ember and as uncommitted changes on
   Nikola's dev box. Blocking and `POWER_CASE` on the old base are also only local branches
   (again, with `.patch` copies in tt-ember).
2. **n300 decoder-block data** for Figs. 3–4 / Table 4 is not committed anywhere.
3. **Newer loose copies of the op-breakdown scripts** exist on Nikola's dev box
   (`ttnn_ops_workload.py`, `preview_op_breakdown.py` with bytes/iter, pJ/byte and a normalised
   chart), plus `probe_op_core_grids.py`, `ttnn_dutycycle_probe.py`. Ask for them before
   starting PR 6.
4. **The paper's numbers predate the baseline fixes.** Merging `fix-first-interval-baseline` /
   `fix-baseline-drift` changes the computed values. Either regenerate the results after they
   land or record which parser version produced each result.
5. **Raw captures** (the `telemetry.txt` / `summary.txt` behind every committed CSV) live only
   on the dev boxes, not in git. That is fine — the CSVs are enough to regenerate every figure —
   but it means a run cannot be re-parsed with a new parser without re-measuring.
6. Every result is **one measurement per grid**. Scatter between neighbouring grids above ~36
   cores is about ±1 A.

---

## 5. Target end state

### 5.1 tt-metal: one example

`tt_metal/programming_examples/high_power_matmul/` on `main`, with:

* all the knobs in section 2 (`fixed_tiles_per_core`, `POWER_CASE` 0–5, `HIGH_POWER_DISABLE_*`,
  write amplification, `HIGH_POWER_BLOCK_M/N`, `HIGH_POWER_OP`);
* the generated grid list, so Wormhole and Blackhole (and anything bigger) get a sensible sweep
  with no hard-coded list;
* a startup banner that prints every effective knob, so the `summary.txt` of any run says
  exactly what was measured (tt-ember already relies on the `POWER_CASE` confirmation line);
* a README covering every knob. `ember-p100-build`'s README already documents `POWER_CASE`.

Optional but useful: an explicit `--grids` argument (e.g. `--grids 4x4,8x7`) to run a subset
without editing code.

### 5.2 tt-ember: one driver

Today there are three drivers: `run_all.py` (the six-run prefill + fixed-tpc sweep on `main`),
`run_power_cases.py` (the `POWER_CASE` × op sweep) and `tools/op_power_breakdown/` (the ttnn
decoder-block workload). They all do the same thing — reset the board, run `auto.py` with some
env vars and app args into a named subdir, then call a comparison script — but with different
hard-coded lists.

Replace them with one entry point. Suggested shape (the name is a suggestion):

```bash
python3 run_sweep.py --config sweeps/paper_fig1.yaml   \
  --telemetry-exe ... --app-exe ... --tt-metal-root ... --output-root ./out
```

where a sweep config lists runs as combinations of:

* `workload`: `high_power_matmul` or `ttnn_ops`;
* `app_args`: M N K iters [fixed_tiles_per_core];
* `power_cases`, `ops`, `block` (`[M, N]` or none), `write_amp_pct`;
* optional `aiclk_mhz` (uses the existing `tt-smi-aiclk1000.sh` / `set_aiclk.py` approach: the
  clock has to be re-pinned after every reset);
* which analyses to run afterwards (`compare_runs2`, per-engine pJ/FLOP, cross-op, naive vs
  blocked, op breakdown).

Requirements that the existing scripts already get right and must be kept:

* reset (`tt-smi -r`) before every run, then a cool-down;
* skip any output dir that already exists, unless `--force` (sweeps take hours; they get
  resumed);
* `--dry-run` that prints every command;
* write every env var and argument into the run's `launcher.log`.

Ship one config per paper result under `sweeps/` (Table 2/3 + Fig. 1, Fig. 2, cross-op, decoder
block), so each figure is one command. The per-figure `make_pj_per_flop*.py` scripts become
analysis steps of the driver, not files scattered under `docs/results/*/`.

### 5.3 tt-ember: results layout

One directory per paper result, containing only CSVs, the config that produced them, the exact
tt-metal and tt-ember commits, and the figure. For example:

```
docs/results/paper/
  fig1_stage_breakdown/{n300,p100a}/*.csv  sweep.yaml  PROVENANCE.md  fig1.png
  fig2_blocked/…
  fig3_4_decoder_block/{n300,p100a}/…
  crossop/{n300,p100a_1350,p100a_1000}/…
```

No raw telemetry, no `generated/`, no `__pycache__`, no `old/` / `new/` staging dirs.

---

## 6. Suggested PR sequence

Each step is small and leaves both repos working.

1. **tt-ember — TDP regex.** Merge `ncvetkovic/fix-telemetry-tdp-regex`. Without it tt-ember
   `main` only works with tt-metal `1dca802`.
2. **tt-metal — the example on main.** Open a PR from `ncvetkovic/ember-p100-build`, or
   first extend it with steps 3–4 and open one PR. Test on Wormhole and Blackhole: every grid
   completes and the `POWER_CASE` banner matches.
3. **tt-metal — blocking.** Port `tt-metal-blocked-matmul.patch` onto the branch. Expect
   conflicts in `mm_power.cpp` because of the API drift in section 3.1. Check: 2x4 on a p100a
   at 11x10 should give about 2.6x the throughput of naive at essentially the same board power.
4. **tt-metal — `HIGH_POWER_OP`, case 5, generated grids.** Port the remainder of
   `tt-metal-high-power-op.patch`. Check: on Wormhole the grid list is still exactly the 15 grids
   3x2 … 8x7; on Blackhole it is 24 grids ending at 11x10.
5. **tt-ember — driver scripts.** Land `compare_runs2.py` + the base `run_power_cases.py`
   (`haris/compare-runs2-power-cases`), then Haris's extensions from
   `haris/n300-p100a-power-case-results` as a separate PR (code only; no results; drop
   `generated/`). Salvage the `auto.py` fixes from `ncvetkovic/p150b-power-sweep-results`.
6. **tt-ember — parser fixes.** One `settle-trim` (drop the duplicate), then
   `fix-first-interval-baseline`, then `fix-baseline-drift`.
7. **tt-ember — op-breakdown tool**, from `ncvetkovic/ttnn-op-power-breakdown` plus the newer
   loose scripts.
8. **tt-ember — unify** into the single driver of section 5.2, deleting `run_all.py` /
   `run_power_cases.py` once the configs reproduce their output.
9. **tt-ember — results.** Re-home the paper results as in section 5.3, pinned to the merged
   commits. Delete the `.patch` files once steps 2–4 are on tt-metal `main`.

### Branch cleanup

After each branch is merged, or its content has been moved elsewhere, **tag it before
deleting it**, so the exact state behind the published paper stays reachable:

```bash
git tag archive/<branch-name> <branch> && git push origin archive/<branch-name>
git push origin --delete <branch>
```

Candidates, once their content has landed: every tt-ember branch in section 3.2 except `main`;
in tt-metal, `ncvetkovic/ember-p100-build`, `ncvetkovic/high_power_usage_workload` and this
branch. Leave `turkmanovic/*` branches to Haris. Tag `1dca802` on the fork (or ask Haris to
keep it), because the published paper's data was produced on it.

---

## 7. Quick reference: reproducing a paper result today (before cleanup)

Build tt-metal at `1dca802` from Haris's fork, apply the patch(es), then run tt-ember from the
branch in section 4:

```bash
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git remote add haris https://github.com/turkmanovic/tt-metal.git
git fetch haris ncvetkovic/high_power_usage_workload
git checkout -b repro 1dca802d332
git submodule update --init --recursive        # tt-umd 38577fd
git am < …/tt-metal-power-case.patch           # + tt-metal-blocked-matmul.patch for Fig. 2
./build_metal.sh --build-programming-examples --build-type Release

cd …/tt-ember && git checkout haris/n300-p100a-power-case-results
python3 run_power_cases.py \
  --telemetry-exe $TT_METAL_HOME/build_Release/tools/umd/telemetry \
  --app-exe $TT_METAL_HOME/build_Release/programming_examples/metal_example_high_power_matmul \
  --parser-script ./parser.py --tt-venv-activate /path/to/venv/bin/activate \
  --tt-metal-root $TT_METAL_HOME --output-root ./out_power \
  --trim-ms 1.0 --power-cases 0 1 2 4 --ops matmul \
  --app-args 1024 2048 2048 160
```

`--app-args` must be the last argument (it swallows everything after it). For Fig. 2, run
`auto.py` directly with `HIGH_POWER_BLOCK_M=2 HIGH_POWER_BLOCK_N=4` exported, because
`run_power_cases.py` only forwards `POWER_CASE` and `HIGH_POWER_OP`.
