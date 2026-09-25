# Where this directory came from

This is a one-shot import of [tt-ember](https://github.com/tenstorrent/tt-ember), the power
telemetry pipeline used for the Wormhole-vs-Blackhole energy paper, into tt-metal. It is a
**reference drop, not a merge candidate**: the intent is to split it into small PRs, following
the plan in
[`tt_metal/programming_examples/high_power_matmul/CLEANUP_GUIDE.md`](../../tt_metal/programming_examples/high_power_matmul/CLEANUP_GUIDE.md),
and then delete this directory.

tt-ember's own git history is not carried over; the tt-ember repo keeps it.

## Source

Everything is the newest version available on 2026-09-25, taken from tt-ember
`ncvetkovic/p100a-crossop-iso-clock` @ `07a516f536` (which contains Haris's
`haris/n300-p100a-power-case-results` and everything below it), plus
`ncvetkovic/fix-telemetry-tdp-regex` @ `ab1d163585` cherry-picked on top.

| Here | From |
|---|---|
| `auto.py`, `parser.py`, `run_all.py`, `compare_runs.py`, `compare_runs2.py`, `run_power_cases.py`, `README.md`, `CLAUDE.md`, `.gitignore`, `diagnostics/`, `docs/` | tt-ember top level |
| `analysis/make_pj_per_flop.py`, `analysis/make_pj_per_flop_naive_vs_blocked.py` | `docs/results/old/p100a_power_cases/` |
| `analysis/make_pj_per_flop_by_engine.py` | `docs/results/old/n300_power_cases/` |
| `analysis/crossop_corrected_flops.py`, `aiclk/set_aiclk.py`, `aiclk/tt-smi-aiclk1000.sh` | `docs/results/p100a_crossop/tools/` |
| `patches/tt-metal-power-case.patch`, `patches/tt-metal-blocked-matmul.patch` | `docs/results/old/p100a_power_cases/patches/` |
| `patches/tt-metal-high-power-op.patch` | `docs/results/p100a_crossop/patches/` |
| `op_power_breakdown/README.md` | `ncvetkovic/ttnn-op-power-breakdown` @ `6512a93209` |
| `op_power_breakdown/ttnn_ops_workload.py`, `preview_op_breakdown.py` | Nikola's working copies, newer than that branch (add bytes/iter, pJ/byte, a normalised chart) |
| `op_power_breakdown/probe_op_core_grids.py`, `ttnn_dutycycle_probe.py` | Nikola's working copies, never committed to tt-ember |

## Changes made while importing

* `analysis/make_pj_per_flop_naive_vs_blocked.py` took its two input CSVs and its output path
  from hard-coded dev-box paths; they are now arguments. Verified to regenerate the paper's
  Fig. 2 from the original captures.
* `aiclk/tt-smi-aiclk1000.sh` hard-coded the venv Python and `set_aiclk.py` location; it now
  uses `$PYTHON` (default `python3`) and `$SET_AICLK` (default: `set_aiclk.py` next to the
  script).
* `op_power_breakdown/ttnn_dutycycle_probe.py` hard-coded a venv shebang and a default
  `TT_METAL_HOME`; it now uses `/usr/bin/env python3` and requires `TT_METAL_HOME`, like
  `probe_op_core_grids.py`.
* `README.md` links to `LICENSE`, `CONTRIBUTING.md` etc. now point at tt-metal's root copies
  (tt-ember's were the same Apache-2.0 texts).
* `.gitignore`: dropped two personal output directories.

No other code was changed. In particular, nothing was reformatted, so `black`/`isort`/`pylint`
findings from tt-metal's pre-commit are expected and are part of the cleanup.

## Deliberately left out

* **Results**: every CSV, PNG and analysis write-up under tt-ember `docs/results/` and
  `tools/op_power_breakdown/results/`. They support a paper that is not published yet, and
  results do not belong next to the tool anyway (see CLEANUP_GUIDE.md section 5.3).
* Raw telemetry captures, tt-metal runtime output (`generated/`), `__pycache__`.
* tt-ember's `LICENSE`, `LICENSE_understanding.txt`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`:
  duplicates of tt-metal's.
* The pre-open-source `upstream/master` and the June `p150b-power-sweep-results` branches.
  The latter has three small `auto.py` fixes for Blackhole (telemetry frequency units, venv
  path, `TT_METAL_HOME`) that are worth porting separately.

## Open decision: how the idle baseline is estimated

Two incompatible changes to `parser.py` exist, and this import takes only the first:

1. **One fixed reference per run** (Haris, tt-ember commit `9248a1d`, included here). The
   baseline for every interval is the idle current in the pause after the *first* interval.
   Rationale: the idle floor drifts upward as the board warms, so two cases measured at
   different points of a long sweep would otherwise get baselines from different points on the
   drift curve, and a case with higher raw current could come out with lower dynamic current.
2. **Follow the drift** (`ncvetkovic/fix-first-interval-baseline` →
   `ncvetkovic/fix-baseline-drift`, not included). The baseline is interpolated in time
   between the pauses on either side of each interval, and the one-sided first/last intervals
   are flagged (`base_one_sided`, `base_unusable`).

(1) makes cases within one sweep comparable; (2) is more accurate within one interval. They
could coexist behind a `--baseline {fixed,interpolated}` flag. Decide with Haris and Nikola
before touching `compute_interval_metrics()`. The paper draft's numbers predate both.
