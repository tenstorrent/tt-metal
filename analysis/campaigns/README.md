# analysis/campaigns: the SDPA and TopK measurement tooling

Everything in this directory drives, records or reduces one device measurement of the September 2026
SDPA and TopK campaign on a Blackhole p100a. Every script takes its configuration from environment
variables, runs one point per process, writes a PROVENANCE line with the card, the firmware bundle,
the git sha and branch, the exact command and the compiled zone config, and then calls a reducer.
No script hardcodes a user directory, so a fresh clone runs unedited.

## What is in the directory

- **The runners**, one process each, all writing into `$DD`: `run_zone.sh` (one
  `analysis/zone_sweep.py` prefill point, `plain` or `mp` for the perf-counter multipass capture),
  `run_regime.sh` and `run_regime_fresh.sh` (any pytest target, with an ops report),
  `run_decode.sh` (`analysis/decode_sweep.py` paged decode, always `tracy -r`), `run_fresh.sh`
  (`run_zone.sh` on the main-tip checkout), `run_zone_tax.sh` (the nop-kernel zone cost harness) and
  `set_zone_config.sh`, which rewrites the five compile-time kernel toggles
  `ZONES READER_STUB MASK_OFF EXP_STUB BARRIER_THR`.
- **The campaign scripts**, one block each, each a fixed list of runner calls with the toggles set
  back to all zeros at the end: `campaign_t21.sh` (the grid), `campaign_t22.sh` (the ablations),
  `campaign_t23.sh` and `campaign_t23r.sh` (the DRAM law and the regimes), `campaign_t24_zoff.sh`
  and `campaign_t24_zon.sh` (the production configuration), `campaign_t25.sh` (the model-level
  attention runs), `campaign_t26.sh` (TopK), `campaign_t27.sh` (main-tip drift), `campaign_t28.sh`
  (decode), `campaign_t29.sh` (indexer and TopK counters), and `campaign_r1.sh`,
  `campaign_r1_add.sh`, `campaign_r1f.sh` and `campaign_r1g.sh` (the re-measurement blocks).
  `campaign_t21.sh`, `campaign_t22.sh` and `campaign_t24_zoff.sh` accept `POINTS`, `MODES`, `ABL`,
  `WITH_MP`, `PROD_Q_DTYPE` and `PROD_KV_DTYPE` overrides.
- **The reducers**, all offline, all run with no arguments and all writing beside themselves:
  `zone_decomp.py` (the per-tag decomposition that joins the zones-off wall, the zones-on parts and
  the counters), `report_tables.py`, `regime_tables.py`, `reduce_decode.py`, `r1_tables.py`,
  `r1b_decode_table.py`, `r1f_counters.py`, `r1f_table.py`, `r1g_threads.py` and `r1_cardlog.py`.
- **The TopK campaign driver**, `topk_campaign.py`: a standalone script, not a pytest module, that
  walks a fixed grid of 300 cells. `--list` enumerates without a card, `--classes` selects a class
  group, `--out` names the output directory, `--exclude-op` skips a cell op, and `--postprocess`
  joins a tracy ops report to a cells CSV offline.
- **The instrumentation appliers**, `apply_zone_patch.py` (streaming kernel path) and
  `apply_zone_patch_common.py` (non-streaming path): anchor-matching rewriters for porting the SDPA
  zone instrumentation onto a different kernel base. Both refuse to run twice and both abort when an
  expected source line has changed.
- **The figure and model scripts**, `figs/` and `model/`: `figs/a_figs_part1.py`,
  `figs/a_figs_part2.py`, `figs/m_figs.py` and `figs/tm_fit_figs.py` draw the report figures, and
  `model/floor_verification.py`, `model/refit_r2_floor_r1f.py` and `model/refit_r2_fit.py` fit and
  score the roofline floor constants against the campaign counters. All of them are read-only on
  their inputs and all of them are run with the polaris virtual environment interpreter, because they
  need matplotlib 3.11, numpy 2.5 and, in some cases, the model imported in place. Two further figure
  scripts are internal and are absent from this branch.

## How the paths resolve

`campaign_paths.sh` is sourced by every runner and every campaign script, and the python scripts
resolve the same names through `os.environ`. It supports two layouts with no edit to any script:

1. **Analysis workspace layout.** The scripts sit in the `data/bh_zones` directory of an analysis
   workspace, `DD` is that directory, `WORK` is four levels above it, and `HANDOFF` is the workspace
   root two levels above.
2. **Checkout layout, which is this one.** The scripts sit in `analysis/campaigns` of a tt-metal
   checkout. `TTM` is found by walking up from the script's own location for the nearest directory
   that holds both `ttnn` and `tt_metal`, `WORK` is the directory above it, and `DD` defaults to
   `$PWD` so that nothing is ever written into the repository tree. A `DD` that does land inside the
   checkout prints a note.

Every value is an environment override that wins over the resolved default: `DD` (where captures and
derived tables are written), `SDPA_WORK`, `TTM`, `TTM_FRESH`, `PYENV` (the tt-metal virtual
environment activate script, `$TTM/python_env/bin/activate` by default) and `HANDOFF`. Set `DD` to a
directory of your own before any new measurement, or the first write fails. The full contract is
`PORTABLE_CONTRACT.md` beside this file.

## The full runbook

`BLACKHOLE_RUNBOOK.md`, beside this file, is the complete procedure: the clones and the two
checkouts, the profiler build, every sweep harness and its environment variables, the zone
instrumentation and the ablations, the perf counters, the ops reports, the TopK campaign, the polaris
model and its validation, the gotchas, and the provenance of every data folder.

## Quick start

```bash
export DD=$HOME/sdpa_campaign && mkdir -p $DD                                  # captures land here
cd analysis/campaigns && ./set_zone_config.sh 0 0 0 0 0                        # kernel toggles off
SDPA_CAUSAL=1 SDPA_SEQ=4096 SDPA_QCHUNK=128 SDPA_KCHUNK=128 ./run_zone.sh anchor_zoff
```
