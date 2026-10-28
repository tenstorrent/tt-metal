# Prefill stage 1 (baseline): work log

Stage folder (logs, raw records, raw Tracy CSVs, scripts): `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/01-baseline/`
(below: `$R`). References: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/refs/`. Host
sjc-snva-tp100, card 0 (P150a). Base commit `4624946b1b`. Every device command ran through
`$R/scripts/run.sh` (card-free check, env dump, `timeout -s KILL`, `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`,
log with `__start` / `__end wall_s` / `__tt_exit`), one at a time.

## Commits

| commit | content |
|---|---|
| `3eed8629a2` | E8 review doc fixes (datatype_sweep README vision worst-token attribution; benchmark REPORT.md; README) - docs only |
| `7ef04f02da` | stage-1 harness: `tests/perf/prefill.py`, `prefill_refs.py`, `prefill_telemetry.py`, `prefill_op_table.py`, `prefill_summary.py`; `perf.py::test_tracy_forward` contract buckets. No model code. |
| `f6b3ded953` | `doc/prefill/target_contract.json`, `doc/prefill/01-baseline/` (this log, README, perf_summary, per-op tables) - docs only |
| `5f597d364f` | E9.1-r1 part A: stage-1 review doc fixes - docs only |
| `7224692da5`, `b0ae1e5ead` | E9.1-r1 part B (WIP): 12 x 10 grid switch (`tt/config.py`, `linear.py`, `attention.py`, `masks.py`, `conftest.py`, `prefill.py`); device opens of the perf / demo scripts, `dtype_probe --inputs e9` |
| `d2e85760dd`, `0c6b153b18` | model folder moved to `models/demos/embedding_gemma2`; serving wrapper `demo/generator_vllm.py` (v1, v2) - not this unit |
| `bc9cb3106c` | E9.1-r1 part B final: perf.py records the grid, demo `__main__` checks it; "E9.1-r1: compute grid 12 x 10" section |
| `325fbd415d` | E9.1-r1 part C: text precision default `p3_qkv_gu` -> `p2_sdpa_fast` (`selected_precision_config.json`) |
| `87aa096ffe` | E9.1-r1 part C docs: contract change, datatype_sweep README / work_log, REPORT addendum, README |
| `4eed20c34c` | E9.1-r2a: text precision default `p2_sdpa_fast` -> `sdpa_nofp32acc` (`selected_precision_config.json` text section, config_id); contract / datatype_sweep / REPORT / work_log amendments |
| `a903d517c6` | E9.1-r2a: root README cost and fallback fixes; `prefill_op_table.py` roofline uses the profiled grid (`--grid-cores`, CSV, `device_info.json`) |
| `637e2b5292` | E9.1-r2b commit 1, E9.1-r2a review minors: README cost label, this table and the config hash, datatype_sweep README layout and gate keys, `--grid-cores 0` error + CPU test, 13 x 10 tables regenerated with `grid_cores` |
| (E9.1-r2b re-baseline commit) | 12 x 10 re-baseline: README (12 x 10 headline first, 13 x 10 kept as superseded), `perf_summary.json` (12 x 10 top level, 13 x 10 nested), `profiles_12x10/`, this log; `prefill_summary.py --superseded` and extra keys - docs and tooling only |

The measurements ran on the working tree of `7ef04f02da` before two non-semantic changes: black
reformatting at commit time, and `@pytest.mark.timeout(3600)` (added after the 65536 sustained case
hit the 300 s default; the 65536 rerun ran with it). Commit check on `7ef04f02da` (run
`20261028-081422`, quick buckets, gate + burst): gates 0.99170 / 0.99233 / 0.99446 (identical to the
sweep), burst 4.814 / 10.759 / 92.927 ms (sweep 4.813 / 10.758 / 92.970).

## Environment

```bash
source /local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e1/env.sh   # TT_METAL_HOME, PYTHONPATH (transformers overlay), TT_VISIBLE_DEVICES=0, OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 HF_MODEL=/local/ttuser/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b
export EMBEDDING_GEMMA2_PREFILL_REFS=/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/refs
export EMBEDDING_GEMMA2_PREFILL_OUT=$R/run      # run_gate, run_controls, run_baseline_precision for the other runs
```

## Commands (chronological)

| time | command | result |
|---|---|---|
| 07:11-07:13 | `scripts/refs.sh 8192 / 16384 50 / 32768 50` = `/usr/bin/time -v python -m ...tests.perf.prefill_refs --bucket B --out refs --rss-cap-gb 50` | 3.2 / 5.4 / 13.7 GB, forward 6.4 / 20.3 / 69.4 s |
| 07:14-07:18 | `scripts/refs.sh 65536 52` (L 49152), tmux `..._e901_ref65536` | 44.5 GB, forward 248.3 s, exit 0 |
| 07:19-07:20 | `scripts/refs.sh B 50` for 128 ... 4096 | 3.2 GB each, exit 0 |
| 07:20 | `scripts/ref_full65536.sh` (L 65536, cap 52 GB; not a contract input) | `RSS_CAP_EXCEEDED rss 52.4 GB` at 25.5 s, exit 3 |
| 07:21 | smoke: `HF_CHECK=1 HOST=1 TT_METAL_TRACE_ALLOC_TRACKING=1 pytest prefill.py -s -q` (out: scratchpad) | 9 passed |
| 07:22-07:27 | gate run: `SHAPES=1 HF_CHECK=1 TT_METAL_TRACE_ALLOC_TRACKING=1 pytest prefill.py` -> `run_gate/` (`logs/gate_alloc_tracking.log`) | 19 passed, 1 failed (b1_s256 gate 0.98872) |
| 07:27:08-07:27:38 | `scripts/gate256_probe.py` selected / baseline precision (`logs/gate256_probe_{selected,baseline}.log`, `gate256_probe.jsonl`) | see README "Gate failure at b1_s256" |
| 07:27:49-07:46:31 | sweep: `EMBEDDING_GEMMA2_PREFILL_ALL=1 pytest prefill.py -s -q` -> `run/` (`logs/sweep_all.log`, tmux `..._e901_sweep`) | 38 passed, 2 failed: b1_s256 gate; b1_s65536-sustained pytest-timeout 300 s |
| 07:46:57-07:58:49 | `EMBEDDING_GEMMA2_PREFILL_BUCKETS=65536 SUSTAINED=1 TELEMETRY=1 pytest prefill.py` -> `run/` (`logs/sweep_s65536_sustained.log`) | 2 passed |
| 07:59-08:04 | `scripts/tracy_all.sh` = per bucket `python -m tracy -r -p -v -m pytest "perf.py::test_tracy_forward[b1_sB]" -q -s` with `EMBEDDING_GEMMA2_MAX_SEQ_LEN` 32768 (65536 for 65536), then `tt-perf-report` and `python -m ...prefill_op_table <csv> --bucket B` | 10 x exit 0 (`logs/tracy_b1_s*.log`) |
| 08:05-08:10 | `scripts/controls.sh`: telemetry A/B at 2048 (`ctl_tel_off_2048`, `ctl_tel_on_2048`); baseline precision gate + burst <= 8192 (`baseline_precision_le8192`); end control (`ctl_quick_end`) | 2 / 2 / 14 / 3 passed |
| 08:10-08:11 | regression: `perf.py` default (`EMBEDDING_GEMMA2_BENCH_DIR=$R/run_perf_default`); `test_trace.py test_full_model_trace.py`; `pytest tests --collect-only` | 4 passed; 17 passed; 362 collected |
| 08:14 | commit check on `7ef04f02da`: `HF_CHECK=1 pytest prefill.py` (`logs/commit_check_quick.log`) | 6 passed |

Analysis (CPU only): `python3 $R/scripts/analyze.py $R/analysis/tables $R/run/prefill.jsonl $R/analysis/reconcile.json`;
`python -m ...tests.perf.prefill_summary --timing $R/run/prefill.jsonl --gate $R/run_gate/prefill.jsonl --tables $R/analysis/tables --stage 01-baseline --extra $R/analysis/extra.json --out doc/prefill/01-baseline/perf_summary.json`.
`run/prefill.jsonl` holds two run ids; for 65536 burst and sustained the later run (`20261028-074659`) is used (last record per shape and phase).

## Decisions and findings

- **Logical length 3/4 of the bucket** (masked path with real padding): exercises key padding in every
  bucket and keeps the 65536 HF reference feasible (49152 tokens, 44.5 GB). Device work does not
  depend on it (static shapes and masks).
- **Device sessions**: buckets <= 8192 in one session (max_seq_len 8192, as `perf.py`); 16384,
  32768, 65536 one each (max_seq_len = bucket). One trace per bucket, 128 MiB trace region for all.
- **Telemetry**: sysfs (driver) reads work while the card is busy; sampler in a second process; A/B
  at 2048 shows no effect (+0.015 % sustained). tt-smi not used during runs.
- **b1_s256 gate failure**: measured, controlled (baseline precision passes the same input),
  reported as an open decision in `target_contract.json` (`accuracy.open_decisions`); no gate or
  threshold changed, no precision change in this stage (stage 1 changes no model code).
- **65536 sustained timeout**: the repo pytest-timeout (300 s, pytest.ini) killed the case during a
  sync, about 5 min into a 10 min run. Not a device hang: teardown released the traces and closed the
  device (log ends with UMD close), the next run opened the card normally. Fix: per-test timeout
  mark. No reset needed; no `tt-smi -r`.
- **perf.py run_info side effect**: the Tracy runs execute `perf.py`, whose autouse fixture rewrites
  `doc/benchmark/run/run_info.json` unless `EMBEDDING_GEMMA2_BENCH_DIR` is set. It was overwritten
  once and restored with Edit to the committed content (diff empty); `scripts/tracy_case.sh` now
  sets `EMBEDDING_GEMMA2_BENCH_DIR`.
- **Op table model fixes** before the tables were committed: gather / slice count only the bytes
  they move; rotary counts only the cos / sin rows it reads (tables are `max_seq_len` long);
  `MinimalMatmul` is a matmul family member (analysis script).

## Anomaly ledger

| observation | evidence | control | resolution |
|---|---|---|---|
| b1_s256 last-hidden PCC 0.98872 < 0.99 | `run/prefill.jsonl`, `run_gate/prefill.jsonl` (identical) | baseline precision 0.99692 on the same input; offsets 997 / 5003 >= 0.9955 | open, orchestrator decision (README options) |
| 65536 "burst" is not a full-clock figure: AICLK drops inside the first window | sustained telemetry started after the same 3 s idle (`run/telemetry_20261028-072751_b1_s65536.csv`, `run/telemetry_20261028-074659_b1_s65536.csv`): 1350 MHz / 68-69 W at start, 1343 MHz at 1 s, 1287 MHz at 2 s; burst windows of run `20261028-074659` rise 2041.6 -> 2055.2 ms. The burst `telemetry_start` (1287 / 1306 MHz) is sampled after warm-up, before the first 3 s idle, so it is not the window-start clock | 2 runs, 2057.2 / 2049.8 ms | documented (corrected in E9.1-r1; the first version said the clock was already at 1287-1306 MHz at burst start, which the window-start telemetry contradicts) |
| traced residual 36 ms at 65536 vs <= 3.2 ms elsewhere | `analysis/reconcile.json` | kernel sum from a cold eager profile at full clock | attributed to the lower clock [I]; not a dispatch cost |
| host work negative at >= 16384 | host records (N = 14 / 5) | replay + sync measured in the same phase | run-to-run noise at small N under clock drift; reported as noise |
| 65536 sustained timed out | `logs/sweep_all.log` | rerun with the timeout mark passed | fixed (harness limit) |

## Self-check against prefill-stage-review (not independent; the orchestrator runs the review)

- Contract: hardware identity, power limit, firmware, PCIe width, all 10 shapes and the path, metric
  and reference settings, gates: present. No override applied; the 256 failure is an open decision.
- PCC gate on traced replay output: 9 of 10 pass; **b1_s256 fails** (requirement not met).
- Burst and sustained for every bucket, with telemetry in sustained: present.
- One warmed forward profiled per bucket; per-op table with roofline columns per bucket: present.
- Reconciliation, host round trip / H2D / D2H per bucket, top three with owners: present.
- Trace evidence: alloc tracking run on all buckets, replay == eager bitwise, 128 MiB region.
- Known gaps: one sustained run per bucket (no repeat), roofline is a model, traced gaps not profiled.

## E9.1-r1: compute grid 12 x 10 (2026-10-28)

Person's instruction (2026-10-28): optimize for a 10-row x 12-column compute grid (Galaxy target),
not the P150a's 13 x 10. Run folder: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/01r-geometry-precision/`
(below: `$G`); every device command through `$G/scripts/run.sh` (card-free check, env dump,
`timeout -s KILL`, `__tt_exit`). Code: commits `7224692da5`, `b0ae1e5ead` and the E9.1-r1 part B
final commit (Commits table below).

### How 12 x 10 is reached on a P150a [M]

| attempt | result | log |
|---|---|---|
| default device open | `compute_with_storage_grid_size()` = (13, 10) | `$G/logs/grid_probe_default.log` |
| `DispatchCoreConfig(WORKER, axis=COL)` | (13, 10): column dispatch is already the Blackhole default | `grid_probe_col.log` |
| `DispatchCoreConfig(WORKER, axis=ROW)` | open fails: `TT_THROW: ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX is enabled` | `grid_probe_row.log` (exit 1) |
| `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=11,9` in the environment | (12, 10), an op runs | `grid_probe_envvar_col.log` |
| same, set from Python after `import ttnn`, before the first open | (12, 10) | `grid_probe_env_in_python.log`, `grid_probe_dyn_after_import.log` |
| same, set after a first open in the same process | stays (13, 10): the runtime reads it once per process | `grid_probe_dyn_toggle.log` |

So the model uses the runtime override. `tt/config.py`: `EMBEDDING_GEMMA2_GRID=<cols>x<rows>`
selects the grid; `request_device_grid` sets the override before the device opens and
`check_device_grid` (in `conftest.open_eg2_device`, used by the test fixture, `perf.py`,
`prefill.py`, `dtype_sweep.py`, `dtype_probe.py`, `bench_sustained.py`; `demo.py __main__` calls the
same two helpers) raises if the open device reports another grid. `compute_grid(device)` is the only
grid read in the text code (`linear.py`, `attention.py`, `masks.py`, `config.act_memory_config`);
the override also caps the ops that pick their own grid (norms, eltwise, `minimal_matmul`). Defaults:
`tests/perf/prefill.py` 12 x 10; every other entry point the device's own grid (13 x 10 on a P150a).

### Evidence

- **13 x 10 path unchanged [M]**: traced outputs for buckets 128-8192 are bitwise equal to the base
  commit `f6b3ded953` (`$G/logs/cmp_base13_new13.log`, `ALL_BITWISE_EQUAL True`).
- **12 x 10 vs 13 x 10 [M]**: bitwise equal for 128-4096; 8192 differs (`$G/logs/cmp_new13_new12.log`:
  last-hidden PCC between the grids 0.99750, max abs diff 492 (values: abs max 2784, std 46); vs HF fp32
  0.99415 at 12 x 10, 0.99446 at 13 x 10, `$G/logs/dump_vs_ref_base.log`). The base commit run
  with only the override (no model change) gives the same 12 x 10 output (`cmp_base12_new12.log`).
- **Which op makes 8192 differ [M]** (`$G/scripts/dump_ops.py`, `compare_ops.py`; one eager 1 x 8192
  forward per grid with a ttnn post-operation hook that fingerprints every op output;
  `$G/logs/b4_compare_ops.log`; eager-with-hook output equals the traced dump at both grids,
  `b4_eager_vs_traced.log`): the first 25 op outputs are bitwise equal; op #25, layer 0, the MLP gate
  projection with fused `gelu_tanh` ([1, 1, 8192, 512] x [512, 2048]), runs as `ttnn.linear` (2D
  multicast config) at 13 x 10 and as `ttnn.experimental.minimal_matmul` at 12 x 10, in every layer
  (24 swaps, the only op-name differences). Everything after it differs.
  Cause [I] (`tt/linear.py mcast2d_config`, arithmetic in `$G/logs/b4_cb_budget.log`): at M = 8192
  per_core_M = 26; per_core_N = ceil(64 / 13) = 5 fits the 900 KiB CB budget only at in0_block_w 1
  (916,096 B <= 921,600 B); at 12 columns per_core_N = 6 needs 1,078,016 B at in0_block_w 1, so no
  2D config fits and `linear` falls back to `minimal_matmul` with the fused activation. The two
  kernels apply gelu at different points (packer vs minimal_matmul's fused activation), so the bfp8
  outputs round differently. Both pass the gate. Perf note for stage 3: the code comment records
  minimal_matmul's fused activation as slow (218 us vs 40 us without, PLE gate at M = 4096); the
  12 x 10 gate projection at >= 8192 should be measured there.
- **No op uses more than 120 cores at 12 x 10 [M]**: Tracy, one warmed eager 1 x 2048 forward
  (`$G/scripts/tracy_case.sh 2048 12x10`, CSV `$G/tracy/b1_s2048_12x10/ops_perf_results.csv`),
  `$G/scripts/core_counts.py <csv> 120` (`$G/logs/b3_core_counts_12x10.log`): 646 forward ops, max
  CORE COUNT 120, AVAILABLE WORKER CORE COUNT 120 on every op, 0 ops above 120. Control on the
  stage-1 13 x 10 profile (`b3_core_counts_13x10_control.log`): max 130, 449 / 646 ops above 120.
  At 12 x 10 the 2D matmuls with N = 2048 / 512 run on 110 cores (11 columns: ceil(Nt / 12)); a
  stage-3 tuning item.
- **Tests at both grids, precision `p3_qkv_gu` [M]** (`$G/logs/gates_summary.log`): text PCC +
  trace + precision config + full model + full model trace 147 passed, `test_text_model[b1_s32768]`
  passed, demo 2 passed, at 12 x 10 and at 13 x 10.
- **Entry points at both grids [M]** (`$G/logs/chain_b2_summary.log`): `perf.py` default 4 passed
  at 13 x 10 (text 1 x 512 forward 6.879 ms, `run_b1_perf_13x10/`) and 12 x 10 (7.078 ms,
  `run_b1_perf_12x10/`; every record carries `compute_grid`, `device_info.json` [12, 10]); demo
  pytest 2 passed and demo `__main__` exit 0 (logs the grid; retrieval argmax TT == HF) at both
  grids; `dtype_probe.py --inputs e9` runs at 12 x 10 (`b1_probe_e9_12x10.jsonl`, `compute_grid`
  [12, 10]; `p3_qkv_gu` min last hidden 0.98836 at L 200, the b1_s256 family).
- **Contract gate at 12 x 10, `p3_qkv_gu`, traced replay vs cached HF fp32 [M]** (last-hidden PCC /
  embedding cosine): 128 0.99170 / 0.99747, 256 **0.98872** / 0.99763 (fails), 512 0.99366 / 0.99882,
  1024 0.99233 / 0.99911, 2048 0.99321 / 0.99927, 4096 0.99387 / 0.99947, 8192 0.99415 / 0.99955
  (`$G/run_b_gate_12x10/prefill.jsonl`); 16384 0.99496 / 0.99953, 32768 0.99453 / 0.99945, 65536
  0.99420 / 0.99937 (`$G/run_b2_gate_12x10/prefill.jsonl`, `TT_METAL_TRACE_ALLOC_TRACKING=1`).
  Replay equals eager bitwise on every bucket. The 256 failure is the precision issue that the
  E9.1-r re-selection (part C) addresses.

### Contract change (E9.1-r, 2026-10-28; approved by the orchestrator, person's geometry instruction)

`doc/prefill/target_contract.json`:
- `hardware.compute_grid` [13, 10] -> [12, 10], with `compute_grid_note` (how 12 x 10 is reached
  and checked, above).
- `checkout.precision_config`: text `p3_qkv_gu` -> `p2_sdpa_fast` (`doc/datatype_sweep/README.md`,
  "E9.1-r re-selection"; selected at 12 x 10 under the E6 policy with the contract inputs and the
  b1_s256 family in the probe).
- `accuracy.open_decisions.b1_s256` resolved without a `pcc_gate_override` (moved to
  `resolved_decisions`): with `p2_sdpa_fast` the input gives 0.99438 / 0.99873.
- New `accuracy.open_items.baseline_precision_12x10`: the E5a fallback precision crashes at
  12 x 10 in multi-bucket sessions <= 8192 (static CBs clash with L1 buffers at the 1 x 4096
  full-attention SDPA); open for stage 5.
- Stage-1 timing numbers (13 x 10, `p3_qkv_gu`) are not comparable with the new contract; the
  12 x 10 re-baseline is the next unit.

### Contract change (E9.1-r2a, 2026-10-28; orchestrator decision, E9.1-r review finding 4)

`doc/prefill/target_contract.json`: `checkout.precision_config` text `p2_sdpa_fast` ->
`sdpa_nofp32acc`, `checkout.contract_changes` extended, `accuracy.resolved_decisions.b1_s256`
gains the new default's value on the input (0.99432 / 0.99861). Reason, gates, soaks and cost:
`doc/datatype_sweep/README.md`, "Amendment E9.1-r2a". Run folder
`tt-ops/runs/embeddinggemma-2/e9/01r2a/`.

Config identity of the r2a runs (E9.1-r2b, CPU only) **[M]**: every r2a device log
(`01r2a/logs/*.log`, first `__start` 11:45:32, last 12:02:38) ran on `87aa096ffe` with
`selected_precision_config.json` modified in the working tree (log headers: `dirty: M ...selected_precision_config.json`)
and printed `default precision sdpa_nofp32acc from .../selected_precision_config.json`.
`git hash-object doc/datatype_sweep/selected_precision_config.json` = `96c2c0b20b3ce99b623ac8a5b00af7491bbbbe91`,
equal to the committed blob (`git rev-parse 4eed20c34c:models/demos/embedding_gemma2/doc/datatype_sweep/selected_precision_config.json`
= `96c2c0b20b...`, unchanged at `a903d517c6`). The file's mtime is 11:44:41, before the first r2a log,
so the logs read this content **[I: from mtime; the logs do not print a hash]**.

E9.1-r2b commit 1 (r2a review minors, CPU only) **[M]**: `prefill_op_table.py --grid-cores 0` (or
below) is now an error (argparse exit 2; `table(grid_cores=0)` raises) instead of falling back to
the CSV; `tests/perf/test_prefill_op_table.py` (CPU only: grid from the CSV column, from
`device_info.json`, column over `device_info.json`, no record, mixed values, explicit override,
0 / -120, CLI 0): 9 passed (`e9/01r2b/logs/c1_test_prefill_op_table.log`). The committed 13 x 10
tables (`profiles/b1_s*/`) were regenerated from their raw CSVs (`e9/01-baseline/tracy/`): JSON equal
except the new `grid_cores` 130 key, markdown equal except the header's "130 cores", all 10 buckets
(`e9/01r2b/logs/c1_regen_13x10_tables.log`).

## E9.1-r2b: stage-1 re-baseline at 12 x 10 with `sdpa_nofp32acc` (2026-10-28)

Run folder `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/01r2b/` (below: `$B`). Every device
command through `$B/scripts/run.sh` (card-free check polling every 60 s, env dump, `timeout -s KILL`,
`TT_METAL_OPERATION_TIMEOUT_SECONDS=60`, `__start` / `__end wall_s` / `__tt_exit`), one at a time, in
tmux `tt-bringup_embeddinggemma2_e901r2b_timing` (`chain.sh`) then `..._e901r2b_ctl2` (`chain2.sh`).
The card was never held by another process (`card_wait.log` absent). Environment as above plus
`EMBEDDING_GEMMA2_GRID`. All runs on HEAD `637e2b5292`; dirty files at run start (log headers) were
only `doc/prefill/01-baseline/profiles_12x10/` (new tables), `tests/perf/prefill_summary.py` (not
imported by `prefill.py`) and `perf_summary.json`; no model code.

| time | label | command (`$B/scripts/run.sh <label> <timeout> ...`) | result |
|---|---|---|---|
| 12:23 (CPU) | verify captures | `python3 $B/scripts/verify_captures.py roofline/capture/out .../selected_precision_config.json a903d517c6` | 10 / 10 OK (`logs/verify_captures.log`) |
| 12:23 (CPU) | op tables | per bucket `python -m ...tests.perf.prefill_op_table roofline/capture/out/text_b1_sB/ops_perf_results.csv --bucket B --grid-cores 120 --out-md profiles_12x10/b1_sB/op_table.md --out-json .../op_table.json` | 10 x exit 0 (`logs/op_tables_12x10.log`, with CSV sha256) |
| 12:24:06 | `ctl_start` | `EMBEDDING_GEMMA2_GRID=12x10 EMBEDDING_GEMMA2_PREFILL_OUT=$B/run_ctl_start pytest prefill.py -s -q -p no:cacheprovider` | 3 passed, exit 0 |
| 12:25:03-12:47:25 | `sweep_all` | `EMBEDDING_GEMMA2_GRID=12x10 EMBEDDING_GEMMA2_PREFILL_ALL=1 EMBEDDING_GEMMA2_PREFILL_OUT=$B/run pytest prefill.py -s -q -p no:cacheprovider` (timeout 5400) | 40 passed in 1339 s, exit 0 (run `20261028-122505`) |
| 12:47:26 | `ctl13` | `EMBEDDING_GEMMA2_GRID=13x10 EMBEDDING_GEMMA2_PREFILL_BUCKETS=512,2048,8192 EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 EMBEDDING_GEMMA2_PREFILL_SUSTAINED=1 EMBEDDING_GEMMA2_PREFILL_TELEMETRY=1 EMBEDDING_GEMMA2_PREFILL_OUT=$B/run_ctl13 pytest prefill.py ...` | 9 passed, exit 0 |
| 12:50:04 | `ctl12` | same with `EMBEDDING_GEMMA2_GRID=12x10`, `OUT=$B/run_ctl12` | 9 passed, exit 0 |
| 12:52:43 | `ctl_end` | as `ctl_start`, `OUT=$B/run_ctl_end` | 3 passed, exit 0 |
| 12:54:23 | `tracy_b1_s2048_13x10` | `EMBEDDING_GEMMA2_GRID=13x10 EMBEDDING_GEMMA2_BENCH_DIR=$B/tracy/b1_s2048_13x10 EMBEDDING_GEMMA2_MAX_SEQ_LEN=32768 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 python -m tracy -r -p -v -m pytest "perf.py::test_tracy_forward[b1_s2048]" -q -s`; CSV copied to `$B/tracy/b1_s2048_13x10/`, then `prefill_op_table ... --bucket 2048` (130 cores from the CSV) | 1 passed, exit 0 |
| 12:54:59 | `ctl13_32768` | `EMBEDDING_GEMMA2_GRID=13x10 EMBEDDING_GEMMA2_PREFILL_BUCKETS=32768 EMBEDDING_GEMMA2_PREFILL_SUSTAINED=1 EMBEDDING_GEMMA2_PREFILL_TELEMETRY=1 OUT=$B/run_ctl13_32768 pytest prefill.py ...` | 2 passed, exit 0 |
| 12:58:42 | `ctl12_32768` | same with `EMBEDDING_GEMMA2_GRID=12x10`, `OUT=$B/run_ctl12_32768` | 2 passed, exit 0 |

Analysis (CPU): `python3 $B/scripts/analyze.py doc/prefill/01-baseline/profiles_12x10 $B/run/prefill.jsonl
roofline/out/roofline_sdpa_nofp32acc.json $B/analysis/r2b.json` (families, top-3, reconciliation,
% of roofline; `logs/analyze.log`); `python3 $B/scripts/render.py $B` (README tables,
`analysis/render.md`); `python -m ...tests.perf.prefill_summary --timing $B/run/prefill.jsonl --gate
e9/01r2a/run_gate_12x10/prefill.jsonl --tables $B/analysis/tables --stage "01-baseline (E9.1-r2b
re-baseline, 12x10)" --commit "637e2b5292 (...)" --extra $B/analysis/extra.json --superseded
superseded_13x10_p3_qkv_gu=$B/analysis/perf_summary_13x10_committed.json --out
doc/prefill/01-baseline/perf_summary.json` (`analysis/tables` = copies of the 12 x 10 op-table JSONs;
`perf_summary_13x10_committed.json` = the committed 13 x 10 summary at `637e2b5292`). Hardware identity
read from sysfs after the runs: fw bundle 19.8.0.0, DM app 0.24.0.0, KMD 2.11.0, power1_max 150 W,
curr1_max 200 A, PCIe x16 32 GT/s, thermal trip count 0 (unchanged from the contract).

### Decisions and findings

- **Profiles reused, not recaptured** (brief): the roofline captures pass every check (commit, grid,
  precision, signposts). One extra profile: 1 x 2048 at 13 x 10 with the current default, to split
  the 12 x 10 cost at 2048 by op family (the existing 13 x 10 profiles are `p3_qkv_gu`).
- **Gate inside the timing run**: `PREFILL_ALL` includes the HF check; 10 / 10 pass with values equal
  to the r2a alloc-tracking gate run, so `perf_summary.json` cites that run as the alloc-tracking gate.
- **Extra control at 32768** (not in the brief's list): the sustained clock drop at >= 16384 is larger
  than in the 13 x 10 baseline; 13 x 10 vs 12 x 10 at 32768 on the same code shows no grid effect
  (599.3 vs 599.7 ms, both 1237 MHz), so the drop comes with the code / precision change.
- **SDPA at 2048**: the expected 13.8 ms is the `p3_qkv_gu` 12 x 10 profile; at `sdpa_nofp32acc` it
  is 8.66 ms (13 x 10: 6.85 ms). The 2-wave cause stands (q_chunk 64, 128 units); its cost is +1.81 ms.
- `target_contract.json` unchanged. Proposed to the orchestrator (not applied): `checkout.commit`
  still names the 13 x 10 stage-1 base `4624946b1b`, and `compute_grid_note` says stage 1 was measured
  at 13 x 10 and cites only the 1 x 2048 core check; after this unit both could name the 12 x 10
  re-baseline (`637e2b5292` + the r2b docs commit; 0 ops above 120 cores at all 10 buckets).

### Anomaly ledger (E9.1-r2b)

| observation | evidence | control | resolution |
|---|---|---|---|
| sustained / burst 1.028 / 1.054 / 1.035 at 16384 / 32768 / 65536 (AICLK 1293 / 1237 / 1193) vs 1.015-1.019 at 13 x 10 `p3_qkv_gu` | `run/prefill.jsonl` telemetry last third | 13 x 10 vs 12 x 10 at 32768, same code: same clock and time | controlled: not the grid; follows the precision / code change **[I]**; recorded as a named limitation |
| 8192 at 13 x 10 throttles in sustained (1318 MHz), 12 x 10 does not (1350) | `run_ctl13`, `run_ctl12` | burst numbers (no throttle) | documented: the sustained 12 x 10 cost at 8192 (+7.2 %) is below the burst cost (+8.5 %) |
| traced residual 6.1 / 63.4 ms at 32768 / 65536 | `analysis/r2b.json` | kernel sum from a cold 1350 MHz profile | attributed to the clock drop inside burst windows **[I]** (as at 13 x 10) |
| host work negative at >= 16384 | host records N = 14 / 6 / 5 | replay + sync timed in the same phase | noise under clock drift |
| 8192 gate 0.99628 at 12 x 10 vs 0.99674 at 13 x 10 | `run_ctl12`, `run_ctl13` | known: gate projection on `minimal_matmul` at 12 x 10 (E9.1-r1) | expected; both pass |

### Self-check against prefill-stage-review (not independent)

- Every bucket at 12 x 10 has burst, sustained with sysfs telemetry, the gate on the traced replay
  output, an op table (120 cores), reconciliation, host / H2D / D2H, and top three with owners.
- 13 x 10 controls (512, 2048, 8192, plus 32768) and start / end drift controls are in the same session.
- Both regimes measured; headline is sustained (contract). Numbers come from one harness, one commit
  and one power limit. The 13 x 10 numbers in the README are labelled superseded and are not used as a
  "before" for this geometry.
- Gaps: one sustained run per bucket (repeats only at 512 / 2048 / 8192 / 32768); the roofline is a
  model; traced gaps are not profiled; the larger clock drop at >= 16384 is not attributed per op.
