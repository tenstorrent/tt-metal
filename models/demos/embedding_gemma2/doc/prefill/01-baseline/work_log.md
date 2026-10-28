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
