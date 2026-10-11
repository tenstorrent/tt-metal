# Stage 11 run notes

## Refresh on the stage-3 code (2026-10-11), what REPORT.md now shows

- Code measured: `da43bdb9eff` (stage 3A + 3B, reviewed clean pass), clean tree for the model
  directory (`git status` empty at launch, recorded in `stage11_refresh/commit.txt`). No benchmark
  code, template, subset or precision change. Policy logged at load: `act_bf16__w_bfp8_all__hifi2`.
- Same protocol as the first run: subsets copied from `stage11/subsets` (manifest and the three
  jsonl files byte-identical), same weight cache, same commands. Order: bucket harness
  (`PPLX_DECIDER_STAGE6_DIR=$R/perf_buckets pytest .../tests/perf/test_model_perf.py::test_model_perf`,
  1 passed, 696 s), then `run_benchmark --subsets $R/subsets --out $R/run` (exit 0), then
  `report --stage11 $R --doc doc/benchmark` (regenerates `results.json`, `predictions.csv`).
  `R=/local/ttuser/gtobar/artifacts/pplx_decider/stage11_refresh`. One tmux job, run sequentially
  on the one device.
- Old vs new comparison (predictions, probabilities, latencies): `$R/compare_vs_stage11.json`.
  5 of 600 predictions changed, all near-ties. Accuracy is 174 / 168 / 193 (before: 174 / 165 / 193).
- Logs: `$R/logs/{perf_buckets.log,run_benchmark.log,report.err}` (report.err empty); tables: `$R/report_tables.md`.
- REPORT.md was edited in place; the worker harness blocks whole-file writes of report files.

## First run (2026-10-09, before stage 3)

- Date: 2026-10-09. Host: one Blackhole p150a (31.83 GiB DRAM, 130 Tensix), firmware 19.8.0, KMD 2.6.1.
- Code measured: `b98508eac2c` (stage 12B, last reviewed model change). The stage-11 commit adds only
  `benchmark/`, `doc/benchmark/` and a README section; no model code changed.
- Precision: C0 from `doc/datatype_sweep/selected_precision_config.json`, logged at load as
  `act_bf16__w_bfp8_all__hifi2`; vision tower BF16.
- Weight cache: `/local/ttuser/gtobar/artifacts/pplx_decider/weight_cache/b01a5cbaca53/weights` (BFP8, 27 GB).
- Datasets (pinned revisions in `identity.json`): allenai/winogrande winogrande_xl/validation,
  takala/financial_phrasebank Sentences_50Agree, facebook/belebele eng_Latn/test; seed 20260920, 200 each.
- Order of runs: accuracy pass (file order), throughput pass (seeded shuffle), bucket latency harness,
  image latency, then `benchmark/report.py`.
- Logs: `/local/ttuser/gtobar/artifacts/pplx_decider/stage11/logs/{run_benchmark,perf_buckets}.log`.
- REPORT.md and this file were written from the orchestrating session from the generated
  `report_tables.md` and the run artifacts; the worker's harness blocked report-file creation.
