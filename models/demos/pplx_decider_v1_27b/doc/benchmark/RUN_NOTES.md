# Stage 11 run notes

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
