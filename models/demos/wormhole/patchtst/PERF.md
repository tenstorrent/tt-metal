# PatchTST Performance Notes

The table below records the current public demo targets and the latest measured values kept in this repo.

| Workload | Required Metric | Achieved Metric | Verify Command |
|---|---|---|---|
| Primary forecast, trace batch 1 | `>= 150 seq/s`, `<= 40 ms` | `172.67 seq/s`, `5.50 ms` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k primary_forecast_trace_batch1_targets --timeout=1200` |
| Primary forecast, reproducibility | `3 fresh-process runs all >= 150 seq/s` | `191.66, 188.23, 172.67, ...` on the last saved passing sample set | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k primary_forecast_trace_batch1_reproducibility --timeout=1200` |
| Batch forecast throughput | `>= 800 seq/s`, `<= 15 ms` | `1982.52 seq/s`, `8.07 ms` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k batch16_trace_targets --timeout=1200` |
| Sharded forecast path | sharded forecast path meets perf/quality targets with shard fallback disabled | `latency_ms=19.6901`, `throughput_sps=203.1473`, `parity_corr=0.999658`, `quality_corr=0.754497` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k sharded_attention_evidence_and_quality --timeout=1200` |
| Cached online forecast | cached path faster than full rerun and parity preserved | `3.18 ms` cached vs `5.13 ms` full rerun, `speedup=0.3793` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k cached_streaming_speedup_and_parity --timeout=1200` |
| Shared-encoder forecast + classification | one encoder pass, latency gain over sequential | `latency_gain=0.4812` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf/test_patchtst_perf.py -s -k shared_encoder_multitask_latency_gain --timeout=1200` |
| Long-context forecast | real run at `4096 x 96` | `14.15 ms`, `282.73 seq/s` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/integration/test_public_tasks.py -s -k native_long_context --timeout=1200` |
| High-channel forecast | real run at `862` channels | `517.93 ms`, `7.72 seq/s` | `PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/integration/test_public_tasks.py -s -k native_high_channel --timeout=1200` |

## Notes
- All commands assume the in-repo runtime built with `./create_venv.sh` and `./build_metal.sh`.
- The public demo uses strict no-fallback execution by default.
- The cached streaming path keeps state and a persistent device buffer, but it does not claim decoder-style autoregressive KV-cache semantics.
- The long-context checkpoint is local and shape-matched; it is evaluated on the `etth1` training split because the public test split is too short for `4096 + 96`.
