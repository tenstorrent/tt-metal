# Full model work log (stage 6, prefill-only classifier)

Model `perplexity-ai/pplx-decider-v1-27b`, one Blackhole p150a (13x10 grid, 8 DRAM banks, 31.83 GiB).
Branch `gtobarTT/pplx-decider-bringup`, base `4e2b3a0f480` (stages 1-2). Policy `act_bf16__w_bfp8_all__hifi2`.
Artifacts: `/local/ttuser/gtobar/artifacts/pplx_decider/stage6/` (logs in `logs/`, scratch probes in
`probes/`). Environment for every command: `source python_env/bin/activate`, `PYTHONPATH` and
`TT_METAL_HOME` set to the checkout, `HF_HOME=/local/ttuser/gtobar/hf`.

Labels: **measured** = a command and its output recorded here; **inferred** = from code or arithmetic.

## Steps

| # | what | command / probe | result |
|---|---|---|---|
| 0 | stage-2 review fixes + commit golden files | `git commit` (black/isort hooks reformatted the two golden files; AST-identical check: `ast.dump` equal for both) | `aeefd2cbf9a` |
| 1 | head op probe | `probes/ops_probe.py`: where/ge/multiply/softmax on fp32 [1,1,256] | ops work; `ttnn.softmax` max abs err 4e-4, sum != 1 |
| 2 | softmax precision | `probes/ops_probe2.py` | `ttnn.softmax` max rel 2.1e-3, sum 0.99889; max/sub/exp/sum/divide max rel 4e-7, sum 1.0000001 -> composed softmax kept |
| 3 | reduced model smoke (layers 0, 3) | `probes/smoke.py 0,3` | runs; DRAM per layer 0.3857 / 0.4003 GiB |
| 4 | first full load + 2 rows | `probes/smoke.py all` (log `logs/first_load.log`) | 27.296 GiB allocated, 4.536 free; s01 TT 0.99057 vs HF 0.99073; m06 0.99356 vs 0.99319 |
| 5 | e2e suite | `pytest tests/e2e/test_model.py -q -s` (log `logs/e2e_run1.log`) | 6 passed, 87 s, exit 0; 25/25 agree; logit PCC min 0.99801 |
| 6 | HF answers for the demo rows | `taskset -c 16-23 python -m ...reference.hf_demo_reference --threads 8` (log `logs/hf_demo_reference.log`) | 108.6 s; urgency 0.99300, routing technical_support 0.99181; prompts 100 / 115 tokens |
| 7 | demo | `python demo/demo.py --compare-hf .../demo_hf_reference.json` (log `logs/demo.log`) | exit 0; same decisions; max prob diff 0.0000 / 0.0002 |
| 8 | load times, fresh cache dir | `probes/load_times.py` | first load 162.6 s (0/66 hits); the second open in the same process ALSO 0/66 hits |
| 9 | cache miss root cause | file names end in `_device_1`; `LazyWeight._get_fingerprint` uses `device.id()`, a per-process counter | fix: `CachedWeight` drops the id; existing 819 files renamed (`mv`, name only) |
| 10 | load times after fix | `probes/load_times2.py` | new process 36.8 s (66/66), second open 4.0 s (66/66) |
| 11 | perf, first run | `pytest tests/perf/test_model_perf.py -k "test_model_perf or test_traced_forward"` (log `logs/perf_run1.log`) | device forward 23-37 % above the stage-2 projection at >= 1024; trace speedup <= 1.03x |
| 12 | per-layer times inside the model | `probes/layer_times.py` | with a sync per layer, 1024 sums to 392.6 ms; layer time grows L0 6.15 -> L63 6.6 ms; isolated L0 at the end 7.91 ms (stage 2: 6.03) |
| 13 | cooldown experiment | `probes/cooldown.py` (`cooldown_experiment.json`) | 1024: 363.9 idle / 367 -> 463 back to back / 364.0 idle; 2048: 656.9 / 743 -> 900 / 656.8 |
| 14 | perf with burst + sustained | same test after adding burst mode (log `logs/perf_run2.log`) | README table; burst within 2 % of the projection up to 2048 |
| 15 | DRAM at the 8192 bucket | `probes/dram_peak.py` (`dram_peak_S8192.json`) | peak at layer boundary 27.455 GiB allocated / 4.376 free |
| 16 | watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=stage6/watcher pytest tests/e2e/test_model.py -k "test_full_forward_stays_on_device and s01"` | 1 passed, exit 0; watcher.log 0 error/assert/sanitize/hang/overflow lines |
| 17 | profile, first try | tracy `test_profile_model` | post-process failed: "Device data missing: Op ... not present"; profiler DRAM buffers full (warm-ups + load not flushed) |
| 18 | profile, second try | `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`, `ReadDeviceProfiler` after load and every warm-up | 1713 ops in the signpost window, 0 missing device times, sum 648.0 ms; "buffers full" warnings on 2 cores at the final read |
| 19 | perf report | `tt-perf-report ... --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice --summary-file doc/full_model/perf/full_S2048_summary` | summary CSV in `perf/`; per-op CSV + PNG in artifacts |

## Decisions

- The head slices the last real token **before** the final norm (RMSNorm is row-wise, so this is exact).
  That normalises one row instead of the whole bucket.
- The readout is padded to 256 columns, so the softmax covers whole tiles. Column 255 is always masked.
- Mask, scale and softmax run in fp32. The readout output stays BF16 (as the app's bf16
  `readout(...).float()`).
- The cache is reused only when every weight of a group hits. Otherwise the group is rebuilt from the real
  tensors (no partial mixing).
- The layer stack is unchanged from stage 2 (layer-major, 2048-token chunks). A chunk-major order would
  avoid the per-layer concat at 4096/8192. That is a stage-7 option, not tried.
