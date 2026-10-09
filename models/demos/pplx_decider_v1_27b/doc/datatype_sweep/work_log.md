# Stage 8 work log: datatype sweep

Host: one Blackhole p150a. Branch `gtobarTT/pplx-decider-bringup`, base `8ad9f5a13dd`.
Artifacts: `/local/ttuser/gtobar/artifacts/pplx_decider/stage8/<candidate>/` (e2e JSON, `model_perf.json`,
logs) and `stage8/sweep.log` (driver log). Labels: **measured** (command + value), **inferred**.

## Setup changes

- `tt/optimizations.py`: `PrecisionPolicy.from_json()` and `PrecisionPolicy.default()`. The default reads
  `doc/datatype_sweep/selected_precision_config.json`; `$PPLX_DECIDER_PRECISION_CONFIG` points it at
  another JSON. `from_json` fails on unknown fields and on a role without a dtype or a fidelity.
  `Optimizations.build` reads each role's fidelity with `[role]` (no silent `HiFi2` default any more).
- `tt/model.py`: `from_snapshot()` uses `PrecisionPolicy.default()` and logs the policy. The weight cache
  directory is now `weight_cache/<revision>/weights` for every policy. A cache file name already carries
  the weight's dtype, shape and layout (`LazyWeight._get_fingerprint`), so policies share the files of the
  groups whose dtype they share. The existing BFP8 directory `act_bf16__w_bfp8_all__hifi2` was renamed
  to `weights` (`mv`, no file changed). A candidate adds only the files of its new dtypes.
- `tests/test_utils.py`, `tests/perf/test_prefill_perf.py`: the stage-1 PCC / layer-perf helpers use
  `PrecisionPolicy.default()` (they used `bfp8_weights()`).
- `tests/e2e/test_precision_config.py` (new): builds the model through the default path on a reduced
  stack (layers 0 and 3, embedding, head), reads `tensor.dtype` of every loaded projection weight and spies
  on every `minimal_matmul` / `linear` call during one forward to record the weight dtype, the
  `compute_kernel_config.math_fidelity` and the output dtype per role. Asserts they equal the JSON.
- `doc/datatype_sweep/candidates/*.json`: one policy per candidate. `sweep_report.py` builds
  `sweep_results.{json,csv}` from the artifacts. It draws no charts: the Pareto plots were dropped by
  the person's amendment to the short sweep (see README "Not done").

Driver: `artifacts/pplx_decider/stage8/run_sweep.sh` (queue 1) and `run_sweep_q2.sh` (queue 2). For each
candidate it sets `PPLX_DECIDER_PRECISION_CONFIG=<candidate json>` and
`PPLX_DECIDER_STAGE6_DIR=stage8/<candidate>` and runs, unchanged,
`pytest tests/e2e/test_model.py -q -s` then `pytest tests/perf/test_model_perf.py -k test_model_perf -q -s`.
Queue 1's "e2e exit" field printed the log's last line, not the pytest result; use the pytest summary
lines in `<candidate>/e2e.log` (queue 2 prints them).
`-k test_model_perf` also matches the module name, so each perf run also ran `test_traced_forward` (3) and
`test_profile_model` ("5 passed"). Those add time and do not change `model_perf.json`.

## Log

| # | step | command / evidence | result |
|---|---|---|---|
| 1 | smoke: BFP4 + LoFi through the default path (reduced stack) | `PPLX_DECIDER_PRECISION_CONFIG=candidates/C2_bfp4_mlp_lofi.json pytest tests/e2e/test_precision_config.py -q -s`; `stage8/smoke/precision_C2.log` | 1 passed; MLP roles `BFLOAT4_B` + `LoFi` in device tensors and matmul calls; minimal_matmul with `fuse_swiglu` runs BFP4 |
| 2 | C0 e2e | `stage8/C0_bfp8_all_hifi2/e2e.log` | 6 passed, 56 s; 25/25; logit PCC min 0.99801 |
| 3 | C0 perf | `stage8/C0_bfp8_all_hifi2/perf.log`, `model_perf.json` | 5 passed, 761 s; request burst 148.6 / 366.1 / 665.6 / 1487.0 / 3521.7 ms (stage 6 within 1 %) |
| 4 | C1 e2e (writes the BFP4 MLP cache files) | `stage8/C1_bfp4_mlp_hifi2/e2e.log` | 1 failed, 5 passed, 150 s; 25/25 but `l01_faq_match_230opt` logit PCC 0.98895 < 0.99 |
| 5 | C1 perf | `stage8/C1_bfp4_mlp_hifi2/perf.log` | 5 passed, 761 s; burst 138.8 / 365.5 / 663.4 / 1456.7 / 3371.0 ms |
| 6 | C2 e2e | `stage8/C2_bfp4_mlp_lofi/e2e.log` | 1 failed, 5 passed; same failure; all 25 rows' logits, PCCs and layer traces identical to C1 |
| 7 | C2 perf | `stage8/C2_bfp4_mlp_lofi/perf.log` | 5 passed, 707 s; burst 138.7 / 298.5 / 552.4 / 1197.8 / 2791.5 ms |
| 8 | orchestrator amendment (person): keep C0, stop the sweep | queue 2 (C5 = C2 with `mlp_down` restored to BFP8, then C4, then C3 e2e-only) cancelled before it started; C2 had already run | C3, C4 not run |
| 9 | selection | `selected_precision_config.json` = C0; `python doc/datatype_sweep/sweep_report.py` | fastest passing = C0 (the only passing candidate) |
| 10 | proof on the default path (no env override) | `PPLX_DECIDER_STAGE6_DIR=stage8/default_path pytest tests/e2e/test_precision_config.py tests/e2e/test_model.py -q -s`; `stage8/default_path/pytest.log` | 7 passed, 78 s; all roles `BFLOAT8_B` + `HiFi2`; 25/25, logit PCC min 0.99801 |
| 11 | cache cleanup | `find weight_cache/b01a5cbaca53/weights -name '*dtype_BFLOAT4_B*' -delete` | 128 BFP4 files (9.0 GB) deleted; 819 BFP8 files (27 GB) remain; `df -h /` 36 GB free |

Decisions:

- C1 and C2 were stopped at the gate: one row of 25 below the 0.99 logit-PCC bar. The bar was not changed.
- C1/C2 bit-identical outputs (inferred): LoFi keeps every product bit when the weight is BFP4 and the
  activation is BF16, so LoFi is free accuracy-wise for BFP4 groups on this model.
- Stage-1 per-layer PCC suite not re-run: the selected policy is the stage-1 policy.
