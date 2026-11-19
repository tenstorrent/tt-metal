# Optimized decoder validation plan

This document describes the acceptance queue now completed for the selected optimized defaults; `hard_gates.json` records the current-source aggregate PASS. The queue is reproducible only with the root agent owning chip 0 and no concurrent device work. `tests/validate_optimized.sh` serializes each bounded job, checks chip ownership, and honors `doc/optimized_decoder/STOP_QUEUE` between jobs. Source the existing persistent lane environment and activate `python_env` before invoking it; preserve the lane's device-visibility policy. Do not use the script's filesystem artifacts as proof until each command exits successfully and its JSON/log is inspected.

## Explicit implementation wiring

`tests/stress_optimized_decoder.py` injects the exact `OptimizedDecoder` class into the shared `stress_fused_decoder.run` harness. It checks the class and constructor source resolve to `tt/optimized_decoder.py`, sets optimized provenance, verifies the emitted decoder name, and records that the current production defaults ran. The wrapper keeps the original CLI options and fresh-HF behavior; optional CPU oracle reuse remains available but the acceptance queue does not enable it. A fresh output path is required, preventing old success JSON from surviving a new setup failure.

`tests/test_optimized_decoder.py` uses actual checkpoint weights for layers 0, 3 and 34. It explicitly injects `OptimizedDecoder` into the boundary battery and asserts its MoE method differs from the fused implementation. This tests optimized runtime behavior; inherited shared attention helpers are allowed, functional fallback is not. Tests retain the original thresholds. The shared CLI battery, trace-reuse, long-context and clean benchmark paths select through `decoder_factory` with `QWEN_DECODER=optimized`.

## Required current-path evidence

| Check | Workload | Artifact / coverage |
|---|---|---|
| FUNCTIONAL-shaped correctness and trace replay | Layers 0/3 seed913, layer34 seed154; 128 prefill + 128 changing decode + 33 continuation; three repeats | `stress_{0,3,34}.json`; fresh HF, deterministic replay, program-cache-inclusive allocation tracking. Latency/roofline floor needs separate measured accounting. |
| Repeated stream stress | Same representatives, 128 + 1024 + 33, three repeats | `stress_extended_{0,3,34}.json`; no automatic 8192-step sweep. If changed sensitive math reproduces the old layer34 anomaly, run the specific discriminating longer case as remediation. |
| Nonaligned boundaries, continuation, batch32 | Layers 0/3, real weights, battery `--mode all --batch 32` | `battery_{0,3}.json`; lengths 1/7/31/32/33/255/256/257/513, alternate chunk32, resume starts32/33/256/512/528, negative resume control, traced batch and shared-pool/guard-page assertions. |
| Independent test entrypoint | Real-weight pytest layers0/3/34 | `real_weight_pytest.xml` plus per-test JSON in `pytest/`; validates optimized class rather than a factory default. |
| Watcher and live-trace reuse | Layers 0/3; requests31/32/33/255/257/7 and resumed50@33,49@32 | `trace_reuse_{0,3}.json`, logs and `watcher/layer_{0,3}`; Watcher separated from all profiler jobs. Inspect watcher logs for warnings/errors before claiming clean. |
| Context smoke | Layers 0/3, length32768 | `context_32768_{0,3}.json`; shared harness checks32767/32768 prefills and deterministic final-position traced decode. Advertised262144 remains unchanged. No maximum sweep if KV dtype/layout unchanged. |
| Clean headline latency and PCC | Layers0/3, S256/context512, 10 warmed prefills, 60 traced decodes | `final_{0,3}.json`; tracking/Watcher/profiler disabled. Compare matched serial fused baseline and best correct candidates. |
| Short and boundary timings | Layers0/3, lengths127/128/129/257/4095/4096/4097, context8192, 3 warmed prefills/20 traced decodes | `final_l*_s*.json`; covers public chunk-hint boundary and awkward tails. Profiling must independently expose actual internal chunks64/256. |

The inherited fused default retains stricter per-row/contribution assertions. The optimized wrapper explicitly gates whole-output PCC at .995, as required by the owner, and preserves per-row diagnostics. Layer34 short continuation has an inherited synthetic routing-cutoff outlier; unchanged-fused controls and the same-input HF routing probe are recorded in `AUTOFIX.md`, while natural-input stress passes all rows. A passing 128-step run does not prove the old8192-step recurrent bug absent if an optimization changes those sensitive math paths.

## Additional stage-owned evidence outside this queue

- Matched fused baseline and candidate selection, including repeated serial timing, canonical BFP8 policy and any measured sensitive-group exceptions.
- Prefill/decode tt-perf-report tables/CSV with advice; current-source complete captures, actual dtype/fidelity rows, and measured decisions for material candidates.
- Topology/precision/geometry decisions and stop-rule hand-offs, source/runtime host-fallback audit, context-contract update if persistent layout/dtype changes, and any known-limitations control artifacts.
- Source-bound aggregate manifest/check, formatting, self-review, then independent stage review. Do not call a review before pending measurements or gates finish.

`page_extent_regression.py`, `padded_trace_regression.py` and `audit_watcher.py` now support optimized destinations and were run in the serial acceptance queue. `page_extent_regression.json`, `padded_trace_regression.json` and `watcher_audit.json` retain their results. Clean FUNCTIONAL128/128 timing is separately recorded in `functional_{0,3}.json` and reconciled in `perf_summary.json`; correctness stress is not used as timing evidence.
