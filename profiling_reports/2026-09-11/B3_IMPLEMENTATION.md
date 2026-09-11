# B3: MLA SP8/TP1 row and operation priorities

The B3 test was committed in `532d8833a90` on `akhan/mistral4-prefill-followups`; `96125e3b5f7` added its invocation to the existing LoudBox perf job. This report branch, `ssalice/mistral4-b3-findings`, starts from `e094693594d`. Measurement provenance remains the original run, not the later report-branch revision. No issue has been posted.

## Changes

- `tests/test_mla.py::test_mistral4_mla_chunked_prefill_loudbox` adds a dedicated Blackhole 8×1 Torus-Y worker. It runs exactly one functional scalar-metadata MLA forward with a 51,200-token KV prefix and 5,120 new tokens. Determinism repetitions are disabled. Prefix preparation lies outside the measured signposts.
- `tests/perf/test_mla_perf.py::test_mistral4_mla_chunked_perf_loudbox` invokes that exact worker and reports only the `MLA_START`/`MLA_END` region. It does not call the TP=4 galaxy approximation or borrow a TP=4 baseline.
- `utils/perf_utils.py::run_model_device_perf_test_with_merge` accepts an explicit `None` expected duration to record without a threshold. This path rejects empty, missing, negative or nonfinite source durations and nonpositive totals. Numeric-baseline callers retain their existing gate. The report uses the existing operation merge (max for ordinary ops, mean for collectives); it is accumulated operation work, not elapsed request latency.

All paths above are under `models/demos/deepseek_v3_d_p/`.

## Ranking

[B3_TP1_RANKING.md](B3_TP1_RANKING.md) recomputes the committed historical captures without extrapolation. Stage-0 accumulated-work shares: Dispatch+Combine 35.65%, SDPA 23.44%, all matmuls 17.53%, expert FFN 15.77%, residual collectives 0.16%. Matmul classification is aggregate, not exclusively MLA. These historical eager profiles are not a fresh standalone LoudBox MLA result.

## Validation performed

- `python_env/bin/python profiling_reports/2026-09-11/check_b3_host.py`: passed. This executes production function ASTs without native imports and uses the real device-row merge on synthetic CSV data. It checks record-only output, preservation of numeric threshold gating, invalid/missing-duration rejection, signpost filtering, the single-forward worker/wrapper arguments, and the Galaxy environment skip. It does not run pytest's native fixtures or validate kernels.
- Python syntax checks passed for the test, wrapper, helper and fixture changes; applicable pre-commit checks passed when the implementation was committed.
- Direct Black formatting comparison with Python 3.10 target and line length 120: passed. The initial Black CLI check stalled and was terminated; it is not counted as a successful check.
- `git diff --check`: passed.
- Historical ranking reproduction completed using `analyze_b3.py`; its JSON matches the saved evidence.

Python-only changes: no C++/CMake build required.

## Hardware validation status (updated after visibility probe)

A fresh MLA 8×1 measurement completed successfully on the filtered Galaxy column: one worker and one perf-wrapper case passed, outer exit 0. The merged operation sum is **8,585,052 ns**, including **7,195,124 ns SDPA**, **737,823 ns matmuls**, and **652,105 ns other operations**. This is a single functional/random-weight sample, not a correctness gate, wall-clock latency, or LoudBox calibration. Artifacts are in [b3-glx-column-PzSANj](b3-glx-column-PzSANj/). The initial conclusion that the Galaxy could not run this shape was too broad: `TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8` exposed eight chips, auto-discovery mapped an 8×1 ring, and Torus-Y mesh open/close passed on September 11. The cluster retains its BLACKHOLE_GALAXY identity. The fixture now permits (8,1) Torus-Y for that identity while retaining its exact-visible-device-count check. The wrapper labels this result `glx_column`, separately from `lb`. The local visibility probe used the installed older runtime solely to test hardware capability. Its full logs are not included in this report branch; the measured CSV, validation and runtime provenance are included. The matching b48 native build was recovered and verified before the benchmark; see B3_RUNTIME_RECOVERY.json.

On a Tracy-enabled LoudBox with the normal repository runtime environment and matching native build, run from the repository root:

```bash
python -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py::test_mistral4_mla_chunked_perf_loudbox -v -s
```

Retain the node ID, revision, host, workload, source CSV and reported operation breakdown. Confirm one measured MLA region and eight participating devices. The new row intentionally has no CI performance gate until a real baseline is measured; the existing LoudBox job now invokes this row, without a fabricated threshold. B3 implementation, historical ranking, and a direct Galaxy-column SP8/TP1 MLA sample are now available. An actual LoudBox CI calibration remains separate and unverified.

Galaxy-column invocation used for this sample:

```bash
TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8 MESH_DEVICE=TG python -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py::test_mistral4_mla_chunked_perf_loudbox -v -s
```

The visibility list was checked against live topology first; it is host-specific and must not be copied to another Galaxy without checking its physical ring. The test-fixture fabric allowlist now permits Blackhole Galaxy (8,1) Torus-Y while preserving the exact visible-device count check. The wrapper labels the result `glx_column`, not `lb`.

Independent CSV validation passed: exactly one MLA_START/MLA_END pair, 192 finite positive device rows, 24 operations on each of eight devices, matching operation identities after device-offset normalization, and an exact 8,585,052 ns aggregate. All 192 rows report program-cache misses: this is the first untraced instrumented forward, with no warmup. Kernel counters exclude host compilation gaps, but cold-state effects and instrumentation overhead were not quantified; do not label this steady-state performance. Ring SDPA includes communication. See `b3-glx-column-PzSANj/validation.json` and `DIRECT_BREAKDOWN.md`.

## Published evidence and reproduction

The branch includes a compressed source CSV, its uncompressed SHA-256 in `validation.json`, the operation breakdown, and `validate_capture.py`. Large Tracy binaries and console logs remain local. From the repository root, run:

```bash
python3 profiling_reports/2026-09-11/b3-glx-column-PzSANj/validate_capture.py
```

This validates the archived measurement without hardware. The original CI dispatch is not treated as a passing validation result here; no CI result was used to establish the local findings.
