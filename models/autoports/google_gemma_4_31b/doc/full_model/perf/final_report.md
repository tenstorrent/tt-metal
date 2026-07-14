# Source-current reduced token-out `tt-perf-report`

Date: 2026-07-14 UTC
Hardware: 4x Blackhole P150b, TP4 Linear fabric
Workload: optimized real layers 0 and 5, prompt 149, batch one, six generated tokens, four steady trace replays
Source CSV SHA-256: `cefa4861ae9713bc1d83c117dab38760939997a0a305ee71a03483f1c7d528a3`

The profile was collected after sampler alignment/stride, persistent gathered-output, mixed device-prefill, and explicit teardown fixes. `ttnn.ReadDeviceProfiler(mesh)` flushed setup traffic before the signpost; no profiler-buffer-drop warning occurred.

```bash
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/google_gemma_4_31b/doc/full_model/tracy/reduced_token_out_final \
  -n gemma4_full_model_final -m pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_token_out_perf_signposts

tt-perf-report <source-csv> \
  --start-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY \
  --end-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY_END --no-color \
  --csv perf/final_filtered.csv --summary-file perf/final_summary
```

## Selected rows

```text
window                         552 merged device-op rows, 4 complete replays
device-op sum                  12,377.1445 us (3,094.286 us/replay)
op-to-op gap sum                  680.6290 us
local greedy winner median        299.464 us (299.342-299.586), 8 cores
final greedy reducer median          0.4205 us (0.418-0.424), 1 core
local vocab LM head median        1,740.674 us (1,739.564-1,741.507)
sampler summary share                  9.68%
LM-head summary share                  56.25%
profiled steady e2e                 287.018 t/s/u = 3.484 ms/token
```

The custom sampler is not dominant. Its approximately 0.300 ms is about 8.6% of profiled steady end-to-end time. `tt-perf-report` identifies the vocab-sharded LM head, not sampling, as the largest terminal cost.

Advice notes:

- The LM head receives generic DRAM-sharded/HiFi advice. The accepted contract already shards vocabulary across TP4, uses tied BF16 values, and meets 100% top-5/top-100. No unvalidated intra-device DRAM-sharded rewrite was substituted at stage close.
- LoFi BFP4 MLP rows receive higher-fidelity accuracy advice. The inherited Stage 05 datatype/fidelity policy is accuracy-validated and intentionally preserved.
- Several Gemma/CCL/custom operations are unclassified by `tt-perf-report`; official rows remain in `final_filtered.csv`, and no timing is silently reassigned.

`final_summary.csv`/`.png` and `final_filtered.csv` are the compact official tool outputs. Raw Tracy trees are local-only because they are multi-gigabyte artifacts.
