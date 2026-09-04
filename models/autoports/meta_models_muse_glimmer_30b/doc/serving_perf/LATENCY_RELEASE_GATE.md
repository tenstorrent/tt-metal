# Batch-1 latency release gates

P150 and P150x2 are qualified with two complementary sweeps:

1. the fixed-OSL generator sweep measures the underlying prefill/decode path;
2. the live API sweep sends a tools-enabled chat request at every prompt length
   and fails unless the server returns the expected structured call.

The API sweep is the authority for the user-visible tool-calling path. It
records actual prompt and completion token counts, TTFT to the first semantic
SSE delta, end-to-end latency, and derived TPOT. Run it against each packaged
profile after `/health` is ready:

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/tool_call_latency_sweep.py \
  --profile p150 \
  --base-url http://127.0.0.1:20000 \
  --repeats 3 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/p150_tool_call_latency.json

python models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/tool_call_latency_sweep.py \
  --profile p150x2 \
  --base-url http://127.0.0.1:20000 \
  --repeats 3 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/p150x2_tool_call_latency.json
```

Run the fixed-OSL device sweep once per topology as a lower-level control:

```bash
TT_METAL_VISIBLE_DEVICES=0 \
python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_sweep_batch1.py \
  --mesh-shape 1x1 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/p150_fixed_osl_latency.json

TT_METAL_VISIBLE_DEVICES=0,1 \
TT_MESH_GRAPH_DESC_PATH="$PWD/models/autoports/meta_models_muse_glimmer_30b/tt/p150x2_qb2_mesh_graph_descriptor.textproto" \
python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_sweep_batch1.py \
  --mesh-shape 1x2 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/p150x2_fixed_osl_latency.json
```

Every sample is a hard functional gate: `finish_reason` must be `tool_calls`,
the function must be `record_latency_probe`, its arguments must equal
`{"payload":"ready"}`, and the server-reported prompt length must equal the
requested ISL. A malformed or prose-only response aborts the sweep.

## Performance acceptance

A minor measured degradation is allowed. For a same-topology before/after
comparison, a median regression up to 5% in TTFT, derived TPOT, or E2E is
considered minor when it is stable across three samples and documented. A
larger or unexplained regression blocks publication. Cross-topology numbers
are reported directly and are not treated as regressions: P150, P150x2, and
P150x4 have different compute and collective costs.

Correctness, structured tool-call validity, stable full-context serving,
memory headroom, and clean device release remain hard gates regardless of
latency.

## Existing P150x4 baseline

The original agentic-coding container is gated against the committed
`cardrun2` P150x4 baseline measured on QB2/P300x2. Both the baseline and rerun
use batch 1, OSL 512, per-shape warmup, and ISLs 128, 1K, 4K, 8K, 16K, 32K,
64K, and 130,560.

Do not run these commands until a read-only process check shows the devices
are idle. The generator sweep opens the selected mesh directly.

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_sweep_batch1.py \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_toolcalling_release.json

python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_gate.py \
  --baseline models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_cardrun2.json \
  --candidate models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_toolcalling_release.json \
  --allowed-regression-percent 2 \
  --allowed-ttft-regression-ms 5 \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_gate_toolcalling_release.json
```

The legacy comparator uses the precision displayed in the model card: 0.1 ms
for TTFT/E2E and 0.01 ms for TPOT. Its original four-chip allowance is 2% per
displayed metric, with a 5 ms absolute TTFT allowance for short-input variance.
If a shape exceeds the allowance, rerun it twice with `--isl <shape>`, add both
JSON files as further `--candidate` arguments, and compare the median of all
three samples.

Exit status 0 means every shape passed. Status 2 means a missing shape or two
required retries; status 1 means a confirmed median regression beyond both
allowances. This comparator remains the four-chip reproducibility gate; it is
not used to compare lower-device profiles against P150x4.
