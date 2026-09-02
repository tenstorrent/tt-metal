# Batch-1 latency release gate

The agentic-coding container is gated against the committed `cardrun2`
P300x2 baseline. Both the baseline and rerun use batch 1, OSL 512, per-shape
warmup, and ISLs 128, 1K, 4K, 8K, 16K, 32K, 64K, and 130,560.

Do not run these commands until the user has released the TT devices and a
read-only process check shows the P300x2 is idle. The sweep opens the full mesh.

```bash
python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_sweep_batch1.py \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_toolcalling_release.json

python models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model/bench/latency_gate.py \
  --baseline models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_cardrun2.json \
  --candidate models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_sweep_toolcalling_release.json \
  --out models/autoports/meta_models_muse_glimmer_30b/doc/serving_perf/benchmarks/latency_gate_toolcalling_release.json
```

The comparator uses the precision displayed in the model card: 0.1 ms for
TTFT/E2E and 0.01 ms for TPOT. If an initial shape is slower, rerun that shape
twice with `--isl <shape>`, add both JSON files as further `--candidate`
arguments, and rerun the gate. It compares the median of all three samples.

Exit status 0 means every shape passed. Status 2 means a missing shape or two
required retries; status 1 means a confirmed median regression. Packaging is
blocked unless the final result is status 0.
