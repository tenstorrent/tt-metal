# Completed native versus single-step GDN sweep

[Comparison graph](index.html) · [PNG](comparison.png) · [PDF](comparison.pdf) ·
[CSV](comparison.csv) · [JSON](comparison.json)

Both sweeps completed on one physical TP4 replica of `10.228.203.98`, with all
64 model layers, 128 fixed output tokens, one warmup and three measured runs.
The persistent sweep service exited successfully at 2026-10-08 00:59 UTC and
closed devices cleanly. Each variant has 24 measured cells, one B32/32K
prefill allocator failure, five capacity guards and five B64 implementation
guards. Guards are untested configurations, not measured physical limits.
These are native-generator measurements without HTTP/router overhead.

| Input context | Users per TP4 | Native output tok/s | Single-step output tok/s | Change |
|---:|---:|---:|---:|---:|
| 32,768 | 16 | 209.28 | 259.22 | +23.86% |
| 131,072 | 8 | 123.58 | 123.90 | +0.26% |
| 262,016 | 4 | 65.68 | 64.74 | -1.44% |
| 8,192 | 16 | 237.01 | 302.89 | +27.80% |
| 8,192 | 32 | 289.64 | 434.12 | +49.88% |

The candidate improves every measured B16/B32 cell. B8 is approximately flat;
smaller batches regress. This supports qualifying a batch-dependent recurrence
policy after the new capacity measurements, rather than replacing the native
default at every batch. No serving policy is changed by this report. At 128K
and near-256K the current measured batches do not show an improvement; there
are no B16/128K or B8/near256K measurements in these sweeps.

For the user's 2,680 output tok/s Galaxy goal, eight times the best measured
32K TP4 rate is 2,073.76 output tok/s, or 77.4% of the target. That is an
eight-replica projection, not a Galaxy measurement. The corresponding best
observed TP4 rates at 128K and near-256K remain about 124 and 65.7 output tok/s.
Actual eight-replica scaling and online reference evaluations remain pending.

The comparison verifies that all model source hashes match except the selected
precision-file hash; resolved precision differs only in config ID and recurrence
selection. Topology, runtime knobs, prompts and measurement protocol match.
Every completed cell is rechecked against raw timing samples and within-variant
output repeatability. Warmup compilation is excluded from timings. No missing
measurements are filled in. `validation.json` also records rejection of altered
source, precision, runtime, prompt, summary accounting and cleanup evidence.

The two sweeps ran sequentially, with clean-process recovery after allocator
failures. They are not randomized interleaved A/B trials; sub-percent differences
should not be interpreted as a demonstrated speedup. Output hashes across the
two variants are recorded for diagnosis, not used as an accuracy score. The
timing harness continues past EOS for a fixed workload. GPQA and the other
reference-evaluation gates are not established by these timings.

Full per-variant input/output throughput, TSU, TTFT, request throughput, raw
attempts and original receipts are in
[`../gdn-native-sweep-v4/`](../gdn-native-sweep-v4/README.md) and
[`../gdn-candidate-sweep-v4/`](../gdn-candidate-sweep-v4/README.md).
Reproduce the comparison with a new output directory:

```sh
python -m models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps \
  --native models/demos/qwen38_27b_qb2/galaxy-evidence/gdn-native-sweep-v4/sweep.json \
  --candidate models/demos/qwen38_27b_qb2/galaxy-evidence/gdn-candidate-sweep-v4/sweep.json \
  --output /tmp/qwen-gdn-comparison-NEW
```

The smaller-chunk capacity sweep started automatically after the completed
matched sweep. It tests B16/32K as a control, then B32/32K, B16/128K and
B8/near256K under the native recurrence policy. Placement and reader-barrier
diagnostics follow it. Their launch instructions and fixed-precision scope are
in [`../gdn-model-integration-v1/README.md`](../gdn-model-integration-v1/README.md).
