# Calibrated GLM-5.3 expert placement

`glm_5_3_torus_xy_5120.json` is an opt-in prefill placement for GLM-5.3 on
an eight-bank, 8x4 Blackhole Galaxy with SP8/TP4, TorusXY fabric, BF4 experts,
and 5120-token chunks. It relocates experts in 73 layers (3 and 5–76).
All 256 experts remain present exactly once within their original 64-expert
TP dispatch group; top-8 selection and expert dimensions are unchanged.

The cache tool moves original quantized host shards without requantizing.
Completion markers enable the matching logical-to-physical ID lookup in the
existing grouped-top-k writer, after logical selection and score gathering.
Ordinary caches retain their existing path. The lookup tables total 37,376
bytes per chip for this preset; the number of resident experts is unchanged.

## Measured scope

The base is tt-metal main `4e8a922e982ef9bb0f2d83e0ed303cbaaa97d4f3`.
Measurements used `bh-glx-120-b02u08`, all 32 chips limited to 160 W, two fabric links,
the existing GLM harness, and a 56,320-token scaled-FP8 KV-cache capacity.
The full 78-layer test produces KV caches, with the final layer KV-only;
these timings exclude the LM head, decode, model loading, and cache conversion.
The production gate, sparse-attention behavior, and routed/combine overlap
settings are unchanged.

At 56,320 processed tokens, fixed A/B/B/A runs measured:

| Run | Original placement, seconds | Placed cache, seconds |
| --- | ---: | ---: |
| First pair | 4.255 | 4.089 |
| Second pair | 4.258 | 4.092 |

Each entry is the median of nine retained whole-prefill iterations; iteration
zero is excluded. The median of the two run medians is 4.2565 versus 4.0905 s:
**4.058% higher throughput, 3.900% lower latency**. Whole-prefill timings include
input-copy and loop overhead. Raw chunk trace times are recorded separately.
Both sides used the same rebuilt native libraries. The original-cache control
also matched the clean-main baseline (4.259 s).

A separate 10,240-token A/B confirmation, using the same KV capacity, measured
0.754 versus 0.719 s: **4.868% higher throughput, 4.642% lower latency**. Each
side retained nine iterations, and all 99 cache tensor digests matched exactly.
This shorter input is a prefix of the same calibration workload.

An additional 56,320-token identity-map control allocated and used the same 73
lookup tables while leaving every expert in its original slot. Its median was
4.257 s, with exact clean-main cache digests. The placed cache is 4.070% faster
than this control; the identity maps alone change throughput by -0.012%.

All four runs passed the upstream golden PCC checks. Every valid cache digest
matched clean main exactly: 78 KV tensors and 21 indexer tensors per run.
The repacker verified all 56,064 moved quantized host-shard hashes. Device tests
checked exact selected IDs and weights, ties, padding sentinels, program-cache
rebinding, and trace replay across all 32 chips.

The preset was frozen using only the first five routing chunks (25,600 tokens).
The remaining six chunks (30,720 tokens) were then evaluated without refitting.
Held-out chunk trace replay throughput improved about 3.2%. The traffic model
used integer per-expert Fabric2D ring payload routes; its load estimates are
proxies, not measured latency predictions.

This is one calibration workload, `vllm-glm53-indexer-kcache-55k`, using checkpoint
`GLM-5.3-fp8-aca966e4`. Its 56,320-token ID sequence has SHA256
`aa1ad5e2b5a5a90a0d2505e98f5bc1ac3dc005ca7aaff5c0686c15b37d0789ff`.
These measurements do not establish independent-prompt, decode, serving-batch,
other power-limit, or other mesh performance. Recalibrate and measure those
workloads before making a placement their default.

The preset contains exactly the measured physical permutations. It omits the
redundant inverse arrays and formats eight physical slots per line. Its
`calibration_plan_sha256` identifies the original frozen calibration artifact;
the repacker derives the inverse and records the supplied preset file's hash.

## Prepare and benchmark

Use a build containing this change and the repository's Python environment.
Set `GLM53_HF_MODEL` to the matching checkpoint and `PREFILL_TRACE_DIR` to the
matching golden-trace directory. Run from the repository root with
`TT_METAL_HOME` and `PYTHONPATH` pointing to this checkout. This Slurm node
required raising its soft process limit from 512 to 8192 for device-kernel JIT
compilation, within the existing hard limit. Leave
`TT_MESH_GRAPH_DESC_PATH` unset for the production TorusXY auto-discovery path.

Supply the actual original logical-order cache and a separate, nonexistent
destination:

```bash
python -m models.demos.deepseek_v3_d_p.scripts.prepare_expert_placement \
  --source /path/original/glm_5_3_bh_32dev/8x4 \
  --destination /path/placed/glm_5_3_bh_32dev/8x4 \
  --plan models/demos/deepseek_v3_d_p/scripts/expert_placement_plans/glm_5_3_torus_xy_5120.json
```

The tool rejects already-placed sources and incomplete caches. Keep the source
available because unchanged tensors are symlinked. Offline preparation writes
about 397 GB of selected routed weights for this preset. The marked cache must
be loaded by the matching marker-aware prefill implementation.

Set `TT_GLM53_PREFILL_TTNN_CACHE=/path/original` for fixture resolution. The
benchmark's explicit `--cache` selects the actual original or placed cache:

```bash
python -m models.demos.deepseek_v3_d_p.scripts.benchmark_glm_expert_placement \
  --mode perf --cache /path/original/glm_5_3_bh_32dev --output /path/results/control
python -m models.demos.deepseek_v3_d_p.scripts.benchmark_glm_expert_placement \
  --mode perf --cache /path/placed/glm_5_3_bh_32dev --output /path/results/placed
```

Output directories must not exist. `perf` runs 78 layers and 11 chunks;
`short` runs 78 layers and two chunks, retaining the same KV capacity.
Both collect ten traced iterations and check golden PCC after timing.
`smoke` uses ten layers and two chunks. `routing` collects the untraced
814-file routing workload; collect calibration routing from the original cache.
Routing diagnostics are not performance measurements.

Repeat in counterbalanced order, compare the raw `samples.jsonl` and
`measurement.json` timings, and require matching `cache-digests.jsonl` files.
Correctness readbacks and digest calculation happen after all timed iterations.
The benchmark driver is the measured wrapper with formatting-only changes.
