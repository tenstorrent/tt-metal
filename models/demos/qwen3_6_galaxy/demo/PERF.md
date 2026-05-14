# Qwen3.6-27B Galaxy AR demo — initial perf sheet

Captured via:

    python -m tracy -p -v -r -m pytest \
        models/demos/qwen3_6_galaxy/demo/text_demo.py::test_demo_text \
        -k perf_1L_1T

Run config: 1 decoder layer (layer 0 = `linear_attention` / DeltaNet),
prefill T=32 (prompt: "The capital of France is"), 1 decode step
(`num_tokens=2` so the prefill produces token 1 and the decode loop runs
once for token 2). Mesh: BH GLX 8×4.

Tracy CSV: `generated/profiler/reports/2026_05_14_07_31_52/ops_perf_results_2026_05_14_07_31_52.csv`

The profiled region is bracketed by `tracy.signpost` markers
(`start` → `prefill_done` → `stop`) — the analysis below filters to
just those rows, excluding warmup and setup.

## Wall-clock

| phase | latency |
|---|---|
| Model build (1 layer, on-device weight upload)  | ~26 s (one-time) |
| Warmup prefill + decode (excluded from profile) | ~1.5 s |
| Profiled prefill (T=32, 1 layer)                | **720.2 ms** |
| Profiled decode  (T=1, 1 layer)                 | **51.8 ms** |

## Prefill — device-side op breakdown (signpost: `start` → `prefill_done`)

5 472 op rows = 32 mesh chips × 171 logical ops. `sum_dev_us` is summed
across all chips × calls; chips run concurrently so wall-clock latency
is bounded by the slowest per-chip path, not the sum.

| op | count | sum_dev_us | avg_us | % of dev time |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation                  | 928 | 75 395.0 | 81.24 | **68.3 %** |
| AllGatherDeviceOperation               | 256 | 14 441.3 | 56.41 | **13.1 %** |
| BinaryNgDeviceOperation                | 1 408 | 4 217.4 | 3.00 | 3.8 % |
| LayerNormPostAllGatherDeviceOperation  | 96 | 3 794.0 | 39.52 | 3.4 % |
| ReshapeViewDeviceOperation             | 608 | 2 985.4 | 4.91 | 2.7 % |
| LayerNormPreAllGatherDeviceOperation   | 96 | 2 154.5 | 22.44 | 2.0 % |
| TilizeDeviceOperation                  | 128 | 1 613.2 | 12.60 | 1.5 % |
| UnaryDeviceOperation                   | 416 | 1 012.0 | 2.43 | 0.9 % |
| EmbeddingsDeviceOperation              | 32  | 942.8 | 29.46 | 0.9 % |
| FastReduceNCDeviceOperation            | 64  | 625.5 | 9.77 | 0.6 % |
| TransposeDeviceOperation               | 288 | 535.0 | 1.86 | 0.5 % |
| LayerNormDeviceOperation               | 96  | 443.1 | 4.62 | 0.4 % |
| TilizeWithValPaddingDeviceOperation    | 64  | 392.3 | 6.13 | 0.4 % |
| SliceDeviceOperation                   | 320 | 344.6 | 1.08 | 0.3 % |
| UntilizeDeviceOperation                | 32  | 318.7 | 9.96 | 0.3 % |
| UntilizeWithUnpaddingDeviceOperation   | 64  | 250.7 | 3.92 | 0.2 % |
| TypecastDeviceOperation                | 192 | 244.3 | 1.27 | 0.2 % |
| CloneOperation                         | 96  | 220.5 | 2.30 | 0.2 % |
| ConcatDeviceOperation                  | 128 | 175.8 | 1.37 | 0.2 % |
| MeshPartitionDeviceOperation           | 96  | 110.7 | 1.15 | 0.1 % |
| FillPadDeviceOperation                 | 32  | 90.5 | 2.83 | 0.1 % |
| ReduceDeviceOperation                  | 32  | 37.7 | 1.18 | 0.0 % |
| **PREFILL TOTAL**                      | **5 472** | **110 345** | — | **100 %** |

## Decode — device-side op breakdown (signpost: `prefill_done` → `stop`)

4 000 op rows = 32 mesh chips × 125 logical ops per decode step. Fewer
ops than prefill because the linear projections operate on T=1 instead
of T=32, and the DeltaNet decode path uses the *recurrent* kernel (T
sequential matmuls) instead of the *chunked* prefill kernel.

| op | count | sum_dev_us | avg_us | % of dev time |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation                  | 448 | 71 856.4 | **160.39** | **72.8 %** |
| AllGatherDeviceOperation               | 256 | 13 750.3 | 53.71 | **13.9 %** |
| LayerNormPostAllGatherDeviceOperation  | 96  | 3 798.7 | 39.57 | 3.8 % |
| LayerNormPreAllGatherDeviceOperation   | 96  | 2 154.3 | 22.44 | 2.2 % |
| ReshapeViewDeviceOperation             | 608 | 1 565.7 | 2.58 | 1.6 % |
| BinaryNgDeviceOperation                | 640 | 1 490.7 | 2.33 | 1.5 % |
| TilizeWithValPaddingDeviceOperation    | 224 | 861.0 | 3.84 | 0.9 % |
| FastReduceNCDeviceOperation            | 64  | 627.0 | 9.80 | 0.6 % |
| UnaryDeviceOperation                   | 256 | 531.2 | 2.07 | 0.5 % |
| SliceDeviceOperation                   | 320 | 393.8 | 1.23 | 0.4 % |
| LayerNormDeviceOperation               | 96  | 379.9 | 3.96 | 0.4 % |
| TransposeDeviceOperation               | 192 | 270.9 | 1.41 | 0.3 % |
| TypecastDeviceOperation                | 256 | 266.7 | 1.04 | 0.3 % |
| UntilizeWithUnpaddingDeviceOperation   | 96  | 241.9 | 2.52 | 0.2 % |
| CloneOperation                         | 96  | 222.2 | 2.31 | 0.2 % |
| ConcatDeviceOperation                  | 128 | 128.2 | 1.00 | 0.1 % |
| MeshPartitionDeviceOperation           | 96  | 110.0 | 1.15 | 0.1 % |
| EmbeddingsDeviceOperation              | 32  | 46.9 | 1.47 | 0.0 % |
| **DECODE TOTAL**                       | **4 000** | **98 695** | — | **100 %** |

## Headline findings

1. **Matmul dominates both phases** — 68 % of prefill, **73 % of decode**
   device-kernel time. The decode average matmul (160 µs) is **2× the
   prefill average** (81 µs) because the recurrent DeltaNet kernel
   issues many small matmuls per call, each with low arithmetic
   intensity (matmul of [1, K] × [K, V] is bandwidth-bound).
2. **AllGather (CCL) is the second-largest line item** at ~13 % in both
   phases. DistributedNorm uses `all_gather(cluster_axis=1)` 3× per
   layer, and the MLP uses `all_gather(cluster_axis=0) + fast_reduce_nc`
   for the row-parallel reduction.
3. **Decode wall-clock = 51.8 ms** for 1 layer means a full 64-layer
   decode step would run at roughly **1.0 tok/s eager** (extrapolating
   linearly — actual scaling is sublinear because CCL ops grow with
   layer count). The observed 64-layer wall-clock in the e2e demo
   matches at ~0.66 s/step → ~1.5 tok/s.
4. **No trace capture in this run** — `_TRACE_SUPPORTED=False` (T14b.9
   in progress; see `tt/generator.py` docstring). The same decode loop
   under trace replay would amortize the ~150 µs/op Python+kernel-launch
   overhead, projecting to a 5–7× speedup based on the
   trace-vs-eager ratio observed in llama3_70b_galaxy.

## Next obvious optimization targets

| target | est. savings | effort |
|---|---|---|
| Land T14b.9 trace capture for decode | 5–7× decode latency (~10 tok/s eager → 50–70 tok/s) | medium — one residual host-write site identified (`to_memory_config` in `llama_attention.forward_decode`); see PERF.md sibling files for diagnosis |
| Fuse MLP gate + up projections | ~10–15 % of decode matmul time (~7 ms/step) | small — single matmul + split, llama_70b pattern |
| Tune DeltaNet `chunk_gated_delta_rule_ttnn` chunk size (currently 32) | unknown; depends on memory hierarchy; needs sweep | small — single config knob |
| Replace per-call shard configs with pre-computed configs | small (~1 ms/step) | small |
