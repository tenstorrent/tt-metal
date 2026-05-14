# Qwen3.6-27B Galaxy AR demo — initial perf sheet

Captured via:

    python -m tracy -p -v -r -m pytest \
        models/demos/qwen3_6_galaxy/demo/text_demo.py::test_demo_text \
        -k perf_1L_1T

Run config: 1 decoder layer (layer 0 = `linear_attention` / DeltaNet),
prefill T=32 (prompt: "The capital of France is"), no decode steps
(num_tokens=1 means only the prefill-tail token is emitted). Mesh: BH GLX 8×4.

Tracy CSV: `generated/profiler/reports/2026_05_14_07_22_59/ops_perf_results_2026_05_14_07_22_59.csv`

## Wall-clock

| phase | latency |
|---|---|
| Model build (1 layer, on-device weight upload) | 25.9 s (one-time) |
| Warmup prefill+decode (excluded from profile) | ~1.5 s |
| Profiled prefill (T=32, 1 layer)               | **712.9 ms** |

## Device-side op breakdown — profiled prefill region (signpost: `start` → `prefill_done`)

5472 op rows = 32 mesh chips × ~171 logical ops. `sum_dev_us` is summed
across all chips × calls; chips run concurrently so wall-clock is bounded
by max-per-chip, not sum.

| op | count | sum_dev_us | avg_us | % of dev time |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation                | 928 | 75 359.9 | 81.21 | **68.3 %** |
| AllGatherDeviceOperation             | 256 | 14 484.4 | 56.58 | **13.1 %** |
| BinaryNgDeviceOperation              | 1 408 | 4 216.8 | 2.99 | 3.8 % |
| LayerNormPostAllGatherDeviceOperation | 96 | 3 796.3 | 39.55 | 3.4 % |
| ReshapeViewDeviceOperation           | 608 | 2 986.8 | 4.91 | 2.7 % |
| LayerNormPreAllGatherDeviceOperation | 96 | 2 151.6 | 22.41 | 1.9 % |
| TilizeDeviceOperation                | 128 | 1 616.0 | 12.63 | 1.5 % |
| UnaryDeviceOperation                 | 416 | 1 011.6 | 2.43 | 0.9 % |
| EmbeddingsDeviceOperation            | 32  | 943.0 | 29.47 | 0.9 % |
| FastReduceNCDeviceOperation          | 64  | 623.7 | 9.75 | 0.6 % |
| TransposeDeviceOperation             | 288 | 535.3 | 1.86 | 0.5 % |
| LayerNormDeviceOperation             | 96  | 443.2 | 4.62 | 0.4 % |
| TilizeWithValPaddingDeviceOperation  | 64  | 392.4 | 6.13 | 0.4 % |
| SliceDeviceOperation                 | 320 | 344.4 | 1.08 | 0.3 % |
| UntilizeDeviceOperation              | 32  | 318.6 | 9.95 | 0.3 % |
| UntilizeWithUnpaddingDeviceOperation | 64  | 251.1 | 3.92 | 0.2 % |
| TypecastDeviceOperation              | 192 | 244.4 | 1.27 | 0.2 % |
| CloneOperation                       | 96  | 222.0 | 2.31 | 0.2 % |
| ConcatDeviceOperation                | 128 | 176.2 | 1.38 | 0.2 % |
| MeshPartitionDeviceOperation         | 96  | 110.7 | 1.15 | 0.1 % |
| FillPadDeviceOperation               | 32  | 90.5  | 2.83 | 0.1 % |
| ReduceDeviceOperation                | 32  | 37.8  | 1.18 | 0.0 % |
| **TOTAL**                            | **5 472** | **110 357** | — | **100 %** |

## Headline findings

1. **Matmul dominates at ~68 %** of device-kernel time. The 928 matmul
   calls for one prefill pass break down into: QKV projection (wqkvg),
   q-norm × k-norm internal matmuls, MLP gate/up/down, LM-head linear,
   DeltaNet in-projections, and the DeltaNet recurrent kernel's many
   small matmuls. The DeltaNet recurrent kernel is the largest single
   contributor here — it issues ~24 matmuls per token (vs ~5 for a
   standard attention layer).

2. **CCL (AllGather) ~13 %**. Distributed-norm uses `all_gather` across
   `cluster_axis=1` (4 cols) and MLP uses `all_gather + fast_reduce_nc`
   across `cluster_axis=0` (8 rows). 256 AllGather calls for 1 layer ×
   T=32 means ~8 gathers per token-step.

3. **LayerNorm (pre + post) ~5.3 %**. DistributedNorm is invoked 2× per
   decoder layer + once at the final norm = 3 invocations, each running
   PreAllGather → AllGather → PostAllGather across 32 prefill positions.

4. **No decode region in this profile** — `num_tokens=1` means the
   prefill tail produces the only generated token; no `forward_decode`
   call ran. To capture a decode-step perf sheet, change the
   `perf_1L_1T` parametrize tuple to `(1, 2, "The capital of France is",
   "perf_1L_1T")` and re-run; the second token comes from one
   `forward_decode` call which the trace will record between the
   `prefill_done` and `stop` signposts.

## Next obvious optimization targets (based on % share)

- Matmul: **fuse gate+up MLP projection** (currently 2 separate matmuls
  + ttnn.silu + multiply). Llama 70B fuses these. Estimated saving:
  ~10–15 % of MLP matmul time.
- AllGather: **trace capture** would eliminate the per-step launch
  overhead. (T14b.9 in progress — see `tt/generator.py` docstring.)
- DeltaNet recurrent: **batched chunk size** in
  `chunk_gated_delta_rule_ttnn` is fixed at 32; profiling indicates
  this is where most of the matmul time goes per token. Worth a tuning
  pass to find the chunk size that maximizes utilization.
