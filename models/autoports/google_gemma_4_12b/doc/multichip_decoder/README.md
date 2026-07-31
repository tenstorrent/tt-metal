# Gemma 4 12B multichip decoder

Target: `google/gemma-4-12B`
Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`

This stage adds `tt/multichip_decoder.py` and validates it as the decoder-layer
baseline for a later full-model stack. The implementation is intentionally
targeted at the available 8-device Wormhole T3K mesh and does not try to support
smaller meshes.

## Target mesh

| Item | Choice |
| --- | --- |
| Hardware | Wormhole T3K, 8 devices |
| Mesh | `ttnn.MeshShape(1, 8)` |
| Fabric | `ttnn.FabricConfig.FABRIC_1D_RING` |
| CCL topology | `ttnn.Topology.Ring` |
| Tensor parallelism | TP8 on mesh axis 1 |
| Baseline | `OptimizedDecoder` for TP1 and correctness reference |
| MoE | Not applicable, this model is dense |

## Parallel plan

The residual stream is replicated across the mesh. Decode tensors are kept in
per-device L1 width-sharded layout at the stacked-decoder boundary; the tested
decode output shard spec is `WIDTH_SHARDED`, L1, shard shape `[32, 128]` on the
30-core grid `x=0..5, y=0..4`.

| Tensor family | Strategy |
| --- | --- |
| RMSNorm | Local optimized RMSNorm on replicated residual, no collective |
| Attention QKV | Column-parallel fused WQKV |
| Attention SDPA | Local heads only |
| Attention O | Row-parallel WO followed by ring all-reduce |
| MLP gate/up | Column-parallel |
| MLP activation | Local GeGLU |
| MLP down | Row-parallel followed by ring all-reduce |
| Collectives | TTNN ring all-reduce, lowered in reports as reduce-scatter plus all-gather |
| KV cache | Paged per-device local KV heads, replicated page tables and positions |

Per-device tensor widths:

| Layer kind | Q heads/chip | KV heads/chip | Local Q width | Local K/V width | Local fused QKV width | MLP width/chip | KV replication |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sliding attention | 2 | 1 | 512 | 256 | 1024 | 1920 | No |
| Full attention | 2 | 1 | 1024 | 512 | 2048 | 1920 | Yes, because full layers expose one KV head |

No tensor-parallel padding is required: hidden size 3840, MLP intermediate size
15360, and all local Q/K/V/MLP widths are tile-aligned under TP8. If this model
is later stacked, the expected input/output contract is replicated
`[1, 1, S, 3840]` with decode outputs in local L1 width-sharded form.

Rejected alternatives:

| Alternative | Reason rejected |
| --- | --- |
| Replicate all weights | Correct but gives no useful mesh speedup |
| Hidden-sharded residual stream | Would require distributed RMSNorm and frequent gathers at stack boundaries |
| Sequence-parallel prefill | Worse fit for the autoregressive decode-focused stack contract |
| 2D mesh strategy | The target machine exposes a 1x8 T3K topology |
| Reduce-scatter output contract | Breaks the replicated residual contract needed by the next decoder layer |
| Shard full-attention KV across TP | Full layers have only one KV head, so KV must be replicated |
| HiFi4 full decode QKV | Only marginally improved PCC over HiFi3/fp32 and carries Wormhole warnings |
| SDPA grid tweaks for long full decode | Did not improve the isolated PCC failure |

## Correctness

The multichip decoder is compared against the optimized single-chip TTNN
decoder, not against PyTorch or the functional fallback. Source fallback audit is
clean for `ttnn.from_torch`, `ttnn.to_torch`, and `FunctionalDecoder` tokens in
`tt/multichip_decoder.py`.

PCC results:

| Layer | Seq | Prefill PCC vs optimized | Decode PCC vs optimized | Decode bar |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 128 | 0.9997969662 | 0.9964325808 | 0.993 |
| Full | 128 | 0.9996992886 | 0.9983275303 | 0.995 |
| Sliding | 1024 | 0.9994995075 | 0.9993016740 | 0.992 |
| Full | 1024 | 0.9994667300 | 0.9924891310 | 0.992 |

Trace and layout coverage:

| Layer | Trace replay PCC | Trace determinism PCC | Replica PCCs | KV cache shape at seq 128 |
| --- | ---: | ---: | --- | --- |
| Sliding | 0.9964325808 | 1.0 | all 1.0 | `[7, 1, 64, 256]` |
| Full | 0.9983275303 | 1.0 | all 1.0 | `[7, 1, 64, 512]` |

Paged cache contracts were validated with page tables `[1, 7]` for seq 128 and
`[1, 21]` for seq 1024. Decode `position_idx` and `position_idx_cache` are both
rank-2 tensors with shape `[1, 1]`. Sliding layers keep one local KV head per
chip; full layers keep one replicated KV head per chip.

## Performance

The table uses the same `tt-perf-report` `Device Time` metric as
`doc/optimized_decoder/tracy/perf_summary.json`. Op-to-op gap and host-op counts
are reported separately.

| Layer | Mode | Optimized single-chip us | Multichip us | Speedup | TP8 efficiency | CCL us | Op gap us | Host ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | Prefill | 3294.000 | 1487.611 | 2.214x | 27.7% | 421.360 | 1111.227 | 0 |
| Sliding | Traced decode | 1374.000 | 679.406 | 2.022x | 25.3% | 216.587 | 631.727 | 0 |
| Full | Prefill | 4404.000 | 1713.979 | 2.569x | 32.1% | 420.338 | 906.137 | 0 |
| Full | Traced decode | 1740.000 | 881.606 | 1.974x | 24.7% | 225.801 | 75.282 | 0 |

`tt-perf-report` findings:

- Each layer has two row-parallel reductions, reported as four CCL ops
  (`ReduceScatterDeviceOperation` plus `AllGatherDeviceOperation` pairs).
- Decode windows have zero host ops. Sliding decode still reports a large
  merged-device op-to-op gap inside the traced replay window; this is not a
  runtime fallback.
- Matmuls are still marked `SLOW` by the advice pass. Decode local matmuls reach
  about 118-162 GB/s DRAM bandwidth and about 4.9-10.3 TFLOP/s depending on the
  projection and layer kind.
- Full decode QKV uses HiFi3 with fp32 accumulation to keep long-context PCC
  above the 0.992 bar. Sliding decode uses HiFi2 except the inherited optimized
  long-position path.

## Watcher

Clean watcher evidence was collected with Tensix watcher enabled and ETH watcher
disabled:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/multichip_decoder/watcher_tensix \
pytest -q models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=600
```

Result: `2 passed, 3 warnings in 637.58s`. The watcher log is
`watcher_tensix/generated/watcher/watcher.log`, and it contains no watcher
assert/error matches.

## Artifacts

- PCC and layout JSONL: `pcc_results.jsonl`
- Performance summary: `tracy/perf_summary.json`
- Raw OP CSVs: `tracy/sliding/ops.csv`, `tracy/full/ops.csv`
- Human-readable reports:
  `tracy/sliding/prefill_perf_report.txt`,
  `tracy/sliding/decode_perf_report.txt`,
  `tracy/full/prefill_perf_report.txt`,
  `tracy/full/decode_perf_report.txt`
- CSV/provenance reports:
  `tracy/{sliding,full}/{prefill,decode}_perf_report.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.png`
- Raw Tracy report folders:
  `tracy/sliding/raw/reports/2026_06_09_00_15_16/`,
  `tracy/full/raw/reports/2026_06_09_00_17_32/`
- Watcher evidence: `watcher_tensix/generated/watcher/watcher.log`

## Limitations

- Only the 1x8 T3K TP8 path is implemented.
- The implementation is a decoder-layer baseline only; full-model and vLLM
  work are intentionally out of scope for this stage.
- The residual stream is replicated by design. A future full-model stage should
  preserve this contract or deliberately revalidate every stacked boundary.
