# Gemma 4 12B optimized multichip decoder

Target: `google/gemma-4-12B`
Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
Date: 2026-06-09 UTC

This stage optimizes the completed TP8 multichip decoder path in place. Full
model and vLLM work were not started. The accepted final path is still
`tt/multichip_decoder.py`: every attempted runtime change was rejected because
it either slowed traced decode, failed long-context PCC, or hit a CCL runtime
failure.

## Accepted path

| Item | Final choice |
| --- | --- |
| Hardware | Wormhole T3K, 8 devices |
| Mesh | `ttnn.MeshShape(1, 8)` |
| Fabric | `ttnn.FabricConfig.FABRIC_1D_RING` |
| CCL topology | `ttnn.Topology.Ring` |
| Tensor parallelism | TP8 on mesh axis 1 |
| Residual stream | Replicated hidden state between decoder layers |
| Decode activations | Local L1 width-sharded tensors |
| Prefill activations | DRAM interleaved tensors |
| Attention | Column-parallel fused QKV, local SDPA, row-parallel O, ring all-reduce |
| MLP | Column-parallel gate/up, local GeGLU, row-parallel down, ring all-reduce |
| KV cache | Paged per-device KV heads; full-attention KV is replicated because the layer has one KV head |
| MoE | Not applicable, this model is dense |

The final stack boundary has no CPU fallback and no extra inter-layer collective.
The only inter-device reductions are the two mathematically required TP
all-reduces inside each decoder layer, after attention O and MLP down. Internal
`ShardedToInterleaved`, `InterleavedToSharded`, and `Reshard` ops remain where
TTNN op contracts require them inside the layer; no accepted trial showed that
moving the layer boundary to a scattered contract was faster overall.

## Correctness

Before and after PCC are the same because the completed multichip path was
retained as the optimized path after rejecting all candidate modifications.
The reference is the optimized single-chip TTNN decoder.

| Layer | Seq | Prefill PCC before | Prefill PCC after | Decode PCC before | Decode PCC after | Decode bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | 128 | 0.9997969662 | 0.9997969662 | 0.9964325808 | 0.9964325808 | 0.993 |
| Full | 128 | 0.9996992886 | 0.9996992886 | 0.9983275303 | 0.9983275303 | 0.995 |
| Sliding | 1024 | 0.9994995075 | 0.9994995075 | 0.9993016740 | 0.9993016740 | 0.992 |
| Full | 1024 | 0.9994667300 | 0.9994667300 | 0.9924891310 | 0.9924891310 | 0.992 |

Trace replay remained correct:

| Layer | Trace replay PCC | Determinism PCC | Replica PCCs |
| --- | ---: | ---: | --- |
| Sliding | 0.9964325808 | 1.0 | all 1.0 |
| Full | 0.9983275303 | 1.0 | all 1.0 |

Layout and cache contracts stayed stable:

| Layer | Decode output memory | Seq 128 KV shape | Seq 1024 KV shape |
| --- | --- | --- | --- |
| Sliding | `WIDTH_SHARDED`, L1, shard `[32, 128]` on 30 cores | `[7, 1, 64, 256]` | `[21, 1, 64, 256]` |
| Full | `WIDTH_SHARDED`, L1, shard `[32, 128]` on 30 cores | `[7, 1, 64, 512]` | `[21, 1, 64, 512]` |

Runtime fallback audit is clean for `ttnn.from_torch`, `ttnn.to_torch`,
`FunctionalDecoder`, and the rejected `prefill_l1_inputs` hook in
`tt/multichip_decoder.py`.

## Performance

Metric: `tt-perf-report` `Device Time`, in microseconds. Decode is the traced
warmed decode window.

| Layer | Mode | Before us | After us | Delta us | CCL us | Matmul us | Movement us | Op gap us | Host ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | Prefill | 1487.611 | 1487.611 | 0.000 | 421.360 | 563.834 | 0.000 | 1111.227 | 0 |
| Sliding | Traced decode | 679.406 | 679.406 | 0.000 | 216.587 | 197.993 | 107.574 | 631.727 | 0 |
| Full | Prefill | 1713.979 | 1713.979 | 0.000 | 420.338 | 682.993 | 0.000 | 906.137 | 0 |
| Full | Traced decode | 881.606 | 881.606 | 0.000 | 225.801 | 284.531 | 164.049 | 75.282 | 0 |

The before/after delta is zero because the final accepted code path is the
completed multichip decoder. The optimization pass still produced useful
negative evidence: async CCL, two-link CCL, prefill L1 inputs, and precision
reductions were all tried and rejected.

## tt-perf-report findings

The advice-backed human-readable reports and CSV/provenance files are under
`tracy/{sliding,full}/`.

| Finding | Action | Decision |
| --- | --- | --- |
| CCL is a major cost: two TP all-reduces lower as four CCL ops per layer | Tried `ttnn.experimental.all_reduce_async`; tried a two-link Ring CCL manager setting | Rejected: async was slower for traced decode and two-link CCL hit a runtime fatal |
| Decode matmuls are marked `SLOW` and DRAM-bound | Kept DRAM-sharded decode matmul weights/configs from the completed path; tested lower precision groups | Keep final BF16/BFP8 mix; BFP4 and full-attention BFP8 trials failed decode PCC |
| Advice suggests higher fidelity for accuracy | Full decode QKV already uses HiFi3/fp32 accumulation where long-context PCC requires it; broad HiFi4 was not a performance optimization | Keep selective fidelity only where needed |
| Advice flags high op-to-op gap, especially sliding decode | Decode window is already trace replay with zero host ops; the merged-device report appears to overstate trace savings | Record as tt-perf-report limitation, not a runtime fallback |
| Prefill movement advice favored L1 input placement in the trial profile | Tried L1 placement for attention/MLP prefill inputs | Rejected: long sliding prefill PCC dropped to 0.9915908094 below the 0.995 bar |

Potential report improvement: the sliding traced-decode table says tracing could
save 555 us even though the measured window is already trace replay with zero
host ops. The merged-device op-gap advice should distinguish host gaps from
device scheduling or cross-device merge artifacts in traced windows.

## Rejected trials

| Trial | Evidence | Decision |
| --- | --- | --- |
| `ttnn.experimental.all_reduce_async` simple replacement | Correctness passed; metrics were sliding prefill 1490.959 us, sliding decode 681.769 us, full prefill 1709.462 us, full decode 883.397 us | Rejected because traced decode was slower and CCL time did not improve |
| Ring CCL `num_links=2` | Focused correctness hit `TT_FATAL Unexpected values for event in completion queue`, with `CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST`, length 8208, expected 1024 | Rejected after runtime fatal and device reset |
| Prefill L1 inputs for attention/MLP | Short profile was mixed: sliding prefill 1481.033 us, sliding decode 677.761 us, full prefill 1716.737 us, full decode 882.515 us; long sliding prefill PCC then fell to 0.9915908094 | Rejected and fully removed |
| BFP4 MLP decode weights | Sliding decode PCC 0.9898308853 < 0.993; full decode PCC 0.9917303167 < 0.995 | Rejected |
| Full-attention BFP8 QKV/O | Full decode PCC 0.9838415619 < 0.995 | Rejected |
| Fused matmul-CCL | `all_gather_matmul_async` examples in `models/tt_transformers/tt/attention.py` are gather-then-matmul; `llama_rs_matmul` is reduce-scatter+matmul. This decoder needs row-parallel matmul partials summed back to a replicated residual stream. | Ruled out for the final contract because using those APIs would require extra gather/scatter around layer boundaries |
| Explicit semaphore/preallocated CCL buffer reuse | The simple async API did not improve latency, and the two-link CCL attempt was unstable on this path | No accepted semaphore/buffer change |

## Watcher

Clean watcher evidence was collected with ETH cores active and the repo-documented
dispatch-kernel watcher workaround enabled:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_DISPATCH=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_multichip_decoder/watcher_eth_no_dispatch \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_multichip_decoder.py::test_multichip_paged_prefill_then_decode_pcc_vs_optimized \
  --tb=short --timeout=1200
```

Result: `2 passed, 3 warnings in 556.97s`. The log
`watcher_eth_no_dispatch/generated/watcher/watcher.log` has no matches for
`assert|error|fault|hang|tripped|critical|watcher stopped|invalid NOC`.

A fully enabled ETH+dispatch watcher attempt was also tried. It tripped an
idle-ETH dispatch `cq_prefetch.cpp` watcher assertion before the next kernel.
The failure was isolated to dispatch watcher instrumentation because the same
optimized multichip path passes with ETH watcher active when only dispatch
watcher is disabled, which is documented in `docs/source/tt-metalium/tools/watcher.rst`
for dispatch-kernel watcher trouble.

## Artifacts

- PCC/layout/trace JSONL: `pcc_results.jsonl`
- Repeated short-run evidence: `repeated_short_pcc_results.jsonl`
- Before/after performance summary: `perf_summary.json`
- Precision trial evidence: `precision_trials.jsonl`
- Accepted raw ops CSVs: `tracy/sliding/ops.csv`, `tracy/full/ops.csv`
- Accepted human-readable reports:
  `tracy/sliding/prefill_perf_report.txt`,
  `tracy/sliding/decode_perf_report.txt`,
  `tracy/full/prefill_perf_report.txt`,
  `tracy/full/decode_perf_report.txt`
- Accepted CSV/provenance reports:
  `tracy/{sliding,full}/{prefill,decode}_perf_report.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.csv`,
  `tracy/{sliding,full}/{prefill,decode}_perf_report_stacked.png`
- Rejected async CCL profiles: `tracy/async_ccl/{sliding,full}/`
- Rejected prefill-L1 profiles: `tracy/prefill_l1/{sliding,full}/`
- Watcher clean evidence: `watcher_eth_no_dispatch/generated/watcher/watcher.log`
- Tensix-only fallback watcher evidence: `watcher_tensix/generated/watcher/watcher.log`

## Limitations

- This stage optimizes only the existing 1x8 T3K TP8 decoder-layer path.
- No full-model or vLLM serving path was started.
- The final performance improvement from this pass is zero because all tested
  changes were rejected. This is an evidence-backed optimum for the current
  hardware, TTNN APIs, and acceptance thresholds, not a deferred optimization.
