# GPT-OSS-20B optimized decoder

This stage provides a standalone single-device `OptimizedDecoder` for
`openai/gpt-oss-20b`. It preserves the functional decoder's prefill/decode,
linear paged KV-cache, attention sinks, non-aligned sequence lengths,
determinism, and 21,248-token context contract. The measured path never calls
the functional implementation and contains no Torch/host fallback.

## Selected path

| Area | Final policy |
| --- | --- |
| attention topology | packed QKV linear, TTNN head creation, device-indexed RoPE, SDPA, dedicated decode concat-heads, packed O projection |
| advisor attention layouts | 10-core block-sharded decode RMSNorm; 45/80-core width-sharded QKV input/output; L1 O input; 90-core width-sharded O/residual; required DRAM revert before sparse MoE |
| sliding decode | device-built explicit 128-token mask, non-causal SDPA, explicit 8x8/K=32 program, BF16 destination accumulation |
| attention/cache precision | BF16 weights, activations, and linear KV cache; HiFi4 |
| MoE | routed `ttnn.sparse_matmul`, four active experts, BFP8 weights/intermediates, 9x10 gate/up and down grids, block width 45 |
| prefill | automatic interleaved matmul programs; arbitrary valid logical lengths remain public |
| trace | complete decode layer including cache updates, mutable-position RoPE/mask, attention, routing, and experts |

The mandatory advisor seed materially improved the selected attention path:
the boundary-correct 100-replay timing fell from 0.983908 ms to 0.928144 ms.
The advisor's full dense-MoE layout chain is also implemented behind the
`advisor` A/B variant. After localizing and repairing the router-scatter and
expert-broadcast layout boundaries, it passes real prefill/decode at PCC
`0.99958--0.99973`, but measures 7.727175/5.909603 ms. It is therefore a
correct, measured performance rejection rather than an untried recommendation.

## Correctness

The functional acceptance bar is PCC 0.99. The final suite reports eight
passes and one opt-in performance skip:

| Coverage | Final PCC/result |
| --- | --- |
| concat-head graph rewrite | 0.999844 versus dense functional control |
| synthetic prefill 17 / 33 / 128 | 0.999995 / 0.999996 / 0.999998 |
| real prefill 17 | 0.997100 |
| sliding decode 17 / 18 / 19 | 0.997748 / 0.996880 / 0.996660 |
| full decode 17 / 18 / 19 | 0.997855 / 0.997090 / 0.996897 |
| traced sliding boundary 128 / 129 / 130 | 0.997418 / 0.995303 / 0.997214 |
| traced full boundary 128 / 129 / 130 | 0.997419 / 0.996543 / 0.997395 |
| determinism/stress | identical repeated same-position output and 20 correct fixed-position trace replays for both layer kinds |
| watcher | both real-weight layer-kind tests pass under `TT_METAL_WATCHER=10`; no watcher error/assert |

The PCC delta from the functional BF16 dense path comes from selected BFP8
sparse expert weights. BFP4/LoFi experts were rerun on the final cumulative
advisor-attention topology at 9x10 and 5x6: ordinary prefill is 0.976127 and
boundary prefill is 0.973667 for both layer kinds, below the gate before trace
capture. BFP8 cache was
retested with the final explicit mask: full attention passes the boundary, but
sliding attention fails at position 129 (PCC 0.863325), so the final cache is
BF16.

The native sliding-window kernel is also rejected: its PCC falls to 0.913280
at position 130. The explicit-mask workaround remains correct when both hidden
input and position tensor are changed between trace replays. Automatic, K=32,
and K=64 explicit-mask SDPA configurations all pass positions 128--130; their
100-replay timings are indistinguishable at 0.9280--0.9281 ms, so explicit
K=32 is retained as the smallest contract-matching chunk.

## Performance

Headline timing is batch 1, sequence 17. Prefill is warmed; decode is one
captured trace replayed asynchronously before a final device synchronization.

| Path | Warmed prefill | Traced warmed decode | Relative to functional |
| --- | ---: | ---: | ---: |
| functional baseline | 7.709439 ms | 6.139372 ms | 1.00x / 1.00x |
| concat-head rewrite only | 7.707365 ms | 6.124800 ms | 1.00x / 1.00x |
| boundary-correct sparse path before advisor layouts | 3.884900 ms | 0.983908 ms | 1.98x / 6.24x |
| final selected path | **3.874835 ms** | **0.928144 ms** | **1.99x / 6.61x** |
| full dense advisor A/B | 7.727175 ms | 5.909603 ms | reject, correct but 6.37x slower decode |

At sequence 128, comparable five-repeat prefill measurements are:

| Matmul policy | Warmed prefill | Decision |
| --- | ---: | --- |
| automatic | **13.466493 ms** | keep |
| explicit 2D 8x4 QKV / 6x4 O | 13.545396 ms | reject, 0.59% slower |
| explicit 2D 10x4 QKV / 10x4 O | 13.525096 ms | reject, 0.44% slower |

The final post-AutoFix Tracy run reports 267 decode device ops across three
trace replays, 2.715058 ms aggregate device time (0.905019 ms/replay), 253.808
us aggregate op-to-op gaps, and 0.999758 ms/replay profiler wall time. Prefill
reports 126 device ops over two repeats and 7.602124 ms aggregate device time. There are
zero host ops in either signposted window.

Aggregated decode device time is concentrated in sparse expert matmuls (38.36%),
dense attention/router matmuls (17.95%), and norms (5.09%). The selected
advisor QKV/O rows run on 80/90 cores at 77/62 us per replay; active-expert
gate/up/down rows run at about 117/116/114 us on 90 cores with BFP8/LoFi.

### Roofline

A conservative weight/cache traffic model for one decode token is 159,364,864
bytes: 29.491 MB QKV, 23.593 MB O, 0.184 MB router, 105.754 MB for four active
gate/up/down experts, 0.069 MB expert bias, 0.012 MB norms, and 0.262 MB KV
reads. Against Blackhole's 512 GB/s aggregate DRAM figure, the theoretical
bandwidth floor is 0.311259 ms/token. The profiled wall time is 3.21x that
floor (31.1% bandwidth equivalent); the device-only time is 34.4%. The
independent `tt-perf-report` model reports 33.4% / 171 GB/s, consistent with
this lower-bound calculation.

The remaining block-to-width QKV reshard and L1/DRAM sparse-module boundary
are intentional producer/consumer contracts. Tracy shows no host conversion,
host fallback, explicit untilize/tilize, or redundant same-layout reshard in
the measured source path.

## Rejected attention experiments

DRAM-sharded QKV/O originally looked numerically broken because bias had been
fused into a sharded result. Matching the common Attention1D contract by adding
bias separately restores ordinary-position PCC: BF16/HiFi4 reaches
0.9967--0.9977 and BFP8/HiFi2 reaches 0.9957--0.9968. Corrected timings are
1.042835 ms and 0.957894 ms respectively; BFP4/LoFi remains invalid at about
0.75 PCC.

Neither corrected candidate can be selected. BF16/HiFi4 passes the full
attention boundary but sliding attention fails eagerly and under trace at
position 130 (PCC 0.91306). BFP8/HiFi2 fails full attention at position 129
and sliding at 130. Exact-mask probes, automatic and K=64 SDPA programs, DRAM
query placement, and earlier deallocation do not repair it. The selected
advisor-seeded BF16 attention chain is both correct and faster.

## Context and scope

`doc/context_contract.json` continues to advertise 21,248 tokens. Cache dtype,
shape, and linear indexing are unchanged, so capacity is not reduced. The
public batch contract remains the emitted batch of one. Multichip, full-model,
generation, sampling, and vLLM work are intentionally outside this stage.

## Reproduction and artifacts

```bash
# Correctness
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/test_results.xml

# Final timing
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_decoder_perf

# Watcher, deliberately separate from profiler
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k real_weight_optimized_prefill_decode_pcc
```

Primary artifacts are `test_results.xml`, `logs/final_correctness_after_advisor_autofix.log`,
`logs/perf_final_after_advisor_autofix_100.log`, `logs/watcher_stress.log`, the
`tracy/sliding_attention` reports, and mandatory advisor
`shard_advise/{report.json,final_ir.mlir}`. The complete command ledger,
candidate evidence, advisor dispositions, AutoFix record, and optimization
checklist are in `work_log.md`.

The advisor report's optional `decision_trace` provenance points to the large
greedy-phase debug trace generated and consumed during the run, then pruned.
The retained hard-gate artifacts are the unchanged `report.json` and
authoritative `final_ir.mlir`.
