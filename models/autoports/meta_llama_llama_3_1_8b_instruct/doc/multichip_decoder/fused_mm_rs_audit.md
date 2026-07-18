# Fused matmul plus reduce-scatter audit

Both row-parallel projections were evaluated against the public fused CCL
interfaces. The selected decoder uses DRAM-sharded BFP4 weights; the generic
fused operation requires a 2-D multicast matmul, so the test converts the same
rank-local BFP4 weights to interleaved DRAM and compares fusion against a
separate interleaved matmul plus minimal asynchronous reduce-scatter. Both
sides therefore use identical shapes, BFP4 weights, LoFi math, BF16 output,
two links, ring topology, and an `8x6` matmul grid. The reduce-scattered output
is `[1,1,32,1024]` per rank in both paths.

## Exact-shape measurements

| Projection | TP4-local MM shape | Separate MM + RS | Generic fused MM + RS | Fused penalty | Minimum rank PCC |
| --- | --- | ---: | ---: | ---: | ---: |
| attention O | `32x1024x4096` | 0.052418 ms | 0.066379 ms | 26.6342% | 0.999999940395 |
| MLP down | `32x3584x4096` | 0.098161 ms | 0.112457 ms | 14.5637% | 0.999999940395 |

The generic operation is correct at both material row-parallel boundaries but
slower even before accounting for the distributed normalization and eventual
all-gather needed by a fractured residual stack. It is therefore rejected.
The explicit `allowed_worker_cores` set prevents candidate-only core-grid
auto-population warnings.

Command:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_FUSED_MM_RS_PROBE=1 timeout 300 \
pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k fused_matmul_reduce_scatter_probe
```

The complete output is in `logs/fused_mm_rs_probe.log`; compact results are in
`topology_results.csv`.

## Minimal-strided API disposition

The other public API,
`ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`, is not a safe
Blackhole alternative. The repository's GPT-OSS integration explicitly gates
it off on Blackhole in `models/demos/gpt_oss/tt/attention/operations.py` for
issue `#46181`: its reduce-scatter can read matmul blocks before the producer
completes, yielding nondeterministic values with absolute maxima around
`1e13`. The same source states that it is validated on Wormhole. Because this
machine is Blackhole, launching the known-racy candidate would not be a valid
performance result.

The generic exact-shape measurements and the architecture-specific source
gate cover both exposed fused MM+RS families. The selected decoder retains the
correct, faster separate projection and asynchronous all-reduce topology.
