# Qwen2.5-Coder-32B-Instruct optimized decoder

This stage provides an independent, optimized one-layer TTNN decoder in
`tt/optimized_decoder.py`. It preserves the functional decoder's dense
Qwen2 semantics, arbitrary positive logical prefill lengths, single-token
decode, and in-place paged KV-cache updates. It does not call the functional
decoder and contains no full-model, multichip, generator, LM-head, sampling,
or vLLM work.

## Selected runtime

The default is phase- and batch-aware:

- Batch 32 uses `advisor_packed_bfp8_hifi2_1d`, the layout/program family
  derived from the mandatory shard-advisor run on the rewritten packed
  attention+MLP graph.
- Batch 1-31 uses `packed_mlp_bfp8_hifi2_dram_gate40c`, the fastest correct
  DRAM-sharded decode family measured for the smaller-batch geometry.
- Prefill stays DRAM-interleaved with explicit large 2D matmul programs and
  accepts non-aligned logical sequence lengths; padding is internal.

The selected policy uses BFP8 weights, BFP8 KV caches, BF16 activations and
outputs, HiFi2 matmuls, fused QKV, packed gate/up, fused SiLU+multiply,
fused paged cache update, optimized SDPA, and sharded L1 decode activations.
There are no `torch`, `from_torch`, `to_torch`, tilize/untilize, or host
fallback calls in either runtime forward.

## Correctness and performance

Layer 32 represents the single meaningful decoder-layer kind: all 64 source
layers are dense `Qwen2DecoderLayer` instances with the same topology.

| Check | Functional bar | Selected result |
| --- | ---: | ---: |
| real-weight prefill, batch 1 | 0.998781 PCC | 0.999201 PCC |
| real-weight decode, batch 1 | 0.998929 PCC | 0.999280-0.999336 PCC across three repeated steps |
| real-weight prefill, batch 32 user 0 | 0.998781 PCC | 0.999214 PCC |
| real-weight decode, batch 32 user 0 | 0.998929 PCC | 0.999576 PCC |
| non-aligned synthetic prefill | same functional bar | 0.999752 at seq 17; 0.999847 at seq 33 |

On one Blackhole p300c device, batch 32, representative layer 32, 50 warmed
repetitions:

| Path | Functional | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| prefill, seq 17 | 83.273 ms | 9.997 ms | 8.33x |
| traced decode, mean | 82.374 ms | 1.941 ms | 42.44x |
| traced decode, minimum | 82.341 ms | 1.935 ms | 42.56x |

The final decode also beats the strongest earlier correct optimized candidate
(2.291 ms) by 15.26%. Trace replay is bitwise stable for an unchanged device
input and changes when the traced input is refreshed.

The batch-aware default is also a measured improvement at batch 1:

| Path | Functional | Optimized 40-core packed | Improvement |
| --- | ---: | ---: | ---: |
| prefill, seq 17 | 3.117 ms | 2.949 ms | 1.06x |
| traced decode, mean | 2.903 ms | 2.150 ms | 1.35x |
| traced decode, minimum | 2.895 ms | 2.142 ms | 1.35x |

## Artifacts

- `work_log.md`: topology audit, commands, candidate tables, PCC/performance,
  profiler conclusions, checklist, and stage provenance.
- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: mandatory
  advisor outputs for the rewritten dense packed graph.
- `shard_advise/report.txt`: human-readable advisor report.
- `tracy/layer32/`: compact prefill/decode `tt-perf-report` CSV, text, and
  stacked summaries. Raw Tracy dumps are intentionally not retained.
- `validation/`: retained watcher-enabled correctness/PCC output, clean device
  watcher dump, candidate PCC/geometry evidence, and paired batch-1/batch-32
  50-repetition functional/candidate/final timing logs. Its README maps each
  artifact to the command and decision it supports.
- `AUTOTRIAGE.md` and `triage/`: summary plus a losslessly compressed raw
  `tt-triage` capture from the trace-capture stall and its source-backed
  correction.

`doc/context_contract.json` remains unchanged. The optimized stage adds no
public sequence-alignment restriction and uses no more activation capacity
than the functional path, so it does not reduce the measured batch-32,
one-device 3,999-token contract.

## Test commands

Set the real checkpoint snapshot and run the default optimized suite:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<snapshot> \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py -q -s
```

Run the watcher separately from profiling:

```bash
TT_METAL_WATCHER=10 \
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<snapshot> \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py -q -s
```

The work log contains the opt-in candidate and performance environment
variables. The test defaults exercise the selected optimized implementation;
the only skips are the deliberately opt-in candidate sweep and serialized
performance test.
