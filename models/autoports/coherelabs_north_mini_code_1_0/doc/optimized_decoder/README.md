# North-Mini-Code-1.0 optimized decoder

This stage starts from `tt/fused_decoder.py` and produces the single-device
`tt/optimized_decoder.py` path.  The selected policy keeps activations,
outputs, and paged KV caches BF16, stores attention/MLP/router weights as
BFP8_B, and uses explicit HiFi2 compute configs. Batch-1 dense decode also
keeps advisor-shaped DRAM-sharded copies of its four weights; prefill and all
other layer/batch paths retain the interleaved copies. It does not enter
multichip, full-model, or serving integration work.

## Result

All times are warmed single-device Blackhole p300c measurements. Decode is
trace replay; prefill includes device synchronization. Fifty samples were
used for the final decode numbers and twenty for prefill.

| path | fused baseline (ms) | optimized (ms) | change |
|---|---:|---:|---:|
| dense decode, batch 1 | 0.335773 | 0.254850 | -24.1% |
| dense decode, batch 32 | 6.026796 | 4.957004 | -17.7% |
| dense prefill 128, batch 1 | 0.596576 | 0.579367 | -2.9% |
| dense prefill 128, batch 32 | 12.568575 | 12.137082 | -3.4% |
| sparse decode, batch 1 | 6.707334 | 6.662622 | -0.7% |

The primary batch-1 decode target beats the best correct fused baseline and
batch 32 also improves. The 14-test optimized suite passes. Representative
PCC is 0.999680 for dense decode, 0.999655--0.999669 for non-aligned dense
prefill, 1.0 for sparse prefill, and 0.998161/0.999764 for traced sparse
decode at layer 1 batch 32/layer 4 batch 1. Physical paged-cache slots are
0.999886--0.999907 PCC and repeated decode is bitwise deterministic.

The final path contains no host tensor conversion or fallback in measured
dense prefill/decode. Sparse routing retains required tilize/untilize
boundaries around row-major top-k/scatter and tiled expert matmuls; these are
not host fallbacks. The path retains the fused QKV and packed gate/up projections,
device SwiGLU, device SDPA, paged cache updates, and trace-safe decode.

## Advisor and profiler

The required `ttnn-advise capture` was run on the BFP8 rewritten dense block.
Its machine-readable output is in `shard_advise/report.json`; the captured IR
is `shard_advise/final_ir.mlir`. It considered and advised all four dense
matmuls for DRAM-sharded weights.

The corrected capture uses the pinned 3072 dense intermediate size. Its QKV,
O, packed gate/up, and down recommendations are respectively `(8,1,2)`,
`(16,1,1)`, `(8,1,2)`, and `(12,1,1)`. All four are applied to the batch-1
dense decode specialization, which is correct at 0.999680 PCC and wins at
0.254850 ms. Earlier wrong-shape hang evidence remains only as an anomaly
record; `AUTODEBUG.md` explicitly leaves its root cause uncertain.

Tracy and advice-enabled `tt-perf-report` artifacts are under `tracy/`.
Batch-1 dense decode reaches 160 GB/s modeled DRAM bandwidth (31.2% roofline);
the report classifies the remaining decode as many small device operations,
so BFP8 weight traffic is the material win. The profiler also confirms the
fused projections remain single matmuls and exposes no host fallback.

## Reproduce

```bash
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py

python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --implementation optimized --candidate bfp8_hifi2 --mode decode \
  --layer 0 --batch 1 --warmups 5 --iterations 50

TT_METAL_WATCHER=10 pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

The exact advisor bootstrap/capture and profiler commands, candidate table,
topology audit, and checklist are in `work_log.md`.
