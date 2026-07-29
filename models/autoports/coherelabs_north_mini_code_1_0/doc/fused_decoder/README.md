# North-Mini-Code-1.0 fused decoder

Status: Stage-02 implementation and hard gates complete; independent review
returned `clean-pass`.

## Delivered path and contract

`tt/fused_decoder.py` preserves the completed functional decoder's prefill and
decode APIs, tensor shapes, deterministic traced replay, BF16 paged K/V cache,
page size 32, and batch range through 32. All layer kinds retain the
500000-token contract at batch 1, and dense layers retain it at batch 32.
Sparse layers at batch 32 have a measured hard p300c DRAM limit of 496928
tokens; the next page (496960) has 81536 bytes less free per bank than required.
`doc/context_contract.json` records this narrowly scoped physical reduction.

The default fused runtime uses:

- one packed dense gate/up projection in prefill and decode; unrestricted
  prefill uses two device slices while decode uses the faster device split;
- one packed expert gate/up batched matmul for both sparse layer kinds;
- SiLU fused into the consuming binary multiply for every SwiGLU;
- metadata-only final reshape for batch-1 decode, while serving batch 32 keeps
  the faster transpose geometry.
- exclusive packed-weight ownership: the fused loader never uploads unused
  separate gate/up device tensors; caches are reserved before largest-first
  weight placement to avoid allocator fragmentation.

The inherited attention path already consists of dedicated TTNN operations:
packed QKV projection, QKV/head creation, RoPE, SDPA or paged decode SDPA,
paged cache fill/update, and head concatenation. The measured runtime has no
Torch conversion or host fallback.

## Correctness

```bash
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py \
  -s --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/correctness.xml
```

Result: **11 passed in 42.61 seconds**.

| Coverage | Result |
|---|---:|
| dense prefill, logical length 33 | PCC 0.999722 |
| dense prefill, logical length 65 | PCC 0.999715 |
| dense traced decode | PCC 0.999847 |
| sliding/RoPE/MoE traced decode, batch 32, nonzero experts | PCC 0.998193 |
| full/no-RoPE/MoE traced decode, batch 1, nonzero experts | PCC 0.999823 |
| packed sliding/full MoE prefill, nonzero experts | PCC 0.999821 / 0.999817 |
| paged cache positions 5/17/31/63 | K/V PCC at least 0.999942 |
| repeated identical decode | bitwise equal |

Tests instantiate `FusedDecoder`, assert its dense, sparse, and attention
overrides, and exercise the packed default rather than a functional fallback.
Lengths 33 and 65 prove the public API has no chunk-alignment restriction.

## Warmed performance

Blackhole p300c, 1x1 mesh, synthetic BF16 weights, sequence 128, five warmups
and 50 samples. Decode is complete-forward trace replay. Exact samples and
variant metadata are in `perf/final_*.json`; `performance_matrix.csv` is the
consolidated table.

| Kind | Phase | Batch | Functional | Final fused | Faster |
|---|---|---:|---:|---:|---:|
| dense/full/RoPE | prefill | 1 | 0.6334 ms | 0.5982 ms | 5.57% |
| dense/full/RoPE | prefill | 32 | 13.7495 ms | 12.5829 ms | 8.49% |
| dense/full/RoPE | traced decode | 1 | 0.3605 ms | 0.3344 ms | 7.24% |
| dense/full/RoPE | traced decode | 32 | 6.7506 ms | 6.0389 ms | 10.54% |
| sliding/RoPE/MoE | prefill | 1 | 14.8380 ms | 10.1323 ms | 31.71% |
| sliding/RoPE/MoE | prefill | 32 | 145.9959 ms | 122.0536 ms | 16.40% |
| sliding/RoPE/MoE | traced decode | 1 | 9.5320 ms | 6.7039 ms | 29.67% |
| sliding/RoPE/MoE | traced decode | 32 | 11.2394 ms | 8.4105 ms | 25.17% |
| full/no-RoPE/MoE | prefill | 1 | 14.7575 ms | 10.1504 ms | 31.22% |
| full/no-RoPE/MoE | prefill | 32 | 145.5648 ms | 121.4943 ms | 16.54% |
| full/no-RoPE/MoE | traced decode | 1 | 9.5376 ms | 6.6989 ms | 29.76% |
| full/no-RoPE/MoE | traced decode | 32 | 11.2340 ms | 8.4166 ms | 25.08% |

Every final case beats the best correct functional traced baseline. Fifty
repeated samples per point plus the repeated-cache test provide stress evidence.
Older candidate JSONs that are marginally faster are shorter repeats of the
identical retained default, not different configurations; the largest such
delta is 0.22%. Distinct configurations were selected on reproduced
end-to-end latency, and the table reports the authoritative final 50-sample
repeat.

## Profiler and movement audit

Final signpost-filtered `tt-perf-report` CSV/summary artifacts:

- `tracy/final_dense_prefill_b1/`;
- `tracy/final_dense_decode_b1/`;
- `tracy/final_dense_decode_b32/`;
- `tracy/final_sparse_prefill_b1/`;
- `tracy/final_sparse_decode_b1/`.

Dense measured prefill/decode contains no from/to-Torch, tilize/untilize,
reshard, or host-fallback operation. Sparse routing requires row-major
TopK/scatter indices and the batched expert matmul requires TILE input; the
report therefore shows the unavoidable router/expert boundary conversions.
They are not redundant round trips. Dedicated sparse alternatives were tested:
`sparse_matmul` failed serving-batch correctness (PCC 0.938366), and
`moe_compute` crashed at one token while its 32-token control requires BF4
weights and a different device-opening contract. Dense decode is
weight-traffic bound (47.6% modeled DRAM roofline at batch 1); sparse decode is
dominated by the packed expert matmul.

The complete dedicated gate family was also source-audited. DeepSeek,
generalized, grouped, hash, and `TTMoEGate` paths normalize scores, group/hash
routing, or require DeepSeek-specific preallocated/sharded state. They cannot
express North's unnormalized global `sigmoid(topk(router_logits))` semantics;
`moe_gate_mm` is only the DeepSeek gate projection and does not replace the
remaining routing sequence. Exact contract conclusions are in `work_log.md`.

## Watcher

The final topology was rerun with the profiler disabled:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/watcher_final \
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py \
  -s --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/watcher_correctness.xml
```

Result: **11 passed in 45.46 seconds**. The 2170-line
`watcher_final/generated/watcher/watcher.log` contains no fatal, assert,
invalid-NoC, overflow, sanitizer, timeout, hang, or stuck-waypoint signature.

## Limitation

TTNN's packed TILE `split` specialization does not compile for dense prefill at
logical length 65 because its reader/writer kernels reference undeclared
`single_tile_size_bytes`. The accepted packed path uses two device slices,
preserves all logical lengths, and is faster than separate projections.

For sparse layers at batch 32, finite traced decode passed at context 496928
for both representative layer kinds. Context 496960 is the first failing
page-aligned value: total free and largest free block are both 100581760 bytes
per bank, while the packed weight requires 100663296. The exact physical limit
is preserved in `sparse_batch32_context496960_oom.txt`. Cache remains BF16, and
500000 remains valid for sparse batch 1 and dense batch 32.
