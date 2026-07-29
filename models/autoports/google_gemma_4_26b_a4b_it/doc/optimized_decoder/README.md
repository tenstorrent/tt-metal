# Gemma 4 26B A4B optimized decoder

Status: complete pending independent stage review.

This stage owns `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
and this directory. It does not include multichip, full-model, or vLLM work.
The final single-device policy uses BF16 attention/dense weights, BFP8 expert
weights, a local sliding-QKV geometry, an L1 O→norm boundary, load-time packed
dense gate/up, tuned dense-down, fused router scales, and an exact 11x2 expert
grid with independently tuned sparse projections.

The concrete optimized class owns prefill/decode entry points and the changed
dense and sparse implementations. Tests reject a functional alias or an
optimized policy with no selected roles.

## Correctness

| Layer/cache kind | Prefill PCC | Decode PCC | Required |
| --- | ---: | ---: | ---: |
| sliding/shared | 0.998634 | 0.999521 | 0.995 |
| full/natural | 0.998006 | 0.999824 | 0.995 |
| full/shared HMA | 0.998006 | 0.999824 | 0.995 |

All ten logical prefill boundary lengths pass for each attention kind,
including 1, 31, 33, 65/129, 1023, and 1025. Traced decode passes at batch 1
and 32 for both attention kinds, including repeat-replay PCC >= 0.9999.
Advertised-context decode passes at the final HF position. The optimized
policy does not change KV-cache dtype, layout, or capacity, so
`doc/context_contract.json` remains unchanged.

## Performance

Warmed sequence/current-position 1024 measurements on one Blackhole P300:

| Layer | Mode | Batch | Functional host | Optimized host | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| sliding | prefill | 1 | 681.667 ms | 670.522 ms | -1.6% |
| full | prefill | 1 | 682.521 ms | 671.529 ms | -1.6% |
| sliding | prefill | 32 | 21793.675 ms | 21440.032 ms | -1.6% |
| full | prefill | 32 | 21831.195 ms | 21478.793 ms | -1.6% |
| sliding | traced decode | 1 | 3.038 ms | 1.391 ms | -54.2% |
| full | traced decode | 1 | 3.204 ms | 1.572 ms | -50.9% |
| sliding | traced decode | 32 | 68.969 ms | 19.580 ms | -71.6% |
| full | traced decode | 32 | 68.723 ms | 19.392 ms | -71.8% |

The final decode-only Tracy windows report 1.343/1.524 ms summed device
operations for sliding/full attention. The correctness-critical expert gate
fell from 1.177 ms BF16/2-core to 0.093 ms BFP8/22-core. Expert up uses the
exact 11x2 grid and `in0_block_w=11` at batch 1; the rectangular 4x6/8x3
attempts were invalid because 22 active cores cannot map to 24 receivers.
Expert down retains K block 1 after independent block, grid, and placement
sweeps regressed. Compact decode reports are under
`tracy/final_{sliding,full}_attention_batch1/`; raw Tracy logs were not retained.

Prefill profiling first targeted seq-1024, but that window overflowed the
device profiler. The adapted bounded retry uses seq-256 for both layer kinds
and retains 172-op reports under
`tracy/final_prefill_{sliding,full}_attention_batch1/`: summed device time is
168.120/168.469 ms and warmed host time is 168.262/168.761 ms. In a focused
packed-dense prefill sweep, TTNN auto was 168.636 ms; 11x8 2D programs with
K blocks 8 and 11 regressed to 171.481 and 171.703 ms, so auto remains selected.

## Final topology and conversion ledger

The final batch-1 decode topology is RMSNorm → QKV → head creation/normalization
→ rotary/cache update/SDPA → concat/O → residual RMSNorm → packed dense
gate+up/slice/GELU/mul/down → residual RMSNorm → FP32 router/TopK/scatter →
BFP8 sparse gate/up/down → expert reduction → final norm and residual updates.
Sliding QKV uses the selected local `in0_block_w=1` program; full QKV keeps the
correct default geometry because its 10,240-wide output cannot fit the
sliding geometry's 88-core grid. O produces L1-interleaved data for its
consumer. At batch 1, dense gate/up is one load-time-packed projection, dense
down uses `in0_block_w=3`, and sparse gate/up use the exact 11x2 grid with
11-wide K blocks. Sparse down retains K block 1. At batch 32, expert up uses
`in0_block_w=88`; TTNN-auto separate dense layout is retained because packed
and packed-plus-down candidates were within 0.020 ms measurement noise.

The retained profiler reports contain no host operations. Their intentional
device conversions are:

| Boundary | Sliding/full cost | Why it remains |
| --- | ---: | --- |
| sharded ↔ interleaved | 10.248/9.905 us | Head creation, rotary/cache/SDPA, concat, packed slices, and RMSNorm have different memory contracts. The adapted L1-interleaved O→RMSNorm boundary is the fastest correct persistent form. |
| tile ↔ row-major around routing | 12.687/12.792 us | TopK indices/weights and scatter must enter sparse matmul's row-major sparsity contract; the result is re-tilized for device compute. |
| router typecasts | 6.002/6.064 us | The router matmul is intentionally FP32 for correctness, with BF16 outputs for TopK/sparse compute; the final BF16-to-BF16 cast is the sparse-input contract. |
| device copies | 6.966/5.812 us | Materialize inputs at selected consumer boundaries; these are device copies, not host fallback. |

After sparse expert reduction, the profiler shows L1 expert output feeding
RMSNorm directly, followed by DRAM residual adds/norms (rows 344–351 in both
final reports). There is no layout conversion between the expert reduction
and that final normalization boundary.

## Required artifacts

- `shard_advise/report.json` and `shard_advise/final_ir.mlir`
- `tracy/final_sliding_attention_batch1/decode_perf_report.csv`
- `tracy/final_full_attention_batch1/decode_perf_report.csv`
- `tracy/final_prefill_sliding_attention_batch1/prefill_perf_report.csv`
- `tracy/final_prefill_full_attention_batch1/prefill_perf_report.csv`
- `candidate_matrix.json`
- PCC, context, boundary, trace-contract, and host-timing JSON files here
- commands, candidate evidence, and checklist in `work_log.md`

## Limitations

Batch 32 retains TTNN-auto separate dense layouts because the batch-1 shard
geometries do not apply; its speedup comes from BFP8 experts with the selected
K block 88. Dense BFP8 is rejected because
a non-aligned sliding prefill case fell below 0.995. The router matmul remains
FP32, while its two static input scales are fused at load time. The measured
path has no runtime torch, `from_torch`, `to_torch`, or host fallback. The
remaining conversions are the device-side contract boundaries itemized above,
rather than unaccounted host or layout churn.
