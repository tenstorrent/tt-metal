# North-Mini-Code-1.0 optimized decoder

Status: complete; exact-final hardware/watcher validation and independent
stage-review clean-pass recorded.

This stage starts from `FusedDecoder`. It preserves packed QKV, packed
gate/up, paged KV-cache, arbitrary logical sequence lengths, prefill
chunking, and all representative layer kinds. The optimized default adds:

- A dense layer-0 batch-1 decode path with BFP8 DRAM-width-sharded weights,
  matching L1-width-sharded activations, sharded RMSNorm/residuals, BFP8/LoFi
  decode math, and the largest legal K block for each matmul.
- A batch-1 MoE path that carries a guaranteed exact-count top-8 mask through
  gate/up and down. Routing scores remain separate and are applied once after
  down projection. The fused predecessor evaluated all 128 down experts. Its
  matched 32x32 sparse activation chain remains in L1 through down input.
- Legal 24/32-core prefill programs with K-block 8 for the dominant packed
  all-expert gate/up and down projections.

Batch 32 was explicitly out of scope. It was not measured, swept, or used as
a gate, and the existing context contract was not reduced.

## Selected policy

| Component | Selected configuration |
|---|---|
| Dense decode attention/MLP | BFP8 weights, LoFi, DRAM-width-sharded weights, L1-width-sharded input/output |
| MoE attention | BFP8 weights, HiFi2 |
| Router/expert | BFP8 weights; router HiFi2, experts LoFi |
| Activations, norm, cache, output | BF16 |
| Dense K blocks | QKV=8, O=16, gate/up=8, down=12 on the fixed 8-bank DRAM grid |
| Dense residual | 8-core L1 width shard, sharded RMSNorm and adds |
| Sparse gate/up | exact active 8/128, 24 cores, `in0_block_w=16`, L1 output |
| Sparse activation/down | matched 32x32 tiles; SwiGLU, reshape, transpose, down input, and output in L1; down uses 32 cores, `in0_block_w=24` |
| Router output | layer 1 L1; layer 4 DRAM to preserve near-boundary real-weight top-8 selection |
| Prefill | fused packed topology; explicit 24/32 cores, gate/up and down `in0_block_w=8` |
| SDPA/cache | default TTNN SDPA selection; BF16 cache |

There is no functional-decoder runtime fallback. Prefill and multi-token MoE
use the fused packed path; batch-1 MoE uses the active-expert path. Runtime
methods contain no Torch, `from_torch`, or `to_torch`.

## Performance

Wall results are warmed, traced batch-1 decode (50 samples) or warmed batch-1
prefill at logical sequence 128 (20 samples).

| Layer kind | Fused decode | Optimized decode | Delta | Fused prefill | Optimized prefill | Delta |
|---|---:|---:|---:|---:|---:|---:|
| dense full attention, layer 0 | 0.320013 ms | 0.174017 ms | 45.6% faster | 0.580181 ms | 0.575725 ms | 0.8% faster |
| sliding RoPE MoE, layer 1 | 2.129856 ms | 0.551025 ms | 74.1% faster | 10.079501 ms | 8.124654 ms | 19.4% faster |
| full no-RoPE MoE, layer 4 | 2.131000 ms | 0.530140 ms | 75.1% faster | 10.067067 ms | 8.090731 ms | 19.6% faster |

The primary batch-1 decode target beats the best correct fused baseline for
every layer kind. A paired 100-sample completion audit also confirms the
layer-1 matched L1 chain beats the otherwise-identical no-chain policy,
0.550780 versus 0.561696 ms. Prefill does not regress.

Final Tracy runs were collected separately from watcher. `Profiler wall` is
the end-to-end harness result from the same two-iteration instrumented run;
its difference from the longer uninstrumented headline run includes profiler
overhead and run-to-run variance.

| Path | Headline / profiler wall | Device / op gaps | Ops | DRAM roofline |
|---|---:|---:|---:|---:|
| dense decode | 174.017 / 191.916 us | 153.390 / 23.257 us | 25 | 48.1%, 246 GB/s |
| dense prefill | 575.725 / 603.945 us | 537.843 / 14.789 us | 17 | 16.2%, 83 GB/s |
| sliding MoE decode | 551.025 / 588.410 us | 522.982 / 45.990 us | 50 | 21.9%, 112 GB/s |
| sliding MoE prefill | 8124.654 / 8059.170 us | 7975.713 / 35.829 us | 33 | 20.5%, 105 GB/s |
| full MoE decode | 530.140 / 563.952 us | 497.373 / 47.557 us | 46 | 23.0%, 118 GB/s |
| full MoE prefill | 8090.731 / 7955.774 us | 7864.111 / 30.990 us | 31 | 20.8%, 106 GB/s |

Dense decode moves about 37.75 MB of active BFP8 weights, with a 73.7-us
ideal lower bound at 512 GB/s. MoE decode moves about 56.89 MB, with a
111.1-us bound. Exact-final MoE gzip-compressed raw ops CSV, filtered report
CSV, human table, summary/plot, advice, and capture log are under
`tracy/<moe-kind>/<phase>/`.
The unchanged dense path retains its filtered report, human table,
summary/plot, and capture log.

## Correctness and capacity

The PCC threshold is 0.995.

| Selected-path check | PCC |
|---|---:|
| dense synthetic traced decode | 0.999344 |
| real layer-0 prefill / cache-consuming traced decode | 0.999772 / 0.999259 |
| real layer-1 decode | 0.999721 |
| real layer-4 prefill / cache-consuming traced decode versus functional | 0.999650 / 0.999604 |
| sliding/full active-expert prefill | 0.999516 / 0.999577 |
| sliding/full active-expert traced decode | 0.999821 / 0.999800 |
| adversarial zero routing scores, 20 trace replays | 1.0 |

The exact-count mask scatters ones at the eight top-k indices separately from
sigmoid routing values. Thus `nnz=8` remains true even if all eight BF16
sigmoid scores underflow to zero. The adversarial repeated-trace test proves
that case without a hang.

Coverage also includes non-aligned prefill 1, 31, 32, 33, and 65; sparse
sequence 1025 and layer-4 sequence 33; paged physical cache placement;
deterministic batch-2/batch-4 replay; sliding history at 4097; and layers 0,
1, and 4.

Fresh exact-final optimized decode at position 499999 passed with finite
outputs for all layer kinds: 44.054 ms dense, 0.935 ms sliding MoE, and
44.475 ms full MoE. BF16 KV storage remains 1,024,000,000 bytes at context
500000.

The exact-final normal suite passed 21/21 in 67.51 seconds. The exact-final
watcher suite passed 21/21 in 77.44 seconds, and the gzip-compressed
22,905-line raw watcher log has no watcher fatal/assert, illegal NoC, device
timeout, hang, or stuck marker.

## Reproduce

```bash
python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode decode --batch 1 --layer 0 --iterations 20

python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode prefill --batch 1 --layer 1 --sequence 128 --iterations 20

pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py

TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher_logs_final \
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

See `work_log.md` for the topology audit, candidate matrix, rejected configs,
adapted-path blockers, profiler commands, checklist, review remediation, and
commit SHAs.
