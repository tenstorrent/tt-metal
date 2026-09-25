# pplx-embed-v1-4B Blackhole P150 reference numbers

Reference numbers for the ISL 512 batch sweep on one Blackhole P150 of a Galaxy (12×10 = 120 worker
cores). Record new numbers against these before and after a change. Qwen3-Embedding-4B runs the same
stack and lands within 1% of every row (`HF_MODEL=Qwen/Qwen3-Embedding-4B`).

## How to reproduce

Wall time, per batch (30 iterations; the sustained figure is the median of iterations 15–29):

```
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py --iterations 30
bash models/demos/blackhole/pplx_embed_4b/perf_tools/sustained_run.sh 8 0 30 bs8     # <batch> <chip> <iters> <tag>; adds tt-smi clock/power sampling
```

Per-op device time, per batch:

```
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  python -m tracy -p -r -v -m pytest models/demos/blackhole/pplx_embed_4b/tests/perf/new_perf_bs8_isl512.py -sv
```

Run one Tracy capture at a time (concurrent captures collide on the same port) and keep the 40000 program
budget, or the batched-shape ops get no device data.

## Reading the two numbers

Wall time from the trace replay is the number that counts. The Tracy figures below sum
`DEVICE KERNEL DURATION` over the ops between the `start` and `stop` signposts, taken in file order
(ops replayed from a trace keep their capture-time host timestamps, so sorting by timestamp empties the
window). Kernel cycles are converted at the nominal 1.35 GHz, while a sustained run settles at ≈1.1–1.3 GHz,
so the kernel sums read below the wall time at the large batches. Use them to rank ops, not as a latency.

## Wall time, trace replay (sustained)

| Batch | Sustained latency | Embeddings/s | Tokens/s | Tokens/s, 32 cards |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.6 ms | 56.8 | 29,091 | 0.93 M |
| 8 | 120.9 ms | 66.2 | 33,879 | 1.08 M |
| 16 | 227.6 ms | 70.3 | 35,993 | 1.15 M |
| 32 | 446.4 ms | 71.7 | 36,703 | 1.17 M |

## Device kernel time, per op

36 decoder layers. 180 matmuls is 5 per layer: QKV, attention output, MLP gate, up and down (at bs 8 and 16
the gate and up projections run as one fused SwiGLU matmul, 144 calls). The model-local kernels are the
fused head split + Q/K RMSNorm + RoPE (36 calls), the fused residual add + RMSNorm (71 calls at bs≥8) and the
SwiGLU product (36 calls at bs32; a binary op at bs1; inside the matmul at bs 8 and 16).

### B1/S512 — 509 ops, kernel sum 16,647.5 us

| Op | Calls | Total us | % |
| --- | ---: | ---: | ---: |
| Matmul | 180 | 10,060.3 | 60.4 |
| SwiGLU product (binary op) | 36 | 1,877.5 | 11.3 |
| SDPA | 36 | 1,769.6 | 10.6 |
| Fused heads (QKV split + Q/K RMSNorm + RoPE) | 36 | 1,599.1 | 9.6 |
| LayerNorm | 73 | 602.1 | 3.6 |
| Residual add (binary op) | 72 | 500.7 | 3.0 |
| ShardedToInterleaved | 72 | 199.4 | 1.2 |
| Embeddings | 1 | 30.0 | 0.2 |
| InterleavedToSharded | 1 | 3.9 | 0.0 |
| UntilizeCodegen | 1 | 3.1 | 0.0 |
| Slice | 1 | 1.7 | 0.0 |

### B8/S512 — 293 ops, kernel sum 106,632.6 us

| Op | Calls | Total us | % |
| --- | ---: | ---: | ---: |
| Matmul | 144 | 82,721.8 | 77.6 |
| Fused residual add + RMSNorm | 71 | 8,998.2 | 8.4 |
| SDPA | 36 | 8,188.9 | 7.7 |
| Fused heads (QKV split + Q/K RMSNorm + RoPE) | 36 | 6,371.0 | 6.0 |
| LayerNorm | 2 | 167.9 | 0.2 |
| Embeddings | 1 | 103.6 | 0.1 |
| Residual add (binary op) | 1 | 76.4 | 0.1 |
| UntilizeCodegen | 1 | 3.2 | 0.0 |
| Slice | 1 | 1.7 | 0.0 |

### B16/S512 — 293 ops, kernel sum 199,829.1 us

| Op | Calls | Total us | % |
| --- | ---: | ---: | ---: |
| Matmul | 144 | 155,043.0 | 77.6 |
| SDPA | 36 | 16,849.9 | 8.4 |
| Fused residual add + RMSNorm | 71 | 15,870.5 | 7.9 |
| Fused heads (QKV split + Q/K RMSNorm + RoPE) | 36 | 11,447.0 | 5.7 |
| LayerNorm | 2 | 254.6 | 0.1 |
| Embeddings | 1 | 205.5 | 0.1 |
| Residual add (binary op) | 1 | 153.6 | 0.1 |
| UntilizeCodegen | 1 | 3.2 | 0.0 |
| Slice | 1 | 1.7 | 0.0 |

### B32/S512 — 365 ops, kernel sum 365,317.7 us

| Op | Calls | Total us | % |
| --- | ---: | ---: | ---: |
| Matmul | 180 | 236,837.9 | 64.8 |
| SwiGLU product (silu_mul) | 36 | 49,036.8 | 13.4 |
| SDPA | 36 | 29,103.6 | 8.0 |
| Fused residual add + RMSNorm | 71 | 28,340.9 | 7.8 |
| Fused heads (QKV split + Q/K RMSNorm + RoPE) | 36 | 20,869.3 | 5.7 |
| LayerNorm | 2 | 441.2 | 0.1 |
| Embeddings | 1 | 408.1 | 0.1 |
| Residual add (binary op) | 1 | 275.2 | 0.1 |
| UntilizeCodegen | 1 | 3.2 | 0.0 |
| Slice | 1 | 1.5 | 0.0 |

## Where the time goes

| Batch | Matmul | SDPA | Fused heads | Norm + residual | SwiGLU product | Matmul + SDPA | Kernel sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.4% | 10.6% | 9.6% | 6.6% | 11.3% | 71.1% | 16.6 ms |
| 8 | 77.6% | 7.7% | 6.0% | 8.7% | 0.0% | 85.3% | 106.6 ms |
| 16 | 77.6% | 8.4% | 5.7% | 8.1% | 0.0% | 86.0% | 199.8 ms |
| 32 | 64.8% | 8.0% | 5.7% | 8.0% | 13.4% | 72.8% | 365.3 ms |

Matmul and SDPA hold 70–75% of kernel time at every batch, so they carry any large win. At bs1 the
remaining share is the SwiGLU product and the fused heads kernel; at bs32 it is the SwiGLU product
(13%) and the two fused kernels (14%).
