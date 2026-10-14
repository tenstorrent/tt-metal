# BGE-M3 Blackhole p150 baseline

Reference numbers for the S512 batch sweep on one Blackhole p150a. Record new
numbers against these before and after a change.

## How to reproduce

Wall time, per batch:

```
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py \
  -k "forward and b1_s512" -s
```

Per-op device time, per batch:

```
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \
  --no-runtime-analysis -v -m pytest \
  "models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py::test_bge_m3_tracy_perf[device_params0-batch1]" -sv
```

Name the exact test id. `-k "batch1"` also selects `batch16`, and the two runs
then land in one CSV.

## Reading the two numbers

Wall time from `execute_trace` is the number that counts. The Tracy figures
below sum `DEVICE KERNEL DURATION` only.

Do not add the op-to-op gaps. A gap is dispatch, not kernel work, and the Tracy
run is untraced so every op pays a dispatch that trace replay removes. This is
why the kernel sum is far above the wall time: 3.7 ms of kernels against 4.3 ms
of wall at B1, and 38.8 ms against 39.0 ms at B16. Use the kernel sums to rank
the ops against each other, never as a latency.

## Wall time, trace replay

| Batch | Avg latency | Best latency | Embeddings/s | Tokens/s |
| ----- | ----------- | ------------ | ------------ | -------- |
| 1     | 4.310 ms    | 4.315 ms     | 232.0        | 118,791  |
| 8     | 23.699 ms   | 23.68 ms     | 337.6        | 172,833  |
| 16    | 38.963 ms   | 38.94 ms     | 410.6        | 210,250  |
| 32    | see below   | —            | —            | —        |

Per-embedding cost falls from 4.31 ms at B1 to 2.96 ms at B8 to 2.44 ms at B16.

## Device kernel time, per op

24 encoder layers. 96 matmuls is 4 per layer: QKV, attention output, MLP wi, MLP
wo. 48 GenericOp calls are the head-split QKV and concat-heads custom ops, 2 per
layer.

### B1/S512 — 350 ops, kernel sum 3,746.8 us

| OP CODE                   | Calls | Total us | %    |
| ------------------------- | ----- | -------- | ---- |
| Matmul                    | 96    | 2,105.4  | 56.2 |
| SDPA                      | 24    | 679.1    | 18.1 |
| LayerNorm                 | 49    | 434.1    | 11.6 |
| GenericOp (custom heads)  | 48    | 139.9    | 3.7  |
| ShardedToInterleaved      | 49    | 98.3     | 2.6  |
| InterleavedToSharded      | 50    | 96.2     | 2.6  |
| BinaryNg                  | 25    | 78.8     | 2.1  |
| TilizeWithValPadding      | 1     | 39.3     | 1.0  |
| Embeddings                | 2     | 28.5     | 0.8  |
| Repeat                    | 1     | 16.1     | 0.4  |
| Tilize                    | 1     | 14.3     | 0.4  |
| UntilizeWithUnpadding     | 1     | 8.1      | 0.2  |
| Typecast                  | 2     | 7.2      | 0.2  |
| Unary                     | 1     | 1.6      | 0.0  |

### B8/S512 — 252 ops, kernel sum 23,573.1 us

| OP CODE                  | Calls | Total us | %    |
| ------------------------ | ----- | -------- | ---- |
| Matmul                   | 96    | 9,545.7  | 40.5 |
| SDPA                     | 24    | 7,317.2  | 31.0 |
| GenericOp (custom heads) | 48    | 3,409.5  | 14.5 |
| LayerNorm                | 49    | 2,297.7  | 9.7  |
| BinaryNg                 | 25    | 733.6    | 3.1  |
| Embeddings               | 2     | 87.8     | 0.4  |
| Repeat                   | 1     | 47.3     | 0.2  |
| Tilize                   | 1     | 44.6     | 0.2  |
| Typecast                 | 2     | 42.7     | 0.2  |
| TilizeWithValPadding     | 1     | 31.4     | 0.1  |
| UntilizeWithUnpadding    | 1     | 9.3      | 0.0  |
| ReshapeView              | 1     | 3.3      | 0.0  |
| Unary                    | 1     | 3.0      | 0.0  |

### B16/S512 — 276 ops, kernel sum 38,828.2 us

| OP CODE                  | Calls | Total us | %    |
| ------------------------ | ----- | -------- | ---- |
| Matmul                   | 96    | 19,318.2 | 49.8 |
| SDPA                     | 24    | 8,252.0  | 21.3 |
| GenericOp (custom heads) | 48    | 4,884.9  | 12.6 |
| LayerNorm                | 49    | 4,473.8  | 11.5 |
| Typecast                 | 26    | 900.9    | 2.3  |
| BinaryNg                 | 25    | 605.7    | 1.6  |
| Embeddings               | 2     | 165.0    | 0.4  |
| Repeat                   | 1     | 95.7     | 0.2  |
| Tilize                   | 1     | 84.9     | 0.2  |
| TilizeWithValPadding     | 1     | 22.5     | 0.1  |
| UntilizeWithUnpadding    | 1     | 10.0     | 0.0  |
| ReshapeView              | 1     | 8.0      | 0.0  |
| Unary                    | 1     | 6.7      | 0.0  |

## Where the time goes

| Batch | Matmul | SDPA | Custom heads | LayerNorm | Matmul + SDPA |
| ----- | ------ | ---- | ------------ | --------- | ------------- |
| 1     | 56.2%  | 18.1% | 3.7%        | 11.6%     | 74.3%         |
| 8     | 40.5%  | 31.0% | 14.5%       | 9.7%      | 71.5%         |
| 16    | 49.8%  | 21.3% | 12.6%       | 11.5%     | 71.1%         |

Matmul and SDPA hold about 71% of kernel time at every batch, so they carry any
large win.

Two shifts between B1 and B8 are worth noting. SDPA grows from 18.1% to 31.0%,
which is the S512 attention cost rising with batch. The custom head ops grow
from 3.7% to 14.5%, so the head-split kernels pay off most at B1 and cost more
as the batch grows.

B1 also runs 98 shard conversions that the larger batches do not, because the
sharded LayerNorm handoff is B1-only. They cost 194.5 us, about 5.2% of B1.

## B32 does not run

B32/S512 fails before the first forward:

```
Statically allocated circular buffers in program 39 clash with L1 buffers
on core range [0-0 - 12-9]. L1 buffer allocated at 1158336 and static
circular buffer region ends at 1290368
```

129 KB over. The failure is in the MLP wi matmul, at `tt/mlp.py:104`.

This is not a regression in this model. `origin/main` fails the same way at the
same addresses, and the BGE config matches the p150 branch line for line. The
circular buffers that upstream matmul allocates grew after this shape was tuned,
so the L1 budget no longer fits.

Disabling the four Blackhole L1 gates in `optimizations.py` lets B32 run at
102.197 ms and 313.1 embeddings/s. That is slower per embedding than B16, and it
throws away the L1 work that the gates carry, so it is a measurement and not a
fix. The p150 branch recorded 65.7 ms for this shape.

Recovering B32 means finding an L1 budget that keeps the sharded handoffs under
the larger upstream buffers. DRAM sharding for the MLP weights is the first
thing to try: nothing in this model is DRAM sharded today, the weights sit in
interleaved DRAM, and a DRAM-sharded matmul holds less L1 per core.

## Environment

- Blackhole p150a, board 0000040331911033, one card
- tt-metal main at `3bc24c6dfb`, branch `gtobarTT/bge_m3_p150_optimizations`
- bfloat8_b weights, `NUM_ITERATIONS = 10`
- 2026-09-21
