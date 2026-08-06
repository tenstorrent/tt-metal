# Stage: 19-qkv-unshard-to-l1

- source commit: `da1f869da23`
- kernel time (mean of replays 2-10): **11.570 ms**
- change from the previous stage: **-0.167 ms**
- device ops: **344**
- note: qkv unshard targets L1 rather than DRAM.

## What this change was

**qkv unshard into L1 rather than DRAM.** `nlp_create_qkv_heads` **cannot take the shard**: its sharded path requires a grid dividing
`num_q_heads`, so 16 cores at most against qkv's 48. The shard must therefore be undone
explicitly.

Sending that unshard to L1 rather than DRAM is worth 0.16 ms — the conversion drops 8.81 → 7.33 us
and its consumer 50.69 → 48.34. L1 interleaved is a poor *matmul* input (see the interleaved-L1 row in
[`DEAD_ENDS.md`](DEAD_ENDS.md)); **this consumer is not a matmul**, which is why the
usual rule inverts here.

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 59.54 | 5.895 | 51.0 |
| SDPAOperation | 24 | 94.97 | 2.279 | 19.7 |
| NlpCreateHeadsDeviceOperation | 24 | 48.14 | 1.155 | 10.0 |
| LayerNormDeviceOperation | 49 | 19.52 | 0.957 | 8.3 |
| ShardedToInterleavedDeviceOperation | 72 | 7.32 | 0.527 | 4.6 |
| NLPConcatHeadsDeviceOperation | 24 | 17.84 | 0.428 | 3.7 |
| BinaryNgDeviceOperation | 50 | 3.93 | 0.196 | 1.7 |
| UnaryDeviceOperation | 1 | 122.79 | 0.123 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | 7.83 | 0.008 | 0.1 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 84.0 | 2.099 | 64 | 44.9 | 20.5 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 66.3 | 1.592 | 48 | 73.9 | 22.0 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 55.4 | 1.330 | 48 | 66.3 | 23.4 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 21.6 | 0.519 | 48 | 56.6 | 26.3 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 312.2 | 0.312 | 48 | 62.7 | 29.2 | HiFi2 |
| 576 x 768 x 1024 | 1 | 42.5 | 0.042 | 48 | 21.6 | 29.7 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
