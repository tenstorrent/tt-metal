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

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 59.54 | -0.01 | 5.895 | 51.0 |
| SDPAOperation | 24 | +0 | 94.97 | -0.02 | 2.279 | 19.7 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.14 | -2.73 | 1.155 | 10.0 |
| LayerNormDeviceOperation | 49 | +0 | 19.52 | +0.00 | 0.957 | 8.3 |
| ShardedToInterleavedDeviceOperation | 72 | +0 | 7.32 | -1.51 | 0.527 | 4.6 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.84 | -0.06 | 0.428 | 3.7 |
| BinaryNgDeviceOperation | 50 | +0 | 3.93 | +0.06 | 0.196 | 1.7 |
| UnaryDeviceOperation | 1 | +0 | 122.79 | -0.82 | 0.123 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.83 | +0.21 | 0.008 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 80.7 | +0.1 | 1.937 | 64 | 45.5 | 20.6 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 66.3 | +0.0 | 1.592 | 48 | 73.9 | 22.0 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 55.4 | +0.0 | 1.330 | 48 | 66.3 | 23.4 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 21.6 | -0.1 | 0.519 | 48 | 56.6 | 26.3 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.2 | -0.3 | 0.312 | 48 | 62.7 | 29.2 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 161.7 | -0.8 | 0.162 | 48 | 30.3 | 19.1 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 42.5 | -1.2 | 0.042 | 48 | 21.6 | 29.7 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
