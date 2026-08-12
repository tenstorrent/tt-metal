# Stage: 13-asymmetric-sdpa-chunks

- source commit: `d9b9a3ba0d2`
- kernel time (mean of replays 2-10): **14.883 ms**
- change from the previous stage: **-0.190 ms**
- device ops: **271**
- note: k_chunk over the whole key sequence.

## What this change was

**Asymmetric SDPA chunks.** `k_chunk` covering the whole key sequence gives each q block **one** inner iteration instead of
three, so the softmax updates its running max and sum once instead of three times. **103 → 94.6 us
per op, and PCC rose** on the shorter reduction chain.

`q_chunk` stays at 192 because it sets the parallelism, and 192 is where two curves cross: 6
blocks of 96 fill all 64 cores but every block re-reads the whole of K and V, while 2 blocks of
288 leave 32 cores idle. Numerics do not depend on `q_chunk` once there is a single k iteration.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 83.77 | +0.04 | 8.294 | 55.7 |
| SDPAOperation | 24 | +0 | 94.62 | -7.98 | 2.271 | 15.3 |
| LayerNormDeviceOperation | 49 | +0 | 33.06 | +0.39 | 1.620 | 10.9 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 50.98 | -0.90 | 1.223 | 8.2 |
| BinaryNgDeviceOperation | 50 | +0 | 18.64 | +0.07 | 0.932 | 6.3 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.60 | -0.18 | 0.422 | 2.8 |
| UnaryDeviceOperation | 1 | +0 | 124.60 | +1.07 | 0.125 | 0.8 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 123.1 | +0.1 | 2.955 | 64 | 29.8 | 15.2 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 87.0 | -0.1 | 2.087 | 48 | 56.3 | 19.1 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 83.2 | +0.3 | 1.997 | 48 | 44.1 | 25.4 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 31.5 | -0.3 | 0.757 | 48 | 38.8 | 24.5 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 314.6 | +2.0 | 0.315 | 48 | 62.3 | 28.9 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 139.0 | +0.2 | 0.139 | 48 | 35.2 | 25.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.8 | +0.3 | 0.044 | 48 | 20.9 | 28.8 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
