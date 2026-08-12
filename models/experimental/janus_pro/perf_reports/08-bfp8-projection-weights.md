# Stage: 08-bfp8-projection-weights

- source commit: `0c3409ab2d6`
- kernel time (mean of replays 2-10): **17.571 ms**
- change from the previous stage: **-1.373 ms**
- device ops: **271**
- note: All body projection weights narrowed to bfloat8_b.

## What this change was

**bfloat8_b projection weights.** Weight traffic was 604 MB per pass against a 19.8 ms span, i.e. **10.6% of DRAM peak**, so halving
it "should" have bought nothing. It bought **1.56 ms**.

These matmuls are latency-bound on per-core delivery: bytes on the critical path matter while the
fabric sits idle.

**Trap: aggregate bandwidth utilization does not predict whether cutting bytes helps.** Reading
"10.6% of DRAM peak" as "not bandwidth-bound, so bytes are free" is the natural inference and it
is wrong here.

Who benefited was decided by reuse pattern, not size: `qkv` fell 26 us and `c_proj` 23, while
`c_fc` moved only 4.6 — it runs 1D with in0 multicast, so each core already held just two N-tiles
of weight.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 101.23 | -14.28 | 10.022 | 57.1 |
| SDPAOperation | 24 | +0 | 112.29 | +0.64 | 2.695 | 15.3 |
| LayerNormDeviceOperation | 49 | +0 | 32.55 | +0.11 | 1.595 | 9.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 62.97 | -0.09 | 1.511 | 8.6 |
| BinaryNgDeviceOperation | 50 | +0 | 22.34 | -0.02 | 1.117 | 6.4 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 20.68 | +0.05 | 0.496 | 2.8 |
| UnaryDeviceOperation | 1 | +0 | 124.31 | +0.61 | 0.124 | 0.7 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 134.0 | -4.3 | 3.216 | 64 | 27.4 | 13.9 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 108.9 | -25.3 | 2.614 | 48 | 33.7 | 25.1 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 102.9 | -23.6 | 2.470 | 48 | 47.6 | 18.1 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 43.1 | -4.8 | 1.033 | 48 | 28.5 | 27.5 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 470.2 | -18.4 | 0.470 | 48 | 41.7 | 31.7 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 175.8 | -1.2 | 0.176 | 48 | 27.9 | 28.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.2 | -0.9 | 0.043 | 48 | 21.2 | 29.2 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
