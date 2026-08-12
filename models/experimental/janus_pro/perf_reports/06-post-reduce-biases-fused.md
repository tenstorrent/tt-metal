# Stage: 06-post-reduce-biases-fused

- source commit: `8becfcd193e`
- kernel time (mean of replays 2-10): **19.135 ms**
- change from the previous stage: **-0.902 ms**
- device ops: **273**
- note: wo and c_proj biases folded in, valid on a single device only.

## What this change was

**Post-reduce biases fused, single-device only.** `bo` and `c_proj`'s bias sit **after** the all-reduce. Fusing them would add the bias once per
device. They fuse only when `num_devices == 1`, gated on the same test the reduce itself uses —
correct at every device count, not an N150 shortcut.

Changes 5 and 6 together took elementwise ops from 124 to 52.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 115.34 | +0.39 | 11.418 | 59.7 |
| SDPAOperation | 24 | +0 | 112.18 | +2.08 | 2.692 | 14.1 |
| LayerNormDeviceOperation | 49 | +0 | 32.40 | +0.06 | 1.587 | 8.3 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 63.08 | -0.58 | 1.514 | 7.9 |
| BinaryNgDeviceOperation | 52 | -48 | 24.90 | +2.13 | 1.295 | 6.8 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 20.67 | +0.09 | 0.496 | 2.6 |
| UnaryDeviceOperation | 1 | +0 | 123.17 | -0.12 | 0.123 | 0.6 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 138.3 | -0.3 | 3.320 | 64 | 26.6 | 24.0 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 133.6 | -1.0 | 3.207 | 48 | 27.5 | 28.6 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 126.1 | +1.2 | 3.026 | 48 | 38.8 | 26.4 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 48.0 | +1.7 | 1.153 | 48 | 25.5 | 32.2 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 177.2 | -3.7 | 0.177 | 48 | 55.3 | 28.0 | HiFi4 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 491.9 | +1.3 | 0.492 | 48 | 79.6 | 30.3 | HiFi4 |
| patch embed | 576 x 768 x 1024 | 1 | 43.5 | +0.1 | 0.044 | 48 | 21.1 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
