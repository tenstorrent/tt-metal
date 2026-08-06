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

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 115.34 | 11.418 | 59.7 |
| SDPAOperation | 24 | 112.18 | 2.692 | 14.1 |
| LayerNormDeviceOperation | 49 | 32.40 | 1.587 | 8.3 |
| NlpCreateHeadsDeviceOperation | 24 | 63.08 | 1.514 | 7.9 |
| BinaryNgDeviceOperation | 52 | 24.90 | 1.295 | 6.8 |
| NLPConcatHeadsDeviceOperation | 24 | 20.67 | 0.496 | 2.6 |
| UnaryDeviceOperation | 1 | 123.17 | 0.123 | 0.6 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 139.9 | 3.497 | 64 | 27.7 | 24.2 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 133.6 | 3.207 | 48 | 27.5 | 28.6 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 126.1 | 3.026 | 48 | 38.8 | 26.4 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 48.0 | 1.153 | 48 | 25.5 | 32.2 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 491.9 | 0.492 | 48 | 79.6 | 30.3 | HiFi4 |
| 576 x 768 x 1024 | 1 | 43.5 | 0.044 | 48 | 21.1 | 29.0 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
