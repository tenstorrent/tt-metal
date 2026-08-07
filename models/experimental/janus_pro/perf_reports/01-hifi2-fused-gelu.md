# Stage: 01-hifi2-fused-gelu

- source commit: `16f0bf214c9`
- kernel time (mean of replays 2-10): **26.174 ms**
- change from the previous stage: **-3.327 ms**
- device ops: **345**
- note: HiFi2 across the tower plus the MLP gelu fused into c_fc's matmul. Tower PCC 0.995519.

## What this change was

**HiFi2 across the tower, fused MLP gelu.** Two defaults undone at once. The tower ran every matmul at **HiFi4** — four passes over the
operands' mantissa — and applied gelu as a **separate `ttnn.gelu` op** after `c_fc`.

Fidelity is how many passes the matrix engine makes; peak per core on Wormhole is HiFi4 1.028,
HiFi2 2.056 TFLOPs (`perf_report.py:311-320`). Nothing had *chosen* HiFi4 — it was simply what
the inherited config used.

Folding gelu into `c_fc`'s matmul removes 24 ops outright. It must ride *inside* the program
config: passing `activation=` alongside an explicit config appends a **second** gelu op
(`matmul.cpp`).

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 146.49 | +18.47 | 14.503 | 55.3 |
| SDPAOperation | 24 | +0 | 181.63 | -0.26 | 4.359 | 16.6 |
| BinaryNgDeviceOperation | 124 | -24 | 28.93 | -9.77 | 3.588 | 13.7 |
| LayerNormDeviceOperation | 49 | +0 | 32.37 | +0.16 | 1.586 | 6.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 64.70 | +0.90 | 1.553 | 5.9 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 20.46 | -0.36 | 0.491 | 1.9 |
| UnaryDeviceOperation | 1 | -24 | 123.02 | +0.68 | 0.123 | 0.5 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 273.8 | +94.6 | 6.570 | 48 | 17.9 | 18.1 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 128.7 | -9.4 | 3.090 | 48 | 28.5 | 29.7 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 125.1 | -5.4 | 3.003 | 48 | 39.1 | 39.6 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 46.9 | -3.6 | 1.125 | 48 | 26.1 | 33.0 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 181.1 | +3.7 | 0.181 | 48 | 54.1 | 27.4 | HiFi4 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 490.4 | -0.4 | 0.490 | 48 | 79.9 | 30.4 | HiFi4 |
| patch embed | 576 x 768 x 1024 | 1 | 42.7 | -2.2 | 0.043 | 48 | 21.5 | 29.6 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
