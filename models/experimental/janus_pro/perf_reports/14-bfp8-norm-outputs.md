# Stage: 14-bfp8-norm-outputs

- source commit: `9fcbc1edfbf`
- kernel time (mean of replays 2-10): **14.375 ms**
- change from the previous stage: **-0.508 ms**
- device ops: **319**
- note: Layer-norm outputs narrowed to bfloat8_b via typecast.

## What this change was

**bfloat8_b layer-norm outputs.** Each norm output is read exactly once, by the projection that follows it, so it satisfies the
read-once rule. Narrowing it halves what `qkv` and `c_fc` multicast as in0 — the two matmuls that
dominate the tower.

It costs an op per norm, because **`ttnn.layer_norm` has no output-dtype argument**
(`layernorm_nanobind.cpp:182`; the `dtype` field at `:232` is primitive-only and `ttnn.prim` is
not exposed to Python). Narrowing the norm's *input* instead would mean a bfloat8_b residual,
which the encoder's PCC gate rejects — see
[`DEAD_ENDS.md`](DEAD_ENDS.md#bfloat8b-residual-stream). 48 typecasts, 0.564 ms, buying back four times that
in matmul.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 73.11 | -10.66 | 7.238 | 50.3 |
| SDPAOperation | 24 | +0 | 95.15 | +0.53 | 2.284 | 15.9 |
| LayerNormDeviceOperation | 49 | +0 | 32.80 | -0.26 | 1.607 | 11.2 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 50.76 | -0.22 | 1.218 | 8.5 |
| BinaryNgDeviceOperation | 50 | +0 | 18.87 | +0.23 | 0.943 | 6.6 |
| TypecastDeviceOperation | 48 | new | 11.17 | new | 0.536 | 3.7 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.98 | +0.38 | 0.431 | 3.0 |
| UnaryDeviceOperation | 1 | +0 | 123.70 | -0.90 | 0.124 | 0.9 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_proj | 576 x 4096 x 1024 | 24 | 86.6 | -0.4 | 2.078 | 48 | 56.6 | 19.2 | HiFi2 |
| mlp c_fc | 576 x 1024 x 4096 | 24 | 81.8 | -41.3 | 1.963 | 64 | 44.9 | 20.3 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 81.3 | -1.9 | 1.950 | 48 | 45.2 | 23.5 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 31.4 | -0.1 | 0.753 | 48 | 39.0 | 24.7 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.8 | -1.8 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 139.1 | +0.1 | 0.139 | 48 | 35.2 | 25.2 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 42.3 | -1.5 | 0.042 | 48 | 21.7 | 29.9 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
