# Stage: 04-approx-gelu

- source commit: `56e4f367883`
- kernel time (mean of replays 2-10): **21.215 ms**
- change from the previous stage: **-2.321 ms**
- device ops: **345**
- note: Piecewise-linear gelu approximation in c_fc.

## What this change was

**Approximate gelu in c_fc.** `c_fc` measured 235 us against `c_proj`'s 125 for the **identical 4.83 GFLOP**. Since the
arithmetic is the same, the gap cannot be the matmul — it is the fused gelu's SFPU cost, which a
FLOP count does not credit.

`APPROXIMATION_MODE` selects a 6-segment piecewise-linear fit (`ckernel_sfpu_gelu.h:209-215`)
over a MaxULP-1 rational erf (`:301-311`), taking `c_fc` to 139 us.

The fit is coarse: **it costs the tower 0.0176 of PCC on its own**, 0.99453 → 0.97697, the
largest single accuracy price in the tower by an order of magnitude. Every other step moved PCC by
at most 0.0055. **If accuracy headroom is ever needed, this is the first thing to give back.**

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 113.43 | 11.230 | 52.9 |
| BinaryNgDeviceOperation | 124 | 28.76 | 3.567 | 16.8 |
| SDPAOperation | 24 | 111.74 | 2.682 | 12.6 |
| LayerNormDeviceOperation | 49 | 32.44 | 1.590 | 7.5 |
| NlpCreateHeadsDeviceOperation | 24 | 63.60 | 1.526 | 7.2 |
| NLPConcatHeadsDeviceOperation | 24 | 20.92 | 0.502 | 2.4 |
| UnaryDeviceOperation | 1 | 123.25 | 0.123 | 0.6 |

## Matmul instances by shape

| shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---:|---:|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 25 | 140.1 | 3.501 | 64 | 27.7 | 24.1 | HiFi2 |
| 576 x 1024 x 3072 | 24 | 128.4 | 3.082 | 48 | 28.6 | 29.8 | HiFi2 |
| 576 x 4096 x 1024 | 24 | 125.2 | 3.006 | 48 | 39.1 | 26.5 | HiFi2 |
| 576 x 1024 x 1024 | 24 | 46.2 | 1.109 | 48 | 26.5 | 33.5 | HiFi2 |
| 576 x 4096 x 4096 | 1 | 489.0 | 0.489 | 48 | 80.1 | 30.5 | HiFi4 |
| 576 x 768 x 1024 | 1 | 42.8 | 0.043 | 48 | 21.4 | 29.5 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
