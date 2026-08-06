# Stage: 00-baseline-unoptimized

- source commit: `f7f7d7cd87f`
- kernel time (mean of replays 2-10): **29.517 ms**
- device ops: **393**
- note: Baseline compute path with today's trace plumbing; same harness as every later stage (warm run, trace on, 10 replays).

## Kernel time by op code, one replay

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | 128.14 | 12.686 | 43.0 |
| BinaryNgDeviceOperation | 148 | 38.93 | 5.762 | 19.5 |
| SDPAOperation | 24 | 182.60 | 4.382 | 14.9 |
| UnaryDeviceOperation | 25 | 122.57 | 3.064 | 10.4 |
| LayerNormDeviceOperation | 49 | 32.26 | 1.581 | 5.4 |
| NlpCreateHeadsDeviceOperation | 24 | 63.57 | 1.526 | 5.2 |
| NLPConcatHeadsDeviceOperation | 24 | 20.35 | 0.488 | 1.7 |

## Matmul instances by shape

| layer | shape | inst | us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc + aligner fc1 | 576 x 1024 x 4096 | 25 | 179.2 | 4.479 | 48 | 54.7 | 27.7 | HiFi4 |
| attn qkv | 576 x 1024 x 3072 | 24 | 138.5 | 3.325 | 48 | 53.0 | 27.6 | HiFi4 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 130.3 | 3.128 | 48 | 75.1 | 38.1 | HiFi4 |
| attn wo | 576 x 1024 x 1024 | 24 | 50.8 | 1.220 | 48 | 48.2 | 30.5 | HiFi4 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 490.9 | 0.491 | 48 | 79.8 | 30.4 | HiFi4 |
| patch embed | 576 x 768 x 1024 | 1 | 43.4 | 0.043 | 48 | 21.1 | 29.1 | HiFi2 |

**The first row is two projections, not one.** `c_fc` runs 24 times per pass and the aligner's
`fc1` once, and at this stage they share both the shape (576 x 1024 x 4096) and the math fidelity.
Rows are grouped by exactly those two, so nothing in this profile separates them: `us each` is the
average over all 25 instances and belongs to neither. They appear apart wherever their fidelities
differ — changes 1-6 and 25 onward.

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
