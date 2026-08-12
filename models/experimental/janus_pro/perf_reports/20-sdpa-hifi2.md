# Stage: 20-sdpa-hifi2

- source commit: `43c86bd2fa3`
- kernel time (mean of replays 2-10): **11.400 ms**
- change from the previous stage: **-0.170 ms**
- device ops: **344**
- note: SDPA dropped from HiFi4 to HiFi2.

## What this change was

**SDPA HiFi4 → HiFi2.** Per-RISC on the tree at the time: BRISC 100.0%, **TRISC0/1/2 all at 98.5%**, NCRISC 26.6%. Unlike
the matmuls — where BRISC == op duration is structural — here the math pipe itself is saturated,
so SDPA is the tower's only genuinely compute-bound op. And it was the last op still at HiFi4
(`model_config.py:832-837`), 64 cycles per tile against HiFi2's 32.

The op fell 94.86 → 87.76 us, **−7.5%, not the ~50% the cycle count suggests**, because softmax
and exp run on the SFPU and do not scale with matrix-engine fidelity.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 59.59 | +0.05 | 5.900 | 51.7 |
| SDPAOperation | 24 | +0 | 88.04 | -6.93 | 2.113 | 18.5 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.22 | +0.08 | 1.157 | 10.1 |
| LayerNormDeviceOperation | 49 | +0 | 19.51 | -0.01 | 0.956 | 8.4 |
| ShardedToInterleavedDeviceOperation | 72 | +0 | 7.36 | +0.04 | 0.530 | 4.6 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.95 | +0.11 | 0.431 | 3.8 |
| BinaryNgDeviceOperation | 50 | +0 | 3.92 | -0.01 | 0.196 | 1.7 |
| UnaryDeviceOperation | 1 | +0 | 123.24 | +0.45 | 0.123 | 1.1 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.24 | -0.59 | 0.007 | 0.1 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 80.8 | +0.1 | 1.939 | 64 | 45.5 | 20.6 | HiFi2 |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 66.4 | +0.1 | 1.593 | 48 | 73.8 | 22.0 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 55.4 | -0.0 | 1.330 | 48 | 66.3 | 23.4 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 21.6 | +0.0 | 0.519 | 48 | 56.6 | 26.3 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.3 | +0.1 | 0.312 | 48 | 62.7 | 29.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 163.6 | +1.9 | 0.164 | 48 | 29.9 | 18.9 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 43.4 | +0.9 | 0.043 | 48 | 21.2 | 29.1 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
