# Stage: 25-aligner-activation-fused

- source commit: `b24ca03be99`
- kernel time (mean of replays 2-10): **9.983 ms**
- change from the previous stage: **-0.033 ms**
- device ops: **295**

## What this change was

**The aligner's activation moved inside the matmul that produces its input.** Standing alone it was
one `UnaryDeviceOperation` reading the whole 576x4096 intermediate back from DRAM for a single SFPU
pass, 123.6 us -- more than the pass itself costs.

Carrying it needs an explicit program config on both aligner projections, because `ttnn.linear`'s
`activation=` is rejected outright once in0 arrives sharded (`matmul.cpp:235`), which it does behind
the last block's residual add. Only the config's `fused_activation` field folds an activation into
the matmul; `activation=` on a config-less linear emits a separate op instead.

The config is not free. It costs **+0.087 ms** across the two aligner matmuls against the 0.124 ms
the standalone op was worth, so the net is -0.041 ms. `in0_block_w` cannot buy that back: the shard
the aligner inherits is four tiles wide and `matmul_device_operation.cpp:1410` requires that width to
divide it, making 4 a ceiling rather than a swept optimum. Unsharding to lift the bound was measured
and discarded -- see DEAD_ENDS.md.

Accuracy is untouched by construction: a fused `GELU` carries APPROXIMATION_MODE false, the same
value a standalone `ttnn.gelu` defaults to (`unary.hpp:339`), so fusing moves where the SFPU pass
runs without changing what it computes. The aligner's own PCC went 0.9999549642 -> 0.9999549895
against its 0.9999 gate.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 54.69 | +0.95 | 5.414 | 54.2 |
| SDPAOperation | 24 | +0 | 66.83 | -0.27 | 1.604 | 16.1 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 48.53 | +0.18 | 1.165 | 11.7 |
| LayerNormDeviceOperation | 49 | +0 | 19.11 | +0.10 | 0.937 | 9.4 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.82 | -0.44 | 0.428 | 4.3 |
| ShardedToInterleavedDeviceOperation | 24 | +0 | 9.57 | +0.02 | 0.230 | 2.3 |
| BinaryNgDeviceOperation | 50 | +0 | 3.93 | -0.01 | 0.196 | 2.0 |
| InterleavedToShardedDeviceOperation | 1 | +0 | 7.63 | +0.16 | 0.008 | 0.1 |
| UnaryDeviceOperation | 0 | -1 | — | gone | 0.000 | 0.0 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_fc | 576 x 1024 x 4096 | 24 | 78.2 | +0.1 | 1.877 | 48 | 35.4 | 18.6 | LoFi |
| mlp c_proj | 576 x 4096 x 1024 | 24 | 55.5 | +0.3 | 1.332 | 48 | 49.9 | 26.3 | LoFi |
| attn qkv | 576 x 1024 x 3072 | 24 | 48.2 | -0.2 | 1.157 | 48 | 43.0 | 22.7 | LoFi |
| attn wo | 576 x 1024 x 1024 | 24 | 18.3 | +0.0 | 0.440 | 48 | 37.8 | 31.1 | LoFi |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 236.1 | +71.4 | 0.236 | 48 | 20.7 | 13.1 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 327.5 | +15.5 | 0.327 | 48 | 59.8 | 27.8 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.4 | +0.9 | 0.044 | 48 | 20.7 | 28.5 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
