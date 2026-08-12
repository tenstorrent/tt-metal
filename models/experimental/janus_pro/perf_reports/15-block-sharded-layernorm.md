# Stage: 15-block-sharded-layernorm

- source commit: `33aebf75626`
- kernel time (mean of replays 2-10): **13.831 ms**
- change from the previous stage: **-0.544 ms**
- device ops: **367**
- note: TtJanusProLayerNorm block-sharded on 48 cores instead of the interleaved 18.

## What this change was

**Block-sharded layer norm on 48 cores.** The sibling vision norms — `TtLayerNorm` and qwen3_vl's copy — pin the shard to a single tile-row
(`SHARD_HEIGHT = 32`). That suits decode, but for 576 tokens the shard is **18x too small to hold
the tensor**: 32 cores x 1 tile is 32,768 elements against 589,824. So the sharded path was not
merely slow at this shape, it was **unusable**, and prefill fell through to the interleaved
factory's 18 cores. The profiler showed the symptom — LayerNorm at 32.26 us on 18 cores — and
reading the class explained it.

`TtJanusProLayerNorm` (`tt/janus_pro_layernorm.py`) sizes the shard from the sequence on first
call instead: 18 tile-rows split six ways, 32 width-tiles split eight ways, so 48 cores with
`block_h=3`, `block_w=4`. Nothing is passed in to make that happen — the constructor signature
matches the siblings.

Measured in isolation on the tower's shape:

| grid | cores | us | vs interleaved |
|---|---:|---:|---:|
| interleaved (18 cores) | 18 | 33.09 | — |
| 8x6 | 48 | **19.27** | **−41.8%** |
| 4x6 | 24 | 22.50 | −32.0% |
| 8x3 | 24 | 29.68 | −10.3% |
| 8x2 | 16 | 39.70 | +20.0% |
| 2x6 | 12 | 36.54 | +10.4% |
| 8x1 | 8 | 68.64 | +107.4% |

Two things fall out of that table and both are encoded in the class:

- **8x3 and 4x6 have the same core count and differ by 32%.** A layer norm reduces along *width*,
  so splitting width forces that reduction to cross cores while splitting height does not. The
  grid search therefore maximizes the height split first, then width.
- **Below 24 cores the sharded path loses outright.** Hence the guard: shard only when the height
  split is ≥3 and the grid is ≥24 cores, else fall back to the inherited interleaved path.

**The sharded output has to reach the narrowing, or the change is a net loss.** Per norm, in
measured microseconds:

| | |
|---|---:|
| interleaved norm + typecast | 32.73 + 11.16 = **43.89** |
| shard + sharded norm + narrowing unshard | 7.97 + 19.99 + 6.23 = **34.19** |
| sharded norm returning interleaved, then typecast | 7.97 + 19.99 + 6.23 + 11.16 = **45.35** |

The third row is what happens if the caller does not ask for a sharded return: two conversions are
paid and the typecast survives, for a **net loss of 1.5 us per norm**.

Every knob in `LayerNormShardedMultiCoreProgramConfig` was then swept on the 48-core grid, and
none helps: `subblock_w` 2 measured −0.8% (noise) and 1 measured +1.4%; `inplace=True` is legal
here — the input is a temporary shard, not the caller's tensor — but +1.4%; `legacy_reduction`
+1.0%; `legacy_rsqrt` +8.2% and slightly worse PCC. **`use_welford` is unreachable from Python**:
the sharded factory requires a `recip_tensor` that `ttnn.layer_norm` does not expose
(`layernorm_op_multi_core_sharded.cpp:179`). PCC was identical across every variant that ran.

This is the **best accuracy-per-millisecond trade in the whole change log**: 0.000953 of
end-to-end PCC for 0.555 ms, i.e. 0.0017 per ms, against gelu's 0.0078.

## Kernel time by op code, one replay

| Op | inst | Δ inst | us each | Δ us each | ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MatmulDeviceOperation | 99 | +0 | 72.71 | -0.40 | 7.199 | 52.0 |
| SDPAOperation | 24 | +0 | 94.51 | -0.64 | 2.268 | 16.4 |
| NlpCreateHeadsDeviceOperation | 24 | +0 | 51.01 | +0.25 | 1.224 | 8.9 |
| LayerNormDeviceOperation | 49 | +0 | 19.93 | -12.87 | 0.976 | 7.1 |
| BinaryNgDeviceOperation | 50 | +0 | 18.63 | -0.24 | 0.932 | 6.7 |
| NLPConcatHeadsDeviceOperation | 24 | +0 | 17.63 | -0.35 | 0.423 | 3.1 |
| InterleavedToShardedDeviceOperation | 48 | new | 8.01 | new | 0.384 | 2.8 |
| ShardedToInterleavedDeviceOperation | 48 | new | 6.30 | new | 0.302 | 2.2 |
| UnaryDeviceOperation | 1 | +0 | 123.18 | -0.52 | 0.123 | 0.9 |
| TypecastDeviceOperation | 0 | -48 | — | gone | 0.000 | 0.0 |

## Matmul instances by shape

| layer | shape | inst | us each | Δ us each | ms | cores | FLOPs % | DRAM % | fidelity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp c_proj | 576 x 4096 x 1024 | 24 | 86.4 | -0.2 | 2.074 | 48 | 56.7 | 19.2 | HiFi2 |
| attn qkv | 576 x 1024 x 3072 | 24 | 80.7 | -0.6 | 1.938 | 48 | 45.5 | 23.7 | HiFi2 |
| mlp c_fc | 576 x 1024 x 4096 | 24 | 80.7 | -1.1 | 1.936 | 64 | 45.5 | 20.6 | HiFi2 |
| attn wo | 576 x 1024 x 1024 | 24 | 31.4 | +0.0 | 0.755 | 48 | 38.9 | 24.6 | HiFi2 |
| aligner hidden | 576 x 4096 x 4096 | 1 | 312.8 | +0.0 | 0.313 | 48 | 62.6 | 29.1 | HiFi2 |
| aligner fc1 | 576 x 1024 x 4096 | 1 | 138.5 | -0.6 | 0.139 | 48 | 35.3 | 25.3 | HiFi2 |
| patch embed | 576 x 768 x 1024 | 1 | 44.8 | +2.5 | 0.045 | 48 | 20.5 | 28.2 | HiFi2 |

`FLOPs %` is achieved FLOPs over `peak_per_core(fidelity) x cores`, so **it is not a ranking of how
well a matmul runs**. It rises when an op uses fewer cores and when fidelity goes up, which is why
`cores` and `fidelity` are next to it. See PROFILER_NOTES.md for a worked case where the number moved
14 points while the op's time did not change at all.
