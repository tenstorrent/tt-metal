# Fused KDA causal conv in the linear-attention prefill chunk

The new tt-metal base carries the KDA (Kimi Delta Attention) op family, which
did not exist on the base this autoport was brought up against. One of those
ops, `ttnn.experimental.kda.qkv_causal_conv1d_silu`, is exactly the depthwise
causal convolution this model's Gated DeltaNet layer hand-rolls: a four-tap
causal conv, SiLU, and the Q/K/V split. The model's `linear_conv_kernel_dim` is
4 and its Q/K/V widths are 2048/2048/6144, so every one of the op's shape
constraints is already met.

This experiment swaps that one block and measures it. Nothing else changes.

## Result

Layer 0 (`linear_attention`), prefill, batch 1, sequence 128 (four 32-token
chunks), synthetic weights, single Blackhole P150 chip.

| | control (`linear_final`) | candidate (`linear_kda_conv`) | delta |
|---|---:|---:|---:|
| Device time, signposted window | 272.774 ms | 268.805 ms | **-3.969 ms (-1.46%)** |
| Dispatched ops in window | 643 | 515 | **-128** |
| End-to-end host latency, median | 274.105 ms | 270.132 ms | -3.973 ms |
| PCC vs the HF layer (bar 0.995) | 0.9999958 | 0.9999956 | -0.0000002 |

The candidate policy is `dataclasses.replace(_LINEAR_FINAL,
linear_kda_conv=True)`, so the two arms differ in exactly one knob.

## Where the time went

| Op | base us | base n | KDA us | KDA n | Δ us | Δ n |
|---|---:|---:|---:|---:|---:|---:|
| `SliceDeviceOperation` | 13193.4 | 136 | 11307.3 | 96 | -1886.1 | -40 |
| `UntilizeWithUnpaddingDeviceOperation` | 1040.6 | 36 | 42.6 | 8 | -998.1 | -28 |
| `TilizeDeviceOperation` | 317.9 | 16 | 0.0 | 0 | -317.9 | -16 |
| `TilizeWithValPaddingDeviceOperation` | 370.5 | 16 | 61.4 | 8 | -309.1 | -8 |
| `PermuteDeviceOperation` | 287.7 | 12 | 0.0 | 0 | -287.7 | -12 |
| `BinaryNgDeviceOperation` | 14854.1 | 115 | 14596.5 | 87 | -257.6 | -28 |
| `UnaryDeviceOperation` | 136.3 | 28 | 110.5 | 24 | -25.9 | -4 |
| `ConcatDeviceOperation` | 10781.8 | 47 | 10756.6 | 43 | -25.1 | -4 |
| `MatmulDeviceOperation b={1536} x 128 x 128 x 128` | 192914.6 | 44 | 192895.0 | 44 | -19.7 | +0 |
| `MatmulDeviceOperation b={1536} x 128 x 32 x 128` | 18974.7 | 8 | 18965.3 | 8 | -9.4 | +0 |
| `TypecastDeviceOperation` | 394.3 | 24 | 385.4 | 24 | -8.9 | +0 |
| `RepeatInterleaveCodegenDeviceOperation` | 56.7 | 8 | 50.4 | 8 | -6.3 | +0 |
| `ReshapeViewDeviceOperation` | 2604.0 | 52 | 2599.8 | 52 | -4.2 | +0 |
| `MatmulDeviceOperation 32 x 5120 x 64` | 481.0 | 8 | 482.2 | 8 | +1.2 | +0 |
| `MatmulDeviceOperation 32 x 6144 x 5120` | 608.5 | 4 | 610.4 | 4 | +1.8 | +0 |
| `MatmulDeviceOperation b={1536} x 32 x 128 x 128` | 9948.7 | 4 | 9950.6 | 4 | +1.9 | +0 |
| `CopyDeviceOperation` | 93.4 | 8 | 98.9 | 8 | +5.5 | +0 |
| `RepeatCodegenDeviceOperation` | 1890.8 | 12 | 1896.5 | 12 | +5.7 | +0 |
| `TransposeDeviceOperation` | 555.4 | 40 | 567.8 | 40 | +12.4 | +0 |
| `UntilizeCodegenDeviceOperation` | 0.0 | 0 | 38.0 | 8 | +38.0 | +8 |
| `QkvCausalConv1dSiluOperation` | 0.0 | 0 | 120.7 | 4 | +120.7 | +4 |
The fused op costs 120.7 us across the four chunks and removes roughly 4156 us
of primitive work, a **22x improvement on the block it replaces**. The removals
are what the op subsumes: the `permute` pair, the four `slice`+`multiply`+`add`
tap terms, and -- the largest single item -- the tilize/untilize conversions
the old block paid because the conv, reshape, and recurrent kernels required
interleaved tensors. `UntilizeWithUnpadding` alone drops from 1040.6 us over 36
calls to 42.6 us over 8.

The layer total only moves 1.46% because the block was never the bottleneck.
71% of the window is the affine-scan recurrence
(`MatmulDeviceOperation b={1536} x 128 x 128 x 128`, 192.9 ms over 44 calls),
and the swap does not touch it. That matmul stack, not the convolution, is
where the remaining prefill headroom is; `affine_exclusive_scan` and
`reduce_affine_transforms` from the same KDA family are the candidates for it
and are not attempted here.

## Tuning the op

`channel_chunk_size` is the op's only program knob and it matters a lot. Device
time for one 32-token, 10240-channel chunk, measured with the real-time
profiler (`sweep_channel_chunk.py`, median of 7):

| channel_chunk_size | 64 | 128 | 160 | 256 | **320** | 512 | 640 | 1024 | 1280 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| device us | 51.9 | 35.3 | 30.3 | 33.2 | **29.2** | 37.4 | 42.6 | 60.8 | 73.8 | 111.8 |

320 ships. The first measured A/B used 1280 and saw only -1.39%; retuning to
320 took the op from 73.8 to 29.2 us per chunk. 2560 and above fail to
allocate. The 73.8 us sweep value matches the 72.9 us/chunk that tt-perf-report
attributed to the op in the 1280 run, which cross-checks both measurements.

## Scope and what it does not cover

- **Prefill only.** The op needs a tile-aligned sequence, so it cannot serve
  batch-32 decode where T is 1. Decode PCC is bit-identical to the control
  (0.999997412709174 both), confirming the knob does not reach that path.
- **Tile-aligned chunks only.** A ragged tail chunk from a non-aligned prompt
  fails the op's `sequence % 32 == 0` check, so `_linear_causal_conv_prefill`
  falls back to the inherited implementation for it. Verified at sequences 33
  and 65, which still pass PCC.
- **Batch 1 only.** The op takes a single `[1, T, C]` sequence; batched prefill
  keeps the inherited path.
- **`sigmoid_gated_rms_norm` was rejected, not measured.** It looked like a
  second 1:1 swap for the `rms_norm` + `multiply(out, silu(z))` pair, but it
  gates with `sigmoid(gate)` while this model gates with `silu(z)`. Those are
  not the same function, so the op is not applicable here.
- Synthetic weights and a single layer. This is a per-layer op-level result,
  not a full-model claim.

## Reproducing

```bash
# A/B under tracy, then tt-perf-report over the PERF_PREFILL window
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/run_ab.py \
    --sequence 128 --iterations 1

# the channel_chunk_size sweep
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/sweep_channel_chunk.py
```

`run_ab.py` raises `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` to 8192. At the
default cap the device profiler silently drops rows past roughly program 1070
of the ~1440 this harness dispatches, and the post-processor then aborts with
`Device data missing: Op ... not present in cpp_device_perf_report.csv`. The
dropped ops are real matmuls in the scan loop, so the run cannot simply be
reported with a hole in it.

Retained per arm: `perf.csv`, `perf_stacked.csv`, `perf_stacked.png`,
`tt_perf_report.txt`, and `profile.log`. The raw captures under `.logs/` and
`reports/` are not retained, per this autoport's `.gitignore` policy -- the
`profile_log_device.csv` alone is 347 MB per arm.
