# Fused KDA causal conv in the linear-attention layer

The new tt-metal base carries the KDA (Kimi Delta Attention) op family, which
did not exist on the base this autoport was brought up against. One of those
ops, `ttnn.experimental.kda.qkv_causal_conv1d_silu`, is exactly the depthwise
causal convolution this model's Gated DeltaNet layer hand-rolls: a four-tap
causal conv, SiLU, and the Q/K/V split. The model's `linear_conv_kernel_dim` is
4 and its Q/K/V widths are 2048/2048/6144, so the op's shape constraints are
already met.

This swaps that one block on both the prefill and the decode path and measures
each. Nothing else changes. The knob is `linear_kda_conv`, default off; the
candidate policy is `dataclasses.replace(_LINEAR_FINAL, linear_kda_conv=True)`,
so the two arms differ in exactly one field.

## Result

Layer 0 (`linear_attention`), synthetic weights, single Blackhole P150 chip.

### Decode, batch 32, traced

The shape the layer actually spends its time in: 48 of this model's 64 layers
are linear-attention, and the autoport's own
`doc/advisor_challenger/README.md` measured this layer at 15.844 ms against
1.449 ms for a full-attention layer.

| | control (`linear_final`) | candidate (`linear_kda_conv`) | delta |
|---|---:|---:|---:|
| Device time per step | 15.908 ms | 8.254 ms | **-7.654 ms (-48.1%)** |
| Trace replay, median | 15.825 ms | 8.194 ms | **-7.631 ms (1.93x)** |
| Dispatched ops per step | 87 | 76 | -11 |
| PCC vs the HF layer, step 1 / step 2 | 0.9999677 / 0.9999912 | 0.9999674 / 0.9999909 | — |

### Multichip decode, TP4, batch 32, traced

The shipped path: four Blackhole p300c devices, `MeshShape(1,4)`, `FABRIC_1D_RING`,
12 value heads per device. Same knob, same one-field policy difference.

| | control (`linear_final`) | candidate (`linear_kda_conv`) | delta |
|---|---:|---:|---:|
| TP4 trace replay, median | 4.3335 ms | 2.3691 ms | **-1.9644 ms (1.83x)** |
| single-chip baseline in the same run | 15.8556 ms | 8.2056 ms | (1.93x, matches the single-chip result) |
| PCC vs that baseline | 1.0 | 1.0 | — |

Per-device the widths are Q/K/V = 512/512/1536 with a 2560-wide conv, all
tile-aligned, and `K*B` is 128 at batch 32, so the user-major packing carries
over unchanged. As on single chip, the state is stored *as* the window rather
than beside it.

The one addition is `_active_mask`. The composite computes the convolution from
the **unmasked** advanced state and blends only what it stores, so inactive rows
keep their old state while still producing (discarded) outputs. The fused path
preserves that exactly: the op reads the advanced window, and the mask is
applied to the stored state afterwards.

Full-attention layers have no conv cache, so the setup is a no-op for them.

**No op-level table for this arm.** The device profiler captures only 1 of the 4
devices during a multi-device *trace replay* on this base — the raw
`ops_perf_results` CSV contains device 3 alone, with single-chip op shapes
(`b={1536}`, full-width `32 x 5120 x 17408` MLP) and no CCL ops, i.e. the
signposted window picked up the harness's single-chip baseline instead. The same
capture on a *non-traced* multichip run returns all four devices, and the
autoport's own recorded artifact
(`doc/multichip_decoder/artifacts/tracy/linear_b32_dram_sharded/decode_perf_report.csv`)
has devices 0-3, `b={384}` and CCL ops from this same command — so traced
multichip profiling did work before. Recorded, not diagnosed. The trace-replay
median above is the metric the autoport's own multichip evidence uses.

### Prefill, batch 1, sequence 128 (four 32-token chunks)

| | control | candidate | delta |
|---|---:|---:|---:|
| Device time, signposted window | 272.785 ms | 268.896 ms | **-3.889 ms (-1.43%)** |
| Dispatched ops in window | 643 | 515 | -128 |
| End-to-end host latency, median | 273.860 ms | 270.043 ms | -3.817 ms |
| PCC vs the HF layer | 0.9999958 | 0.9999956 | — |

## How decode uses a single-stream op

The op is causal over its sequence dimension and requires a tile-aligned
sequence. Decode is B independent single-token streams, so T=1 -- which is not
tile-aligned and would be the wrong reduction anyway.

The way through, taken from `ayerofieiev/qwen38/gdn-conv-kda` which did this
first on the `models/demos/blackhole/qwen36` implementation, is to pack the
per-user tap windows **user-major into the sequence dimension** as one
`[1, K*B, C]` stream. With K=4 and B=32 that is T=128, tile-aligned. User b
occupies rows `[K*b, K*b+K)` oldest-first, so only row `K*b + K-1` has a
four-tap window entirely inside its own user's block; the rows that straddle a
user boundary are discarded by a constant one-hot select matmul
(`[1, B, K*B] @ [1, K*B, W]`, run at HiFi4 with FP32 accumulation so the select
is exactly value-preserving). This needs `K*B` tile-aligned, so B must be a
multiple of 8; a batch-1 decode keeps the composite path.

Rather than keep a second window buffer beside `caches["conv"]`, this stores the
state **as** the window: `caches["conv"]` becomes `[B, K, C]` row-major when the
fused decode path is active. One source of truth means an external writer --
prefill, a vLLM slot edit, a test restoring the cache -- is seen by the fused
path with no sync step and no per-token write-back. The composite
implementations still index the state as `[1, B, C, K]`, so the two paths that
can still reach them (a ragged prefill tail, and batch-32 prefill) convert in
and out around the inherited call; that conversion round-trips bit-exactly
(fused vs composite PCC = 1.0 at batch-32 prefill).

Dropping the write-back was worth 0.42 ms/step on its own: an earlier revision
that kept `caches["conv"]` current alongside the window measured 8.612 ms.

## Where the time went

### Decode (window covers 2 steps)

| Op | base us | base n | KDA us | KDA n | Δ us | Δ n |
|---|---:|---:|---:|---:|---:|---:|
| `UntilizeWithUnpaddingDeviceOperation` | 4899.2 | 6 | 110.0 | 2 | -4789.2 | -4 |
| `PermuteDeviceOperation` | 3981.7 | 12 | 97.8 | 2 | -3883.9 | -10 |
| `SliceDeviceOperation` | 3282.1 | 16 | 56.3 | 10 | -3225.8 | -6 |
| `TilizeWithValPaddingDeviceOperation` | 2739.7 | 6 | 6.9 | 2 | -2732.8 | -4 |
| `BinaryNgDeviceOperation` | 3875.4 | 38 | 3596.4 | 36 | -279.0 | -2 |
| `FillPadDeviceOperation` | 527.8 | 6 | 288.3 | 4 | -239.5 | -2 |
| `ReduceDeviceOperation` | 403.6 | 6 | 177.1 | 4 | -226.5 | -2 |
| `UnaryDeviceOperation` | 562.1 | 14 | 363.0 | 12 | -199.0 | -2 |
| `CopyDeviceOperation` | 491.0 | 4 | 296.8 | 4 | -194.2 | +0 |
| `TypecastDeviceOperation` | 910.3 | 8 | 907.2 | 8 | -3.1 | +0 |
| `RepeatInterleaveCodegenDeviceOperation` | 245.2 | 4 | 246.4 | 4 | +1.2 | +0 |
| `MatmulDeviceOperation b={1536} x 32 x 128 x 128` | 6834.1 | 4 | 6838.0 | 4 | +3.9 | +0 |
| `MatmulDeviceOperation 32 x 128 x 6144` | 0.0 | 0 | 17.4 | 2 | +17.4 | +2 |
| `MatmulDeviceOperation 32 x 128 x 2048` | 0.0 | 0 | 17.9 | 4 | +17.9 | +4 |
| `QkvCausalConv1dSiluOperation` | 0.0 | 0 | 161.1 | 2 | +161.1 | +2 |
| `ReshapeViewDeviceOperation` | 844.0 | 14 | 1109.3 | 16 | +265.3 | +2 |
The fused op plus its three select matmuls costs 196 us and removes about 15.5
ms. Four ops account for 14.9 ms of that: `UntilizeWithUnpadding`
4899 -> 110 us, `Permute` 3982 -> 98 us, `Slice` 3282 -> 56 us, and
`TilizeWithValPadding` 2740 -> 7 us. **The block goes from ~15.6 ms to
~0.46 ms, about 34x.**

The reason decode was so much worse than prefill is layout, not arithmetic. The
composite decode conv works in `[1, B, C, K]` where the tap axis is the last
dimension and K=4 -- tile-padded to 32, so seven eighths of every tile is
padding -- and it pays untilize/tilize round-trips plus permutes of 1.3M-element
tensors to get there and back. The user-major packing puts the tap axis into a
tile-aligned sequence dimension and none of that traffic is needed.

### Prefill

| Op | base us | base n | KDA us | KDA n | Δ us | Δ n |
|---|---:|---:|---:|---:|---:|---:|
| `SliceDeviceOperation` | 13204.4 | 136 | 11283.2 | 96 | -1921.2 | -40 |
| `UntilizeWithUnpaddingDeviceOperation` | 1040.9 | 36 | 44.8 | 8 | -996.1 | -28 |
| `TilizeDeviceOperation` | 317.6 | 16 | 0.0 | 0 | -317.6 | -16 |
| `TilizeWithValPaddingDeviceOperation` | 370.4 | 16 | 61.0 | 8 | -309.4 | -8 |
| `PermuteDeviceOperation` | 287.7 | 12 | 0.0 | 0 | -287.7 | -12 |
| `BinaryNgDeviceOperation` | 14867.9 | 115 | 14644.5 | 87 | -223.4 | -28 |
| `UnaryDeviceOperation` | 141.9 | 28 | 110.1 | 24 | -31.8 | -4 |
| `ConcatDeviceOperation` | 10780.1 | 47 | 10771.2 | 43 | -8.9 | -4 |
| `MatmulDeviceOperation b={1536} x 128 x 32 x 128` | 18975.1 | 8 | 18971.5 | 8 | -3.7 | +0 |
| `MatmulDeviceOperation 32 x 5120 x 6144` | 596.3 | 4 | 594.7 | 4 | -1.6 | +0 |
| `LayerNormDeviceOperation` | 169.7 | 6 | 168.4 | 6 | -1.3 | +0 |
| `MatmulDeviceOperation 32 x 6144 x 5120` | 611.4 | 4 | 610.2 | 4 | -1.2 | +0 |
| `MatmulDeviceOperation 32 x 5120 x 10240` | 1031.9 | 4 | 1030.8 | 4 | -1.1 | +0 |
| `ReduceDeviceOperation` | 30.9 | 8 | 32.0 | 8 | +1.1 | +0 |
| `MatmulDeviceOperation b={1536} x 32 x 128 x 128` | 9949.6 | 4 | 9951.0 | 4 | +1.4 | +0 |
| `TransposeDeviceOperation` | 554.9 | 40 | 557.5 | 40 | +2.5 | +0 |
| `CopyDeviceOperation` | 93.4 | 8 | 97.4 | 8 | +4.0 | +0 |
| `RepeatInterleaveCodegenDeviceOperation` | 55.0 | 8 | 62.7 | 8 | +7.7 | +0 |
| `ReshapeViewDeviceOperation` | 2595.7 | 52 | 2609.2 | 52 | +13.5 | +0 |
| `MatmulDeviceOperation b={1536} x 128 x 128 x 128` | 192897.7 | 44 | 192921.7 | 44 | +24.0 | +0 |
| `UntilizeCodegenDeviceOperation` | 0.0 | 0 | 38.0 | 8 | +38.0 | +8 |
| `QkvCausalConv1dSiluOperation` | 0.0 | 0 | 123.8 | 4 | +123.8 | +4 |
Same removals, far smaller: ~4156 us of primitive work replaced by ~187 us,
22x on the block. The layer total moves only 1.43% because 71% of the prefill
window is the affine-scan recurrence
(`MatmulDeviceOperation b={1536} x 128 x 128 x 128`, 192.9 ms over 44 calls),
which this does not touch. That matmul stack is where the remaining prefill
headroom is; `affine_exclusive_scan` and `reduce_affine_transforms` from the
same KDA family are the candidates for it and are not attempted here.

## Tuning the op

`channel_chunk_size` is the op's only program knob and it is load-bearing.
Device time for one call, measured with the real-time profiler
(`sweep_channel_chunk.py`, median of 7):

| channel_chunk_size | 64 | 128 | 160 | 256 | 320 | 512 | 640 | 1024 | 1280 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill, T=32 | 51.9 | 35.3 | 30.3 | 33.2 | **29.2** | 37.4 | 42.6 | 60.8 | 73.8 | 111.8 |
| decode, T=128 | 179.4 | 107.8 | 94.6 | 82.9 | 85.0 | **81.6** | 82.3 | 101.8 | 113.0 | 145.0 |

The optimum moves with T, so the two phases tune separately: 320 for prefill,
512 for decode. Both defaults were wrong on the first pass and both retunings
mattered -- prefill's first A/B used 1280 and understated the win as -1.39%,
and decode's first default of 128 cost 26 us/step against 512. 2560 and above
fail to allocate. The T=32 sweep's 73.8 us at 1280 matches the 72.9 us/chunk
that tt-perf-report attributed to the op in the 1280 run, which cross-checks
the two measurements against each other.

Note that a plausible heuristic -- largest tile-multiple divisor of C up to 128
-- lands on 128 for this C, which is the 107.8 us point at decode's T. Worth
sweeping rather than deriving.

## Correctness, and a defect in the shared harness

**The shared synthetic weights cannot see a conv bug.**
`linear_attention_synthetic_pcc._state` sets `conv[:, 0, -1] = 0.5` and leaves
taps 0-2 at zero, so the convolution degenerates to `silu(0.5 * x_t)`. Under
that weight the tap history is multiplied by zero, and a stale window, a wrong
shift, or a mis-seeded state is invisible. Every PCC number in the tables above
inherits that blind spot -- they are still valid perf measurements, since the op
does the same work whatever the tap values, but on their own they do not
validate the convolution.

`check_conv_taps.py` closes the gap: it reuses the same synthetic state with a
dense random four-tap kernel and compares the fused path against both the
composite path and the HF layer.

| case | composite vs HF | fused vs HF | fused vs composite |
|---|---:|---:|---:|
| decode batch 32, 4 sequential steps (worst step) | 0.9999970 | 0.9999967 | 0.9999997 |
| prefill batch 1, sequence 128 | 0.9999953 | 0.9999950 | 0.9999997 |
| prefill batch 32, sequence 128 (fallback conversion) | 0.9999953 | 0.9999953 | **1.0** |
| prefill batch 1, sequence 33 (ragged tail fallback) | 0.9999954 | 0.9999952 | 0.9999997 |
| **TP4** decode batch 32, 4 steps (worst step) | 0.9999970 | 0.9999967 | 0.9999997 |
| **TP4** decode batch 32 with `active_mask`, 4 steps (worst step) | 0.9999661 | 0.9999662 | 0.9999998 |
| **TP4** prefill batch 32, sequence 128 (state conversion) | 0.9999896 | 0.9999896 | **1.0** |

Multiple sequential decode steps matter here: a single step cannot distinguish
a correct window from one that never advances.

One harness fix was needed. `traced_synthetic_pcc.py` snapshots and restores
`decoder.caches[...]` between the warmup decode and trace capture, and its
`_copy_host` hardcoded `layout=ttnn.TILE_LAYOUT` while already deriving `dtype`
from the destination. A row-major conv state failed with `Host tensor has
different page config`. The restore now takes both layout and dtype from the
destination tensor, which is what it was already doing for the compressed KV
cache dtype.

## Scope

- **Decode needs B a multiple of 8** (`K*B` tile-aligned). Batch 1 decode keeps
  the composite path.
- **Prefill needs tile-aligned chunks and batch 1.** A ragged tail chunk and
  batch-32 prefill both fall back, converting the state layout around the
  inherited call.
- **`sigmoid_gated_rms_norm` was rejected, not measured.** It looked like a
  second 1:1 swap for the `rms_norm` + `multiply(out, silu(z))` pair, but it
  gates with `sigmoid(gate)` while this model gates with `silu(z)`. Not the
  same function, so not applicable.
- Synthetic weights, one layer. These are per-layer results; the 48x scaling to
  a full-model decode estimate would be arithmetic, not measured.
- Multichip decode needs `B` a multiple of 8 as well; at batch 1 `K*B` is 4 and
  the composite path is kept (measured unchanged at 0.7929 ms).

## Reproducing

```bash
# A/B under tracy, then tt-perf-report over the signposted window
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/run_ab.py --phase decode --iterations 2
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/run_ab.py --phase prefill --sequence 128 --iterations 1

# correctness with a real four-tap kernel
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/check_conv_taps.py --mode decode --batch 32 --steps 4
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/check_conv_taps.py --mode prefill --batch 32 --sequence 128
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/check_conv_taps.py --mode decode --batch 32 --steps 4 --multichip
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/check_conv_taps.py --mode decode --batch 32 --steps 4 --multichip --active-mask

# TP4 A/B (wall-clock trace replay; run each arm and compare)
python models/autoports/qwen_qwen3_6_27b/tests/multichip_traced_decode.py --kind linear --batch 32 --steps 4 \
    --candidate linear_kda_conv --baseline-candidate linear_kda_conv

# the channel_chunk_size sweeps
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/sweep_channel_chunk.py --sequence 32
python models/autoports/qwen_qwen3_6_27b/doc/kda_conv_swap/sweep_channel_chunk.py --sequence 128
```

`run_ab.py` raises `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` to 8192. At the
default cap the device profiler silently drops rows past roughly program 1070
of the ~1440 the prefill harness dispatches, and the post-processor then aborts
with `Device data missing: Op ... not present in cpp_device_perf_report.csv`.
The dropped ops are real matmuls in the scan loop, so the run cannot just be
reported with a hole in it.

Retained per arm: `perf.csv`, `perf_stacked.csv`, `perf_stacked.png`,
`tt_perf_report.txt`, `profile.log`. The raw captures under `.logs/` and
`reports/` are not retained, per this autoport's `.gitignore` policy -- the
`profile_log_device.csv` alone is 347 MB per arm.

## Prior art

`ayerofieiev/qwen38/gdn-conv-kda` applied this op to the decode conv of
`models/demos/blackhole/qwen36` first, and the user-major packing and one-hot
select above are that branch's technique. That branch carries no perf
measurement and is default-off behind `QWEN36_GDN_CONV_KDA`; it also builds
prefill/decode state sync, per-slot writes and vLLM batch-condense helpers that
this change does not need, because it makes the cache the window instead of
maintaining a second copy. It does not cover prefill, and its
`default_channel_chunk` heuristic is unswept.
