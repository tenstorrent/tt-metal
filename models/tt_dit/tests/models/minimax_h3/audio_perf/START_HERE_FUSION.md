# Fusion implementation — start here, build only

For a fresh session. **Do not re-derive anything in this file.** The design is settled and the maths is
verified; what is missing is build-and-verify cycles. The previous session produced ~1.16x against a
5-100x goal because it kept converting implementation into analysis. Don't repeat that.

## The framing to work from

From `plan-audio-encoder.md` (the original meeting): *"Fuse multiple ops to keep intermediate values in
SRAM and eliminate redundant memory traffic"*, and crucially *"tiling strategy may allow full fusion
without requiring entire tensor to be resident."*

Tiling is not a workaround for L1 being small -- **it is the thing that makes full fusion possible.**
Scope the resident chunk as widely as L1 allows and run the whole chain on it, rather than fusing one
band and stopping. 53->4 ops per band is the timid version; the chain the doc points at is
upsample + resblock branches, where the arithmetic is nearer 500->4.

## Facts already established -- trust these, do not re-measure

| fact | value | source |
|---|---|---|
| per-op floor | **180.3 us fp32**, 127.6 us bf16, flat to ~16k rows | `op_floor.py` |
| decode op count | 6955 -> 1254 ms of pure floor vs ~1.1 s actual | same |
| 60 ms budget | **~332 ops** fp32, ~470 bf16 | same |
| Concat | 469 calls, 285.3 ms, **20.4 % -- largest single line item** | PROFILE_2026_08_06.txt |
| all convolution | 292 ms / 21 % | same |
| data movement total | ~900 ms of ~1400 ms | same |
| conv invocations per band | **6.85** for 2 logical convs (DRAM slicing, 5-op wrapper per slice) | same + audio_ops.py:375 |
| arithmetic cost | free -- `sin` cheaper than `add` | `compute_intensity.py` |
| row effect | same elements cost **11.6x** more at C=8 than C=224 | same |
| current decode | 1.107 s fp32, PSNR 49.45 dB vs CPU | `cpu_vs_device.py` |

## Dead ends -- do not retry, all measured

trace (1.00x) · conv1d L1_FULL (1 of 42 shapes fit) · operand splitting · algebraic band fusion
(**-2.4 %**) · L1-sharded intermediates · `_zero_pad_t` concat->pad (2.2x on the op, **0 % end to
end**) · conv3d UnpackToDestFp32 · 32-chip hypothesis · act_block_h · grouped-conv branch batching
(**0.94-1.11x**, and lossy at C>=64).

They fail for one reason: they optimise ops that are already at the floor, or merge ops far above it.
Only removing large numbers of floor-bound ops moves the decode.

## Verified maths -- transcribe, don't re-derive

`band_stencil_cpu.py`, **maxdiff 3.331e-16** against the literal band in float64. With `xp` the
once-replicate-padded input:

    u[m], m even -> sum_j h[K-1-2j] * xp[m/2 + j]        odd-indexed taps, reversed
    u[m], m odd  -> sum_j h[K-2-2j] * xp[(m+1)/2 + j]    even-indexed taps, reversed
    s            =  u + inv_beta * sin(alpha*u)^2
    y[n]         =  sum_k g[k] * s[2n + K-1-k]

The phase-to-tap-subset mapping is **the reverse of the obvious guess** -- getting it backwards costs
maxdiff 0.598. Replicate padding does **not** decompose per phase (the pad region's parity
alternates), so chunk edges take halo from real neighbours; only the outermost chunks apply the
replicate rule.

## What is already built and verified

* `apply_snake_beta` in `compute_depthwise_conv1d.cpp` -- per-channel snake in DST, **3 DST slots**
  (reuses DST_A/DST_B after the last tap), **does not pop** (params are per-block, popping destroys
  them for following tiles).
* `Conv2dCb::SNAKE_PARAMS` + `CBInfo` entry + `SNAKE_PARAMS_CB_ID` define. **0 pages unless
  `TT_CONV1D_SNAKE_PARAMS` is set**, so `post_conv2d_op_memory_checks` is untouched on the default
  path -- that check is a known hard-failure source (tt-metal #35207) and this sidesteps it.
* Built and regression-tested: **17/17, `conv1d=8.082e-08 mac=8.082e-08`**, bit-identical.

## Next three steps to make the snake fusion live

1. Fill the CB once in the weights reader, **outside** the block loop.
2. Host: alpha and `inv_beta = 1/(beta+eps)` as tiles, per-channel value replicated down all 32 rows.
3. Check against `band_stencil_cpu.py`'s float64 golden, target ~1e-07 (GELU got 7.6e-08 through the
   scalar seam).

Then widen the resident chunk from the band to the resblock chain, which is where the order of
magnitude is.

## Build procedure -- this bites

**Host C++ must be built from the main checkout.** The worktree's `build_Release` is a symlink to
`/data/rshirvani/tt-metal`'s, whose cmake cache is configured against the main checkout, so
`build_metal.sh` from the worktree refuses outright -- and a build that did run would compile the
wrong sources. Copy the files across, build there, keep backups.

Device kernels are JIT-compiled from `TT_METAL_HOME` and **do** take effect from the worktree. That
asymmetry is why kernel changes were verifiable in place all session and host changes were not.

**Cost:** a header change (e.g. `conv2d_op_program_factory_common.hpp`) plus unity builds invalidates
most of ttnn -- **~2 h**. A .cpp-only change is minutes. Prefer .cpp-only edits while iterating.

After any kernel edit, immediately re-run
`pytest test_audio_vae_minimax_h3.py -k "depthwise_mac or channel_padding"` (~20 s). A template
defined outside an `#ifdef` still needs its callees declared -- "gated off" does not mean "cannot
break the build", and that check catches it in seconds.

## Gates

`pytest models/tt_dit/tests/models/minimax_h3/test_audio_vae_minimax_h3.py` -- 17 must pass.
`cpu_vs_device.py` for timing and PSNR. `full_table.py` for the whole comparison from saved WAVs.
Commit and push to `rouzbeh/audio-decode-exact-fp32`; the user has asked for this explicitly, which
supersedes the "don't commit" line in plan-audio-encoder.md.
