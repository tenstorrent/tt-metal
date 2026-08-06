# MiniMax-H3 audio decode — what changed, measured

Branch `worktree-audio-kernels`. All figures are one warmed forward at the shipping working point
(207 latents x 2 channels -> 165,600 samples, mesh 1x1 Blackhole), reproduced with the commands in
`AUDIO_KERNELS_BENCH.md`. STATE.md am. 82 records ±8 % run-to-run at identical shape and seed, so
single-run differences below ~8 % are not signal.

## Achieved

| metric | before | after |
|---|---|---|
| decode 5 s (fp32) | 1.286 s | **1.169 s** |
| decode 10 s / 15 s (fp32) | 2.482 / 3.929 s | **2.260 / 3.623 s** |
| decode 5 s (bf16, opt-in) | did not run | **0.959 s** |
| PSNR vs `MINIMAX_H3_AUDIO_ACCURATE=1` | 39.46 dB | **42.86 dB** |
| PSNR, bf16 | — | 34.93 dB (gate: 28 dB) |
| accurate mode | 5.863 s | **2.407 s**, `rel_rmse 0.000e+00` (bit-identical) |
| depthwise conv1d fp32 error | 1.563e-03 | **7.06e-08** (= the exact elementwise form) |
| MAC fallbacks per decode | 36 | **0** |
| precision gates | — | 6 passed |

That is ~1.10x on the default path, 1.34x taking the bf16 option, and **2.44x on the accurate path at
bit-identical output**.

## What produced it

**1. Exact fp32 depthwise conv1d** (`a7bbb62a233`) — the keystone. Two changes that only work as a
pair: SFPU tap accumulation in `compute_depthwise_conv1d.cpp`, and `UnpackToDestFp32` on four CBs in
the conv2d sharded program factory. `ACT` matters as much as `ACT_TILIZED`: the tilize step rounds
fp32 to TF32 before the tilized buffer is written, which is what left a residual 4.154e-04 (2^-11.2)
when only the latter was overridden.

conv1d became bit-equal to the exact MAC form at 5-26x its speed, which is why everything else below
was possible: it removed the accuracy tax that had been blocking every speed lever.

**2. MAC fallback eliminated** (`a7bbb62a233`) — with conv1d exact, the one shape the DRAM slicer
cannot configure (C=512, K=7) runs as 4 depthwise conv1ds over channel chunks instead. 3.36 -> 1.33 ms,
identical error. No MAC calls remain in a decode.

**3. Snake tile-fold** (`a7bbb62a233`) — at C=8 a 32-wide tile carries 8 useful lanes. Folding
timesteps into channels is exact re-indexing for an elementwise per-channel op: 6.94 -> 1.91 ms at s6,
`maxdiff 0.0`.

**4. bf16 made runnable** (`14572d860ce`) — bf16 previously could not run at all (L1 overflow,
1981312 B against a 1572864 limit), because `get_conv3d_config` consults the tuned H3 blocking table
only under `weights_dtype == float32` and otherwise falls back to `C_in_block = in_channels`. It
clears the 28 dB gate only because change 1 raised the baseline first; from the old 39.46 dB it would
have landed near the floor.

**5. Stale gate re-derived** (`2b761a35e70`) — `test_depthwise_mac_is_more_accurate_than_conv1d`
asserted MAC beats conv1d by 100x, which change 1 made false. Both are now gated at fp32 grade.

## Not achieved

The target was 5-100x. This is ~1.3x. The remaining work is documented in `AUDIO_FUSION_PLAN.md`,
with the measurements that size it.

## Defaults

Everything new is off by default; the achieved numbers above are the default path except where the
row says bf16. See the knob table in `AUDIO_KERNELS_BENCH.md`.
