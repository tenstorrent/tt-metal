# MiniMax-H3 audio decode — what changed, measured

Branch `worktree-audio-kernels`. All figures are one warmed forward at the shipping working point
(207 latents x 2 channels -> 165,600 samples, mesh 1x1 Blackhole), reproduced with the commands in
`AUDIO_KERNELS_BENCH.md`. STATE.md am. 82 records ±8 % run-to-run at identical shape and seed, so
single-run differences below ~8 % are not signal.

## Achieved

| metric | before | after |
|---|---|---|
| decode 5 s (fp32) | 1.286 s | **1.157 s** |
| decode 10 s / 15 s (fp32) | 2.482 / 3.929 s | **2.256 / 3.583 s** |
| decode 5 s (bf16, opt-in) | did not run | **0.959 s** |
| PSNR vs `MINIMAX_H3_AUDIO_ACCURATE=1` | 39.46 dB | **42.86 dB** |
| PSNR vs the **CPU** reference, 5 s | — | **41.41 dB** (gate: 28 dB) |
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

**6. Wider timestep fold** (`a805896d13e`) — the fold in change 3 stopped at one tile width. Lifting
that to a target of C=256 (the factor need only divide T, and need not be a power of two) is worth
4.5-5.2x on the snake at the tail, bit-exact. End to end it is ~1 %, because the snake is only 140 ms
of the stage; see `AUDIO_FUSION_PLAN.md` for why that closes the cheap row-widening work.

## Not achieved

The target was 5-100x. This is ~1.3x. The remaining work is documented in `AUDIO_FUSION_PLAN.md`,
with the measurements that size it.

## Pre-fix vs post-fix on real audio, measured against a full pre-fix build

The earlier attempt at a pre-fix baseline failed because reverting the JIT kernel while the host half
of the keystone stayed in the built `.so` yields NaN, not old behaviour. The clean way is a **separate
full build of `cglagovich/minimax-h3`** (`3fdb75f55e5`) in its own worktree, where both halves are
consistently absent. Every loaded artefact was verified to come from that tree -- `_ttnn.so`,
`_ttnncpp.so`, `libtt_metal.so` and the model code -- because the venv's editable-install
MetaPathFinder hard-maps `ttnn` to the main checkout and silently wins over `PYTHONPATH`.

Four real clips, fp32, 5.17 s each, scored against the same torch/diffusers CPU decode. Source and CPU
files are bit-identical across the two runs, so the decoder is the only variable:

| clip | device pre | device post | PSNR pre | PSNR post | Δ |
|---|---|---|---|---|---|
| voice_libri1 | 1.264 s | **1.108 s** | 44.54 dB | **47.87 dB** | +3.33 |
| voice_libri2 | 1.263 s | **1.107 s** | 44.32 dB | **47.82 dB** | +3.50 |
| music_trumpet | 1.262 s | **1.108 s** | 49.14 dB | **52.83 dB** | +3.69 |
| music_brahms | 1.261 s | **1.109 s** | 45.75 dB | **49.28 dB** | +3.53 |
| **mean** | **1.262 s** | **1.108 s** | **45.94 dB** | **49.45 dB** | **+3.51** |

**1.14x faster and +3.51 dB**, with `rel_rmse` down from ~3.9e-02 to ~2.6e-02 (~33 % less error). The
gain is uniform across speech and music (+3.33 to +3.69 dB), so nothing here is content-specific.

Both builds are flat in time across content -- pre 1.261-1.264 s, post 1.107-1.109 s -- while the CPU
takes 1.365-2.412 s on identical shapes. That flatness is the op-count-bound signature: device cost
follows tensor shape, not audio content.

WAVs (source / cpu / device-post / device-pre per clip) are in `/data/rshirvani/audio_compare/clips/`
with a README; `cpu_vs_device.py` regenerates them.

## Two different PSNRs, and why the padding fix did not move either

They were conflated once (in `825cc1ffda3`) and are worth keeping apart:

* **42.86 dB** — default path vs `MINIMAX_H3_AUDIO_ACCURATE=1`. Device against device, so it is blind
  to any error both paths share.
* **41.41 dB** — `test_decode` vs the torch/diffusers CPU reference, the only figure that can see a
  defect present in every device path. Now logged by the test rather than discarded.

The fp32 channel-padding corruption fixed in `825cc1ffda3` was real and provable at the bit level, but
measured A/B on the same seed and shape it is **worth nothing end to end**:

    pre-fix  (audio_ops.py @ 1a4ffb00df4)   41.40 dB, log-mel 0.054
    post-fix                                41.41 dB, log-mel 0.054

0.01 dB is noise. That is the honest result: the fix is a **correctness and speed** change -- it makes
the padding op exact and 4-30x faster -- not an accuracy win. A ~1e-03 perturbation on the narrow-C
padding sits far below whatever dominates the residual 41 dB, so removing it is invisible at the model
level. Worth having, and worth not overselling.

## Defaults

Everything new is off by default; the achieved numbers above are the default path except where the
row says bf16. See the knob table in `AUDIO_KERNELS_BENCH.md`.
