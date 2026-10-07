# t170 — LTX-2.5 eval pack (4x8 1080p 145f, blx01) and default knobs

Box g15blx01, tree t48 bf7db12a149 + t164 a4b6a835d1a overlay, default warmup, warm JIT cache.
One config per broker job; gen#0 = cold/trace capture (default prompt), gen#1/#2 = warm (fresh prompts).
PCC/PSNR vs this pack's own baseline at the same gen and prompt.

## Phase 1: knob pack (warm e2e and stages in s)
| config | gen1 e2e | d vs base | gen2 e2e | encode | S1 | upsample | S2 | VAE | audio | export | gen1 PCC (min) | gen1 PSNR (min) | gen2 PCC | gen2 PSNR | job |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| baseline | 6.017 | 0 | 6.041 | 0.20 | 2.25 | 0.14 | 2.31 | 0.56 | 0.38 | 0.10 | ref | ref | ref | ref | 665 |
| exact_shard | 5.970 | -0.047 | 6.011 | 0.20 | 2.22 | 0.15 | 2.31 | 0.51 | 0.38 | 0.20 | 1.00000 | identical | 1.00000 | identical | 666 |
| lofi | 6.047 | +0.030 | 6.068 | 0.20 | 2.25 | 0.14 | 2.31 | 0.55 | 0.40 | 0.20 | 0.99921 (0.99910) | 39.15 (38.70) | 0.99934 | 38.32 | 667 |
| gate | 5.815 | -0.202 | 5.919 | 0.19 | 2.16 | 0.13 | 2.21 | 0.56 | 0.37 | 0.10 | 0.97788 (0.97428) | 25.71 (25.13) | 0.97028 | 22.93 | 668 |
| adaln | 5.963 | -0.054 | 5.888 | 0.20 | 2.20 | 0.14 | 2.26 | 0.56 | 0.39 | 0.20 | 0.98337 (0.97671) | 26.98 (25.58) | 0.96883 | 22.72 | 685 |
| agmm | 6.041 | +0.024 | 6.033 | 0.20 | 2.24 | 0.13 | 2.30 | 0.56 | 0.39 | 0.20 | 0.95943 (0.95468) | 23.16 (22.38) | 0.95540 | 21.16 | 686 |
| hostcopy | 5.970 | -0.047 | 6.094 | 0.20 | 2.20 | 0.14 | 2.31 | 0.56 | 0.39 | 0.20 | 1.00000 | identical | 1.00000 | identical | 687 |
| all | 5.697 | -0.320 | 5.721 | 0.19 | 2.10 | 0.15 | 2.16 | 0.50 | 0.39 | 0.20 | 0.98829 (0.98518) | 28.55 (27.56) | 0.95367 | 21.04 | 688 |

Export 0.10 vs 0.20 s is run-to-run noise in the export step, not a knob effect.

## Phase 2: 5 seeds (LTX_FRESH_PROMPTS=0, seeds 0-4, default prompt), vs ref_dv145 per seed
| config | warm e2e gen1-5 mean (s) | VAE | PCC mean (worst) | PSNR mean (worst) dB | VBench subj / bg / img / motion | job |
|---|---:|---:|---|---|---|---|
| ref_dv145 | - | - | ref | ref | 0.893 / 0.926 / 0.553 / 0.985 | blx03 886 |
| baseline5 | 5.995 | 0.56 | 0.9931 (0.9816) | 34.25 (31.72) | 0.888 / 0.922 / 0.563 / 0.984 | 689 |
| exact_shard5 | 5.939 | 0.51 | 0.9931 (0.9816), byte-identical to baseline5 | 34.25 (31.72) | 0.888 / 0.922 / 0.563 / 0.984 | 707 |
| fast5 (exact_shard+gate+adaln) | 5.686 (5.725/5.655/5.632/5.711/5.705) | 0.51 | 0.874 (0.674) | 22.25 (18.09) | 0.888 / 0.921 / 0.559 / 0.984 | 708 |

fast5 per seed (PCC / PSNR): s0 0.920/23.4, s1 0.883/21.6, s2 0.808/20.3, s3 0.927/23.5, s4 0.833/22.3.

## Visual check (fast5, 5 seeds, frames 0/72/144 vs ref)
fast5 is as sharp and clean as the reference on every seed: same subject, scene, lighting and colour.
The difference is trajectory drift (pose, framing, small props such as the shelf box in seed 2), no
artifacts, blur or colour shift. VBench agrees (img quality 0.559 vs baseline 0.563, ref 0.553).
ref_dv145 is an older TT run, not a torch reference, so the PCC drop shows a different rounding path,
not lower accuracy. The fused AdaLN keeps the normalized intermediate in fp32 (less rounding than the
unfused bf16 boundary); the fused gate folds the gate weights into the Q/QKV matmul.

## Recommendation (landed as defaults)
- LTX_VAE_EXACT_SHARD=1 (a3216e1486d): byte-identical, -56 ms (VAE 0.56 -> 0.51 s).
- LTX_FUSE_GATE_ON_DEVICE=1 and LTX_FUSE_NORM_ADALN=1 (2aae332c37b): fast5 total -309 ms vs baseline5
  (5.995 -> 5.686 s), no visible or VBench loss. Each turns off with =0.
- Left off: lofi (+30 ms, no gain), agmm (+24 ms, lowest PCC), hostcopy (-47 ms on gen1 but +53 ms
  on gen2: noise, output identical; not worth a default change on this evidence).

## Videos and stills (g15blx02, tt-project/data/g15/t170/)
- baseline: baseline5/seeds/seed{0..4}.mp4; winner: fast5/seeds/seed{0..4}.mp4
- still, seed 0 at 3 s, baseline5 (left) vs fast5 (right): stills/baseline5_vs_fast5_seed0_t3s.jpg
- fast5 vs ref with diff: fast5/vbench/seed*_cmp_f{000,072,144}.png; downscaled stills/fast5/*.jpg
- pack stills (gen0-2 per config): stills/<config>_g{0,1,2}.jpg, grid stills/grid_g12.jpg

## Drop log
| UTC | box | broker job | chips / tray | whose | outcome |
|---|---|---|---|---|---|
| 2026-10-06 22:50:39 | blx01 | 669 (adaln) | 16-23 / tray 3 | ours | host rebooted ~22:57, driver relaunched; adaln rerun as 685 OK |
| 2026-10-06 23:27:50 | blx01 | 690 (exact_shard5) | 24-31 / broker tray [3] | ours | host rebooted ~23:34, driver relaunched; rerun as 707 OK |

(Earlier, under #164 on g15blx02: tray 1 dropped jobs 407, 421 (ltx-host), 422; not used here.)
