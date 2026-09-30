# t19: conv (tuned 2.3) vs DiffVAE decoder on real LTX-2.5 latents

## Inputs (deviation from spec)
The 22:10 rescope bars full 4x8 runs on blx03, and generating new 2.5 latents at 1080p/145f needs the full mesh.
So no new device job was run. I used the saved stage-2 latents from t37's blx03 job 931 (commit 16ba9a383d,
conv decoder, LTX_DUMP_LATENTS, 1080p/145f, latent 19x34x60, S1 8 / S2 3 steps):
- gen0: default prompt (rapper, low light, fast head motion), seed 0
- gen1: "red paper boat on a pond at dusk", seed 0
- gen2: "old fisherman mends a net on a pier at sunrise", seed 0

These are 3 prompts at seed 0, not 3 seeds of one prompt (run25.sh default LTX_FRESH_PROMPTS=1).
That gives more content variety, not less. A device DiffVAE decode of the same latent exists only for gen0
(t16 ref_dv145/seed0.mp4, same prompt and seed; the DiT output matches, see below).

## Device, full video, gen0: device conv vs device DiffVAE (t16 job 886)
- ffmpeg YUV: PSNR avg 38.78 dB (min frame 35.43), SSIM 0.982.
- RGB per frame: mean 35.05 dB, min 31.88, p10 33.52.
- Face region (448x640): mean 34.94 dB. It drops below 33 dB only on fast-motion frames (17-26, 80-82, 97-112).
  Sharpness (Laplacian var, frames 65-80): conv 6.3, DiffVAE 5.8.
- Visual: on most frames the two can't be told apart (stills/dev_gen0_face_grid_conv_left_dv_right.png, frames 24/60/100/130).
  On motion-blur frames conv smears the face slightly more than DiffVAE (stills/dev_gen0_face_f72_conv_dv_diff.png, frame 72).

## CPU crop A/B (t12 cpu_ab.py + score.py), 3 latent frames at t0=8 (~3 s), conv vs DiffVAE
| gen | crop (h0,w0,HxW latent) | PSNR crop / interior dB | min frame | PCC interior | SSIM interior | sharpness conv / DiffVAE | CPU conv vs device mp4 |
|---|---|---|---|---|---|---|---|
| 0 | face (4,12,14x20) | 31.08 / 32.10 | 24.41 | 0.9896 | 0.9378 | 4.4 / 2.1 | 35.41 |
| 0 | jacket (18,10,8x16) | 31.51 / 35.25 | 29.84 | 0.9759 | 0.9282 | 4.1 / 2.3 | 37.31 |
| 1 | boat (10,18,14x20) | 31.24 / 33.71 | 29.07 | 0.9976 | 0.9234 | 39.6 / 32.0 | 35.30 |
| 1 | ripples (2,2,8x16) | 27.74 / 31.22 | 26.90 | 0.9737 | 0.8581 | 32.7 / 17.8 | 33.55 |
| 2 | fisherman (8,26,14x20) | 30.91 / 33.73 | 29.42 | 0.9981 | 0.9506 | 59.9 / 44.4 | 35.24 |
| 2 | pier/net (24,34,8x16) | 30.06 / 33.48 | 28.31 | 0.9310 | 0.8677 | 34.5 / 16.3 | 33.98 |

The crop numbers include frame 0 of each crop, which is the causal "first frame" of a mid-video crop. The crop's
DiffVAE also sees less context than a full decode. So crops read lower than the full-video number (gen0 face:
32.1 dB crop interior vs 34.9 dB on the full device videos).

## Compared with t12 (2.3 latents: 33-39 dB, PCC 0.996-0.998)
On 2.5 latents the gap is wider: interior 31.2-35.3 dB, PCC 0.931-0.998. Conv output is consistently sharper
(1.2-2.1x Laplacian variance). DiffVAE is softer on water, wood grain and net texture. Conv keeps more fine detail there.
The one place conv looks worse is face smear on fast-motion frames (gen0 frames ~72-82).

## Verdict
Conv decoder stays acceptable as the main 2.5 decode path (option A). No visible degradation on static or
slow content. Differences are texture sharpness (conv sharper) and slight face smear on motion-blur frames.
This does not change the decoder choice. Once full-mesh runs are allowed again, it needs a 5-seed visual check
plus a VBench comparison on motion-heavy prompts (conv vs ref_dv145) to settle it.

## Artifacts (g15blx02, not in git)
tt-project/baselines/t19/: conv_gen{0,1,2}.mp4 (device conv decode, full 1080p/145f), stills/, scores/.
Reference DiffVAE video for gen0: tt-project/baselines/ltx25_1080p_6s/ref_dv145/seed0.mp4.
