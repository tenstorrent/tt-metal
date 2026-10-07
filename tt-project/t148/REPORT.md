# t148: CPU-only fast-motion and export-preset quality check (2026-10-07)

No device jobs. Inputs (all DEFAULT_LTX_PROMPT, seeds 0-4, prompt/seed checked in the sidecar jsons):
- ref = `baselines/ltx25_1080p_6s/ref_dv145` (DiffVAE decode, x264 veryfast-like/crf23, 4.0 Mb/s)
- base = `data/g15/t170/baseline5/seeds` (conv VAE decode, ultrafast/crf20; same motion path as ref)
- fast = `data/g15/ref_t48_f6b8` = #171 job 710 clips (t48 f6b806516cc defaults: conv VAE + gate/adaln
  fusion, ultrafast/crf20, 10.8 Mb/s). #142's clips were replaced by these (#171 continues #142).

base vs ref isolates conv decode + export. fast vs ref also contains the gate/adaln motion-path change that
#170 accepted (PCC 0.874), so its PSNR is low for a known reason and is not a decode/export signal.

Scripts: `t148/analyze.py` (Farneback flow at 480 wide, top-5 / bottom-5 motion frames of ref per seed,
PSNR + gray SSIM, warp error = frame t-1 warped onto t by its own flow, Haar face crop at the fastest frame
with a face), `t148/export_test.sh`. Outputs: `data/g15/t148/` (motion_report.json, export_test*.txt,
faces_ref_base_fast.png, face_s*_full.png, still_s*_ref_base_fast.jpg).

## 1. Fastest-motion frames vs ref_dv145 (mean of 5 frames)

| seed | peak flow px | base PSNR top / low | base SSIM top / low | base worst top frame | corr(motion, PSNR) | fast PSNR top / low |
|---|---|---|---|---|---|---|
| 0 | 26.6 | 34.29 / 36.96 | 0.956 / 0.970 | 33.11 | -0.51 | 23.67 / 26.22 |
| 1 | 28.0 | 33.80 / 34.40 | 0.957 / 0.961 | 33.51 | -0.34 | 22.66 / 24.25 |
| 2 | 20.7 | 33.14 / 34.62 | 0.951 / 0.959 | 32.07 | -0.58 | 19.38 / 19.75 |
| 3 | 24.5 | 34.11 / 35.96 | 0.954 / 0.966 | 33.59 | -0.35 | 22.87 / 24.33 |
| 4 | 26.6 | 34.86 / 34.68 | 0.959 / 0.956 | 34.06 | -0.09 | 22.71 / 22.34 |

Conv decode + ultrafast export loses 0-2.7 dB on fast-motion frames compared with still frames, and never
drops below 32 dB. The difference is mild and gets larger with motion (correlation -0.09 to -0.58).

## 2. Temporal consistency (no reference needed), mean over all frames

| seed | flow px ref / base / fast | warp err ref / base / fast | warp err on top-5 motion frames |
|---|---|---|---|
| 0 | 14.47 / 15.33 / 15.41 | 4.32 / 4.61 / 4.70 | 5.37 / 5.43 / 5.63 |
| 1 | 14.13 / 14.89 / 15.33 | 4.32 / 4.67 / 4.77 | 7.39 / 7.53 / 6.95 |
| 2 | 12.37 / 12.83 / 12.43 | 3.55 / 3.95 / 3.71 | 6.33 / 6.77 / 5.43 |
| 3 | 12.59 / 13.36 / 12.88 | 3.73 / 3.98 / 4.05 | 6.42 / 6.54 / 6.56 |
| 4 | 15.91 / 17.56 / 17.17 | 4.78 / 5.01 / 4.98 | 6.80 / 6.76 / 6.54 |

The conv/ultrafast clips have about 5-10% more warp error and flow than ref. Most likely cause: the encoders
differ. The ultrafast clips have no deblocking and 2.7x the bitrate, so they keep more fine texture, and that
texture counts as residual. On the fastest frames, fast is within ±15% of ref (better on seeds 1, 2 and 4).

## 3. VBench (reused from #170 fast5, which is byte-identical to these #171 clips; baseline5 from the same run)

| dim, mean of 5 seeds | ref_dv145 | base (conv) | fast (f6b8 defaults) |
|---|---|---|---|
| subject_consistency | 0.8929 | 0.8884 | 0.8883 |
| motion_smoothness | 0.9850 | 0.9836 | 0.9836 |
| background_consistency | 0.9256 | - | 0.9206 |
| imaging_quality | 0.5535 | - | 0.5590 |

Per seed, subject consistency drops by 0.0002-0.0065 and motion smoothness by 0.0011-0.0020. Both are under the
0.01 tolerance on every seed for both arms.

## 4. Export preset cost (re-encode of a decoded clip; proxy, the pre-encode frames need a device dump)

Source ref_dv145 seed2 (veryfast/crf23 encode, so it does not favour ultrafast):

| preset/crf | MB | encode s (64 cores) | PSNR avg | PSNR min | SSIM | PSNR on seed-2 top-5 motion frames |
|---|---|---|---|---|---|---|
| ultrafast/20 (t48 default) | 7.32 | 0.28 | 48.96 | 47.80 | 0.9919 | 48.97 (min 48.41) |
| veryfast/23 (ref_dv145's export) | 2.88 | 1.16 | 49.29 | 47.49 | 0.9952 | 50.17 (min 49.05) |
| slow/20 | 3.65 | 1.78 | 51.44 | 50.02 | 0.9964 | 51.88 (min 51.19) |

Source #171 seed0 (ultrafast-encoded already, so it favours ultrafast): ultrafast/20 50.21, veryfast/20 48.52,
medium/20 49.11, slow/20 49.13, slow/18 50.07, veryfast/23 47.14 dB.

Compared with slow/crf20, ultrafast/crf20 costs about 2.5 dB (49 vs 51.4 dB). It matches the ref's own export
(veryfast/23) to within 0.3 dB, and stays 1.2 dB lower on fast-motion frames. All of these are above 47 dB, while
the decode differences are 33-35 dB, so the export is not what limits quality. The cost is about 2x the file size.

## 5. Visual

`faces_ref_base_fast.png` shows ref | base | fast face crops at the fastest frame with a detected face for each
seed. `still_s*_ref_base_fast.jpg` shows half-size full frames at the fastest frame. I see no blocking,
smearing, ghosting or face distortion in any seed. Base and fast look slightly crisper than ref. Ref shows the
same motion blur (seed 2 hand, seed 3 close-up). Fast's pose and framing differ a little from ref, which
matches the gate/adaln path that #170 accepted.

## Verdict

PASS. On fast-motion frames, conv VAE decode plus ultrafast/crf20 export show no visible degradation on any of
the 5 seeds. Base keeps 32-35 dB and SSIM 0.95-0.96 vs DiffVAE on the fastest frames. VBench motion and subject
dims are within 0.007, and the face crops look the same. The export preset costs about 2.5 dB vs slow/crf20 at
about 49 dB, which is invisible, so keeping ultrafast for its roughly 1.5 s encode saving is the right call. One
optional follow-up: ultrafast/crf20 writes about 2x the bytes of slow/crf20. If file size ever matters,
veryfast/crf20 is the cheap middle point (+0.5 s encode on 64 cores, overlapped by the async export).
