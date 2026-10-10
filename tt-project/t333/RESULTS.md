# t333 results: LTX-2.5 standard e2e, conv VAE (LTX-2.3 weights swap), blx01 4x8 — RELATIVE ONLY (AICLK clamped 900 MHz)

- Box: blx01 (g15blx01), 4x8 Blackhole galaxy, broker jobs 386 (c6a cold, timeout), 388 (c6, 6 s), 392 (c10, 10 s).
- Clock: every job logged 33 "AICLK failed to settle ... clamped ... at 900 MHz" (expected 1350). Numbers are relative only (user #230).
- Code: ttp/t48-ltx25-integrated @ f6547442b304d744711e80e2281f6bd368291673 (also on origin/ltx-rt), unmodified test
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py.
- Command (run333.sh, under setsid via the broker):
  python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=570 \
    "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]"
  env: LTX_VERSION=2.5 LTX25_DIFFVAE=0 LTX25_ROOT=/var/tmp/fasth3/models/ltx-2.5 (local sha256-verified copy)
       LTX25_VIDEO_VAE=ltx-2.3-22b-distilled-1.1.safetensors (2.3 conv VAE weights, same arch; NOT the 2.5 conv VAE)
       RUN_VBENCH=0 RUN_CLIP=0; c10 adds NUM_FRAMES=241. Test defaults: 1088x1920, 24 fps, 8+3 steps, CFG 1, x2 upsampler, seed 10.
- c6a (cold JIT) hit pytest-timeout at 570 s while compiling; it served as the JIT cache fill. Not a drop.

## 6 s (145 frames), job 388, gen#1 (warm) — verbatim
┌──────────────────────────────────────────────────────────────────────────┐
│                       LTX DISTILLED — PERFORMANCE                        │
│ Resolution   1088x1920 · 145 frames                                      │
│ Mesh         (4, 8) · sp=8 tp=4 · Ring                                   │
│ Output       /var/tmp/fasth3/t333/out_c6/ltx_av_fast_1920x1088_1.mp4     │
│ Prompt       A confident rapper in a black leather jacket and gold cha...│
├─────────────────────────────────────────────────────────────────┬────────┤
│ Stage                                                           │   Time │
├─────────────────────────────────────────────────────────────────┼────────┤
│ Encoder                                                         │ 0.26 s │
│ Stage 1 denoise                                                 │ 2.79 s │
│ Latent upsample                                                 │ 0.23 s │
│ Stage 2 denoise                                                 │ 3.05 s │
│ VAE decode                                                      │ 0.70 s │
│ Audio decode                                                    │ 0.55 s │
├─────────────────────────────────────────────────────────────────┼────────┤
│ Total                                                           │ 7.58 s │
└─────────────────────────────────────────────────────────────────┴────────┘

E2E_WALL_S gen#1: 7.801 (test log). gen#0 (cold, in-process compile) total 50.32 s; full tables in results/timing_c6_job388.txt.

## 10 s (241 frames), job 392, gen#1 (warm) — verbatim
┌──────────────────────────────────────────────────────────────────────────┐
│                       LTX DISTILLED — PERFORMANCE                        │
│ Resolution   1088x1920 · 241 frames                                      │
│ Mesh         (4, 8) · sp=8 tp=4 · Ring                                   │
│ Output       /var/tmp/fasth3/t333/out_c10/ltx_av_fast_1920x1088_1.mp4    │
│ Prompt       A confident rapper in a black leather jacket and gold cha...│
├────────────────────────────────────────────────────────────────┬─────────┤
│ Stage                                                          │    Time │
├────────────────────────────────────────────────────────────────┼─────────┤
│ Encoder                                                        │  0.25 s │
│ Stage 1 denoise                                                │  3.89 s │
│ Latent upsample                                                │  0.28 s │
│ Stage 2 denoise                                                │  5.84 s │
│ VAE decode                                                     │  1.09 s │
│ Audio decode                                                   │  0.70 s │
├────────────────────────────────────────────────────────────────┼─────────┤
│ Total                                                          │ 12.06 s │
└────────────────────────────────────────────────────────────────┴─────────┘

E2E_WALL_S gen#1: 12.287 (test log). gen#0 total 36.57 s.

## VAE decode vs DiffVAE
- conv VAE (2.3 weights), 1080p 145 f: 0.70 s in the e2e table (log: "VAE decode (forward): 0.7s"), at 900 MHz clamp.
- best DiffVAE decode, 1080p 145 f: 2.313 s (t48 @a5a774ea17f, blx03; clock of that run not re-checked here).
- conv VAE is ~3.3x faster even clamped. 241 f: 1.09-1.10 s.

## Outputs (g15blx02)
- tt-project/t333/results/ltx25_c6_gen1.mp4 + ltx25_c6_gen1_t3s.png (still at t=3 s): rapper in leather jacket, sharp, no artifacts.
- tt-project/t333/results/ltx25_c10_gen1.mp4 + ltx25_c10_gen1_t3s.png
- logs: results/run_c6_job388.log.gz, run_c10_job392.log.gz, run_c6a_job386.log.gz
