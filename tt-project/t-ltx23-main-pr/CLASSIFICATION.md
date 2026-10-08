# LTX-2.3 8+3: which ltx-rt/t48 gains over main are safe (task #296, 2026-10-08)

Scope: LTX-2.3 distilled AV, BH Galaxy 4x8 Ring, 1088x1920, 145 frames, 8+3 steps.
LTX 2.5 (DiffVAE, 2.5 DiT), H3 and 8+2 are out of scope. Moving S2 from 3 steps to 2 (b21f12b93a2) is
NOT counted as a gain; #295 restored 3 steps on t48 (f6547442b30).

Refs: main c398f2b6178, ltx-rt b9f8587ce6c (merge base c2a4d40104d), t48 f6547442b30.
This file comes from reading code and commits only; no device jobs were run.
There are no main vs t48 8+3 e2e numbers yet: #293 has not run (no box reachable). The time
estimates come from commit messages and the ltx-rt 8+3 baseline (research/ltx_baseline.md, job 552:
S1 2.31, S2 2.57, VAE 0.68, audio 0.38, export 0.70, e2e 7.12 s).

Classes: a = bit-identical (by construction or md5), b = quality-neutral with evidence,
c = numerics change, no evidence yet (needs device A/B), d = degrading.

## Gains over main

| # | Gain | Commits | Default ltx-rt / t48 | Est. saved (8+3) | Class | Evidence | Shared ops? |
|---|---|---|---|---|---|---|---|
| 1 | QK-RoPE fusion (bf16-preserving) + active-core RoPE | ltx-rt LTX_FUSE_QK_ROPE code; f8a8611de5a | off / on | 0.24 s (S1 2.31->2.18, S2 2.57->2.46) | a | commit: latents and mp4 unchanged, 4x8 | LTX code (uses rotary op) |
| 2 | Drop per-step replay syncs + latent stats | 4194cd98852 | off / on | <=0.15 s (depends on main's sync points) | a | by construction: same CQ order | LTX only |
| 3 | Seeded noise once, on a host thread | cc51d4a21bc | off / on | ~0.1 s (23+92 ms draws, S1 drawn twice) | a | same values with an explicit generator | LTX only |
| 4 | Reuse the S1 prompt in traced S2 | 5d993cd2f7c | off / on | 33-45 ms | a | same buffers, same data | LTX only |
| 5 | Skip the V2A pad-mask multiply | f793ec1c64a | off / on | ~0.04 s (-0.29 ms per S2 block) | a | md5-identical, blx03 job 161 | LTX only |
| 6 | Async mp4 video encode under the audio decode | 63902277007 | off / on | ~0.15 s (hidden under audio 0.38 s) | a | mp4 byte-identical | LTX only |
| 7 | Fast x264 preset (ultrafast, crf 20, zero-copy) | e7588fb8718, eee3baf7c0d | off / on | ~0.5 s (0.65 -> 0.15 s encode) | b | Y PSNR vs source 47.9 dB vs 45.6 dB (better); file ~3.5x larger | LTX only |
| 8 | Audio VAE + BWE decoder traces | ltx-rt (4e7d0cefcdb squash) | on / on (main: off) | unmeasured, part of audio 0.38 s | a | trace replay is the same program | LTX only |
| 9 | Keep the latent upsampler resident | ab67c9e86b2 | on / on | 0 to 1.5 s: main's reload is a no-op when loaded; whether main evicts it per gen on BH is unverified | a | host/residency only | LTX only |
| 10 | On-device YUV export (LTX_YUV_EXPORT=1) | code already in main, default off; ltx-rt flips it | on / on (main: off) | ~1.3 s (VAE 0.9->0.68, export 1.8->0.7) | c | no recorded PSNR vs main's float path | LTX only |
| 11 | Fold the 3 remaining gated residuals into to_out | ltx-rt (4e7d0cefcdb), LTX_FOLD_GATED_RESIDUAL | on / on | unmeasured (3 programs/block fewer) | c | comment says math-identical; epilogue rounding may differ; no md5 | LTX code, matmul epilogue |
| 12 | Fused RMSNorm+AdaLN | a613d669eef (+ f6b806516cc default flip) | off / on | part of 0.25 s measured with the gate fold | c | only measured together with the gate fold (PCC 0.87, VBench same); never alone vs main | LTX code |
| 13 | Conv VAE trims: exact shard, T-pad fold, W-mask fold, halo-only reader, blockings | 83c11ee2b34 07a5df97490, 1968790b040 558c15e0b97, 0e1a2585b17 2fbc74d9567 e21bcbad8d9, ce356b8815a f5ac9ecaa72 64571a953b2, 7e25dc0dbad ac892b57003 | off / on | vs ltx-rt 0.72->0.50 s; vs main unknown | a vs ltx-rt; c vs main | md5/byte-identical vs ltx-rt (jobs 033/034, 043, 469, t170 5 seeds) | shared C++: neighbor_pad_async (logical_w), conv3d (halo-only reader) |

## Not gains over main

- Gate fold into QKV (a4b20b00918, LTX_FUSE_GATE_ON_DEVICE): main already has it on by default.
  ltx-rt turned it off because it cost 1-4 VBench subject-consistency points (d per ltx-rt's own note;
  t170 saw VBench match). Keep main as is; do not port.
- Fabric AGMM to_out (290869f3e56): the commit says "measured neutral" for speed. Not a gain.
- Host core pinning: main already has host_affinity.py with LTX_PIN_CORES on.
- Matmul/AGMM/conv3d sweeps from #57265: already in main.
- Device-resident stage hand-off: off on Ring (LTX_DEVICE_RESIDENT unset), so it does not help on 4x8.
- Gemma encode trace at warmup, kernel prewarm, warmup captures: first request only.
- S2 3->2 steps (8+2): degrading (#190), excluded.
- Rejected knobs: LTX_SDPA_EXP_APPROX, LTX_FUSE_NORM_ADD, LTX_SDPA_MM_LOFI, LTX_AGMM_K2048, LTX_BATCH_ADALN_ADDS.

Main is ahead in one place: main's vae_ltx.py has the halo conv, persistent pad and in-kernel pad
masking (NP_NO_HALO_CONV, TT_LTX_NO_PERSIST_PAD). ltx-rt lost them in the 4e7d0cefcdb squash, and t48
re-built a similar path (row 13, plus 11cdcb2caea, which is only in t48). So row 13's gain vs main
has to be measured, not assumed.

## Proposed minimal draft PR to main (default on)

Rebase by hand: the ltx-rt commits sit on top of the 4e7d0cefcdb squash, so most need porting rather
than cherry-picking.

Port now (class a/b, LTX code only):
1. Row 1, QK-RoPE fusion + active-core RoPE (models/tt_dit/models/transformers/ltx/ attention code).
2. Row 2, sync removal + latent stats opt-in (pipelines/ltx/pipeline_ltx_distilled.py).
3. Row 3, host-thread noise (pipeline_ltx_distilled.py).
4. Row 4, S2 prompt reuse (pipeline_ltx_distilled.py).
5. Row 5, V2A pad-mask skip (transformer_ltx.py).
6. Rows 6+7, async encode + ultrafast/crf 20 export (pipeline export helper).
7. Row 8, audio VAE/BWE traces on by default (audio_decoder_ltx.py and the BWE decoder).
8. Row 9, resident upsampler, only if main is shown to evict and reload it per gen.

Port only after a device A/B against main (8+3, 4x8, 5 seeds, md5 first, then PCC/PSNR, VBench if they differ):
- A/B-1: LTX_YUV_EXPORT=1 vs main's float export: mp4 PSNR and e2e time.
- A/B-2: LTX_FOLD_GATED_RESIDUAL on vs off: md5 of latents, and time.
- A/B-3: fused RMSNorm+AdaLN alone vs main (gate fold on in both): md5/PCC, VBench, time.
- A/B-4: t48 conv VAE (row 13) vs main's halo/persist-pad VAE: decode time and md5. Only if faster
  does it go in, and it brings shared-op C++ (neighbor_pad_async, conv3d) that needs op tests.
- Plus #293: one standard e2e run each of main and t48 at 8+3, to give the real total.
