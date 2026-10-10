# ttp/ltx-rt-next: project LTX gains on top of ltx-rt (task #313, 2026-10-10)

Branch for the user's ltx-server test (#212). Use git ref `origin/ttp/ltx-rt-next`.

How it was built: `origin/ltx-rt` (b9f8587ce6c) has not moved since t48 branched from it, so
`ttp/ltx-rt-next` is a fast-forward of `ltx-rt` to `origin/ttp/t48-ltx25-integrated` (f6547442b30,
356 non-merge / 371 total commits ahead), plus this file. No conflicts; ltx-rt behaviour is unchanged
except where a gain below replaces it. The 7 gains on `ttp/ltx23-main-pr` (#194 main PR) are all
already in t48 or in ltx-rt itself (mapping at the end), so nothing extra was cherry-picked.

Every flag below is read at pipeline/model construction. Unset = the default shown. ltx-server
needs no new env vars to get the defaults.

Classes: **bit-identical** (same latents/mp4 bytes), **quality-neutral** (output changes, measured
no worse), **numerics change** (on by default in t48, output drifts, checked by PCC/VBench/stills
but not bit-identical), **opt-in** (off by default).

## LTX 2.3 (the pipeline ltx-server runs)

| Gain | Flag (default) | Class | Evidence | ltx-server notes |
|---|---|---|---|---|
| Stage 2 runs 3 steps (8+3, Lightricks default) | `LTX_S2_SIGMAS` (unset = 0.909375,0.725,0.421875,0.0) | baseline | #295 f6547442b30; 8+2 is degrading (user #190) | Same as ltx-rt. Never set an 8+2 schedule. |
| QK-RoPE fused into Q/K RMSNorm, bf16-preserving | `LTX_FUSE_QK_ROPE_PRESERVE_BF16` (1; ltx-rt 0) | bit-identical | f8a8611de5a: latents and mp4 unchanged, S1 2.31->2.18 s, S2 2.57->2.46 s on 4x8 | |
| RoPE on active cores only | `LTX_ROPE_ACTIVE_CORES_ONLY` (1; ltx-rt 0) | bit-identical | f8a8611de5a | |
| No per-step replay syncs; latent stats opt-in | `LTX_STEP_SYNC` (off), `LTX_LATENT_STATS` (0) | bit-identical | 4194cd98852: same CQ order; ~17 ms stats between stages | Per-step STEP_MS log lines are gone unless `LTX_STEP_SYNC=1`. |
| Seeded noise drawn once, on a host thread | none | bit-identical | cc51d4a21bc: explicit generator gives the same values; saves ~0.1 s | |
| Stage 2 reuses stage 1's persisted prompt | `LTX_S2_PROMPT_REUSE` (1) | bit-identical | 5d993cd2f7c: same buffers, same data; 33-45 ms | |
| Skip the V2A pad-mask multiply | `LTX_V2A_SKIP_PAD_MUL` (1) | bit-identical | f793ec1c64a: md5-identical, blx03 job 161 | |
| mp4 video encode on a worker under the audio decode | `LTX_ASYNC_EXPORT` (1) | bit-identical | 63902277007: mp4 byte-identical; hides ~0.15 s | |
| Audio (AAC) encode alongside the video encode | none | bit-identical | eee3baf7c0d: same streams; export 0.35-0.40 -> 0.19 s | |
| x264 export at ultrafast crf 20, zero-copy frames | `LTX_EXPORT_PRESET` (ultrafast), `LTX_EXPORT_CRF` (20) | quality-neutral | e7588fb8718, #296, user OK #198: Y PSNR vs source 45.6 -> 47.9 dB; encode 0.65 -> 0.15 s | **mp4 files are ~3.5x larger.** `LTX_EXPORT_PRESET=veryfast` restores veryfast/crf 23. |
| Conv VAE trims: exact shard, T-pad fold, W-mask fold, halo-only reader | `LTX_VAE_EXACT_SHARD`, `LTX_VAE_FOLD_TIME_PAD`, `LTX_VAE_FOLD_W_MASK`, `LTX_VAE_HALO_ONLY` (all 1) | bit-identical vs ltx-rt | #296 row 13: md5/byte-identical vs ltx-rt (blx03 jobs 033/034, 043, 469; t170 5 seeds); VAE 0.72 -> 0.50 s | Touches shared C++ ops (neighbor_pad_async, conv3d): needs this branch's build, not a Python-only swap. |
| Gate folded into Q/QKV on device | `LTX_FUSE_GATE_ON_DEVICE` (1; not in ltx-rt) | numerics change | f6b806516cc, t170 eval pack (blx01 4x8, 5 seeds): PCC 0.87 vs reference with fused norm below, stills clean, VBench subj 0.888, img 0.559 vs 0.563. ltx-rt kept it off over a 1-4 pt VBench subject-consistency loss (#296) | Clips drift in pose/framing vs ltx-rt. `=0` turns it off. |
| Fused RMSNorm + AdaLN | `LTX_FUSE_NORM_ADALN` (1; ltx-rt 0) | numerics change | f6b806516cc, t170 (measured only together with the gate fold): warm e2e 5.939 -> 5.686 s for both | `=0` turns it off. |
| Warmup captures the Gemma encode trace and the encode path | none | first request only | e7588fb8718, ee87923c0f2 | Gen #0 pays a ~1.6 s encode-trace capture after its mp4 is written. |

Already on in ltx-rt (not project gains, listed so nobody hunts for them): on-device YUV export
`LTX_YUV_EXPORT=1`, gated-residual fold `LTX_FOLD_GATED_RESIDUAL=1`, resident upsampler
`LTX_UPSAMPLER_RESIDENT=1`, audio VAE/BWE/vocoder traces `LTX_VAE_TRACE`/`LTX_BWE_TRACE`/`LTX_VOC_TRACE=1`.

Opt-in, off by default (rejected or unproven, do not turn on for the test): `LTX_AGMM_K2048`,
`LTX_BATCH_ADALN_ADDS`, `LTX_AUDIO_OVERLAP`, `LTX_QUANT` (bf8 presets), `LTX_EULER_TAIL_TRACE`.

## LTX 2.5 (separate pipeline; only if ltx-server instantiates it)

`LTX25DistilledPipeline` (pipelines/ltx/pipeline_ltx25_distilled.py) is new code; ltx-rt has no 2.5
path. It decodes with the LTX-2.3 conv VAE by default (#207: 1080p 6s ~5.0-5.5 s warm on 4x8).
The official 2.5 DiffVAE is **opt-in** with `LTX25_DIFFVAE=1` (decode 2.313 s at 1080p 145f on 4x8,
#277/#278); DiffVAE work is on hold (users #173/#194). Under `LTX25_DIFFVAE=1` these DiffVAE paths
are on by default: `DIFFVAE_NA_EDGE_ORDER` (md5-identical, #277/#278), `DIFFVAE_NA_KEY_PHASE`,
`DIFFVAE_NA_APPROX_EXP`, `DIFFVAE_S5_2D`, `DIFFVAE_S5_PACKED_LANES`, `DIFFVAE_S5_LEAN`,
`DIFFVAE_NA_GATHER_REPHASE`. Opt-in only (not in this branch beyond t48): ttp/t81, t82, t98, t241,
t262, t263, t276.

## FastH3 / H3

None. Nothing project-made is ready for ltx-server.

## ltx23-main-pr gains -> where they are in this branch

| main-pr commit | Gain | Here |
|---|---|---|
| 70157c213e6 | QK-RoPE fusion | f8a8611de5a (t48) |
| bfc93527e92 | host-thread noise | cc51d4a21bc (t48) |
| e6b23ad361c | S2 prompt reuse | 5d993cd2f7c (t48) |
| dca60c1c777 | V2A pad-mask skip | f793ec1c64a (t48) |
| 5565232f12c | audio track alongside video encode | eee3baf7c0d (t48) |
| 05401286709 | mel-VAE trace by default | already in ltx-rt (audio_decoder_ltx.py, `LTX_VAE_TRACE` default on) |
| df9e5ecaac6 | ultrafast crf 20 export | e7588fb8718 / ee87923c0f2 (t48), `LTX_EXPORT_PRESET` |

## Not yet verified on device

This branch's tree equals t48 head (f6547442b30) plus this file. No 4x8 run of this exact ref was made
in #313 (blx03 unreachable, blx01 clamped at 900 MHz). Follow-up: one LTX-2.3 8+3 standard e2e run
through a box's broker at `-t 570`.
