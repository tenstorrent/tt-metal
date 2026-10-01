# LTX-2.3 distilled: prompt-adherence investigation — next steps

Context (2026-09-28): poor prompt adherence at the served shape (153 frames, 25 fps, 1088x1920).
Golden CPU connector embeddings fed into the ttnn pipeline did not fix it, so the Gemma encoder and
connectors are ruled out. Code comb found two divergences from the diffusers reference downstream of
the encoder, plus a test-coverage gap.

## Step 1 — fp32 timestep embedding path (highest expected impact)
The sinusoidal timestep embedding runs in bf16 on device: `LTXAdaLayerNormSingle` is built with
`dtype=ttnn.bfloat16` in `models/tt_dit/models/transformers/ltx/transformer_ltx.py` and the pipeline
uploads the timestep as bf16 (`pipeline_ltx_distilled.py` `_tt_timestep.update`). `t * freq` in bf16 at
t~1000 corrupts ~50/256 sinusoid channels. CPU emulation through the real checkpoint weights:
`prompt_adaln_single` (prompt K/V shift/scale, every block) output rel. error 29-35% at stage-1 sigmas;
`adaln_single` <1%. Wan already uses fp32 for this path; `tests/models/ltx/test_embeddings_ltx.py`
tests fp32 only.
Action: compute the sinusoid in fp32 (fp32 timestep upload + fp32 `Timesteps`), cast to bf16 before the
MLP, regenerate the failing prompt and compare.
Emulation script: /tmp/claude-4236/-home-rsalman-tt-metal/bfcf32f3-50e9-4979-852f-24bc6bf54837/scratchpad/ts_emb_bf16.py

## Step 2 — pass fps to the video RoPE
`prepare_video_rope` divides the time axis by `fps` (matches reference `transformer_ltx2.py` rope), but
neither call site passes it: `pipeline_ltx_distilled.py` `_prepare_stage_statics` and
`pipeline_ltx.py` `call_av`. At 25 fps the video self-attention runs on a 24 fps timeline while the
A/V cross-PE uses 25. Action: add `fps=self.fps` at both call sites; extend
`test_fps_conditioning_ltx.py` to assert the pipeline passes it.

## Step 3 — full-depth parity test at the served shape
`test_transformer_ltx.py` covers 1 layer, PROMPT_LEN=32 (prod 1024), random prompts, 145-frame shapes,
PCC floor 0.992 / RMSE 0.15. Action: add a 48-layer single-step parity test at (20, 34, 60) and
(20, 17, 30) using the golden embeddings in `ltx_ref_dumps/` and real latents, comparing velocity
against the diffusers reference on CPU, so adherence-relevant drift is measured directly.

## Step 1 — executed 2026-09-28 (uncommitted working-tree changes)
Changed: `layers/embeddings.py` (`_LTXTimestepEmbedding`: sinusoid always fp32, cast to MLP dtype after),
`transformer_ltx.py` (host entry uploads timestep / per-token timestep in fp32),
`pipeline_ltx_distilled.py` (`_tt_timestep`, `_tt_video_ts_pair` uploaded fp32),
`tests/models/ltx/test_transformer_ltx.py` (two helpers upload fp32).
Device check (1 chip, real prompt_adaln_single weights, scratchpad/adaln_device_check.py):
bf16 sinusoid rel. error 15-34% at the distilled sigmas -> fp32 path 0.7%.
A/B at the served shape (153f, 25fps, 1088x1920, seed 10, prompt "sculptor, accelerating, salt flat,
sunrise, gradual zoom out, anamorphic film", 4x8 ring, untraced, RUN_VBENCH=0):
  ltx_ab/A_fp32ts_153f25_seed10.mp4 (fix)  vs  ltx_ab/B_bf16ts_153f25_seed10.mp4 (baseline)
  Both render a bronze figure on a plinth on a salt flat at sunrise; the fix changes lighting/texture
  detail, not the composition. Neither clip reproduces the reported adherence failure at this seed.
  Frame strips: ltx_ab/frames/AB_strips.png. Logs: ltx_ab/A_fp32ts.log, ltx_ab/B_bf16ts.log.
Open: need the exact failing seed / prompt / config to reproduce the reported failure. Steps 2 and 3
still pending. Pre-existing 145f/24fps clips backed up to ltx_baseline_pre_fp32/.

## Side issue — noisy audio from the media server (2026-09-29)
Server (~/tt-inference-server/tt-media-server/start_ltx_fast.sh): LTX_TRACED=1, LTX_YUV_EXPORT=1,
TT_DIT_CACHE_DIR=tt-metal/tt_dit_cache, 153f/25fps. Same ttnn build, same pipeline source, same weight
cache contents as the clean pytest runs.
Evidence: server clips today (153f@25, traced) have hiss (62% of energy in 2-8 kHz, 18% above 8 kHz,
rms 0.066); server clips from Sep 25 (145f@24, traced) were clean (93% below 300 Hz, rms 0.24);
pytest clips at 153f@25 untraced were clean. Noise is structured (kurtosis 7.6, flatness 0.42,
L/R corr 0.79 at lag 0), no periodicity at the 8-way T-shard period -> mis-processed signal, not a
gather seam. Video frames are fine.
Not yet tested (needs the device, server holds it): traced pytest run at 153f@25 (LTX_TRACED=1,
NUM_FRAMES=153 FPS=25) with LTX_DUMP_AUDIO_LATENT set, then decode the dumped latent eagerly and via
the traced path to bisect denoise-trace vs vocoder/BWE trace. Note: the BWE trace has no env gate
(use_trace_bwe = traced), so LTX_VOC_TRACE=0 alone does not make the audio path eager.

### Audio bisect results (2026-09-29, device runs; machine may be shared — check `fuser /dev/tenstorrent/*` before launching)
- Full pipeline traced at 153f/25 (test_pipeline_distilled, LTX_TRACED=1): all 3 gens hissy
  (ltx_audio_trace/clips/), stage-2 audio latent healthy (whiteness 0.50). Latent dumped to
  ltx_audio_trace/audio_latent_153.pt. => denoise OK, traced decode broken.
- test_audio_decode_girl on that latent (4x8 line, 153 tokens):
  A eager: PCC 0.9955 vs torch oracle (correct).
  B vocoder+BWE traced: first decode returned, second call hung (spinning thread, 14 min), killed.
  C BWE-only traced (LTX_VOC_TRACE=0): output dead/NaN (rmse/sigma 0.98 every interval, PCC nan,
    t=1s interval NaN). Log shows Metal "Allocating device buffers ... active trace" warning.
  D traced, 145 frames, synthetic latent: hung inside the FIRST cold decode (threads asleep, no
    compiles) -> the standalone traced test is itself unreliable here; killed. E skipped.
Conclusion so far: the BWE (and likely vocoder) trace path corrupts/NaNs output at the served shape;
eager decode of the same latent is correct. Workaround: LTX_TRACED=0 for audio (no finer env gate
exists; use_trace_bwe = traced). Fix direction: allocate all lazily-created audio-decode device state
(resampler _conv1d_cache, tpad masks, mel-STFT constants, trace I/O) before any capture, or add an
LTX_BWE_TRACE gate and validate traced decode against the oracle in CI at 153f/25.

### Traced-audio timeline (2026-09-29 afternoon)
- Traced pipeline at 145f/24 (HEAD + fp32 edits): all 3 gens hissy. Same with the fp32 edits stashed
  (pre-edit code): hissy. => not shape-specific, not caused by the timestep edits.
- Yesterday's "clean" 22:15 clips are byte-identical copies of one file (not 3 traced gens); the 21:10
  VBench replay clips (traced) already had broken audio. 3 of 7 Sep 25 server clips were near-silent
  (matches the dead traced-BWE output of bisect config C). So traced audio has been flaky/broken on this
  box at least since Sep 25, and consistently broken since Sep 28.
- Machine rebooted 2026-09-29 06:57; still hissy after. Firmware bundle 19.12.0.0, clang-20 from July,
  no apt changes. Sep 28 build was a clean full build (ninja log: 1538 steps, all 20:07-20:10) of the
  same native source as the Sep 25 build (HEAD == merge-base; fast-eval = HEAD + 1 python commit that
  only adds fps=self.fps to prepare_video_rope + tests). Submodules identical (umd 53809412 both).
- Chip is harvested (NUM_L1_BANKS=120, NUM_DRAM_BANKS=8; two-board 8+24 chip topology) — a plausible
  machine-specific input to allocator layout if the bug is a post-capture allocation clobbered on replay.
- Instrument: TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 (tt_metal allocator
  trace allocation tracker) names buffers allocated while a trace is live. Run launched:
  ltx_audio_trace/traced_145_tracked.log.
- Server workaround already applied locally: start_ltx_fast.sh defaults LTX_TRACED=0.

### CORRECTION (2026-09-29 18:10) — traced audio path is NOT corrupted
- Traced vs untraced stage-2 audio latent, same prompt/seed (pond prompt, 145f/24): bit-identical
  (PCC 1.000000, torch.equal True). Untraced clip for that prompt has the same broadband spectrum I had
  been calling "hiss": the prompt asks for water laps + crickets. My spectral yardstick compared
  different prompts (wind-rumble sculptor clip vs guitar/crickets clips) and was invalid.
- Stage-by-stage eager-vs-traced bisection of the vocoder and BWE device graphs: bit-exact at every
  stage (capture and replay). ttnn.chunk under trace: exact (micro-test). Trace allocation tracker found
  nothing audio-critical. => Retract: "traced BWE/vocoder decode corrupts audio". The dead/NaN output
  and hangs seen in test_audio_decode_girl under LTX_TRACED=1 are a standalone-harness artifact (no
  eager warm before capture), not the pipeline path.
- All diagnostic edits reverted (audio CCL manager, vocoder prep_run/stage hook, chunk->slice, step
  temporaries dealloc, temp test file). Working tree = fp32 timestep change only (+ user's files).
- Pending: same prompt + seed at 153f/25, traced vs untraced, latent PCC and both clips kept for
  listening (ltx_audio_trace/clips/same153_*). The reported "terrible noisy audio" is therefore either
  prompt-appropriate broadband sound, or something specific to the server request path / 153f-25fps
  conditioning that untraced pytest reproduces too — to be judged by ear on these paired clips.

### ROOT CAUSE (2026-09-29 18:40) — stale mel-decoder weight cache, not tracing
User listened: every clip in ltx_audio_trace/clips (traced AND the untraced pond clip) is crackle with
the right sound envelope (lip-synced) -> decoder-side corruption. Yesterday's good untraced clips used
the DEFAULT cache (~/.cache/tt-dit); the server and every run today used TT_DIT_CACHE_DIR=tt_dit_cache.
Full checksum: only `audio_dec_cin55f0111e` differs between the two caches — 22 mel-decoder conv
tensorbins (mid.*, up.* conv weights), 80-87% of bytes, both weight-like (a different prepared layout).
tt_dit_cache copy written 2026-09-25 17:41 (binary then = previous main build, python = fast-eval);
default copy written 2026-09-28 20:14 by the current HEAD build. Prepared conv3d weights come from the
device op prepare_conv3d_weights (layout depends on C_in_block and the op version); the cache key has
no binary version, AND `audio_dec_*` is keyed by conv3d_blocking_hash(self._vocoder_with_bwe) — the
VOCODER's blocking, not the mel decoder's (audio_decoder_ltx.py ~L253). Same-prompt evidence:
traced vs untraced audio at 145f/24 bit-identical (both bad, same stale cache); traced(tt_dit_cache)
vs untraced(default cache) sculptor at 153f/25 PCC 0.006 — that was the cache, not the trace.
Fix (server): move aside tt_dit_cache/ltx-2.3-22b-distilled-1.1/audio_dec_cin55f0111e so it
regenerates (or point TT_DIT_CACHE_DIR at ~/.cache/tt-dit); LTX_TRACED=1 can stay.
Fix (code): key audio_dec by the mel decoder's own blocking hash; include a tt-metal build id in
tt_dit cache paths so a rebuild can never load prepared weights from another binary.
Pending confirmation: E1/E2 (default cache, 153f/25 untraced+traced, clips_defaultcache_*), and a
scratch-dir regeneration with the current binary compared byte-for-byte to both caches.

### CONFIRMED (2026-09-29 20:35, after device reset)
- E1 untraced + E2 traced, sculptor seed 10, 153f/25, DEFAULT cache: audio PCC 1.0000 vs yesterday's
  good clip; PCC 0.006 vs today's tt_dit_cache clip. Tracing exonerated completely.
- Regenerating the audio caches with the current binary into a scratch dir: audio_dec matches the
  Sep 28 default cache 57/57 files, differs from tt_dit_cache in the same 22 files; audio_voc matches
  both. Preparation is deterministic; tt_dit_cache/audio_dec_cin55f0111e was stale (older build).
- Action taken: tt_dit_cache/ltx-2.3-22b-distilled-1.1/audio_dec_cin55f0111e renamed to
  *.stale-2026-09-25 (reversible). Next server start regenerates it (~1 min). LTX_TRACED=1 is fine.
- Remaining code work: key audio_dec by the mel decoder's own blocking hash (audio_decoder_ltx.py
  ~L253) and include a tt-metal build id in tt_dit cache paths. Also: test_audio_decode_girl under
  LTX_TRACED=1 hangs/returns dead output (standalone harness captures without an eager warm) — a
  separate test-harness bug worth fixing so traced audio has a real parity gate.
- Device hang during the untraced pond run (PCIe chip 16, 18:19) required a board reset; cause unknown.

### Audio reference suite on 4x8 Ring, 145f/24 (2026-09-30, regenerated cache)
PASS: test_stage_a_audio_decoder (mel-VAE vs diffusers, PCC 0.998), test_stage_b_vocoder (PCC 0.99),
test_audio_decode_e2e_psnr (full chain vs CPU reference, >= 28 dB). => cache regeneration confirmed.
FAIL: test_stage_c_vocoder_with_bwe at _assert_sharded_matches_unsharded("stage_c_full"): max|d| 3.1e-2
(gate 5e-3). Located with a temporary locator test (log: ltx_audio_trace/stagec_spike.log):
- vocoder half and BWE residual: 7e-5 to 1e-4 in every config/length (clean).
- resample_skip (VocoderWithBWE._resample_device -> UpSample1d, built WITHOUT a parallel config, i.e.
  unsharded in both objects): 3.5e-2 / ~160-200 samples over 5e-3 at the test's 120-frame synthetic
  mel (19200-sample input, 57600 out), max index differs between calls (100467 vs 85587 vs 108627)
  => non-reproducible output at that shape: uninitialized-read/garbage stretch in the depthwise tap
  filter plan for (B=1, T_pad~19242, C=2, K=43). At production length (601 frames, 96160 in) the skip
  path is bit-exact (0.0) and stage_c_full is 1.0e-4 to 1.3e-4, in both T+C (test) and T-only (prod).
- Note: the test's 4x8 config is T+C sharding (_audio_parallel_config), production is T-only
  (LTX_AUDIO_CHANNEL_TP unset); both behave the same here.
Impact: not the served 6 s clips; would affect clips ~1.2 s or shorter. Action: file the tap-filter
short-T bug (audio_ops.depthwise_tap_filter plan selection at C=2,K=43), and either fix or make the
stage C test use a length whose plan is exact on factor-8 meshes.

### New quality gates for the served 6 s shape (2026-09-30)
File: models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_quality_4x8.py (4x8 BH Ring, traced,
145f/24fps/1088x1920). Both PASS on this box with the regenerated cache:
- test_ltx_6s_audio_matches_torch_reference: real traced pipeline, audio latent captured on the way into
  decode_audio, decoded on CPU with the diffusers reference from the same latent. Measured gen0 and gen1
  (pure replay): PSNR 28.72 dB, PCC 0.984, rms 0.19, 0% clipped; latent whiteness 0.40, zeros 0.05%;
  replay vs capture: audio latent PCC 1.000000, video PSNR inf (bit-identical). Gates: PSNR >= 28 dB,
  PCC >= 0.95 (0.99 was too tight for real content), rms > 5e-3, clipped < 1%, latent sanity, replay
  latent PCC >= 0.999, replay video PSNR >= 35 dB. ~5 min.
- test_ltx_audio_weight_cache_matches_regeneration: regenerates audio_dec/audio_voc with the running
  binary into a scratch dir and requires the live TT_DIT_CACHE_DIR copy to match byte-for-byte
  (57/57 and 827/827 today). Skips when TT_DIT_CACHE_DIR is unset. ~2 min.
Not yet wired into CI (tests/pipeline_reorg/models_e2e_tests.yaml); needs LTX_CHECKPOINT as a local
path and TT_DIT_CACHE_DIR pointing at the cache the server uses.

### Ported (2026-09-30): real audio latent fixture for test_audio_decode_girl
From origin/smarton/ltx-rt-pr2-traced-avgen-20260904 (the named branch smarton/ltx-av-conv-perf does not
exist; no branch adds audio test functions HEAD lacks). Added
models/tt_dit/tests/models/ltx/fixtures/girl_audio_latent.npy (39 KB fp16, (1,151,128), real 145f@24
generation) and made test_audio_decode_girl default to it (AUDIO_LATENT still overrides; synthetic
fallback when NUM_FRAMES changes the latent length). Validated on 4x8 line, untraced: fixture picked up,
oracle PCC 0.99936, worst 1 s window -22.5 dB, PASSED in 62 s.

## 2026-09-30 — Experiments against ~/LTX-2 (reference source)

### Experiment 3: Euler step precision (bf16 in-place vs reference-style fp32 accumulate)
Code: `LTX_EULER_FP32=1` env gate in `_denoise_no_guidance` (pipeline_ltx_distilled.py, T2V branch, video+audio):
`lat = bf16( f32(lat) + v_out_f32 * dt * mask )` — one rounding per step, like ltx_core `EulerDiffusionStep`.
Default path rounds velocity to bf16, multiplies by dt in bf16, adds in bf16.
Runs (untraced, 4x8 ring, seed 10, default guitar prompt, 1088x1920): `ltx_exp3/euler_{bf16,fp32}_{145f24,153f25}_seed10.mp4`,
audio latents `ltx_exp3/audio_latent_*.pt`, logs `ltx_exp3/run_*.log`, analysis `python ltx_exp3/analyze.py <tag>`.
145f/24 result: audio latent PCC 0.992, audio waveform PCC 0.83, video mean PSNR 25.6 dB (frame PCC 0.956).
153f/25 (served) result: audio latent PCC 0.850, audio waveform PCC 0.44, video mean PSNR 21.9 dB (frame PCC 0.902).
Noise floor: an identical bf16 repeat is bit-exact (latent PCC 1.00000, video PSNR 108 dB), so all of the above is the step precision.
=> the step precision alone moves the trajectory a lot (8-step distilled sampler is that sensitive); it does not say
which is closer to Lightricks. Experiment 1 is the arbiter (compare both variants to the CPU reference under injected noise).

### Experiment 1: injected-noise end-to-end parity vs Lightricks DistilledPipeline (CPU)
- Reference venv: `~/ltx2-ref-venv` (torch 2.14 cpu, editable ltx-core + ltx-pipelines from ~/LTX-2, torchaudio).
- Driver: `ltx_exp1/ref_driver.py` (run via `ltx_exp1/run_ref_served.sh`): dumps `embeds.pt`, `noise_{s1,s2}_{video,audio}.pt`
  (GaussianNoiser draws — already patchified `b (f h w) c` / `b t (c f)`, i.e. our token layout), `s1_*`, `upsampled`, `s2_*`, `audio.wav`, `out.mp4`.
  Reference facts confirmed while wiring: noise is drawn in bf16 from a seeded torch.Generator on the patchified state
  (so seeds are NOT comparable with ours -> injection is required); stage outputs are unpatchified; same generator continues into stage 2.
- Our hooks (env-gated, pipeline_ltx_distilled.py): `LTX_INJECT_NOISE_DIR=<ref dump dir>` replaces the s1/s2 video+audio noise draws;
  `LTX_EMBEDS_OVERRIDE=<embeds.pt>` replaces the connector embeddings; `LTX_DUMP_LATENT_DIR=<dir>` dumps embeds/s1/upsampled/s2 latents.
- Compare: `python ltx_exp1/compare.py <ref dir> <tt dir>` (PCC / rel err per stage).
- Plan: (1) reference run at 153f/25 (hours on CPU); (2) our run with injected noise, then with injected noise + ref embeds,
  each for bf16 and fp32 Euler; (3) per-stage PCC tells where we diverge (s1 = transformer numerics, upsampled = upsampler,
  s2 = stage-2 numerics + RoPE fps) and which Euler variant tracks the reference.

#### Status 2026-09-30 03:30
- Reference CPU run (`ltx_exp1/ref_153f25_seed10`, log `ltx_exp1/ref_served.log`): stage 1 = 1273 s (8 x ~160 s), upsample 6 s,
  stage 2 ~1036 s/step (3 steps) then audio + video decode. First attempt was OOM-killed at 540 GB: driver lacked
  `torch.inference_mode()` (autograd kept all activations) — fixed; second run peaks at ~45 GB.
- Device side: the first injected run (`ltx_exp1/run_tt_injected.sh bf16`) hung at stage-1 step 1 (03:09) and wedged the board:
  tt-smi reports `Read 0xffffffff over PCIe ID 16`. `tt-smi -r` and two `tt-smi -glx_reset` attempts failed with
  `POST_RESET failed for device 16`. Needs the hardware-level reset used on 09-29. Same symptom as the 09-29 untraced hang;
  the injected noise is plain N(0,1) bf16 (same distribution as our seeded draw), so the hang is unlikely to be data-driven.
- Once the device is back: `for c in bf16 fp32 bf16_refemb fp32_refemb; do ltx_exp1/run_tt_injected.sh $c; done`
  then `python ltx_exp1/compare.py ltx_exp1/ref_153f25_seed10 ltx_exp1/tt_<cfg>` for each.

#### Audio decode chain cleared against ltx-core (2026-09-30 04:05, CPU only)
Same reference s2 audio latent decoded three ways (`ltx_exp1/audio_oracle_check.py`, `ltx_exp1/ref_audio_decode_dtype.py`):
- our torch oracle (`test_audio_ltx._decode_audio_reference`) vs ltx-core with fp32 mel decoder: **PCC 1.00000** (identical chain);
- ltx-core default (bf16 mel decoder, fp32 vocoder) vs its own fp32-decoder variant: PCC 0.9905 / ~30.5 dB — that is the entire
  gap between our oracle and Lightricks' serving output, and it is decoder dtype, not an implementation difference.
- Our device mel decoder runs bf16 like Lightricks'; device vs oracle was PCC 0.984 / 28.7 dB (quality test), i.e. the same order
  as the bf16-vs-fp32 decoder gap. => residual audio-quality complaints are upstream of the decoder (latent quality: sampler precision,
  transformer numerics), not in the decode chain.

#### 2026-10-01 — fp32 Euler under tracing produced static (server), fixed in the working tree
Server (LTX_TRACED=True in dit_runners.py) with LTX_EULER_FP32=1: output was complete noise; flag off: normal.
Cause: the first fp32 Euler implementation allocated fresh device tensors every step (typecast, multiply, add,
typecast back) while the captured traces were live. Per the pipeline's own trace-I/O notes, buffers allocated
after capture land in a trace's activation region and are clobbered on replay. Exp3 ran untraced, so it never hit this.
Fix (uncommitted): `_euler_step_fp32` uses only in-place ops / `output_tensor=` into scratch that
`_prepare_stage_statics` reserves alongside the baked trace inputs (fp32 masks, fp32 latent shadows, fp32 velocity),
so the step allocates nothing. New state fields in pipeline_ltx.py (`_tt_*_pad_mask32`, `_tt_*_lat32`, `_tt_*_vel32`).
Verify (device must be free): `ltx_exp3/run_exp3_traced_fp32.sh` — traced fp32 at 153f/25 must match the untraced
fp32 clip (traced vs untraced was bit-identical for the bf16 step).
Update 23:15 — the traced noise is NOT the Euler step. With the zero-allocation fp32 step, traced video is bit-identical
to untraced fp32 (PSNR 108 dB) but the mp4 audio is crackle (oracle-vs-mp4 PCC 0.01 after AAC-lag alignment, vs 0.99
for the untraced clip). Traced **bf16** (flag off) on today's binary shows the same: video bit-identical, audio PCC 0.01.
So the traced audio decode is broken on today's build (Oct 1 20:17, umd bump f70cc57->5380941) regardless of the flag;
the server's "normal" bf16 run very likely has bad audio too. Queued: untraced fp32 on today's binary (cache/binary
check), traced fp32 with per-gen latent dumps (gen-0 latent vs untraced), eager audio suite, traced audio tests
(6s quality test, test_audio_decode_girl). Tools: ltx_exp3/oracle_vs_mp4.py (latent -> torch oracle vs mp4 track).

#### 2026-10-01 23:30 — RESOLVED: the "stale audio cache" is a shared-key conflict between the server and pytest
Facts (all on today's binary, served shape, same seed/prompt; tools in ltx_exp3/):
- Zero-allocation fp32 Euler step: traced == untraced bit-exact for video (PSNR 108 dB) AND the audio latent (PCC 1.00000).
  The Euler fix is verified under tracing. (ltx_exp3/euler_fp32traced3_153f25_seed10.mp4, audio_latent_fp32traced3_153f25.pt)
- The crackly audio in every pytest run today came from the mel-decoder weight folder, not from tracing or the Euler step:
  untraced bf16/fp32 pytest runs with the server-regenerated folder (md5 fffc5dfcc5bb) -> crackle (oracle-vs-mp4 PCC 0.01);
  the same run with a pytest-regenerated folder (d8de07dce4fd, at 145f or 153f alike) -> PCC 0.99.
- The server decodes ITS OWN regeneration fine (22:50 server clip: natural audio), pytest decodes its own fine. Each
  consumer prepares `audio_dec_cin55f0111e` differently (same file sizes, different content; 22/57 files) under the SAME
  cache key (keyed by the vocoder's blocking hash). Whoever regenerates first wins; the other consumer gets crackle.
  This also explains 09-25/09-29 ("stale cache") and my 10-01 21:50 "cleanup", which installed the pytest layout into the
  server cache and broke the user's next server start (together with the fp32 allocation bug -> noise video + crackle).
- Why the two layouts differ is NOT yet determined (candidates: mel-decoder C_in_block / exact-table hit differing between
  the server's construction path and the test harness). Next: log get_conv3d_config picks for the mel decoder in both.
Practical rule: NEVER point pytest at the server's TT_DIT_CACHE_DIR. Server: tt-metal/tt_dit_cache (its own layout, intact).
Pytest: tt-metal/tt_dit_cache_pytest (hardlinked clone of the server cache with a pytest-made audio_dec; costs no disk).
Code fix direction: key audio_dec by the mel decoder's own blocking hash AND whatever construction input differs.
