# Stage 06 — end-to-end pipeline (caption + lyrics -> stereo wav) on one Blackhole chip

Work log for `MiniMaxMusic3Pipeline`: diffusers' `MiniMaxMusic3Blocks` sequence (tokenize -> autoregressive
frames -> 200-frame chunk denoising -> vocoder -> crop / stitch) running the stage-02/03/04/05 device code with the
Flow-VAE vocoder on the host.

* implementation: [`../../tt/pipeline.py`](../../tt/pipeline.py) (`MiniMaxMusic3Pipeline.load / generate`),
  [`../../reference/vocoder_ref.py`](../../reference/vocoder_ref.py) (vendored `MiniMaxMusic3Vocoder` + the
  crop / stitch of `decoders.py`, Apache-2.0 header kept, `weight_norm` folded at load),
  [`../../tt/audio_metrics.py`](../../tt/audio_metrics.py) (log-mel distance, audio statistics, code repetition statistics)
* changes to earlier stages: `ARGenerator.generate(..., generator=)` accepts an existing CPU generator
  ([`../../tt/ar_generator.py`](../../tt/ar_generator.py)); `FlowTransformer.clear_caches()` frees the per-shape
  RoPE / mask / selector tensors ([`../../tt/flow_transformer.py`](../../tt/flow_transformer.py)). Stage 02-05 APIs and tests are unchanged.
* tests: [`../../tests/test_pipeline.py`](../../tests/test_pipeline.py) (gate, 7 tests, 3 min 8 s on device incl. load)
* scripts: [`../../scripts/vocoder_control_cpu.py`](../../scripts/vocoder_control_cpu.py) (host: the log-mel bar controls),
  [`../../scripts/generate_song.py`](../../scripts/generate_song.py) (device: one song + qualitative statistics JSON)
* measured numbers: [`pcc/results.json`](pcc/results.json) (written by the gate tests, `_meta` per entry),
  [`pcc/vocoder_control.json`](pcc/vocoder_control.json) (host controls), [`qualitative/*.json`](qualitative/)
  (60 s songs: statistics, timings, DRAM)
* local-only (gitignored): `generated/*.wav` (`golden_replay_seed7_10s.wav`, `free_running_seed7_10s.wav`,
  `golden_seed7_60s.wav`, `techno_seed7_60s.wav`), `generated/*.codes.pt`, `generated/gate06_*.log`,
  `generated/song_*_60s.log`, `generated/vocoder_control.log`, `generated/watcher_pipeline/`

Hardware: one chip of the P300x2 host `qbge-devex-02` (`TT_METAL_VISIBLE_DEVICES=0`, `tt-smi -s` board id
`000004613193411b`, p300c reported as P150, PCI `0000:01:00.0`), 1x1 mesh, program cache on, trace region 90 MB,
no fabric. Software: ttnn from `~/tt-metal` at `e946955cc15` (shared prebuilt binary), model code in worktree branch
`jashan/minimax-music3`; stage 04 at `d14adecff49`, stage 05 at `83093c810d9`; this stage's code commit
`22334d12389` (+ the follow-up commits listed by `git log --oneline -- models/autoports/minimaxai_minimax_music3/doc/pipeline`).
Golden: the stage-01 fp32 diffusers run (`~/mm3-bringup/reference`, seed 7, 10 s, 30 steps, 250 frames, 2 windows).
Weights: `MiniMaxAI/MiniMax-Music3` snapshot `fbdf52fbaaca799592917417eb05f1899f1255ec`.

## What was built

```python
from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline
pipe = MiniMaxMusic3Pipeline.load(mesh_device)                      # $MM3_WEIGHTS, dtype_policy="functional", warms traces
out = pipe.generate(prompt, lyrics, audio_duration=60.0, seed=7, num_inference_steps=30)
out["audio"]          # np.float32 [2, samples] in [-1, 1]      out["sampling_rate"]  # 44100
out["frames"]         # AR frames emitted (25 / s)               out["timings"]        # prefill, ar, ar_frames_per_s, dit_per_chunk, vocoder_per_chunk, total
out["codes"]          # [F, 8] semantic + residual codes         out["latents"]        # per-window uncropped [1, 128, L_k]
```

`load` puts the components on the chip largest first (backbone -> depth decoder -> DiT + condition encoder), loads the
vocoder on the host (fp32), builds the `ARGenerator` (decode inputs, depth traces) and warms: a 1-frame song captures the
backbone trace, one 1-step 200-frame DiT window compiles the DiT programs. `generate` transcribes the reference blocks:

| reference block | here |
|---|---|
| `TokenizeStep`, `AutoregressiveStep` (`max_frames = min(int(audio_duration * 25), 9000)`) | `ARGenerator.generate` with the pipeline's `torch.Generator` |
| `PrepareChunksStep` (`[0]` if F <= 200 else `range(0, F - 100, 100)`) | `chunk_starts_for` (stage 05) |
| per window: `ChunkConditionStep`, `ChunkPrepareLatentsStep` (`randn_tensor(..., generator)`), `ChunkSetTimestepsStep`, `ChunkDenoiseInner`, `ChunkUpdateStep` | `ChunkDenoiser.denoise_chunk` (stage 05) with `noise = torch.randn((1, 128, L), generator=generator)` drawn from the *same* generator the AR draws used - one generator threaded through the whole song, as in the reference |
| `VocoderDecodeStep`: vocoder per window, crop 86 latents left (windows > 0) / 258 right (windows < last) x 512 samples, `cat`, `clamp(-1, 1)` | `vocoder_ref.MiniMaxMusic3Vocoder` (fp32 host) + `stitch_waveforms` |

Trace-lifetime rule (stage 04): the AR traces are captured after every weight tensor exists; whatever the DiT allocates
later (per-shape RoPE / mask / selector caches, the per-window condition projection) is freed in a `finally` before
`generate` returns (`FlowTransformer.clear_caches`), so no buffer allocated after a capture is alive during a later
trace replay. tt-metal's once-per-process "Allocating device buffers is unsafe due to the existence of an active trace"
warning fires (as in stage 04: the prefill and the DiT allocate after the depth traces exist). Evidence that nothing is
corrupted: the determinism test (two songs back to back in one process, bit-identical latents and audio), the golden
replay (frame hiddens PCC 0.99941 = stage 04's value, measured *after* the warm-up DiT window allocated and freed its
caches), and the free-running 10 s song, whose codes and audio statistics are identical to every printed digit across three
separate processes with different preceding DiT-allocation histories (the dev run after the golden replay, the watcher run
with only the warm-up window before it, and the official gate run).

### Vendored vocoder

`reference/vocoder_ref.py` is diffusers' `minimax_music3_vocoder.py` as plain torch (54.2 M parameters: `dec_in_proj`
1x1, `conv_in` k=7, four blocks of Snake -> `ConvTranspose1d` (strides 8, 8, 4, 2) -> three dilated residual units
(dilations 1, 3, 9), Snake -> `conv_out` -> `tanh`; the two audio channels are the two folded 64-channel latent
streams). The safetensors store `weight_g` / `weight_v`; `fold_weight_norm` computes `torch._weight_norm(v, g, 0)` once
(the kernel `torch.nn.utils.weight_norm` runs on every forward) and loads it as the plain `weight`. Check: the fp32 module
on the golden latents reproduces the golden `audio.wav` to within one 16-bit PCM step (max abs sample error 3.07e-5, wav
PCC 0.999999997, log-mel RMS 0.011 dB - `pcc/vocoder_control.json::fp32`).

## Commands

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# host controls for the log-mel bar (no device, ~3 min)
$MM3_PY $MM3_MODEL_DIR/scripts/vocoder_control_cpu.py --threads 8          # -> doc/pipeline/pcc/vocoder_control.json
# gate (also: ~/mm3-bringup/checks/06.sh)
with_hw_lock timeout 5400 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_pipeline.py -m "not slow" -x -q -p no:cacheprovider
# qualitative 60 s songs (wav under generated/, statistics under doc/pipeline/qualitative/)
with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/generate_song.py --preset golden --seed 7 --duration 60
with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/generate_song.py --preset techno --seed 7 --duration 60
# watcher run of the free-running 10 s test
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=$MM3_MODEL_DIR/generated/watcher_pipeline \
  with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_pipeline.py -k free_running_10s -q -p no:cacheprovider
```

## Evidence

All numbers are from this stage's runs on the board above (`pcc/results.json`, entries carry `_meta` with the
timestamp, commit and 1-minute load average; the committed copy is from the official gate run `~/mm3-bringup/checks/06.sh`
after the stage-review fixes (`generated/gate06_final2.log`, `_meta.log_hint` = `gate06_final2`, 7 passed in 187.6 s, `GATE_OK`). The two earlier
full runs (`generated/gate06_dev1.log`, 6 passed in 190 s, uncommitted tree; `generated/gate06_final.log`, 6 passed in 188 s,
`GATE_OK`, commit `22334d12389`) produced the same PCC / log-mel / determinism values to every printed digit; only host wall
timings differ between runs.)

### Log-mel spectral-distance bar (host control, computed first)

`tt/audio_metrics.log_mel_distance`: RMS (dB) of the difference of the two log-power mel spectrograms (128 mel bands,
2048-point FFT, 512 hop, both channels, floored at -80 dB below the louder spectrogram's peak). Controls on the golden
latents / wav (`scripts/vocoder_control_cpu.py`, `pcc/vocoder_control.json`):

| control | log-mel RMS (dB) | mean abs (dB) | wav PCC vs golden |
|---|---|---|---|
| vendored fp32 vocoder on the golden latents (measures the 16-bit PCM of `audio.wav`) | 0.011 | 0.004 | 1.000000 |
| **vendored vocoder in bf16** on the golden latents (torch-vs-torch bf16 control) | **1.115** | 0.506 | 0.999973 |
| fp32 vocoder, golden latents + white noise to latent PCC 0.9996 (stage-05's per-window DiT PCC) | 0.578 | 0.346 | 0.999245 |
| fp32 vocoder, golden latents + white noise to latent PCC 0.99 | 2.739 | 1.719 | 0.985815 |
| fp32 vocoder, golden latents + white noise to latent PCC 0.98 (the latent bar) | 3.839 | 2.470 | 0.974819 |
| fp32 vocoder on unrelated noise latents (scale of the metric) | 23.780 | 19.825 | -0.007 |

**Bar: 2.0 dB RMS** = about twice the bf16 torch control (1.115 dB), three times what the measured DiT latent error
alone contributes (0.578 dB), and about half of what the latent PCC bar itself (0.98 -> 3.84 dB) would tolerate.

### Gate tests (`tests/test_pipeline.py`, 7 passed in 187.6 s, `GATE_OK`, `generated/gate06_final2.log`)

| test | bar | measured |
|---|---|---|
| golden replay (teacher-forced `sampled_codes.pt` + frame-0 codes from `sampled_raw.pt`, golden noises, 30 steps): `text_ids`, `codes`, `chunk_starts` | equal | equal |
| golden replay: `frame_hiddens` PCC vs `frame_hiddens.pt` | >= 0.99 | 0.99941 (stage 04's value) |
| golden replay: latents PCC vs golden `latents[0]` (689) / `latents[1]` (516) | >= 0.98 each | **0.99951 / 0.99923** |
| golden replay: stitched wav vs `audio.wav` (440832 samples each) | log-mel RMS <= 2.0 dB | **0.754 dB** (mean abs 0.530 dB, max 11.2 dB on single bins), wav PCC 0.99875 |
| free-running 10 s, seed 7, 30 steps | 44.1 kHz stereo, RMS > 1e-3, finite, 250 frames, duration within 1 s of frames / 25 | 9.996 s, RMS 0.0927 (-20.7 dBFS), peak 0.99999, 0 NaN, 250 frames (`max_frames`), windows `[0, 100]` |
| free-running 10 s: semantic-code distribution | most common code <= 30 % | 178 distinct codes in 250 frames, most common 2.8 %, adjacent repeat 10.4 %, longest run 5, repeated 4-grams 0.4 %, full-frame repeats 0 |
| `audio_duration` 0.4 s (10 frames -> 1 window, L = 34: the shortest realistic end-token song), 2 s (50 frames -> L = 172), 10 s (2 windows), 13 s (325 frames -> 3 windows `[0, 100, 200]`, tail 125 frames -> L = 430), seed 11, 6 steps | run; duration within 1 s | 0.395 s / 1.997 s / 9.996 s / 13.003 s, RMS 0.0025 / 0.017 / 0.025 / 0.064 |
| golden replay: per-second wav PCC vs golden across the whole clip (covers the window join at latent 431 = sample 220672) | min >= 0.99 | min 0.9905 (second 0, the quiet intro), mean 0.9979; seconds 4-6 around the join 0.9994 / 0.9988 / 0.9984 |
| golden replay: band energy (< 250 Hz / 250-2 kHz / 2-8 kHz / > 8 kHz) vs the golden clip's | each band within 0.02 | replay 0.364 / 0.512 / 0.087 / 0.037 vs golden 0.360 / 0.520 / 0.088 / 0.033 |
| same seed twice (2 s, seed 3, 6 steps) | identical | codes equal, max latent diff 0.0, max audio diff 0.0; seed 4 differs (max audio diff 0.64, different codes) |

Stage 05 measured the windows at PCC 0.99962 / 0.99953 with golden inputs; here window 0 sees the *device* frame hiddens
(0.9994 PCC) and window 1 additionally our own carry, hence 0.99951 / 0.99923 - the same ordering as stage 05's chained run
(0.99962 / 0.99932). The wav's 0.75 dB is between the DiT-only control (0.58 dB) and the bf16 vocoder control (1.1 dB):
the device pipeline with the fp32 host vocoder is closer to the golden than a bf16 torch vocoder alone would be.

### Device memory (functional policy, `pcc/results.json::load`)

| after | DRAM allocated | free |
|---|---|---|
| backbone (`MusicLLM`, bf16 attention / KV, bfp8 MLP, bf16 embedding + LM head, 10240-token paged KV cache) | 15.42 GB | 18.04 GB |
| + depth decoder | 16.77 GB (+1.35 GB) | 16.69 GB |
| + DiT (bf16) + condition encoder | 21.72 GB (+4.95 GB) | 11.74 GB |
| resident after warm-up (traces, decode buffers) | **21.74 GB of 33.46 GB** | 11.72 GB (largest free block per bank 1.46 GB of 4.18 GB) |

No allocation failure; the bfp8-DiT fallback (`load(dit_dtype="bfp8")`) exists but was not needed and is untested.
Load time with warm weight caches: 9.3 s (backbone 6.1 s, depth 1.1 s, DiT 0.8 s from `TT_DIT_CACHE_DIR`, AR init 0.6 s,
warm-up 0.5 s); the first-ever load converts the weights (minutes, stages 02 / 05).

### Timings (host wall, idle host, `pcc/results.json` and `qualitative/*.json`)

| clip | prefill | AR (frames/s) | DiT per window (30 steps) | vocoder per window (host fp32, 12 threads) | total |
|---|---|---|---|---|---|
| golden replay 10 s (250 frames, 2 windows) | 0.13 s | 17.6 s (14.4) | 3.49 s (689 latents, 116 ms/step) / 2.79 s (516, 93 ms/step) | 6.74 s / 5.08 s | 35.7 s |
| free-running 10 s, seed 7 | 0.13 s | 18.9 s (13.3) | 3.50 s / 2.76 s | 6.86 s / 5.22 s | 37.2 s |
| 60 s golden prompt, seed 7 (1500 frames, 14 windows of 200 frames) | 0.13 s | 113.5 s (13.2) | 3.54-3.58 s each (118-119 ms/step), 49.9 s total | 6.57-6.96 s each, 94.2 s total | 257.6 s |
| 60 s techno prompt, seed 7 (1500 frames, 14 windows) | 0.13 s | 113.5 s (13.2) | 3.57-3.61 s each, 50.2 s total | 6.66-7.05 s each, 95.7 s total | 259.3 s |

The AR stage runs at stage 04's 13.3-14.4 frames/s (1.8x slower than the 25 frames/s realtime); the DiT at stage 05's
~116 ms per step for a full window; the host vocoder at 6.8 s per 689-latent window is now the largest single item per
window (it is stage 07's TTNN port). First use of a new window length compiles the DiT programs once (the 13 s clip's
430-latent tail: 3.97 s for 6 steps vs 0.71 s for the two warm 689-latent windows).

### Qualitative: two 60 s songs (`scripts/generate_song.py`, `qualitative/*.json`)

Both songs were generated with `seed 7`, 30 steps, `audio_duration=60` on an otherwise idle host (load average 8.4 / 8.9 =
the pipeline process itself). Neither ended by itself: 1500 frames (`max_frames`), 14 windows of 200 frames, 60.07 s of
audio each. Listening is not possible in this headless run; the statistics below (`tt/audio_metrics.audio_stats`,
1-second windows, mono mix for the spectra) are the qualitative evidence, with the golden 10 s fp32 reference clip
(`pcc/vocoder_control.json::golden_stats`) and our free-running 10 s clip as points of comparison.

| clip | RMS (dBFS) | peak / clipped samples | 1 s windows < -60 dBFS | stereo correlation | band energy < 250 Hz / 250-2 kHz / 2-8 kHz / > 8 kHz | spectral flatness | adjacent-second log-mel correlation mean / max |
|---|---|---|---|---|---|---|---|
| golden fp32 reference, 10 s | -18.7 | 0.9999 / 1.0e-4 | 0 % | 0.50 | 0.36 / 0.52 / 0.09 / 0.03 | 0.023 | 0.74 / 0.85 |
| device golden replay 10 s (same codes and noises as the golden: the path-matched control) | -18.7 | — | 0 % | 0.53 | 0.36 / 0.51 / 0.09 / 0.04 | 0.025 | 0.74 / 0.84 |
| device free-running 10 s, seed 7 (gate) | -20.7 | 1.0000 / 2.4e-5 | 0 % | 0.55 | 0.34 / 0.53 / 0.11 / 0.02 | 0.006 | 0.87 / 0.98 |
| **device golden prompt 60 s, seed 7** (`golden_seed7_60s.json`) | -17.5 | 1.0000 / 9.7e-5 | 0 % | 0.73 | 0.50 / 0.43 / 0.06 / 0.01 | 0.005 | 0.87 / 0.99 |
| **device techno prompt 60 s, seed 7** (`techno_seed7_60s.json`) | -16.6 | 0.9999 / 1.1e-6 | 0 % | 0.72 | 0.56 / 0.36 / 0.06 / 0.01 | 0.004 | 0.89 / 1.00 |

Band energy and loudness over time (means over consecutive 10 s spans):

| clip | span | 1 s RMS (dBFS) | < 250 Hz | 250-2 kHz | 2-8 kHz | > 8 kHz |
|---|---|---|---|---|---|---|
| golden prompt 60 s | 0-10 s | -23.2 | 0.32 | 0.55 | 0.11 | 0.016 |
| | 10-20 s | -20.1 | 0.45 | 0.50 | 0.03 | 0.020 |
| | 20-30 s | -17.3 | 0.60 | 0.35 | 0.04 | 0.013 |
| | 30-40 s | -18.9 | 0.54 | 0.40 | 0.05 | 0.011 |
| | 40-50 s | -18.3 | 0.58 | 0.36 | 0.05 | 0.011 |
| | 50-60 s | -16.7 | 0.54 | 0.40 | 0.05 | 0.009 |
| techno prompt 60 s | 0-10 s | -14.3 | 0.87 | 0.09 | 0.04 | 0.002 |
| | 10-20 s | -22.3 | 0.34 | 0.57 | 0.07 | 0.027 |
| | 20-30 s | -19.4 | 0.32 | 0.57 | 0.08 | 0.023 |
| | 30-40 s | -17.9 | 0.83 | 0.12 | 0.05 | 0.001 |
| | 40-50 s | -17.8 | 0.61 | 0.32 | 0.06 | 0.006 |
| | 50-60 s | -16.9 | 0.41 | 0.49 | 0.09 | 0.008 |

Per-second RMS ranges: golden prompt -32.2 .. -12.8 dBFS (median -18.0), techno -33.0 .. -10.1 dBFS (median -17.7): the
songs have dynamics (quiet intro, louder later sections) and no silent second. Reading: the acoustic-pop prompt starts
quiet and treble-rich (a fingerpicked intro: 11 % of the energy above 2 kHz in the first 10 s), then shifts to a bass /
low-mid balance from 20 s on (the "drums and upright bass enter" of the caption) and gets louder towards the end. The
techno prompt shows a sub-heavy intro (87 % below 250 Hz, -14 dBFS: kick and sub bass), a quieter mid-rich section
(10-30 s), a second bass-dominated section (30-40 s, the "drop"), and a mixed final section. Both are far from noise
(spectral flatness 0.004-0.005; white noise is 1.0, the golden clip 0.023) and stereo but not mono (correlation 0.72-0.73;
the golden clip 0.50).

**Is the difference from the golden clip precision or sampling?** The free-running clips differ from the fp32 golden clip
(flatness 0.005-0.006 vs 0.023, stereo correlation 0.55-0.73 vs 0.50), but they are different sample paths, so that
comparison cannot separate the two causes. The path-matched control is the golden *replay* (same codes, same noises,
device DiT + host vocoder): its statistics sit on top of the golden clip's - RMS -18.66 vs -18.71 dBFS, flatness 0.0249
vs 0.0231, stereo correlation 0.526 vs 0.500, band energy 0.364 / 0.512 / 0.087 / 0.037 vs 0.360 / 0.520 / 0.088 / 0.033,
adjacent-second log-mel correlation 0.74 / 0.84 vs 0.74 / 0.85 (`pcc/results.json::golden_replay.replay_audio_stats`,
`golden_audio_stats`). The device pipeline does not tilt the spectrum or narrow the stereo image; the free-running
differences are properties of the songs the device sampled (bf16 logits -> different top-50 draws from frame 1 on).

Rhythm: the envelope-modulation spectrum (RMS in 10 ms hops, mono, 0.5-4.5 Hz) of the techno clip peaks at 2.0 Hz over
20-30 s and 2.27 Hz over the whole clip (120-136 BPM against the prompted 126 BPM); the acoustic-pop clip peaks at 3.4 Hz
throughout (204 per minute = an eighth-note pulse at ~102 BPM against the prompted 96 BPM). Both clips therefore have a
steady pulse in the prompted range; the resolution of a 10 s window is 0.1 Hz (6 BPM), so this is a coarse check only.

**Repetition / degeneracy checks.** Semantic codes (`code_stats`): golden prompt 768 distinct codes in 1500 frames,
most common 1.7 %, adjacent-frame repeat 11.8 %, longest identical run 5 frames (0.2 s), repeated 4-grams 1.4 %,
no frame whose full 8-code tuple repeats the previous one. Techno: 658 distinct, most common 1.4 %, adjacent repeat
15.9 %, longest run **17 frames (0.68 s)**, repeated 4-grams 4.7 %, no full-frame repeats. The spectral repetition
indicator (correlation of consecutive 1 s log-mel means) averages 0.87 / 0.89 with a maximum of 0.99 / 1.00; the golden
fp32 clip has 0.74 / 0.85 over its 10 s. The 17-frame run and the 1.00 maximum are the two things to flag: both are in
the techno clip, consistent with the genre (a sustained kick / sub pattern repeats bar to bar and the AR stage emitting
the same semantic code for a held sound), and both are far from a degenerate loop (the stage-04 degeneracy bar is a
most-common-code share of 30 %; we measure 1.4-1.7 %, and the 4-gram repetition is under 5 %). Nothing in either clip
looks like noise (flatness) or silence (0 % of seconds below -60 dBFS). Peaks touch 1.0 in the golden-prompt clip (0.01 %
of samples clipped by the final `clamp`; the reference clip clips 0.01 % as well).

### Watcher

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1` over `test_free_running_10s` (the full pipeline: load, warm-up, the
10 s seed-7 song through the AR traces, the DiT and the host vocoder): **1 passed in 144 s** (`generated/watcher_pipeline/pytest.log`;
the test's own log line reports the same 9.996 s / RMS 0.0927 / 178 distinct codes as the plain run, i.e. identical
codes under the watcher). `generated/watcher_pipeline/generated/watcher/watcher.log` has 16442 lines = 30 periodic dumps
of the 4 devices' core states (`k_ids`, waypoints, attach / detach) and **zero** lines matching
`exception | assert | sanitiz | overflow | fault | hang | stall | error`. Watcher runs do not write `pcc/results.json`
(`_record` skips when `TT_METAL_WATCHER` is set), so the committed evidence is the plain gate run.

## Decisions taken without anyone to ask

1. **Vocoder on the host in fp32** (as the stage prompt prescribes; stage 07 ports it). 12 intra-op threads; 6.8 s per
   full window. The bf16 host variant was measured only as the metric control, not used.
2. **One generator for the whole song.** The reference threads one `torch.Generator` through the AR top-k draws and the
   per-window `randn_tensor`; `ARGenerator.generate` gained a `generator=` argument so the pipeline does the same. The
   device's draws differ from the fp32 reference's after the first frame where the top-50 sets differ (stage 04), so
   free-running device audio is a different sample path from the golden for the same seed; the golden replay test
   injects the golden codes and noises instead. `seed=None` draws a seed from torch's global RNG and reports it.
3. **Log-mel bar 2.0 dB** from the controls above (the prompt asks for a bar justified by a torch-vs-torch bf16 control).
4. **DiT caches cleared after every song** (`FlowTransformer.clear_caches`) so that no tensor allocated after the AR
   traces were captured survives into the next song's trace replays (see the trace-lifetime rule). Cost: the RoPE / mask /
   selector tensors are re-created per window (milliseconds).
5. **Reduced step count in the duration sweep** (6 Euler steps for the 2 s / 10 s / 13 s runs and the determinism test)
   to keep the gate at about three minutes; the golden replay and the 10 s free run use the reference's 30 steps.
6. **Warm-up in `load`**: a 1-frame song on a fixed dummy prompt (captures the backbone trace) and one 1-step 200-frame
   DiT window on zero hiddens (compiles the DiT programs at S_pad 768). Other window lengths compile on first use.
7. **Context cap kept at the checkpoint's 10240 positions.** The backbone runs with `max_seq_len = 10240`
   (`max_position_embeddings`, the stage-02 context contract), so a song holds at most `10240 - prompt_len - 1` frames;
   the reference's separate caps (5000 prompt tokens, 9000 frames) can exceed that and HF would extrapolate RoPE past the
   trained positions. Raising `max_seq_len` to 14000 would fit in DRAM (the paged KV cache is 3.3 GB at 10240; 11.7 GB
   are free) but would run the model outside its advertised positions, so it is not done here. `generate` now warns up
   front when `max_frames` exceeds the room left by the prompt, returns `context_frames` and `truncated_by_context`, and
   the AR loop's `stopped_by == "context"` marks the cut. With the prompts used here (104-140 tokens) the cap is
   10099-10135 frames > the 9000-frame reference cap, i.e. it only binds for prompts longer than ~1240 tokens at the
   full 6-minute duration.
8. **`generate` returns the latents and frame hiddens by default** (`keep_latents=True`) for the tests and the server's
   diagnostics; the server stage can turn it off.

## Open risks / hand-off

* **Throughput**: a 10 s clip takes about 36 s (AR 18 s, DiT 6 s, vocoder 12 s); a 60 s clip about 258 s (AR 114 s, DiT 50 s, vocoder 95 s): 4.3x slower than realtime. Levers,
  all stage 07: TTNN vocoder, traced DiT step, bfp8 weights / KV, overlapping depth(f) with backbone(f+1).
* **Same-seed audio differs from the fp32 reference** by construction (bf16 device logits -> different top-50 draws);
  only teacher-forced replay is comparable. The 0.75 dB log-mel distance of the replay is the precision evidence.
* **Listening** is not possible in this headless run; the qualitative section reports spectral statistics only (and the
  path-matched replay control shows the device does not change the spectrum; musical quality and lyric intelligibility
  remain unverified).
* The stage-02 long-prompt precision (hidden PCC 0.982 at 5000 tokens) is inherited and untested end to end here
  (the prompts used are 104-140 tokens).
* `load(dit_dtype="bfp8")` is untested (not needed: 11.7 GB DRAM headroom).
* **Context cap** (decision 7): `frames <= 10240 - prompt_len - 1`. A 5000-token prompt yields at most 5239 frames
  (209.6 s) however long `audio_duration` is; the result carries `truncated_by_context = True` and `stopped_by = "context"`.
* End-token termination is not exercised end to end in this stage (every recorded run stopped at `max_frames` or the
  teacher codes); the `stopped_by = "end_token"` path is stage-04 evidence (seeds 5-42 all ended by themselves), and the
  chunking of an arbitrary frame count is covered by the 0.4 s / 2 s / 13 s cases.
* `keep_latents=True` returns about 196 MB of frame hiddens per 60 s song; a server that keeps results must pass
  `keep_latents=False`.
* The bar-setting latent perturbations are white noise, whose spectral signature is harsher than the DiT's structured
  bf16 error, so the "PCC 0.9996 -> 0.58 dB" mapping is an order-of-magnitude guide; the bar rests on the requested bf16
  torch control (1.115 dB) and the measured 0.754 dB sits below both.

## Stage review

REVIEW_SECTION
