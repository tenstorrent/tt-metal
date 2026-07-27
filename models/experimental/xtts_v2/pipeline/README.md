<!-- SPDX-License-Identifier: Apache-2.0 -->
# XTTS-v2 reference pipeline (TT-in-the-loop)

End-to-end [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) text-to-speech with the neural
blocks running on Tenstorrent. This is the bring-up harness used to validate the hand-written
TTNN blocks against the CPU reference and to measure per-block performance.

The four neural blocks run on device:

| Block | What | dtype |
|-------|------|-------|
| **1** | conditioning encoder + Perceiver resampler → `gpt_cond_latent` `[1,32,1024]` | fp32 |
| **2** | ResNet speaker encoder → d-vector `[1,512,1]` | bf16 |
| **3** | GPT decoder (30 layers): one-shot parallel **prefill** + KV-cached **decode** | bf16 |
| **4** | HiFi-GAN vocoder → waveform | fp32 |

The CPU-only front-end stays in coqui's `TTS` package: text tokenizer, the mel/STFT
front-ends that feed Blocks 1 & 2, and the host-side sampling glue for the decode loop.

## Why two Python environments

`ttnn` (tt-metal) and `TTS` (coqui) have conflicting dependencies and **cannot** share one
venv. The pipeline is therefore split into three phases across two interpreters:

- **Phase A** (`phase_coqui_pre.py`, coqui venv): tokenize, run the CPU mel front-ends, run a
  full-CPU inference to produce a **baseline wav**, and capture the exact block inputs
  (`cond_mel_in.pt`, `speaker_logmel.pt`, `prefix_emb.pt`, …) into a work dir.
- **Phase B** (`phase_tt.py`, tt-metal venv): run Blocks 1–4 on device from those captured
  inputs; autoregressively generate audio codes; write `gpt_latents_tt.pt` and (if the TT
  HiFi-GAN is available) `vocoder_wav_tt.pt`.
- **Phase C** (`phase_coqui_voc.py`, coqui venv): emit the final wav — from the TT vocoder
  output if present, otherwise coqui's HiFi-GAN on the TT latents as a fallback.

`run_pipeline.sh` orchestrates all three.

## Prerequisites

1. **tt-metal built for your card** and its `python_env` (this repo). See the top-level
   tt-metal build docs. Blackhole (p150) notes below.
2. **A coqui venv** with the `TTS` package installed (Python 3.12 works):
   ```bash
   python -m venv xtts_cpu_venv && ./xtts_cpu_venv/bin/pip install coqui-tts soundfile
   ```
3. **The XTTS-v2 checkpoint dir** containing `config.json`, `model.pth`, `vocab.json`
   (e.g. `huggingface-cli download coqui/XTTS-v2 --local-dir xtts_ref`).
4. **A reference-voice waveform** saved as a `.pt`: a torch-saved mono waveform tensor, a
   `(tensor, sr)` tuple, or a HuggingFace `{"audio":{"array":..., "sampling_rate":...}}` dict.
   ```python
   import torch, soundfile as sf
   wav, sr = sf.read("my_voice.wav")
   torch.save(torch.tensor(wav).float(), "ref.pt")   # pass --sr <sr> if not 22050
   ```

## Running

```bash
export XTTS_COQUI_PY=/path/to/xtts_cpu_venv/bin/python   # coqui venv python
export XTTS_CKPT_DIR=/path/to/xtts_ref                   # dir with config.json/model.pth/vocab.json
# TT_METAL_HOME is auto-derived from this script's location; override if needed.

cd models/experimental/xtts_v2/pipeline
./run_pipeline.sh "Your text to speak." /path/to/ref.pt ./out 22050 en
```

Positional args: `"<text>" <ref.pt> [out_dir] [sr] [lang] [dtype]`.

Outputs in `out/`:
- `work/baseline_cpu.wav` — full-CPU coqui reference.
- `tt_in_loop.wav` — TT-in-the-loop result (GPT + vocoder on device).
- `work/*.pt` — captured block I/O and intermediate latents (reused by the benches).

### Limits (model hard caps)

- Text ≤ ~400 GPT tokens (`text_pos_embedding` is `[404,1024]`); coqui warns and truncates
  past its 250-char soft limit but real caps are on tokens.
- Audio ≤ 605 codes (`mel_pos` is `[608,1024]`) ≈ **~28 s** of speech per utterance.
- Generation uses coqui's default stochastic sampling (temperature 0.75 / top-k 50 /
  top-p 0.85 / repetition-penalty 10). Because sampling is stochastic with an independent RNG,
  the TT codes will **not** match coqui's token-for-token — that is expected, not a regression.

## Performance benches

Two standalone scripts reproduce the per-component perf tables (warm, program-cache hot):

```bash
# TT per-block (tt-metal venv). Point it at a work dir from a pipeline run:
XTTS_WORK=./out/work XTTS_CKPT_DIR=$XTTS_CKPT_DIR \
  python bench_warm.py

# CPU per-component baseline (coqui venv), same sub-networks TT accelerates:
XTTS_CKPT_DIR=$XTTS_CKPT_DIR \
  $XTTS_COQUI_PY bench_cpu.py --ref /path/to/ref.pt --sr 22050 --text "Your text to speak."
```

`bench_warm.py` reports Block 1/2/4 as ms/call, Block 3 as one-shot prefill (ms) + decode
(ms/token). Reference numbers on a single **Wormhole N150** (bf16 decode), P=425 prompt:

| Component | CPU (12-thread) | N150 | Speedup |
|-----------|-----------------|------|---------|
| Block 1 conditioning | 88 ms | 18 ms | 5.0× |
| Block 2 speaker encoder | 61 ms | 34 ms | 1.8× |
| Block 3 GPT decode | 93 ms/tok | 10 ms/tok | 9.3× |
| Block 3 GPT prefill (P=425) | (in 1st fwd) | 30 ms one-shot | ~140× vs stepping |
| Block 4 HiFi-GAN vocoder | 7115 ms | 998 ms | 7.1× |

## Running on Blackhole (p150)

The scripts are arch-agnostic — they open `device_id=0` through `ttnn`, so a tt-metal build
that targets Blackhole will pick up the p150 with no script changes. To move from Wormhole to
a p150:

1. **Build tt-metal for Blackhole** and use that build's `python_env` for Phase B / `bench_warm.py`.
2. Run exactly as above; `device_id=0` selects the p150.
3. **Perf will differ** — the reference table above is Wormhole N150. Re-run `bench_warm.py` on
   the p150 to get Blackhole numbers.

Things to watch (these blocks pin specific core coordinates / L1-sharded configs that were
tuned on Wormhole; Blackhole has a different core grid and larger L1, so if anything asserts it
will be here):

- **Block 3 decode** uses `nlp_create_qkv_heads_decode` + `paged_fused_update_cache` with V
  placed on core `(1,0)` and a width-sharded LayerNorm — see `tt/ttnn_xtts_gpt_decode.py` and
  `tt/ttnn_xtts_layernorm.py`. If a sharding/grid assert fires on Blackhole, retune the core
  ranges there.
- **fp32 accumulation**: on Wormhole HiFi4+fp32 has a known accuracy caveat (the vocoder uses
  HiFi3); revisit the compute-kernel config for Blackhole if vocoder PCC shifts.
- `trace_region_size` / `l1_small_size` at device open are generous; adjust only if allocation
  fails.

Everything has been correctness- and perf-validated on Wormhole N150; the p150 path is
expected to work through the same `ttnn` abstraction but has not yet been run here.

## Files

| File | Env | Role |
|------|-----|------|
| `run_pipeline.sh` | — | orchestrates phases A→B→C |
| `phase_coqui_pre.py` | coqui | Phase A: front-end, baseline, capture block inputs |
| `phase_tt.py` | tt-metal | Phase B: Blocks 1–4 on device, generate codes + latents |
| `phase_coqui_voc.py` | coqui | Phase C: emit final wav |
| `bench_warm.py` | tt-metal | warm TT per-block perf table |
| `bench_cpu.py` | coqui | CPU per-component perf baseline |
