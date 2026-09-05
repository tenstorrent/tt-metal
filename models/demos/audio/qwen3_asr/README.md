# Qwen3-ASR-1.7B on Tenstorrent (Blackhole / P150a)

Port of `Qwen/Qwen3-ASR-1.7B` to ttnn. Target: a single Blackhole (P150a; on the
dev host = one chip of a P300 posing as a P150). No upstream tt-metal branch exists
for this model — fresh port. Closest structural reference is `models/demos/qwen3_vl`
(Qwen3 decoder + encoder tower + projector + multimodal token splice); the audio
front-end borrows from `models/demos/audio/whisper`.

## Architecture (verified against HF config.json + qwen_asr modeling)

Single `Qwen3ASRForConditionalGeneration` ("Thinker"), three parts:

1. **AuT audio encoder** (`thinker.audio_tower`, ~300M):
   WhisperFeatureExtractor mel (`num_mel_bins=128`) →
   `conv2d1/2/3` (3×3, stride 2, pad 1, `downsample_hidden_size=480`, GELU) = 8× downsample →
   `conv_out` Linear(480·16 → `d_model=1024`, no bias) →
   `+ SinusoidsPositionEmbedding` (`max_source_positions=1500`) →
   24 × `Qwen3ASRAudioEncoderLayer` (`d_model=1024`, `heads=16`, `ffn=4096`, GELU, qkv bias=True) →
   `ln_post` LayerNorm → `proj1` Linear(1024→1024) → GELU → `proj2` Linear(1024→`output_dim=2048`).
   **Windowed attention**: bidirectional within blocks defined by `cu_seqlens`;
   block size = `n_window_infer=800` mel frames (offline). `conv_chunksize=500`.
2. **Projector** = `proj1`/`proj2` (audio 1024 → LLM hidden 2048).
3. **Qwen3 text decoder** (`thinker.model`, = Qwen3-1.7B): `hidden=2048`, `28` layers,
   `16/8` heads (GQA), `head_dim=128`, `intermediate=6144`, SiLU, RMSNorm `eps=1e-6`,
   **qk-norm**, RoPE `theta=1e6`, `vocab=151936`, `max_pos=65536`.
4. **Multimodal glue**: processor = WhisperFeatureExtractor + Qwen2Tokenizer; audio
   embeddings replace placeholder `audio_token_id=151676` (`audio_start=151669`) in the
   input-embedding sequence, then standard causal prefill + greedy decode. 30 languages.

## Prefill seqlen rule
Prefill embeds are padded to a **multiple of 512** (`tt/qwen3_asr_decoder.py:prefill_logits`), min 512.
Trailing pad rows are causal-masked from the last real token, so padding does not change the last real
token's logits. Two reasons for 512 specifically:
- Attention shards seqlen across the core grid and each shard must be tile(32)-aligned; a 128-multiple
  can yield 48-row shards → TT_FATAL. 256 satisfies alignment on its own.
- **Different 512-buckets cannot be mixed in one long-lived process** — see *Known limitations* below.
  The decoder MLP reshapes prefill `x` to `[1, S_pad//512, 512, -1]` for `S_pad >= 512`, so different
  padded lengths differ only in the batch dim `-3`, which the prefill matmul program-cache hash does not
  distinguish (512→1024 TT_FATALs). Since real prompts are always ≤512 tokens, forcing min-512 pins every
  request to the single `[1,1,512,d]` program shape and sidesteps the collision.

## Known limitations

**Length-keyed prefill corruption → fixed-length prefill workaround.**
Interleaving prefills whose padded lengths fall in *different* 512-buckets corrupts / crashes the decoder
in one long-lived process. **Root cause (confirmed on device, Blackhole P150, 2026-07-07):** a tt-metal
**program-cache collision across the MLP prefill reshape**, not a bug in this model's code.

`models/tt_transformers/tt/mlp.py` reshapes the prefill activation to `[1, S_pad//512, 512, -1]` when
`S_pad >= prefill_len_cutoff` (512 on Blackhole). So a 512-pad prefill is `[1, 1, 512, d]` and a 1024-pad
prefill is `[1, 2, 512, d]` — they differ **only in the batch dim `-3`**, which the downstream matmul's
(`ttnn.experimental.minimal_matmul` / the attention `wo` matmul) program-cache hash does not distinguish.
The program compiled for the first bucket is then wrongly reused for the second.

Reproduced (see the repro under the PR discussion):
- A **1024-token prefill in isolation runs fine** (verified directly, all drivers).
- A **512-token prefill followed by a 1024-token prefill `TT_FATAL`s** in the attention output matmul
  (`a_shape[-1] == b_shape[-2]`, "width=3072 height=2048") — the reused program has the wrong shape.
- On the current tree a 256-pad vs 512-pad mix no longer reproduces corruption (partially improved
  upstream), but the 512↔1024 collision above is deterministic.

Why the shipped model works despite this: real ASR prompts are always ≤512 tokens (a 14 s clip ≈ 200
tokens), so every request pads to **exactly** 512 → one program shape → no collision. The workaround is
therefore effectively "pin to the single 512 bucket", enforced at two layers:
- **Op level** (`tt/qwen3_asr_decoder.py`): pad every prefill to a 512-multiple, min 512.
- **Server level** (`server/qwen3_asr_server.py`, `FIXED_INFER_SEC = 14.0`): pin every `_infer` to a
  fixed 14 s audio length (pad short clips with silence, silence-chunk long audio into ≤14 s windows), so
  every request stays in the 512 bucket. Cost: a small accuracy trade-off from more/shorter chunks
  (full-clip CER 0.045 → 0.065, accepted for stability) and wasted compute on padded silence for short
  clips. See `server/LONGFORM_DESIGN.md` for the tiered chunking design.

Removing the fixed-14 s pin (to allow long single-shot / variable-length prefill) requires the tt-metal
program-cache fix — the batch dim `-3` must be part of the prefill matmul program hash. This cannot be
fixed at the model layer (bucketing still collides). Tracking issue + repro:
`docs/prefill_program_cache_collision_issue.md` in this PR.

## Install

Two environments, deliberately separate:

```bash
# 1) device side (server, demos, tests) — on top of a built tt-metal
pip install -r models/demos/audio/qwen3_asr/requirements.txt
pip install --no-deps -r models/demos/audio/qwen3_asr/requirements-processor.txt

# 2) CPU reference / golden tooling — its own venv, NEVER the tt-metal env
python3 -m venv /tmp/qwen3-asr-ref
/tmp/qwen3-asr-ref/bin/pip install -r models/demos/audio/qwen3_asr/requirements-reference.txt
```

`qwen-asr` is pinned and installed with `--no-deps` on the device side because it declares
an older `transformers` than tt-metal pins. Only its processor (prompt template + log-mel)
is used there, and `reference/qwen_asr_processor.py` imports that module without executing
the package `__init__` chain that would pull in the CPU modeling stack. The reference
tooling does need the full package, hence its own venv.

**Encoder golden requires a chunk-aligned clip (no partial final chunk).**
The AuT front-end consumes mel in 1 s chunks (100 frames at the 10 ms hop) and emits 13 encoder
rows per chunk. The CPU reference masks the audio tower's output down to `feature_lens`, so on a
clip that does not fill whole chunks the reference emits fewer rows than the ttnn port's
chunk-aligned output (e.g. a 7.62 s clip: reference `7*13 + ceil(62/8) = 99` rows vs the port's
`8*13 = 104`), and because the windowed-attention blocks then differ the encoder PCC drops to
~0.96 across *all* rows — not just the trailing ones. The port does not implement the reference's
partial-final-chunk masking; the shipped pipeline sidesteps it instead by padding every request to
a fixed length and trimming the encoder output to the processor's audio-token count
(`server/_infer`, `demo/`, `tests/test_e2e.py`). Consequence for the PCC suite: generate goldens
from a **whole-second** clip (`reference/dump_reference.py` defaults to 7.0 s and warns otherwise;
`tests/test_audio_encoder.py` fails with this explanation if the golden is misaligned).
Implementing feature-lens masking in the ttnn encoder would remove the constraint.

## Reference golden

`reference/dump_reference.py` (run in the reference venv above) loads the CPU model, hooks
submodules, transcribes a short clip, and saves per-stage tensors + `manifest.json`. The
defaults need nothing outside a clean checkout (an in-repo 16 kHz wav; output to
`$QWEN3ASR_GOLDEN_DIR` or `/tmp/qwen3_asr_golden` — tensors are large, so they stay out of
the repo). Captured stages:
`conv2d1`, `conv_out`, `enc_layer0`, `ln_post`, `audio_tower`/`proj2` (= audio embeds),
`lm_head` (prefill + decode logits), plus end-to-end token text.

Verified shapes on a 12 s clip: conv_out `(12,13,1024)`, audio embeds `(156,2048)`,
prefill logits `(1,174,151936)`.

## Tests and CI

```bash
# staged artifacts (regenerate with the reference venv, see above)
export QWEN3ASR_SNAP=<hf-hub>/models--Qwen--Qwen3-ASR-1.7B/snapshots
export QWEN3ASR_GOLDEN_DIR=/tmp/qwen3_asr_golden
export QWEN3ASR_TEXT_DECODER=/tmp/qwen3_asr_text_decoder
pytest models/demos/audio/qwen3_asr/tests/test_audio_encoder.py
pytest models/demos/audio/qwen3_asr/tests/test_decoder.py
QWEN3ASR_E2E_WAV=<16k-mono.wav> QWEN3ASR_E2E_TEXT="<expected words>" \
  pytest models/demos/audio/qwen3_asr/tests/test_e2e.py -s
```

The golden tensors and the extracted text-decoder checkpoint are too large for the repo, so
the fixtures skip when they are absent — convenient on a dev box, but on a runner that is
supposed to have them staged a skip would hide exactly the breakage these tests exist to
catch. Set **`QWEN3ASR_REQUIRE_ARTIFACTS=1`** to turn every such skip into a hard failure.

Use that flag whenever this suite runs unattended (a pipeline leg, a nightly, a bring-up
script): point `QWEN3ASR_GOLDEN_DIR` / `QWEN3ASR_TEXT_DECODER` at the staged artifacts and set
the flag, and a missing artifact, a dependency break or a `tt_transformers` API change fails
loudly instead of reporting a green skip. `tests/test_e2e.py` needs no staged audio — it runs
on the in-repo clip above.

> Wiring this into the shared model pipelines (`tests/pipeline_reorg/`) is deliberately left
> out of this PR; see the follow-ups in the PR description.
