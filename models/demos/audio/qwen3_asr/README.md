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
- **But 256 cannot be mixed with ≥512 in one long-lived process** — see *Known limitations* below. The
  decoder MLP reshapes prefill `x` to `[1, S_pad//512, 512, -1]` only when `S_pad >= 512`; a 256-pad
  prefill takes the no-reshape path. Interleaving the two paths corrupts the model. Forcing every prefill
  onto the ≥512 reshape path keeps a single, consistent program shape.

## Known limitations

**Length-keyed decoder-state corruption → fixed-length prefill workaround.**
In the long-lived encoder+decoder server, interleaving requests of *different* real prefill (sequence)
lengths corrupts the Qwen3 decoder: it "locks" to the first request's length and thereafter emits only
the language tag + EOS (an empty transcript) for other lengths. Decoder-alone and encoder-alone are each
stable across lengths in isolation (proven by `tests/test_decoder.py::test_prefill_length_state_regression`
and `tests/test_audio_encoder.py::test_encoder_length_stability`); the trigger is the encoder/decoder
interleaving with a *varying* real sequence length — a length-keyed program-cache / KV-shape issue in
tt-metal, not in this model's code.

Workarounds shipped here (both keep the prefill length constant so the bug never fires):
- **Op level** (`tt/qwen3_asr_decoder.py`): pad every prefill to the same 512-multiple bucket.
- **Server level** (`server/qwen3_asr_server.py`, `FIXED_INFER_SEC = 14.0`): pin every `_infer` to a
  fixed 14 s audio length (pad short clips with silence, silence-chunk long audio into ≤14 s windows), so
  every request prefills at exactly one length. Cost: a small accuracy trade-off from more/shorter chunks
  (full-clip CER 0.045 → 0.065, accepted for stability) and wasted compute on padded silence for short
  clips. See `server/LONGFORM_DESIGN.md` for the tiered chunking design.

This should be root-caused and fixed in tt-metal (length-independent prefill program/KV handling) so the
fixed-length pin can be removed. Tracking issue draft: see the PR description / filed tt-metal issue.

## Reference golden

`reference/dump_reference.py` (run with the `/tmp/qwen3-asr-eval` venv) loads the CPU
model, hooks submodules, transcribes a short clip, and saves per-stage tensors +
`manifest.json`. Default golden lives outside the repo at
`/home/ttuser/ttwork/qwen3_asr_golden/` (tensors are large). Captured stages:
`conv2d1`, `conv_out`, `enc_layer0`, `ln_post`, `audio_tower`/`proj2` (= audio embeds),
`lm_head` (prefill + decode logits), plus end-to-end token text.

Verified shapes on a 12 s clip: conv_out `(12,13,1024)`, audio embeds `(156,2048)`,
prefill logits `(1,174,151936)`.
