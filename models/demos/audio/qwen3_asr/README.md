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

## Reference golden

`reference/dump_reference.py` (run with the `/tmp/qwen3-asr-eval` venv) loads the CPU
model, hooks submodules, transcribes a short clip, and saves per-stage tensors +
`manifest.json`. Default golden lives outside the repo at
`/home/ttuser/ttwork/qwen3_asr_golden/` (tensors are large). Captured stages:
`conv2d1`, `conv_out`, `enc_layer0`, `ln_post`, `audio_tower`/`proj2` (= audio embeds),
`lm_head` (prefill + decode logits), plus end-to-end token text.

Verified shapes on a 12 s clip: conv_out `(12,13,1024)`, audio embeds `(156,2048)`,
prefill logits `(1,174,151936)`.
