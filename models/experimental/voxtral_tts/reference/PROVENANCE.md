# Voxtral-TTS reference — provenance

PyTorch "golden" reference for the TTNN port of Mistral AI's **Voxtral TTS** (`Voxtral-4B-TTS-2603`).
Every TTNN module we build will be validated against these files, same methodology as
`models/experimental/xtts_v2`.

## Sources

| What | Where | Pinned at |
|------|-------|-----------|
| Reference implementation (read, not vendored) | https://github.com/vllm-project/vllm-omni — `vllm_omni/model_executor/models/voxtral_tts/*`, `vllm_omni/transformers_utils/{configs,parsers}/voxtral_tts.py` | commit `8001bb155dae5798a1ae891ae2529a314c6ee99a` (2026-07-27) |
| Weights + `params.json` + `tekken.json` | HuggingFace `mistralai/Voxtral-4B-TTS-2603` | revision `b81be46c3777f88621676791b512bb01dc1cb970` (2026-03-31) |
| Paper | arXiv `2603.25551` — *Voxtral TTS* | v2 |

Unlike the XTTS-v2 effort, upstream is **not** vendored into the tree: it is a live part of
vLLM-Omni (imports `vllm`, `mistral_common`, `transformers`, `einops`, `flash_attn`) and there is
no standalone model repo to copy. Re-fetch the six files above from the pinned commit if you need
to re-read them.

## License — read before upstreaming

- **Upstream reference code** (vLLM-Omni): Apache-2.0. Compatible with tt-metal; we reimplemented
  rather than copied, so nothing needs a license header carried over.
- **Pretrained weights** (`mistralai/Voxtral-4B-TTS-2603`): **CC BY-NC 4.0 — NON-COMMERCIAL**, and
  that explicitly covers the 20 shipped reference voices. Same class of blocker as XTTS-v2's CPML:
  fine for bring-up, benchmarking and demos; **confirm with Tenstorrent legal before any product
  use.** Commercial access is via Mistral's API or a separate licence agreement.
  (Voxtral *Realtime* / *Transcribe* are Apache-2.0, but those are ASR, not TTS.)

## Runtime assets (gitignored)

`consolidated.safetensors` is 8.00 GB and never committed. Fetch into `reference/weights/`:

```
hf download mistralai/Voxtral-4B-TTS-2603 consolidated.safetensors params.json tekken.json \
    --local-dir models/experimental/voxtral_tts/reference/weights
```

`CKPT_MANIFEST.json` **is** committed: it is the checkpoint's 386-tensor manifest (name, dtype,
shape only — 34 KB of metadata, no weights), read out of the safetensors header. It lets the whole
structural test suite run with no download, and it is what the config self-checks compare against.

## Blocks

The model is four networks; **only three are portable from the public release** (see the finding
below). One reference file per block, mirroring the XTTS-v2 layout.

| # | Block | Params | Reference file | Boundary |
|---|-------|--------|----------------|----------|
| 1 | AR backbone (Ministral-derived) | 3.4B | `voxtral_backbone_ref.py` | `inputs_embeds [1,S,3072]` → `hidden_states [1,S,3072]` |
| 2 | Flow-matching acoustic transformer | 390M | `voxtral_flow_ref.py` | `h [B,3072]` → `audio_codes [B,37]` |
| 3 | Codec decoder | ~150M | `voxtral_codec_ref.py` | `codes [B,37,T]` → `waveform [B,1,T*1920]` @ 24 kHz |
| — | Codec encoder | ~150M | **not portable** | weights absent from the release |

`voxtral_common_ref.py` holds what all three share: the safetensors reader, the config constants,
`rms_norm` / `swiglu` / `gqa_attention` / RoPE / `fold_weight_norm`, the 37-codebook offsets, and
`pcc`.

Per-frame flow at 12.5 Hz — each frame is **37 tokens** (1 semantic + 36 acoustic FSQ):

```
text ──[tekken BPE, host]──► ids ─┐
voice preset ──► frames ──────────┼──► [Block 1: AR backbone] ──► h ──► [Block 2: FM, 7 Euler steps]
                                  │                                              │
                                  └──────── embed_frame(37 codes) ◄──────────────┘
                                                                          codes ──► [Block 3: codec] ──► wav
```

## Torch-only

No `vllm`, `vllm_omni`, `mistral_common`, `transformers`, `einops`, `safetensors`, `flash_attn`,
`apex`, or `numpy` is imported anywhere in `reference/`. Substitutions:

| Upstream needs | We use |
|---|---|
| `safetensors` | `SafeTensors` — 40-line header parse + `torch.frombuffer`, **seeks per tensor** so a block reads only its own slice of the 8 GB file |
| vllm's `VoxtralTTSConfigParser` | `load_params()` — `params.json` is plain JSON |
| `apex.normalization.FusedRMSNorm` | `rms_norm()` |
| `flash_attn_func` (ALiBi + sliding window) | `attention_bias()` folding ALiBi + causal + window into one additive pre-softmax term |
| `einops.rearrange` | `view` / `permute` / `reshape` |
| `torch.nn.utils.parametrizations.weight_norm` | `fold_weight_norm()` (`torch._weight_norm`, dim=0) |
| `mistral_common` tokenizer | out of scope — host-side, like XTTS-v2's tokenizer |

## Findings that shape the port

1. **The codec ENCODER is not in the public checkpoint.** Zero tensors under
   `audio_tokenizer.input_proj.*` or `audio_tokenizer.encoder_blocks.*` (verified against all 386).
   Upstream raises `RuntimeError: encode_waveforms requires encoder weights which are not available
   in the open-source checkpoint.` **Consequence: voice cloning from arbitrary reference audio is
   impossible with public weights** — only the 20 `voice_embedding/*.pt` presets work. This is the
   biggest functional difference from XTTS-v2, where we built the full on-device cloning path.
   Guarded by `test_codec_encoder_is_absent_from_released_checkpoint` so a future release that adds
   them shows up as a failing test.
2. **7 Euler steps, not 8.** `params.json` omits `n_decoding_steps`; vLLM-Omni's parser warns and
   defaults to **7**. The paper says "8 NFEs". The shipped config wins.
3. **RoPE is Mistral-native interleaved pairs**, not HF's half-split. The checkpoint is
   `consolidated.safetensors` + `params.json`, so the interleaved convention is correct; getting
   this wrong is an accuracy-only failure with no crash. Pinned by
   `test_rope_is_interleaved_pairs_not_half_split`.
4. **`n_heads * head_dim (4096) != dim (3072)`** in both the backbone and the FM transformer — the
   attention interior is wider than the residual stream, so `wq`/`wo` are not square. Same Mistral
   quirk `tt_transformers` already handles (`model_config.py:1983`).
5. **The codec's `norm_eps` is 1e-2**, three orders off the 1e-5 used everywhere else. It is in
   `params.json` and load-bearing; pinned by `test_codec_norm_eps_is_1e_2`.
6. **The FM transformer's sequence is 3 tokens long** — `[x_t, t_emb, h]` — and the velocity is read
   off position 0. Attention is bidirectional and unmasked; `rope_theta` in its config is inert.
7. **`acoustic_transformer.time_embedding.inv_freq` is absent** from the checkpoint despite being
   registered `persistent=True` upstream. It must be recomputed or loading KeyErrors.
8. **The semantic head is padded to 8320** (8192 codes + 2 specials → 128-multiple). The pad rows
   are live logits and must be masked, or the model can emit an invalid code.
9. **The codec decoder's sliding windows are 2, 4, 8, 16 — narrowest first.** Upstream threads one
   `cur_window_size` through encoder construction (halving 16→2) and the decoder inherits the final
   value and doubles it per upsample. Derived in `decoder_window_sizes()` rather than hard-coded.
10. **Semantic codebook is stored as EMA running sums**, not the codebook: the usable table is
    `embedding_sum / cluster_usage.clamp(min=1e-5)`.

## Validation status

- **Structural + wiring: done.** 45 tests pass with no checkpoint present. Every weight the
  references ask for exists in the released manifest at the right shape; both small blocks map
  1:1 onto their checkpoint tensors in *both* directions; and all three blocks run end-to-end at
  real widths on random weights (the backbone on a shortened stack, since 26 layers of fp32 do not
  fit in RAM), covering every matmul dimension, head split and reshape.
- **Numerical vs upstream: NOT done.** Three tests are skipped pending the 8 GB download, and even
  those only check self-consistency (KV-cache path == full forward, FSQ round-trip, finite output).
  A true PCC=1.0 gate against vLLM-Omni needs `vllm` + `vllm-omni` + `flash-attn` in a separate
  venv, comparing block-by-block against dumped activations. That is the next step, and no accuracy
  claim should be made before it.
