# XTTS-v2 GPT (decoder-only, VQ codes) — TTNN bringup (Block 3)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: **transformer core brought up**, 2026-07-17.
  - **Default = native bf16 + SDPA: PCC 0.99972** (gated at 0.999) — the fast path we
    intend to run on TT.
  - Opt-in `HIGH_ACCURACY=1` = fp32 + manual attention: **PCC 0.99996** (gated at 0.9999).
- Owner: acicovic
- Scope done: prefill forward of the full 30-layer GPT2 transformer core + both final norms,
  single unpadded sequence (batch=1, S=64). Not yet done: KV-cache decode loop,
  embeddings/heads (kept on CPU / out of this block for now).

## Role in pipeline
The autoregressive core. Takes conditioning latents (Block 1) as a prefix + text tokens
(Block 0), and (with `return_latent`) emits `gpt_latents` for the vocoder (Block 4).
**This block, as brought up, is the transformer core only**: `inputs_embeds -> latents`.
Embedding lookups, position embeddings, and the text/mel heads are outside this block
(cheap; done on CPU when building the golden input).

## Block boundary under test (IMPORTANT)
```
inputs_embeds [1, S, 1024]
  -> 30x GPT2 block (causal):  x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
  -> ln_f          (gpt.gpt.ln_f)     # GPT2's final LayerNorm
  -> final_norm    (gpt.final_norm)   # XTTS's extra LayerNorm
= latents [1, S, 1024]
```
- Transformer = HuggingFace `GPT2Model` with the built-in positional embedding (`wpe`)
  **nulled to zeros** → `hidden = inputs_embeds` (no positional addition inside the core;
  XTTS adds learned pos embeddings *before* the core, i.e. upstream of this block).
- For a **single unpadded sequence (batch=1)** the padding mask is all-ones, so attention
  is standard **GPT2 causal**. The golden uses S=64 (n_text=16 + n_mel=48).
- Latents = `final_norm(ln_f(...))` — note **two** trailing LayerNorms.

## Confirmed architecture (from config.json + checkpoint keys)
| Param | Value |
|---|---|
| n_layer | 30 |
| n_embd | 1024 |
| n_head | 16 (head_dim 64) |
| n_inner (FFN) | 4096 |
| activation | `gelu_new` (tanh approx) → `ttnn.gelu(variant=ttnn.GeluVariant.Tanh)` |
| layer_norm_eps | 1e-5 |
| GPT2 Conv1D weights | stored `[in, out]` → matches `ttnn.linear` directly (NO transpose) |

Checkpoint keys per block: `gpt.gpt.h.{i}.{ln_1,attn.c_attn,attn.c_proj,ln_2,mlp.c_fc,
mlp.c_proj}`; endpoints `gpt.gpt.ln_f.*`, `gpt.final_norm.*`.
Other (not in this block): `gpt.text_embedding` (6681,1024), `gpt.mel_embedding` (1026,1024),
`gpt.text_pos_embedding.emb` (404,1024), `gpt.mel_pos_embedding.emb` (608,1024),
`gpt.text_head`, `gpt.mel_head`, `gpt.conditioning_encoder`+`gpt.conditioning_perceiver`
(the latter two = Block 1).

## Reference / golden
- Reference implementation: coqui **XTTS-v2** checkpoint (the real weights), run through HF
  `GPT2Model` (transformers 5.10.2, already in `python_env`) with `wpe` nulled + a separate
  `final_norm` LayerNorm. No coqui-tts install needed (avoids clobbering the torch ttnn
  needs).
- Checkpoint: `/localdev/acicovic/xtts_ref/model.pth` (1.86 GB, from
  https://huggingface.co/coqui/XTTS-v2). Loaded via a stubbed `TTS` module (pickle
  references config classes). GPT-core weights (~380M params, ~1.5 GB fp32) are loaded
  **on demand** from the checkpoint — NOT copied into a separate weight file.
- Golden tensors: `models/experimental/xtts_v2/golden/gpt/{inputs_embeds,last_hidden_state,
  latents,meta}.pt` (small; the weights are not stored here).

## Files
- `models/experimental/xtts_v2/reference/xtts_gpt_ref.py` — checkpoint loader (`load_gpt_core_state`),
  reference builder, golden generator. Also `make_golden_input` (realistic seeded input
  built from the real embedding tables).
- `models/experimental/xtts_v2/tt/ttnn_xtts_gpt.py` — `TTNNGPTCore` + `preprocess_gpt_parameters`.
- `models/experimental/xtts_v2/tests/test_gpt_pcc.py` — PCC gate (target 0.9999).

## How to run
```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
# (re)generate goldens on CPU:
python models/experimental/xtts_v2/reference/xtts_gpt_ref.py
# PCC test on device (default = native bf16 + SDPA, gate 0.999):
python -m pytest models/experimental/xtts_v2/tests/test_gpt_pcc.py -q
# high-accuracy path (fp32 + manual attention, gate 0.9999):
HIGH_ACCURACY=1 python -m pytest models/experimental/xtts_v2/tests/test_gpt_pcc.py -q
```

## PCC results (S=64, HiFi4, fp32_dest_acc)
| activation dtype | attention | PCC | pass (>0.9999) |
|---|---|---|---|
| bf16 | SDPA (flash) | 0.99972 | no |
| bf16 | SDPA, HiFi3 | 0.99968 | no |
| fp32 | SDPA (q/k/v cast to bf16) | 0.99977 | no |
| bf16 | manual matmul+softmax | 0.99975 | no |
| **fp32** | **manual matmul+softmax** | **0.99996** | **YES** |

## Findings log (dated)
- 2026-07-17: Core matches reference. Key precision findings:
  - The PCC gap was **not** the residual stream (fp32 residual + bf16 SDPA only reached
    0.99977) — it was the **attention**. `ttnn.transformer.scaled_dot_product_attention`
    accepts **bf16/bfloat8/bfloat4 q/k/v only** (TT_FATAL on fp32), which caps precision.
  - Clearing >0.9999 over 30 layers required **manual attention** (matmul → ×scale →
    causal additive mask → softmax → matmul) run in **fp32**. Implemented behind
    `attention="manual"` in `TTNNGPTCore`; `"sdpa"` kept as the fast path.
  - `gelu_new` == `ttnn.GeluVariant.Tanh` (do not use the default Accurate/erf gelu).
  - On Wormhole, HiFi4 + fp32_dest_acc triggers a warned HW bug; empirically HiFi4 still
    beat HiFi3 here (0.99972 vs 0.99968), so we keep HiFi4.
  - Causal mask uses additive `-1e9` on the strict upper triangle; batch=1 has no padding.

## Open questions / TODO
- [ ] **KV-cached decode loop** (autoregressive single-step) — this bringup is prefill-only.
      Revisit whether to build on `models/tt_transformers/` decode or extend `TTNNGPTCore`.
- [x] **Precision decision (2026-07-17):** default to native **bf16 + SDPA (~0.9997)** for
      speed; keep fp32 + manual attention behind `HIGH_ACCURACY=1` for accuracy-gated runs.
- [ ] (optional) push bf16 to >0.9999 later if fidelity ever proves insufficient downstream.
- [ ] Validate at longer / realistic sequence lengths (max_audio 605, max_text 402) and
      non-tile-aligned S (padding).
- [ ] Confirm VQ codebook size (1026 incl. start=1024/stop=1025) + mel_head for the
      sampling head when the decode loop is added.
- [ ] Wire real embeddings + position embeddings + prefix construction (currently the
      golden input is built on CPU from the real embedding tables).
- [ ] Confirm this block's latents match coqui's `return_latent` slicing once integrated.
