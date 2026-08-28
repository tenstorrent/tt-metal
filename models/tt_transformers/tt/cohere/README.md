# Command-R (c4ai-command-r-v01) bring-up scaffold — bounty tt-metal#49307

Org-internal scaffold branch (`canada-quant/tt-metal`, pre-approved per tt-rd
`docs/tenstorrent-upstream-tracker.md` §4: "patches land on branches in these forks
first (internal-only)"). **Nothing here goes upstream without explicit owner approval.**

Grounding (verified 2026-08-28 against the gated snapshot on QB2 +
HF reference code, NOT the bounty-issue table):

| Fact | Value | Source |
|---|---|---|
| Checkpoint | `CohereLabs/c4ai-command-r-v01`, snapshot `760ddb6c203d87ebdbe3c9785b49570e1bf95585` (15/15 shards, 70.0 GB, F16) | box HF cache |
| Architecture | 40 layers, hidden 8192, FFN 22528, vocab 256000, SwiGLU (`hidden_act=silu`) | on-box `config.json` |
| Attention | **MHA 64 Q : 64 KV heads** (head_dim 128), `attention_bias=false`, RoPE `theta=8,000,000` | on-box `config.json` + shard-0 header: `q/k/v/o_proj.weight [8192, 8192]` |
| Context | `max_position_embeddings=8192`, **`model_max_length=131072`** (128K via long RoPE theta) | on-box `config.json` |
| Norms | **CohereLayerNorm — mean-centering LayerNorm in fp32, weight-only (no bias), eps 1e-5** — NOT RMSNorm | HF `modeling_cohere.py` v4.39.3 lines 78-94 |
| Decoder block | **Parallel block**: one `input_layernorm` → attn + MLP on the SAME normed input → `residual + attn + mlp` | HF v4.39.3 line 654 |
| LM head | **Tied embeddings** (`tie_word_embeddings=true`) + **`logit_scale = 0.0625`** applied post-linear | on-box config; HF v4.39.3 lines 1027/1114 |

## Corrections vs the bounty-issue table / tt-rd dossier (verified, do not propagate)

1. **GQA 64:8 is WRONG for this checkpoint** — config `num_key_value_heads=64` and the
   K-projection weight `[8192, 8192]` prove MHA 64:64. KV-cache sizing must use 64 KV heads.
2. **QK LayerNorm is NOT PRESENT in Command-R v01** — zero `q_norm`/`query_key_layer_scaling`
   matches in HF v4.39.3 (the checkpoint's `transformers_version` is 4.38.2); `use_qk_norm`
   only appears in transformers ≥4.44 (for the cohere2/Command-A family) and the on-box
   config has **no** `use_qk_norm` key. Attention scaling is plain `1/sqrt(head_dim)`
   (HF v4.39.3: `attn_weights = q @ k^T / math.sqrt(self.head_dim)`).
   → If the bounty submission is later re-scoped to a QK-norm Cohere variant, the existing
   `tt/attention.py` q_norm/k_norm state-dict loader (lines ~289-321, RMSNorm flavor) is the
   insertion point — but it must be swapped to mean-centering LayerNorm semantics.

## Deltas vs the proven Llama-pattern tt_transformers stack

1. **Parallel decoder block** (`cohere_decoder.py`) — single input norm feeding both
   attention and MLP; residual adds BOTH branch outputs. `TransformerBlock`'s sequential
   attn_norm → attn → ff_norm → MLP flow does not apply.
2. **LayerNorm instead of RMSNorm** (`cohere_norm.py`) — input + final norms. `ttnn.layer_norm`
   is a proven op; fp32 internal compute per HF reference must be matched for PCC ≥ 0.99.
3. **`logit_scale` on LM head output** (`cohere_lm_head.py`) — scalar 0.0625 multiply after
   the LM-head linear/all-reduce (HF applies it post-linear, pre-loss).
4. **Tied embeddings** — LM head weight = `embed_tokens.weight`; the checkpoint has no
   separate `lm_head.weight` tensor (weight_map: 322 tensors).
5. **MHA 64:64 KV cache** — per-layer per-token KV = 2 × 64 × 128 elems (F16) = 32 KB;
   ×40 layers = 1.31 MB/token; 131,072 ctx × 4 seqs ≈ 671 MB — fits easily.

## Status / next steps

- P3 (this branch): scaffold landed 2026-08-28 (`b0a6993`).
- P4: DONE 2026-08-27 — CPU reference captured on the QB2 host: fp32 `prompt00-02.npz`
  (43 keys each) at `/root/v4run/results/command-r/reference/` (tt-rd `scripts/command-r/`).
- P5a (this branch, 2026-08-28): **WIRED** — `ModelArgs` picks cohere up from
  config.json (`norm_eps` falls back to `layer_norm_eps`; `logit_scale` read);
  `Transformer` family dispatch swaps in `CohereDecoderLayer` + `TtCohereLayerNorm`
  final norm + `CohereLMHead` when `args.model_type == "cohere"` (lazy imports,
  zero default-path impact); `TtCohereLayerNorm` real weight loader (RMSNorm-mirrored
  reshape/replicate); checkpoint conversion verified compatible — tied weights surface
  both keys via the HF `state_dict`, `input_layernorm -> attention_norm` mapping reused;
  **NO `load_checkpoints.py` change needed**. Authored + `py_compile` clean; NOT yet
  run on box (import smoke + PCC pending).
- P5b next: on-box import smoke + single-layer PCC vs the P4 dumps (P6 harness),
  then full-stack decode smoke + generator/vLLM wiring check.
- Target mesh: QB2 4× Blackhole p150, TP=4, mesh (1,4); bounty target is T3K (TP=8) —
  keep the code arch-agnostic per dossier §3 option 1.

License note: CC-BY-NC-4.0 + Cohere AUP — non-commercial research + bounty use only.
