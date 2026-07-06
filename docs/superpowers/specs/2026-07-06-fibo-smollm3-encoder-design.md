# FIBO on TTNN — Sub-project 1: SmolLM3 Text Encoder

**Date:** 2026-07-06
**Status:** Design approved, ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Scope of this spec:** the SmolLM3 text encoder only. The transformer, VAE wiring, and pipeline are separate sub-projects (see Context).

---

## Context: the larger FIBO effort

We are implementing Bria's **FIBO** text-to-image model in `models/tt_dit`. Reference: `diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py` and `diffusers/models/transformers/transformer_bria_fibo.py`.

FIBO is a flow-matching MMDiT (Flux-shaped) conditioned on an LLM text encoder. It maps cleanly onto the existing tt_dit framework, so the effort is **decomposed into 4 sub-projects**, built in data-flow order, each with its own spec → plan → implementation cycle and each PCC-gated against the diffusers reference:

| # | Sub-project | Strategy | Reuse |
|---|---|---|---|
| **1** | **SmolLM3 text encoder** ← *this spec* | New `encoders/smollm3/`; decoder layer from Qwen25VL, all-hidden-states shell from Gemma | ~70-80% |
| 2 | BriaFibo transformer | New `transformer_bria_fibo.py` from Flux1 + per-layer "concat-halves" text injection | ~90% |
| 3 | Wan VAE + solver wiring | Reuse `vae_wan2_1.py` (T=1 decode) + `EulerSolver` + dynamic-shift scheduler | ~100% |
| 4 | Pipeline + Blackhole bringup | New `pipelines/bria_fibo/`; CFG batched=2; 2×2 mesh (`cfg=(1,0) sp=(2,0) tp=(2,1)`) | glue |

**Component IDs:** transformer `BriaFiboTransformer2DModel`, VAE `AutoencoderKLWan`, text encoder `SmolLM3ForCausalLM`, scheduler `FlowMatchEulerDiscreteScheduler`. Weights available under `briaai/FIBO`.

---

## 1. Goal

Produce a tt_dit encoder that numerically reproduces HuggingFace `SmolLM3ForCausalLM(..., output_hidden_states=True)`, exposing exactly what the FIBO pipeline consumes:

- `prompt_embeds = concat(hidden_states[-1], hidden_states[-2])` along the feature dim → **4096-dim** main conditioning.
- The **full per-layer hidden-state list** (`hidden_states`, length `num_hidden_layers + 1 = 37`), consumed per block by the transformer as `text_encoder_layers` (the exact per-block mapping is a sub-project 2 concern — see Open items).

The encoder is validated **standalone against the HF reference** before any downstream sub-project consumes it.

### In scope
- SmolLM3-3B forward pass as a text encoder (no generation, no KV-cache, no `lm_head`).
- Returning all hidden states with correct pre/post-final-norm semantics.
- Tensor-parallel execution on the Blackhole mesh via tt_dit primitives.
- A PCC test vs. the HF reference on real `briaai/FIBO` weights.

### Out of scope (later sub-projects)
- Tokenization/prompt templating and the VLM-prompt-to-JSON step (pipeline sub-project 4).
- The exact output-layout handoff to the DiT (integration handled in sub-project 4; here we only guarantee correct numerics).
- Autoregressive decoding, sampling, generation.

---

## 2. Background — SmolLM3 architecture (verified against source)

Verified against `HuggingFaceTB/SmolLM3-3B/config.json` and `transformers` `modeling_smollm3.py`. FIBO uses stock `SmolLM3ForCausalLM`; **step 0 of implementation is confirming the gated `briaai/FIBO/text_encoder/config.json` matches** (assumed identical).

| Field | Value |
|---|---|
| num_hidden_layers | **36** |
| hidden_size | **2048** |
| num_attention_heads | **16** |
| num_key_value_heads | **4** (GQA, 4 q-heads per kv-head) |
| head_dim | **128** (= 2048/16; derived, not an explicit field) |
| intermediate_size | **11008** |
| hidden_act | **silu** → SwiGLU MLP `down(silu(gate(x)) * up(x))`, no bias |
| norm | **RMSNorm**, eps **1e-6**, **plain (no +1 offset)** |
| rope_theta | **5e6** |
| vocab_size | **128256** |
| max_position_embeddings | 65536 |
| attention_bias / mlp_bias | **false** / **false** |
| q_norm / k_norm | **none** |
| sliding window | **none** (all layers `full_attention`) |
| no_rope_layer_interval | **4** → NoPE on layers **3, 7, 11, 15, 19, 23, 27, 31, 35** |
| dtype | bfloat16 |

Otherwise a plain Llama-3-style decoder: pre-norm layer (`input_layernorm` before attention, `post_attention_layernorm` before MLP), separate `q/k/v/o_proj` (no bias), final `model.norm`, non-interleaved (rotate-half) RoPE.

### Distinctive feature: NoPE
`SmolLM3Attention` sets `self.use_rope = config.no_rope_layers[layer_idx]`; RoPE is applied only when `use_rope` is truthy. `no_rope_layers` has length `num_hidden_layers`, value `1` = apply RoPE, `0` = skip. Default generation: `int((layer_idx + 1) % no_rope_layer_interval != 0)`. ~9 of 36 layers skip RoPE.

### FIBO's consumption contract (from `pipeline_bria_fibo.py`)
```python
encoder_outputs = text_encoder(input_ids, attention_mask=..., output_hidden_states=True)
hidden_states = encoder_outputs.hidden_states        # tuple length 37 (= num_hidden_layers + 1)
prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)   # 4096
# the full per-layer list is also consumed by the transformer (per-block mapping = sub-project 2)
```
**Critical semantics (HF `output_hidden_states` convention):** the tuple is `[embeddings, out(L0), out(L1), …, out(L34), finalnorm(out(L35))]` — element 0 is the embedding output, elements 1..35 are the outputs of layers L0..L34 (each the input to the next layer), and element 36 is the final-norm output. Therefore `hidden_states[-1]` is **post-final-norm** (== `last_hidden_state`) and `hidden_states[-2]` is the **pre-final-norm** hidden state feeding the last layer (output of L34). The key nuance: `hs[-1]` has the final RMSNorm applied, `hs[-2]` does not. The implementation reproduces the whole tuple, and the test compares all 37 entries directly to the reference, so exact ordering is validated automatically.

---

## 3. Design

### 3.1 Module layout (new `models/tt_dit/encoders/smollm3/`)
```
encoders/smollm3/
├── __init__.py
├── config.py          # SmolLM3Config
└── model_smollm3.py   # SmolLM3Attention, MLP, DecoderLayer, SmolLM3Encoder
tests/encoders/smollm3/
└── test_smollm3_encoder.py
```

Built on tt_dit primitives only (no dependency on `models/tt_transformers/`): `layers/linear.py` (`ColParallelLinear`/`RowParallelLinear`, `bias=False`), `layers/normalization.py` (`RMSNorm`, no offset), `layers/embeddings.py` (`Embedding`), `layers/module.py` (`Module`/`ModuleList`/`Parameter`), `parallel/` (config + CCL), `utils/cache.py` + `utils/substate.py` (weight loading/renaming).

### 3.2 Templates
- **Decoder layer** ← adapt **`encoders/qwen25vl/model_qwen25vl.py`** (closest to SmolLM3: plain RMSNorm no offset `:461-472`, SwiGLU no-bias MLP `:401-458`, rotate-half RoPE `:475-484`, `head_dim = hidden//heads` `:244`, fused QKV + GQA head-packing `:487-514`). **Drop:** Qwen's qkv-bias and 3-section mRoPE.
- **Encoder shell** ← adapt **`encoders/gemma/model_gemma.py`** (all-hidden-states forward `:468-533`, per-layer RoPE dispatch `:523-526`, HF-prefix strip + `rename_substate`/`pop_substate` `:447-466`). **Drop:** Gemma's RMSNorm +1 offset `:83-88`, embedding `*sqrt(hidden)` scaling `:485`, per-head q_norm/k_norm `:176-177`, GELU-tanh MLP and dual global/local RoPE.

### 3.3 Net-new work (exists nowhere in the repo)
1. **NoPE per-layer skip.** Add `no_rope_layers` to config; in the forward loop set `cos, sin = (cos, sin) if no_rope_layers[idx] else (None, None)` (hook = Gemma's per-layer dispatch); guard `_apply_rope` on `cos is not None` (pattern already used in `embeddings_connector.py:165`).
2. **Single-axis RoPE table** at `rope_theta=5e6` (replace Qwen's mRoPE; Gemma's `GemmaRotaryEmbedding` `:91-124` is the shape template — a single table, not dual).
3. **All-hidden-states output + FIBO contract.** `forward` returns the full length-37 list, reproducing HF's tuple ordering exactly (mind the final RMSNorm on `hs[-1]` but not `hs[-2]`); a helper produces `prompt_embeds = ttnn.concat([hs[-1], hs[-2]], dim=-1)`. Note FIBO keeps the full tuple and uses **both** `hs[-1]` and `hs[-2]` (unlike Gemma, whose usage drops `-2`). The pre/post-norm handling is exercised by `tests/encoders/gemma/test_gemma_encoder_all_layers.py:135-146`.
4. **Config + weight names.** New `SmolLM3Config` (Qwen-style explicit kwargs read from HF `config.json`). Weights come from `SmolLM3ForCausalLM`, prefix **`model.`** (not Gemma's `language_model.model.`), no vision keys; strip `model.`, pop `lm_head` (tied, unused).

### 3.4 Parallelization
Mesh-native tensor parallel via `ColParallelLinear`/`RowParallelLinear`, following gemma/qwen encoders. GQA (16 q / 4 kv) uses fused QKV + `optimal_groups` head-packing for uneven TP (Qwen pattern). Develop and PCC-check on a small mesh ((1,1) or (1,2)) for fast iteration, then confirm on the (2,2) BH mesh. The encoder runs once per generation (prompt + negative prompt → batch 2).

### 3.5 Precision
bf16 weights/activations throughout (fp32 forces HiFi4 on Blackhole). Raise math fidelity only for a specific layer if its PCC misses target.

### 3.6 Long context
`max_sequence_length` up to 3000. Handle SDPA chunking (gemma/qwen use chunk 128) and tile padding for non-128-multiple sequence lengths (both templates pad internally). Padding must be applied consistently across every entry of the hidden-state list if a padding-mask path is used.

---

## 4. Public interface (indicative)

```python
class SmolLM3Encoder(Module):
    def __init__(self, config: SmolLM3Config, mesh_device, parallel_config, ccl_manager): ...
    def load_torch_state_dict(self, state_dict) -> None: ...   # strip "model.", pop lm_head

    def forward(self, input_ids, attention_mask=None) -> list[ttnn.Tensor]:
        """Returns all hidden states: [embed, L0..L34, final_norm(L35)] (len 37)."""

    def encode(self, input_ids, attention_mask=None) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """Returns (prompt_embeds[B,T,4096]=concat(hs[-1],hs[-2]), all_hidden_states)."""
```
Exact signatures/layout finalized during planning against the gemma/qwen encoder APIs; tokenization stays on host (HF `AutoTokenizer`), a pipeline concern.

---

## 5. Testing & validation

`tests/encoders/smollm3/test_smollm3_encoder.py`, modeled on `tests/encoders/gemma/test_gemma_encoder_all_layers.py`.

- **Reference:** HF `SmolLM3ForCausalLM.from_pretrained(briaai/FIBO, subfolder="text_encoder")` with `output_hidden_states=True`, on real weights.
- **Assertions:** per-layer hidden-state PCC (all 37 entries), the 4096-dim `prompt_embeds` concat, explicit checks that `hs[-1]` is post-norm and `hs[-2]` is pre-norm.
- **Target:** PCC ≥ **0.99** (bf16) per entry.
- **Cases:** short prompt (tile-aligned) and long prompt near 3000 tokens (chunking/padding); batch 1 and batch 2 (CFG shape); NoPE layers specifically checked (regression if RoPE is wrongly applied there).
- **Mesh:** run on (1,1)/(1,2) during dev, confirm on (2,2).

**Definition of done:** all hidden states + `prompt_embeds` match reference at PCC ≥ 0.99 on (2,2) BH, short and long prompts, on real weights.

---

## 6. Open items (resolve during implementation, not blockers)
- Confirm `briaai/FIBO/text_encoder/config.json` == stock SmolLM3-3B (gated download; `huggingface-cli login`). If it diverges (e.g. different `no_rope_layers`), update `config.py`.
- Confirm the tokenizer/`max_sequence_length` default the pipeline uses (3000 in `__call__`) — informs the long-context test size.
- **[cross-cutting, resolve in sub-project 2]** The transformer reference builds `caption_projection` over all **57** blocks (19 dual + 38 single) and indexes `text_encoder_layers[block_id]`, but SmolLM3 yields only **37** hidden states. The per-block mapping (repeat / subset / slice) must be pinned down when building the transformer by inspecting reference tensor shapes. It does **not** affect this encoder's contract (return all 37 hidden states).

---

## 7. Risks & mitigations
1. **Pre/post-final-norm mix-up** in the 4096 concat → wrong conditioning. *Mitigation:* explicit pre/post-norm assertions in the test.
2. **NoPE applied to wrong layers** (off-by-one in `no_rope_layers`, or RoPE leaking into NoPE layers) → silent accuracy loss. *Mitigation:* per-layer PCC + targeted NoPE-layer checks.
3. **GQA head-packing on uneven TP** (4 kv heads across a 2-wide TP axis). *Mitigation:* reuse Qwen's `optimal_groups` packing; test on (2,2).
4. **Long-context SDPA / tile padding** correctness at ~3000 tokens. *Mitigation:* dedicated long-prompt test case.

---

## 8. Follow-on
On completion, sub-project 2 (BriaFibo transformer) consumes this encoder's outputs as reference inputs for its own PCC gate.
