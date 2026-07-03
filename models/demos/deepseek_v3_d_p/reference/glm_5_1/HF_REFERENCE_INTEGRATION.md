# Wiring `GlmMoeDsaModel` as a GLM-5.1 HF reference (design note)

The GLM-5.1 prefill tests currently validate against a **composed** CPU reference
(`reference.glm_5_1.glm_decoder_layer_reference` = `cpu_deepseek_v32.SparseMLAReference` for the
DSA-MLA + a per-expert noaux_tc MoE), *not* against the HuggingFace model. This note records how to
wire `transformers.GlmMoeDsaModel` as an alternative reference (the "approach A" that DeepSeek/Kimi
use via `reference_model_cls`), what it takes, and why we don't use it as the primary reference.

## TL;DR — verified, but a cross-check, not the primary reference

- `GlmMoeDsaModel` **works** as a reference. Verified numerically (device `TtPrefillBlock` vs HF):
  - **real weights: PCC 0.99996** for the full decoder block (incl. DSA attention).
  - **random weights: ~0.9749** — the untrained lightning-indexer scores make the top-2048 selection
    a knife-edge, so device and HF pick *different* top-k keys. This is an artifact of random
    weights, not a discrepancy.
  - MoE module alone (`GlmMoeDsaMoE` vs our `glm_moe_reference`): **PCC 1.0** (mathematically identical).
- **Why not primary:** the random-weight path (which runs on any box, no cache) would sit at ~0.9749,
  **below the 0.98 block-test gate**, so it would fail there; our composition passes random at 0.9946
  and real at 0.99999964. Plus `GlmMoeDsaModel` needs ~19 GB host RAM for one layer at real dims and
  the plumbing below. So keep the composition; use `GlmMoeDsaModel` as an independent cross-check.

## What `GlmMoeDsaModel` proves about our reference

GLM's router `route_tokens_to_experts` is **line-for-line** the DeepSeek `MoEGate` noaux_tc
(`sigmoid` → `+ e_score_correction_bias` for selection → grouped top-k → *original* sigmoid scores as
weights → normalize → `× routed_scaling_factor`); `GlmMoeDsaNaiveMoe` is the same `silu(gate)*up →
down` expert, just packed. So our composition is the same computation, decomposed per-expert.

## How to run `GlmMoeDsaModel` (the gotchas)

```python
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel

hf_cfg = GlmMoeDsaConfig(          # defaults already carry GLM-5.1 dims (hidden 6144, q_lora 2048, ...)
    num_hidden_layers=1,
    mlp_layer_types=["sparse"],    # <-- selects MoE per layer; NOT first_k_dense_replace
    vocab_size=1024,               # tiny (we pass inputs_embeds; embed is unused)
    max_position_embeddings=max(seq_len, 8192),
)
hf_cfg._attn_implementation = "eager"
hf = GlmMoeDsaModel(hf_cfg).eval().to(torch.bfloat16)
```

- **`num_key_value_heads` must equal `num_attention_heads`** (defaults line up at real dims). If you
  build a reduced-dim config and forget it, the eager DSA attention crashes with an empty-`key_states`
  matmul (`size of tensor a (H) must match tensor b (0)`).
- **The MoE layer is chosen by `config.mlp_layer_types[layer_idx] == "sparse"`**, not
  `first_k_dense_replace`. Force it with `mlp_layer_types=["sparse"]`.
- **Do NOT call a bare `GlmMoeDsaDecoderLayer.forward`** — the DSA attention needs the model's
  cache/mask/rotary/indexer setup. Run the whole model and grab the layer output:

```python
out = hf(inputs_embeds=x, use_cache=False, output_hidden_states=True)
layer_out = out.hidden_states[1]   # hidden_states = [embed_in, layer0_out]; [1] is pre-final-norm
```

## Weight mapping (device/our-format ↔ HF)

HF stores experts **packed**; our checkpoint / `TorchExpert` / `TtMoe` use **per-expert**. Bridge:

```python
# unpack HF -> per-expert (feed device / our refs)
gate_proj[e] = gate_up_proj[e][:inter, :]     # gate_up_proj[e] is [2*inter, hidden]
up_proj[e]   = gate_up_proj[e][inter:, :]
down_proj[e] = down_proj[e]                    # [hidden, inter]

# pack per-expert -> HF (load real checkpoint weights into GlmMoeDsaModel)
gate_up_proj = stack_e( cat([gate_proj[e], up_proj[e]], dim=0) )   # [E, 2*inter, hidden]
down_proj    = stack_e( down_proj[e] )                             # [E, hidden, inter]
```

MLA/indexer names map **1:1** between the canonical `Weights` (cpu_deepseek_v32) and HF
`self_attn.*`, with a single exception: the device keeps the indexer LayerNorm bias split, so
`indexer.k_norm_bias.weight` (canonical) ↔ `indexer.k_norm.bias` (HF). Norms: `input_layernorm`,
`post_attention_layernorm`. Load with `strict=False` (embed/final-norm are unused).

## To wire it as `reference_model_cls` (full approach-A integration)

1. **Adapter** (`tt/runners/adapters/sparse_mla.py`, GLM51Adapter):
   ```python
   reference_model_cls     = GlmMoeDsaModel
   reference_moe_cls       = GlmMoeDsaMoE
   reference_attention_cls = GlmMoeDsaAttention  # if run_reference_mla is needed
   ```
2. **`utils/transformer_helpers.py`**:
   - `create_hf_model`: build a real `GlmMoeDsaConfig` for GLM (the TT side uses the `SimpleNamespace`
     from `glm_hf_config`, which `GlmMoeDsaModel.__init__` can't consume) with `mlp_layer_types`,
     `num_key_value_heads`, `_attn_implementation="eager"`.
   - `extract_layer_state_dict` / `extract_tt_state_dict`: **unpack** the packed experts into the
     per-expert keys the device expects, and add the DSA indexer weights (`self_attn.indexer.*`).
3. **Block reference forward** (`test_prefill_block.py::run_model`): the current call is DeepSeek-shaped
   (`hf_model.layers[i](..., past_key_value=..., use_cache=True)`, no `position_embeddings`). GLM needs
   the full-model path (`model(inputs_embeds=..., output_hidden_states=True)[hidden_states][i+1]`) or a
   layer call with precomputed `position_embeddings`, a 4D causal mask, `past_key_values` (plural), and
   `prev_topk_indices=None`.
4. **Cost:** ~19 GB host RAM per layer at real dims; the CPU DSA forward at seq5120 is minutes.

## Reference experiment scripts

Standalone experiments that produced the numbers above (device vs `GlmMoeDsaModel`, random + real):
`/data/nmilicevic/.claude-tmp/glm_hf_ref_random.py`, `/data/nmilicevic/.claude-tmp/glm_hf_ref_real.py`
(kept outside the repo — copy into `tests/` to run).
