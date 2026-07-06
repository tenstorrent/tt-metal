# FIBO on TTNN — Sub-project 2: BriaFibo Transformer (denoiser)

**Date:** 2026-07-06
**Status:** Design approved, ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Branch:** `fibo-transformer` (stacked on `fibo-smollm3-encoder`)
**Scope of this spec:** the FIBO diffusion transformer (denoiser) only. The SmolLM3 encoder is sub-project 1 (done); Wan VAE wiring and the pipeline are sub-projects 3-4.

---

## Context: the larger FIBO effort

Second of four decomposed sub-projects for bringing up Bria **FIBO** (flow-matching MMDiT text-to-image) in `models/tt_dit`. Reference: `diffusers` `models/transformers/transformer_bria_fibo.py` (`BriaFiboTransformer2DModel`) and `pipelines/bria_fibo/pipeline_bria_fibo.py`. Weights: `briaai/FIBO`, subfolder `transformer` (gated; access confirmed). Each sub-project is PCC-gated against the HF reference on Blackhole.

- Sub-project 1: SmolLM3 text encoder — **DONE** (`fibo-smollm3-encoder`).
- **Sub-project 2: BriaFibo transformer** ← *this spec*.
- Sub-project 3: Wan VAE + flow-match solver wiring — TODO.
- Sub-project 4: pipeline + Blackhole 2×2 bringup — TODO.

---

## 1. Goal

A tt_dit implementation of the FIBO denoiser that reproduces `BriaFiboTransformer2DModel` numerically: given packed latents, the SmolLM3 conditioning (`encoder_hidden_states` = 4096-dim `prompt_embeds`, plus the per-block `text_encoder_layers` list), a timestep, and RoPE position ids, it produces the velocity/noise prediction. Validated standalone against the HF reference transformer on real `briaai/FIBO` weights.

### In scope
- The full MMDiT forward: patch embed (in_channels 48 → 3072), context embed (4096 → 3072), timestep embed, 8 dual blocks + 38 single blocks with per-block "concat-halves" text injection, final AdaLN-continuous norm + output projection.
- Axial RoPE (θ=10000, axes [16,56,56]) over `img_ids`/`txt_ids`.
- Weight loading from HF `transformer/` into tt modules.
- Tensor-parallel execution on the Blackhole mesh.
- Per-unit + full-model PCC tests vs the HF reference transformer.

### Out of scope (other sub-projects)
- Building the 46-entry `text_encoder_layers` list from SmolLM3's 37 hidden states (the slice/pad reconciliation is a **pipeline** concern — sub-project 4). This spec's tests feed the transformer the same 46-entry list the reference receives.
- The exact VAE→latent packing that yields in_channels=48 (sub-project 3/4). Tests use 48-channel reference latents directly.
- CFG batching, the denoise loop, VAE decode, image output.

---

## 2. Background — FIBO transformer architecture (verified from `briaai/FIBO/transformer/config.json`)

| Field | Value |
|---|---|
| `_class_name` | `Bria4Transformer2DModel` (diffusers class `BriaFiboTransformer2DModel`) |
| `num_layers` (dual/MMDiT blocks) | **8** |
| `num_single_layers` | **38** |
| total blocks | **46** |
| `attention_head_dim` / `num_attention_heads` | 128 / 24 → inner_dim **3072** |
| `in_channels` | **48** |
| `joint_attention_dim` | 4096 |
| `text_encoder_dim` | 2048 |
| `pooled_projection_dim` | None |
| `guidance_embeds` | False |
| `patch_size` | 1 |
| `axes_dims_rope` | [16, 56, 56] |
| `rope_theta` / `time_theta` | 10000 / 10000 |

This is a Flux-shaped MMDiT (RMSNorm QK-norm, AdaLN modulation, joint attention) with three FIBO-specific differences: fewer dual blocks (8 vs Flux's 19), in_channels 48 (vs 64), and per-block text injection (below). Timestep embedding is **timestep-only** (no pooled, no guidance).

### 2.1 Per-block "concat-halves" text injection
`__init__` builds `caption_projection`: a list of `num_layers + num_single_layers = 46` `BriaFiboTextProjection(in_features=2048, hidden_size=inner_dim//2=1536)` modules. `context_embedder = Linear(4096 → 3072)` seeds `encoder_hidden_states` once from `prompt_embeds`.

A single running `block_id` counter spans **both** loops (0..45). Before every block (dual and single):
```python
current = caption_projection[block_id](text_encoder_layers[block_id])   # 2048 → 1536
encoder_hidden_states = torch.cat([encoder_hidden_states[:, :, :1536], current], dim=-1)  # → 3072
block_id += 1
```
i.e. each block replaces the **second half** of the running context with its freshly-projected per-block text layer.

### 2.2 Dual vs single block structure
- **Dual blocks (8):** standard MMDiT two-stream block (image + text streams, joint attention), identical in structure to Flux's double block. Injection (2.1) is applied to `encoder_hidden_states` *before* the block.
- **Single blocks (38):** FIBO keeps the two streams (unlike Flux, whose single blocks operate on one merged sequence). Each single-block iteration: inject (2.1) → `hidden_states = cat([encoder_hidden_states, hidden_states], dim=1)` (sequence concat) → run the single-stream block → **re-split** back into `(encoder_hidden_states, hidden_states)`. So text is re-injected every single block. **This re-inject + re-split is the main behavioral delta from Flux and must be verified against the reference.**

### 2.3 The 46-entry `text_encoder_layers` list (pipeline concern, noted for context)
The transformer indexes a 46-entry list. The **pipeline** builds it from SmolLM3's 37 hidden states: since 37 < 46, it appends 9 copies of the last hidden state (`prompt_layers + [prompt_layers[-1]] * (46-37)`). Order: `[hidden_states[0..36], hidden_states[36]×9]`. This spec's tests replicate that construction (or capture it from the reference) to feed identical inputs; the transformer itself only indexes by `block_id`.

### 2.4 RoPE / position ids
Axial RoPE (θ=10000, axes [16,56,56], total 128 = head_dim) over concatenated `txt_ids` (zeros) + `img_ids` (3-axis: [0, h_idx, w_idx]), Flux-style. The tt_dit Flux transformer already implements this pos-embed; reuse with θ=10000 and FIBO's ids.

---

## 3. Design

### 3.1 File layout
```
models/tt_dit/models/transformers/transformer_bria_fibo.py   # new
tests/models/bria_fibo/test_transformer.py                    # new
```
Built on tt_dit primitives + the Flux1 transformer as the base template.

### 3.2 Reuse map (honest ~70-75%)
- **Reuse as-is:** `blocks/attention.py` (joint attention + RMSNorm QK-norm); `blocks/transformer_block.py` (the dual/MMDiT block core — injection happens in the FIBO forward loop *before* the call, so the block is unchanged); the axial RoPE embedding; `layers/{linear,normalization,embeddings}.py`; parallel/CCL wiring; the `Flux1Checkpoint`/`cache.load_model` weight-loading pattern.
- **Net-new (adapt from `transformer_flux1.py`):**
  1. `BriaFiboTextProjection` (2048→1536) and the length-46 `caption_projection` ModuleList.
  2. The `block_id`-threaded concat-halves injection in the forward loop.
  3. The **single-block wrapper**: per-block inject → seq-concat → single-block → re-split (the Flux single-block *core* is reused; the wrapper is new).
  4. `context_embedder` = Linear(4096→3072).
  5. Timestep-only temb (replace Flux's `CombinedTimestepGuidanceTextProjEmbeddings`; no pooled/guidance).
  6. `x_embedder` = Linear(48→3072); final `AdaLayerNormContinuous` + `proj_out`.
  7. `BriaFiboCheckpoint` mapping HF `transformer/` weights (incl. the 46 `caption_projection.*`, `context_embedder`, `x_embedder`, blocks, `norm_out`, `proj_out`) → tt modules.

### 3.3 forward (indicative)
```python
def forward(self, spatial, prompt, timestep, spatial_rope, prompt_rope,
            text_encoder_layers, spatial_seq_len, prompt_seq_len) -> ttnn.Tensor:
    temb = silu(self.time_embed(timestep))                  # timestep-only
    spatial = self.x_embedder(spatial)                       # 48 -> 3072
    prompt  = self.context_embedder(prompt)                  # 4096 -> 3072
    block_id = 0
    for block in self.transformer_blocks:                    # 8 dual
        prompt = inject(prompt, self.caption_projection[block_id](text_encoder_layers[block_id]))
        spatial, prompt = block(spatial, prompt, temb, spatial_rope, prompt_rope, ...)
        block_id += 1
    for block in self.single_transformer_blocks:             # 38 single
        prompt = inject(prompt, self.caption_projection[block_id](text_encoder_layers[block_id]))
        combined = cat([prompt, spatial], dim=seq)
        combined = block(combined, temb, ...)
        prompt, spatial = split(combined, prompt_seq_len)     # re-split
        block_id += 1
    spatial = self.proj_out(self.norm_out(spatial, temb))
    return spatial
```
(`inject(ctx, cur)` = `cat([ctx[..., :1536], cur], dim=-1)`. Exact ttnn ops, memory configs, and the precise single-block re-split indexing are pinned during implementation against the reference.)

### 3.4 Parallelization & precision
Mesh-native tensor parallel following the tt_dit Flux transformer's Blackhole presets (Flux1 already has BH configs). For the 2×2 mesh: `cfg=(1,0), sp=(2,0), tp=(2,1)`. bf16 weights/activations; HiFi2 matmul / HiFi4 for norms & SDPA (mirror Flux). Develop/PCC-check at reduced block count on a small mesh, then full 8+38 on 2×2.

---

## 4. Public interface (indicative)
```python
class BriaFiboTransformer(Module):
    def __init__(self, *, patch_size, in_channels, num_layers, num_single_layers,
                 num_attention_heads, attention_head_dim, joint_attention_dim,
                 text_encoder_dim, axes_dims_rope, rope_theta, mesh_device,
                 parallel_config, ccl_manager): ...
    def load_torch_state_dict(self, state_dict) -> None: ...
    def forward(self, spatial, *, prompt, timestep, text_encoder_layers,
                spatial_rope, prompt_rope, spatial_seq_len, prompt_seq_len) -> ttnn.Tensor: ...
```
Exact signature aligned to the tt_dit Flux transformer during planning.

## 5. Testing & validation
`tests/models/bria_fibo/test_transformer.py`, modeled on the tt_dit Flux transformer tests + the encoder's PCC pattern.
- **Reference:** `BriaFiboTransformer2DModel.from_pretrained("briaai/FIBO", subfolder="transformer")` (diffusers), real weights, `HF_HUB_OFFLINE=1`.
- **Inputs:** random (seeded) 48-channel packed latents, 4096-dim `prompt_embeds`, a 46-entry `text_encoder_layers` list (built via the pipeline's slice/pad rule from a 37-length stand-in, or captured from the reference), a timestep, and `img_ids`/`txt_ids`.
- **Gates (PCC ≥ 0.99, bf16, must run not skip):** (1) one dual block; (2) one single block (verifies re-inject + re-split); (3) the full transformer at reduced block count for iteration; (4) full 8+38 on the 2×2 mesh. Compare the transformer output (velocity prediction) to the reference.
- Reduced-depth env knob (like the encoder's `N_LAYERS`) for fast iteration.

**Definition of done:** full 46-block transformer output matches the reference at PCC ≥ 0.99 (bf16) on the 2×2 Blackhole mesh, on real weights.

## 6. Open items (resolve during implementation, not blockers)
- Pin the exact **single-block re-inject/re-split** ttnn implementation against the reference (the subtlest piece).
- Confirm loading the reference via `BriaFiboTransformer2DModel` vs the config's `_class_name: Bria4Transformer2DModel` (diffusers class mapping); adjust the import/loader accordingly.
- Confirm the `img_ids`/`txt_ids` construction and the resulting `spatial_seq_len` for the test's latent shape (ties to the in_channels=48 packing resolved in sub-project 3/4 — the transformer test can use any self-consistent latent H×W).
- Transformer weights are multi-GB — pre-download `transformer/*` once (offline thereafter), like the encoder.

## 7. Risks & mitigations
1. **Single-block re-inject/re-split fidelity** (main risk) — validated by the dedicated single-block PCC test before the full model.
2. **Full-model bf16 PCC over 46 blocks** — build/validate at reduced depth first; if the full-depth PCC dips below 0.99, report the measured floor rather than silently lowering.
3. **Large weights / memory on the mesh** — tp-shard following Flux BH presets; pre-download offline.
4. **RoPE θ mismatch** — transformer uses θ=10000 (distinct from the encoder's 5e6); test asserts against the reference.

## 8. Follow-on
On completion, sub-project 4 (pipeline) wires this transformer together with the SmolLM3 encoder (sub-project 1), the Wan VAE + solver (sub-project 3), CFG, and the denoise loop, and builds the real 46-entry `text_encoder_layers` list from the encoder's 37 hidden states.
