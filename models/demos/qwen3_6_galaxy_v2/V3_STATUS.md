# V3 (multimodal generator) status

End of autonomous-completion run.

## Completed (this V3 chunk)

All vision-side pieces of the qwen3.6 VLM stack are functionally complete
and PCC-validated on BH GLX 8×4:

| commit | piece | PCC / result |
|---|---|---|
| `0785dc31f7c` | M-RoPE 3D cos/sin helper (`build_mrope_cos_sin`) | matches HF exactly |
| `e3a4dcc1171` | Vision encoder composite (`Qwen36VisionEncoder`) | 0.887 vs HF E2E |
| `a5eccfc46c9` | `get_rope_index()` 3D position_ids helper | exact match, 2 unit tests |
| `6ffeedce7a7` | `Qwen36MMPreprocessor` (HF processor wrapper) | shapes verified on PIL image |
| `b1961e85f9b` | `Qwen36MMPipeline` (preproc + encoder + splice) | 0.76 PCC at image positions (solid-color input; degenerate attention) |
| (this commit) | `build_mrope_tt_tensors` + `Qwen36MMGenerator` skeleton | API documented |

## Remaining work for end-to-end VLM

The bottleneck is **text-decoder integration**. The existing 64-layer
qwen3.6 text decoder owns its own embed lookup + 1D RoPE gather. To run
multimodal:

1. **`tt/llama_rope.py`** — add an `is_qwen36_mrope_external` mode that
   bypasses `get_qwen36_rm_rot_idxs` and accepts a pre-built
   `[1, 1, S, partial_rotary_dim=64]` cos/sin pair built via
   `build_mrope_tt_tensors`.
2. **`tt/llama_attention.py`** (qwen3.6 branch) — when the rope_setup is in
   the external-cos/sin mode, feed the externally provided cos/sin to
   `ttnn.experimental.rotary_embedding_llama` instead of computing from
   rot_idxs.
3. **`tt/llama_model.py`** — thread the cos/sin tensors through
   `forward_prefill_qwen36` signature; add a separate path that accepts
   pre-built input embeddings (skips internal embed lookup).
4. **`tt/qwen36_generator.py`** — add `forward_multimodal_prefill(prompt,
   images, ...)` method that runs the V3 pipeline, builds M-RoPE cos/sin,
   calls the modified prefill with fused embeddings + cos/sin.

**Risk**: each of these touches the production text-decoder forward.
Multimodal mode must be gated by an explicit flag so the text-only fast
path (decode trace, 1D RoPE gather) stays untouched.

## How to use the V3 helpers today

```python
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import Qwen36MMPipeline
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_tt_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs

# Construct pipeline (loads vision encoder weights to mesh)
model_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=256)
pipeline = Qwen36MMPipeline(mesh_device, ccl_manager, model_args, text_embed_weight=text_decoder.embed_weight)

# Per-request: produce fused embeddings + 3D position_ids
inputs, fused_embeddings = pipeline.prepare_decoder_inputs(prompt, images=[pil_image])

# Build M-RoPE cos/sin TT tensors
cos_tt, sin_tt = build_mrope_tt_tensors(
    inputs.position_ids_3d,
    rope_theta=10_000_000,
    partial_rotary_dim=64,
    mrope_section=[11, 11, 10],
    mesh_device=mesh_device,
)

# TODO: pass (fused_embeddings, cos_tt, sin_tt) into modified text decoder.
```

## Out of scope (V4+)

- Video preprocessing path (Qwen3VLVideoProcessor wrapper)
- Decode trace for multimodal (per Molmo2: skip prefill trace for VLM)
- Server (tt-inference-server VLM integration)
- vLLM dual-fork compatibility (try/except for missing
  `Qwen3_5MultiModalProcessor`)
- Real-image PCC at full demo level
