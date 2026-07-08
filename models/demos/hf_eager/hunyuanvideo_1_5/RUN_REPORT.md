# Bring-up run report — `tencent/HunyuanVideo-1.5`

_Generated: 2026-07-08 05:20:52 UTC_

## Outcome

**Converged** after ? iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `hf_eager universal (Video)`
- **Closest template:** `models/demos/hf_eager/demo.py`

## Placement summary

- **ON_DEVICE** (18): graduated, native ttnn, PCC verified
  - `ada_layer_norm_continuous`, `ada_layer_norm_zero`, `combined_timestep_text_proj_embeddings`, `feed_forward`, `hunyuan_video15_ada_norm`, `hunyuan_video15_by_t5_text_projection`, `hunyuan_video15_image_projection`, `hunyuan_video15_individual_token_refiner`, `hunyuan_video15_individual_token_refiner_block`, `hunyuan_video15_patch_embed`, `hunyuan_video15_rotary_pos_embed`, `hunyuan_video15_time_embedding`, `hunyuan_video15_token_refiner`, `hunyuan_video15_transformer_block`, `linear_activation`, `pix_art_alpha_text_projection`, `timestep_embedding`, `timesteps`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `ada_layer_norm_continuous` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous` |
| `ada_layer_norm_zero` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero` |
| `combined_timestep_text_proj_embeddings` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_combined_timestep_text_proj_embeddings.py::test_combined_timestep_text_proj_embeddings` |
| `feed_forward` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_feed_forward.py::test_feed_forward` |
| `hunyuan_video15_ada_norm` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_ada_norm.py::test_hunyuan_video15_ada_norm` |
| `hunyuan_video15_by_t5_text_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_by_t5_text_projection.py::test_hunyuan_video15_by_t5_text_projection` |
| `hunyuan_video15_image_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_image_projection.py::test_hunyuan_video15_image_projection` |
| `hunyuan_video15_individual_token_refiner` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner.py::test_hunyuan_video15_individual_token_refiner` |
| `hunyuan_video15_individual_token_refiner_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner_block.py::test_hunyuan_video15_individual_token_refiner_block` |
| `hunyuan_video15_patch_embed` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_patch_embed.py::test_hunyuan_video15_patch_embed` |
| `hunyuan_video15_rotary_pos_embed` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_rotary_pos_embed.py::test_hunyuan_video15_rotary_pos_embed` |
| `hunyuan_video15_time_embedding` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_time_embedding.py::test_hunyuan_video15_time_embedding` |
| `hunyuan_video15_token_refiner` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_token_refiner.py::test_hunyuan_video15_token_refiner` |
| `hunyuan_video15_transformer_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_transformer_block.py::test_hunyuan_video15_transformer_block` |
| `linear_activation` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_linear_activation.py::test_linear_activation` |
| `pix_art_alpha_text_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_pix_art_alpha_text_projection.py::test_pix_art_alpha_text_projection` |
| `timestep_embedding` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timestep_embedding.py::test_timestep_embedding` |
| `timesteps` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timesteps.py::test_timesteps` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_combined_timestep_text_proj_embeddings.py::test_combined_timestep_text_proj_embeddings -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_feed_forward.py::test_feed_forward -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_ada_norm.py::test_hunyuan_video15_ada_norm -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_by_t5_text_projection.py::test_hunyuan_video15_by_t5_text_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_image_projection.py::test_hunyuan_video15_image_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner.py::test_hunyuan_video15_individual_token_refiner -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner_block.py::test_hunyuan_video15_individual_token_refiner_block -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_patch_embed.py::test_hunyuan_video15_patch_embed -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_rotary_pos_embed.py::test_hunyuan_video15_rotary_pos_embed -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_time_embedding.py::test_hunyuan_video15_time_embedding -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_token_refiner.py::test_hunyuan_video15_token_refiner -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_transformer_block.py::test_hunyuan_video15_transformer_block -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_linear_activation.py::test_linear_activation -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_pix_art_alpha_text_projection.py::test_pix_art_alpha_text_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timestep_embedding.py::test_timestep_embedding -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timesteps.py::test_timesteps -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e tencent/HunyuanVideo-1.5`
