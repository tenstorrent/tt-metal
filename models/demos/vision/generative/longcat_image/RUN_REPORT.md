# Bring-up run report — `meituan-longcat/LongCat-Image`

_Generated: 2026-07-06 23:21:34 UTC_

## Outcome

**Converged** after ? iteration(s).

## Placement summary

- **ON_DEVICE** (25): graduated, native ttnn, PCC verified
  - `ada_layer_norm_continuous`, `ada_layer_norm_zero`, `ada_layer_norm_zero_single`, `autoencoder_k_l`, `decoder`, `down_encoder_block2_d`, `downsample2_d`, `encoder`, `feed_forward`, `long_cat_image_single_transformer_block`, `long_cat_image_timestep_embeddings`, `long_cat_image_transformer2_d_model`, `long_cat_image_transformer_block`, `qwen2_v_l_decoder_layer`, `qwen2_v_l_for_conditional_generation`, `qwen2_v_l_model`, `qwen2_v_l_patch_merger`, `qwen2_v_l_text_model`, `qwen2_v_l_vision_block`, `qwen2_vision_transformer_pretrained_model`, `resnet_block2_d`, `timestep_embedding`, `u_net_mid_block2_d`, `up_decoder_block2_d`, `upsample2_d`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `ada_layer_norm_continuous` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous` |
| `ada_layer_norm_zero` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero` |
| `ada_layer_norm_zero_single` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_zero_single.py::test_ada_layer_norm_zero_single` |
| `autoencoder_k_l` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_autoencoder_k_l.py::test_autoencoder_k_l` |
| `decoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_decoder.py::test_decoder` |
| `down_encoder_block2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_down_encoder_block2_d.py::test_down_encoder_block2_d` |
| `downsample2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_downsample2_d.py::test_downsample2_d` |
| `encoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_encoder.py::test_encoder` |
| `feed_forward` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_feed_forward.py::test_feed_forward` |
| `long_cat_image_single_transformer_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_single_transformer_block.py::test_long_cat_image_single_transformer_block` |
| `long_cat_image_timestep_embeddings` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_timestep_embeddings.py::test_long_cat_image_timestep_embeddings` |
| `long_cat_image_transformer2_d_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_transformer2_d_model.py::test_long_cat_image_transformer2_d_model` |
| `long_cat_image_transformer_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_transformer_block.py::test_long_cat_image_transformer_block` |
| `qwen2_v_l_decoder_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_decoder_layer.py::test_qwen2_v_l_decoder_layer` |
| `qwen2_v_l_for_conditional_generation` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_for_conditional_generation.py::test_qwen2_v_l_for_conditional_generation` |
| `qwen2_v_l_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_model.py::test_qwen2_v_l_model` |
| `qwen2_v_l_patch_merger` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_patch_merger.py::test_qwen2_v_l_patch_merger` |
| `qwen2_v_l_text_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_text_model.py::test_qwen2_v_l_text_model` |
| `qwen2_v_l_vision_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_vision_block.py::test_qwen2_v_l_vision_block` |
| `qwen2_vision_transformer_pretrained_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_vision_transformer_pretrained_model.py::test_qwen2_vision_transformer_pretrained_model` |
| `resnet_block2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_resnet_block2_d.py::test_resnet_block2_d` |
| `timestep_embedding` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_timestep_embedding.py::test_timestep_embedding` |
| `u_net_mid_block2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_u_net_mid_block2_d.py::test_u_net_mid_block2_d` |
| `up_decoder_block2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_up_decoder_block2_d.py::test_up_decoder_block2_d` |
| `upsample2_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/longcat_image/tests/pcc/test_upsample2_d.py::test_upsample2_d` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_ada_layer_norm_zero_single.py::test_ada_layer_norm_zero_single -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_autoencoder_k_l.py::test_autoencoder_k_l -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_decoder.py::test_decoder -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_down_encoder_block2_d.py::test_down_encoder_block2_d -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_downsample2_d.py::test_downsample2_d -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_encoder.py::test_encoder -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_feed_forward.py::test_feed_forward -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_single_transformer_block.py::test_long_cat_image_single_transformer_block -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_timestep_embeddings.py::test_long_cat_image_timestep_embeddings -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_transformer2_d_model.py::test_long_cat_image_transformer2_d_model -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_long_cat_image_transformer_block.py::test_long_cat_image_transformer_block -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_decoder_layer.py::test_qwen2_v_l_decoder_layer -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_for_conditional_generation.py::test_qwen2_v_l_for_conditional_generation -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_model.py::test_qwen2_v_l_model -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_patch_merger.py::test_qwen2_v_l_patch_merger -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_text_model.py::test_qwen2_v_l_text_model -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_v_l_vision_block.py::test_qwen2_v_l_vision_block -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_qwen2_vision_transformer_pretrained_model.py::test_qwen2_vision_transformer_pretrained_model -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_resnet_block2_d.py::test_resnet_block2_d -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_timestep_embedding.py::test_timestep_embedding -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_u_net_mid_block2_d.py::test_u_net_mid_block2_d -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_up_decoder_block2_d.py::test_up_decoder_block2_d -svv
python -m pytest models/demos/vision/generative/longcat_image/tests/pcc/test_upsample2_d.py::test_upsample2_d -svv
```

## Next steps
