# Bring-up run report — `coqui/XTTS-v2`

_Generated: 2026-07-05 20:50:57 UTC_

## Outcome

**Converged** after ? iteration(s).

## Placement summary

- **ON_DEVICE** (29): graduated, native ttnn, PCC verified
  - `adaptive_avg_pool2d`, `attend`, `attention_block`, `conditioning_encoder`, `conv1_d`, `dropout1d`, `g_e_g_l_u`, `g_p_t`, `g_p_t2_block`, `g_p_t2_inference_model`, `g_p_t2_model`, `group_norm32`, `hifi_decoder`, `hifigan_generator`, `instance_norm1d`, `learned_position_embeddings`, `mel_scale`, `mel_spectrogram`, `parametrization_list`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `perceiver_resampler`, `pre_emphasis`, `q_k_v_attention_legacy`, `res_block1`, `res_net_speaker_encoder`, `s_e_basic_block`, `s_e_layer`, `weight_norm`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `adaptive_avg_pool2d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_adaptive_avg_pool2d.py::test_adaptive_avg_pool2d` |
| `attend` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_attend.py::test_attend` |
| `attention_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_attention_block.py::test_attention_block` |
| `conditioning_encoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_conditioning_encoder.py::test_conditioning_encoder` |
| `conv1_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_conv1_d.py::test_conv1_d` |
| `dropout1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_dropout1d.py::test_dropout1d` |
| `g_e_g_l_u` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_g_e_g_l_u.py::test_g_e_g_l_u` |
| `g_p_t` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_g_p_t.py::test_g_p_t` |
| `g_p_t2_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_g_p_t2_block.py::test_g_p_t2_block` |
| `g_p_t2_inference_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_g_p_t2_inference_model.py::test_g_p_t2_inference_model` |
| `g_p_t2_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_g_p_t2_model.py::test_g_p_t2_model` |
| `group_norm32` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_group_norm32.py::test_group_norm32` |
| `hifi_decoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_hifi_decoder.py::test_hifi_decoder` |
| `hifigan_generator` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_hifigan_generator.py::test_hifigan_generator` |
| `instance_norm1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_instance_norm1d.py::test_instance_norm1d` |
| `learned_position_embeddings` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_learned_position_embeddings.py::test_learned_position_embeddings` |
| `mel_scale` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_mel_scale.py::test_mel_scale` |
| `mel_spectrogram` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_mel_spectrogram.py::test_mel_spectrogram` |
| `parametrization_list` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_parametrization_list.py::test_parametrization_list` |
| `parametrized_conv1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_parametrized_conv1d.py::test_parametrized_conv1d` |
| `parametrized_conv_transpose1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_parametrized_conv_transpose1d.py::test_parametrized_conv_transpose1d` |
| `perceiver_resampler` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_perceiver_resampler.py::test_perceiver_resampler` |
| `pre_emphasis` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_pre_emphasis.py::test_pre_emphasis` |
| `q_k_v_attention_legacy` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_q_k_v_attention_legacy.py::test_q_k_v_attention_legacy` |
| `res_block1` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_res_block1.py::test_res_block1` |
| `res_net_speaker_encoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_res_net_speaker_encoder.py::test_res_net_speaker_encoder` |
| `s_e_basic_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_s_e_basic_block.py::test_s_e_basic_block` |
| `s_e_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_s_e_layer.py::test_s_e_layer` |
| `weight_norm` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/xtts_v2/tests/pcc/test_weight_norm.py::test_weight_norm` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/xtts_v2/tests/pcc/test_adaptive_avg_pool2d.py::test_adaptive_avg_pool2d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_attend.py::test_attend -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_attention_block.py::test_attention_block -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_conditioning_encoder.py::test_conditioning_encoder -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_conv1_d.py::test_conv1_d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_dropout1d.py::test_dropout1d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_g_e_g_l_u.py::test_g_e_g_l_u -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_g_p_t.py::test_g_p_t -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_g_p_t2_block.py::test_g_p_t2_block -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_g_p_t2_inference_model.py::test_g_p_t2_inference_model -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_g_p_t2_model.py::test_g_p_t2_model -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_group_norm32.py::test_group_norm32 -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_hifi_decoder.py::test_hifi_decoder -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_hifigan_generator.py::test_hifigan_generator -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_instance_norm1d.py::test_instance_norm1d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_learned_position_embeddings.py::test_learned_position_embeddings -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_mel_scale.py::test_mel_scale -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_mel_spectrogram.py::test_mel_spectrogram -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_parametrization_list.py::test_parametrization_list -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_parametrized_conv1d.py::test_parametrized_conv1d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_parametrized_conv_transpose1d.py::test_parametrized_conv_transpose1d -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_perceiver_resampler.py::test_perceiver_resampler -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_pre_emphasis.py::test_pre_emphasis -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_q_k_v_attention_legacy.py::test_q_k_v_attention_legacy -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_res_block1.py::test_res_block1 -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_res_net_speaker_encoder.py::test_res_net_speaker_encoder -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_s_e_basic_block.py::test_s_e_basic_block -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_s_e_layer.py::test_s_e_layer -svv
python -m pytest models/demos/xtts_v2/tests/pcc/test_weight_norm.py::test_weight_norm -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/xtts_v2/demo/demo.py::test_demo -svv
```

## Next steps

