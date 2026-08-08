# Bring-up run report — `microsoft/VibeVoice-1.5B`

_Generated: 2026-07-08 02:29:04 UTC_

## Outcome

**Converged** after ? iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `XTTS-v2 (multilingual TTS)`
- **Closest template:** `models/demos/xtts_v2`
- **Target model_type:** `vibevoice`
- **Sibling / template base:** `/local/ttuser/apande/models/XTTS-v2-hf`

## Placement summary

- **ON_DEVICE** (19): graduated, native ttnn, PCC verified
  - `block1_d`, `convlayer`, `f_f_n`, `feed_forward_network`, `final_layer`, `head_layer`, `norm_conv1d`, `norm_conv_transpose1d`, `qwen2_decoder_layer`, `qwen2_model`, `s_conv1d`, `s_conv_transpose1d`, `speech_connector`, `timestep_embedder`, `tokenizer_decoder`, `tokenizer_encoder`, `vibe_voice_acoustic_tokenizer_model`, `vibe_voice_diffusion_head`, `vibe_voice_semantic_tokenizer_model`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `block1_d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_block1_d.py::test_block1_d` |
| `convlayer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_convlayer.py::test_convlayer` |
| `f_f_n` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_f_f_n.py::test_f_f_n` |
| `feed_forward_network` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_feed_forward_network.py::test_feed_forward_network` |
| `final_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_final_layer.py::test_final_layer` |
| `head_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_head_layer.py::test_head_layer` |
| `norm_conv1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_norm_conv1d.py::test_norm_conv1d` |
| `norm_conv_transpose1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_norm_conv_transpose1d.py::test_norm_conv_transpose1d` |
| `qwen2_decoder_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_qwen2_decoder_layer.py::test_qwen2_decoder_layer` |
| `qwen2_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_qwen2_model.py::test_qwen2_model` |
| `s_conv1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_s_conv1d.py::test_s_conv1d` |
| `s_conv_transpose1d` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_s_conv_transpose1d.py::test_s_conv_transpose1d` |
| `speech_connector` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_speech_connector.py::test_speech_connector` |
| `timestep_embedder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_timestep_embedder.py::test_timestep_embedder` |
| `tokenizer_decoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_tokenizer_decoder.py::test_tokenizer_decoder` |
| `tokenizer_encoder` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_tokenizer_encoder.py::test_tokenizer_encoder` |
| `vibe_voice_acoustic_tokenizer_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_acoustic_tokenizer_model.py::test_vibe_voice_acoustic_tokenizer_model` |
| `vibe_voice_diffusion_head` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_diffusion_head.py::test_vibe_voice_diffusion_head` |
| `vibe_voice_semantic_tokenizer_model` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_semantic_tokenizer_model.py::test_vibe_voice_semantic_tokenizer_model` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_block1_d.py::test_block1_d -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_convlayer.py::test_convlayer -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_f_f_n.py::test_f_f_n -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_feed_forward_network.py::test_feed_forward_network -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_final_layer.py::test_final_layer -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_head_layer.py::test_head_layer -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_norm_conv1d.py::test_norm_conv1d -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_norm_conv_transpose1d.py::test_norm_conv_transpose1d -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_qwen2_decoder_layer.py::test_qwen2_decoder_layer -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_qwen2_model.py::test_qwen2_model -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_s_conv1d.py::test_s_conv1d -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_s_conv_transpose1d.py::test_s_conv_transpose1d -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_speech_connector.py::test_speech_connector -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_timestep_embedder.py::test_timestep_embedder -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_tokenizer_decoder.py::test_tokenizer_decoder -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_tokenizer_encoder.py::test_tokenizer_encoder -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_acoustic_tokenizer_model.py::test_vibe_voice_acoustic_tokenizer_model -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_diffusion_head.py::test_vibe_voice_diffusion_head -svv
python -m pytest models/demos/vibevoice_1_5b/tests/pcc/test_vibe_voice_semantic_tokenizer_model.py::test_vibe_voice_semantic_tokenizer_model -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_00_forward_on_device.py -svv
python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_e2e_tts.py -svv
python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_trace_2cq.py -svv
python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_trace_2cq_timing.py -svv
python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_tts_perf.py -svv
python -m pytest models/demos/vibevoice_1_5b/demo/demo.py::test_demo -svv
python -m pytest models/demos/vibevoice_1_5b/demo/demo_tts.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e microsoft/VibeVoice-1.5B`
