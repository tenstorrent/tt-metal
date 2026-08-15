<!-- BEGIN bringup -->
# Bring-up run report — `/localdev/lserbedzija/hf_models/voxtral-tts-full`

_Generated: 2026-08-15 18:37:26 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `XTTS-v2 (multilingual TTS)`
- **Closest template:** `models/demos/xtts_v2`
- **Target model_type:** `voxtral_tts`
- **Sibling / template base:** `/local/ttuser/apande/models/XTTS-v2-hf`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `XTTS-v2 (multilingual TTS)` (selected) | 66 | LLM: Closest whole-model structural sibling: Voxtral-TTS is an autoregressive decoder LM that emits discrete acoustic tokens which a codec/vocoder decodes to waveform, exactly the 'autoregressive-GPT TTS [ |
| 2 | `tt_transformers / simple_text_demo` | 65 | LLM: Backbone match for the generative trunk: VoxtralTtsForConditionalGeneration's repeated core block is a Mistral-style decoder-only causal LM layer (RMSNorm + RoPE + GQA attention + SwiGLU MLP). This ba |
| 3 | `Mistral-Small-3.1 (mistral3 VLM)` | 50 | LLM: Voxtral is Mistral-lineage: VoxtralTts wraps a Mistral/Ministral decoder-only causal LM trunk (RMSNorm + GQA + RoPE + SwiGLU MLP) that emits audio codes. mistral3 is the same decoder-layer family, so  |

## Placement summary

- **ON_DEVICE** (7): graduated, native ttnn, PCC verified
  - `attention`, `codec_decoder`, `decoder_layer`, `flow_matching`, `m_l_p`, `r_m_s_norm`, `tts_backbone`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_attention.py::test_attention` |
| `codec_decoder` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_codec_decoder.py::test_codec_decoder` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `flow_matching` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_flow_matching.py::test_flow_matching` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `tts_backbone` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_tts_backbone.py::test_tts_backbone` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_codec_decoder.py::test_codec_decoder -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_flow_matching.py::test_flow_matching -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_tts_backbone.py::test_tts_backbone -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/voxtral_tts_full/demo/demo.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /localdev/lserbedzija/hf_models/voxtral-tts-full`
<!-- END bringup -->
