<!-- BEGIN bringup -->
# Bring-up run report — `/localdev/lserbedzija/hf_models/voxtral-tts-backbone`

_Generated: 2026-08-13 11:25:41 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after bring-up.

## Backend & template match

- **Backend picked:** `Voxtral TTS Backbone (mistral decoder)`  (EXACT (model_type match))
- **Closest template:** `models/demos/voxtral_tts_backbone/`
- **Target model_type:** `mistral`
- **Sibling / template base:** `/localdev/lserbedzija/hf_models/voxtral-tts-backbone` (model_type=`mistral`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `Voxtral TTS Backbone (mistral decoder)` (selected) | 100 | exact model_type 'mistral' |
| 2 | `tt_transformers / simple_text_demo` | 40 | category 'LLM' default (generic runner) |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (5): graduated, native ttnn, PCC verified
  - `attention`, `decoder_layer`, `m_l_p`, `r_m_s_norm`, `rotary_embedding`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_backbone/tests/pcc/test_attention.py::test_attention` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_backbone/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_backbone/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_backbone/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_backbone/tests/pcc/test_rotary_embedding.py::test_rotary_embedding` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/voxtral_tts_backbone/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/demos/voxtral_tts_backbone/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest models/demos/voxtral_tts_backbone/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/demos/voxtral_tts_backbone/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/demos/voxtral_tts_backbone/tests/pcc/test_rotary_embedding.py::test_rotary_embedding -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/voxtral_tts_backbone/tests/e2e/test_e2e_pipeline.py -svv
python -m pytest models/demos/voxtral_tts_backbone/tests/e2e/test_trace_capture.py -svv
python -m pytest models/demos/voxtral_tts_backbone/demo/demo.py::test_demo -svv
python -m pytest models/demos/voxtral_tts_backbone/demo/demo_causal_lm_logits.py::test_demo -svv
python -m pytest models/demos/voxtral_tts_backbone/demo/demo_text_generation.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /localdev/lserbedzija/hf_models/voxtral-tts-backbone`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 5, ungraduated: 0
<!-- END trace-gate -->

<!-- BEGIN emit-e2e -->
# E2E report — `/localdev/lserbedzija/hf_models/voxtral-tts-backbone`

_Generated: 2026-08-13 11:25:41 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 5/5 on device (100%), 0/5 on CPU (0%)
- Graduated (ON_DEVICE) : 1/5 (20%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=4  ADAPT-wired=0  NEW-native=1  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=0
- operations: 5/5 on device (100%), 0/5 on CPU (0%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `causal_lm_logits` | n/a | `models/demos/voxtral_tts_backbone/demo/demo_causal_lm_logits.py` | (none) | (none) |
| `text_generation` | n/a | `models/demos/voxtral_tts_backbone/demo/demo_text_generation.py` | (none) | (none) |

## Reproduce

### causal_lm_logits
```bash
python models/demos/voxtral_tts_backbone/demo/demo_causal_lm_logits.py
```

### text_generation
```bash
python models/demos/voxtral_tts_backbone/demo/demo_text_generation.py
```
<!-- END emit-e2e -->
