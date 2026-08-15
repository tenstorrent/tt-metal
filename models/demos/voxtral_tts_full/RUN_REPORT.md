<!-- BEGIN bringup -->
# Bring-up run report — `/localdev/lserbedzija/hf_models/voxtral-tts-full`

_Generated: 2026-08-15 20:11:43 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after bring-up.

## Backend & template match

- **Backend picked:** `XTTS-v2 (multilingual TTS)`
- **Closest template:** `models/demos/xtts_v2`
- **Target model_type:** `voxtral_tts`
- **Sibling / template base:** `/local/ttuser/apande/models/XTTS-v2-hf`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `hf_eager universal (TTS)` | 40 | category 'TTS' default (generic runner) |
| 2 | `XTTS-v2 (multilingual TTS)` (selected) | 30 | category 'TTS' default |

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
python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_e2e.py -svv
python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -svv
python -m pytest models/demos/voxtral_tts_full/demo/demo.py::test_demo -svv
python -m pytest models/demos/voxtral_tts_full/demo/demo_tts.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /localdev/lserbedzija/hf_models/voxtral-tts-full`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 7, ungraduated: 0
<!-- END trace-gate -->

<!-- BEGIN emit-e2e -->
# E2E report — `/localdev/lserbedzija/hf_models/voxtral-tts-full`

_Generated: 2026-08-15 20:11:43 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 7/7 on device (100%), 0/7 on CPU (0%)
- Graduated (ON_DEVICE) : 4/7 (57%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=3  ADAPT-wired=0  NEW-native=4  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=0
- operations: 114/114 on device (100%), 0/114 on CPU (0%)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `tts` | n/a | `models/demos/voxtral_tts_full/demo/demo_tts.py` | `models/demos/voxtral_tts_full/tests/e2e/test_tts_e2e.py` | `models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py` |

## Reproduce

### tts
```bash
python models/demos/voxtral_tts_full/demo/demo_tts.py
pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_e2e.py -svv
pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -svv
```
<!-- END emit-e2e -->
