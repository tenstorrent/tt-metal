<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 4, ungraduated: 3

fresh capture: invalid ============================= test session starts ==============================
platform linux -- Python 3.10.19, pytest-9.0.3, pluggy-1.6.0 -- /opt/venv/bin/python
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /localdev/lserbedzija/repos/tt-metal-pr46283
configfile: pytest.ini
plugins: anyio-4.14.2, benchmark-5.2.3, cov-7.0.0, dash-2.15.0, split-0.11.0, repeat-0.9.4, timeout-2.4.0, github-actions-annotate-failures-0.3.0
collecting ... ERROR: not found: /localdev/lserbedzija/repos/tt-metal-pr46283/models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py::test_tts_perf
(no match in any of [<Module test_tts_perf.py>])
collected 0 items
- generated xml file: /localdev/lserbedzija/repos/tt-metal-pr46283/generated/test_reports/most_recent_tests.xml -
============================ no tests ran in 0.02s =============================
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `/localdev/lserbedzija/hf_models/voxtral-tts-full`

_Generated: 2026-08-15 15:11:50 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Did not converge** after bring-up.

## Backend & template match

- **Backend picked:** `Voxtral TTS Backbone (mistral decoder)`  (TEMPLATE-FALLBACK (model_type mismatch — closest sibling by category))
- **Closest template:** `models/demos/voxtral_tts_backbone/`
- **Target model_type:** `voxtral_tts`
- **Sibling / template base:** `/localdev/lserbedzija/hf_models/voxtral-tts-backbone` (model_type=`mistral`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` | 40 | category 'LLM' default (generic runner) |
| 2 | `Voxtral TTS Backbone (mistral decoder)` (selected) | 30 | category 'LLM' default |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (5): graduated, native ttnn, PCC verified
  - `attention`, `decoder_layer`, `m_l_p`, `r_m_s_norm`, `tts_backbone`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (2): retry next run
  - `codec_decoder`, `flow_matching`
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_attention.py::test_attention` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `tts_backbone` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/voxtral_tts_full/tests/pcc/test_tts_backbone.py::test_tts_backbone` |
| `codec_decoder` | [wait] | PENDING | retry next run | `models/demos/voxtral_tts_full/tests/pcc/test_codec_decoder.py::test_codec_decoder` |
| `flow_matching` | [wait] | PENDING | retry next run | `models/demos/voxtral_tts_full/tests/pcc/test_flow_matching.py::test_flow_matching` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_tts_backbone.py::test_tts_backbone -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_codec_decoder.py::test_codec_decoder -svv
python -m pytest models/demos/voxtral_tts_full/tests/pcc/test_flow_matching.py::test_flow_matching -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_e2e_tts.py -svv
python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -svv
python -m pytest models/demos/voxtral_tts_full/demo/demo.py::test_demo -svv
python -m pytest models/demos/voxtral_tts_full/demo/demo_tts.py::test_demo -svv
```

## Next steps

- **2 component(s) not graduated** — resume where it left off (already-graduated components are kept):
  - `python -m scripts.tt_hw_planner promote /localdev/lserbedzija/hf_models/voxtral-tts-full --box <BOX> --mesh <MESH>`
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `/localdev/lserbedzija/hf_models/voxtral-tts-full`

_Generated: 2026-08-15 15:11:50 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 5/7 on device (71%), 2/7 on CPU (28%)
- Graduated (ON_DEVICE) : 2/7 (28%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=3  ADAPT-wired=0  NEW-native=2  NEW-partial-CPU=0
- on CPU    : NEW-fallback=2  REUSE/ADAPT-not-wired=0
- operations: 112/114 on device (98%), 2/114 on CPU (1%)
- CPU-fallback modules: (none — fully on device)
- Trace-validated on device (G6, probe false-positive cleared): `codec_decoder`, `flow_matching`

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `tts` | n/a | `models/demos/voxtral_tts_full/demo/demo_tts.py` | `models/demos/voxtral_tts_full/tests/e2e/test_e2e_tts.py` | `models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py` |

## Reproduce

### tts
```bash
python models/demos/voxtral_tts_full/demo/demo_tts.py
pytest models/demos/voxtral_tts_full/tests/e2e/test_e2e_tts.py -svv
pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -svv
```
<!-- END emit-e2e -->
