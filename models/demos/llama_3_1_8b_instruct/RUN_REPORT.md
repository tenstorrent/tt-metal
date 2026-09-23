<!-- BEGIN bringup -->
# Bring-up run report — `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`

_Generated: 2026-09-09 23:13:44 UTC_

_Topology: TP=4 x DP=1 (mesh 1x4, 4 chips) — run emit-e2e / optimize with `--mesh 1x4`._

## Outcome

**Did not converge** after bring-up.

## Backend & template match

- **Backend picked:** `NemotronH (nemotron_h hybrid Mamba2/MoE)`
- **Closest template:** `models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16`
- **Target model_type:** `llama`
- **Sibling / template base:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` (selected) | 30 | category 'LLM' default |
| 2 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |
| 3 | `gemma4 (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (4): graduated, native ttnn, PCC verified
  - `attention`, `m_l_p`, `r_m_s_norm`, `rotary_embedding`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (1): retry next run
  - `decoder_layer`
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/llama_3_1_8b_instruct/tests/pcc/test_attention.py::test_attention` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/llama_3_1_8b_instruct/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/llama_3_1_8b_instruct/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/llama_3_1_8b_instruct/tests/pcc/test_rotary_embedding.py::test_rotary_embedding` |
| `decoder_layer` | [wait] | PENDING | retry next run | `models/demos/llama_3_1_8b_instruct/tests/pcc/test_decoder_layer.py::test_decoder_layer` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/llama_3_1_8b_instruct/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/demos/llama_3_1_8b_instruct/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/demos/llama_3_1_8b_instruct/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/demos/llama_3_1_8b_instruct/tests/pcc/test_rotary_embedding.py::test_rotary_embedding -svv
python -m pytest models/demos/llama_3_1_8b_instruct/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_e2e_pipeline.py -svv
python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_trace_contract.py -svv
python -m pytest models/demos/llama_3_1_8b_instruct/demo/demo.py::test_demo -svv
python -m pytest models/demos/llama_3_1_8b_instruct/demo/demo_text_generation.py::test_demo -svv
```

## Next steps

- **1 component(s) not graduated** — resume where it left off (already-graduated components are kept):
  - `python -m scripts.tt_hw_planner promote /home/ttuser/benchmark-data/Llama-3.1-8B-Instruct --box <BOX> --mesh <MESH>`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **EAGER_WAIVED**

trace not engaged; eager permitted because ungraduated module(s) present: decoder_layer

graduated on-device: 4, ungraduated: 1

fresh capture: no perf test to capture
<!-- END trace-gate -->

<!-- BEGIN emit-e2e -->
# E2E report — `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`

_Generated: 2026-09-09 23:13:44 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 4/5 on device (80%), 1/5 on CPU (20%)
- Graduated (ON_DEVICE) : 4/5 (80%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=0  ADAPT-wired=4  NEW-native=0  NEW-partial-CPU=0
- on CPU    : NEW-fallback=1  REUSE/ADAPT-not-wired=0
- operations: 4/5 on device (80%), 1/5 on CPU (20%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: `decoder_layer`

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `text_generation` | n/a | `models/demos/llama_3_1_8b_instruct/demo/demo_text_generation.py` | (none) | (none) |

## Reproduce

### text_generation
```bash
python models/demos/llama_3_1_8b_instruct/demo/demo_text_generation.py
```
<!-- END emit-e2e -->
