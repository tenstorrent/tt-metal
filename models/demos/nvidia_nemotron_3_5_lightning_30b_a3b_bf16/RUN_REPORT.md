<!-- BEGIN bringup -->
# Bring-up run report — `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

_Generated: 2026-09-05 23:05:58 UTC_

_Topology: TP=2 x DP=2 (mesh 2x2, 4 chips) — run emit-e2e / optimize with `--mesh 2x2`._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `NemotronH (nemotron_h hybrid Mamba2/MoE)`  (EXACT (model_type match))
- **Closest template:** `models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16`
- **Target model_type:** `nemotron_h`
- **Sibling / template base:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (model_type=`nemotron_h`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` (selected) | 100 | exact model_type 'nemotron_h' |
| 2 | `tt_transformers / simple_text_demo` | 40 | category 'LLM' default (generic runner) |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (10): graduated, native ttnn, PCC verified
  - `nemotron_h_attention`, `nemotron_h_block`, `nemotron_h_experts`, `nemotron_h_m_l_p`, `nemotron_h_mamba2_mixer`, `nemotron_h_mo_e`, `nemotron_h_r_m_s_norm`, `nemotron_h_topk_router`, `re_l_u_squared_activation`, `zamba2_r_m_s_norm_gated`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `nemotron_h_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_attention.py::test_nemotron_h_attention` |
| `nemotron_h_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_block.py::test_nemotron_h_block` |
| `nemotron_h_experts` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_experts.py::test_nemotron_h_experts` |
| `nemotron_h_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_m_l_p.py::test_nemotron_h_m_l_p` |
| `nemotron_h_mamba2_mixer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_mamba2_mixer.py::test_nemotron_h_mamba2_mixer` |
| `nemotron_h_mo_e` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_mo_e.py::test_nemotron_h_mo_e` |
| `nemotron_h_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_r_m_s_norm.py::test_nemotron_h_r_m_s_norm` |
| `nemotron_h_topk_router` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_topk_router.py::test_nemotron_h_topk_router` |
| `re_l_u_squared_activation` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_re_l_u_squared_activation.py::test_re_l_u_squared_activation` |
| `zamba2_r_m_s_norm_gated` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_zamba2_r_m_s_norm_gated.py::test_zamba2_r_m_s_norm_gated` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_attention.py::test_nemotron_h_attention -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_block.py::test_nemotron_h_block -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_experts.py::test_nemotron_h_experts -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_m_l_p.py::test_nemotron_h_m_l_p -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_mamba2_mixer.py::test_nemotron_h_mamba2_mixer -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_mo_e.py::test_nemotron_h_mo_e -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_r_m_s_norm.py::test_nemotron_h_r_m_s_norm -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_nemotron_h_topk_router.py::test_nemotron_h_topk_router -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_re_l_u_squared_activation.py::test_re_l_u_squared_activation -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/pcc/test_zamba2_r_m_s_norm_gated.py::test_zamba2_r_m_s_norm_gated -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_main_perf.py -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_perf.py -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_text_generation_perf.py -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/demo/demo.py::test_demo -svv
python -m pytest models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/demo/demo_text_generation.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
<!-- END bringup -->
