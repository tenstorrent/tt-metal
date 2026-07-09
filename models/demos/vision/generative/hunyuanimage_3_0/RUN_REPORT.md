# Bring-up run report — `tencent/HunyuanImage-3.0`

_Generated: 2026-07-09 21:46:50 UTC_

## Outcome

**Converged** after ? iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `Stable Diffusion 1.4`
- **Closest template:** `models/demos/vision/generative/stable_diffusion`
- **Target model_type:** `Hunyuan`
- **Sibling / template base:** `CompVis/stable-diffusion-v1-4`

## Placement summary

- **ON_DEVICE** (2): graduated, native ttnn, PCC verified
  - `image3_decoder_layer`, `top_k_gate`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `image3_decoder_layer` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/hunyuanimage_3_0/tests/pcc/test_image3_decoder_layer.py::test_image3_decoder_layer` |
| `top_k_gate` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/vision/generative/hunyuanimage_3_0/tests/pcc/test_top_k_gate.py::test_top_k_gate` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/pcc/test_image3_decoder_layer.py::test_image3_decoder_layer -svv
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/pcc/test_top_k_gate.py::test_top_k_gate -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e tencent/HunyuanImage-3.0`
