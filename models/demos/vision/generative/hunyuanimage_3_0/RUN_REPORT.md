# Bring-up run report — `tencent/HunyuanImage-3.0`

_Generated: 2026-07-10 00:42:04 UTC_

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

End-to-end / demo:
```bash
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_e2e_prefill.py -svv
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_image3_prefill_perf.py -svv
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_trace_2cq.py -svv
python -m pytest models/demos/vision/generative/hunyuanimage_3_0/demo/demo_image3_prefill.py::test_demo -svv
```

## End-to-end pipeline (emit-e2e) — COMPLETE

Call-1 `hunyuan_image3_transformer_prefill` runs the SHARDED (TP=8) graduated
stubs on the full physical mesh `MeshShape(8,4)` + `FABRIC_1D`, composed along
the real HF nesting (decoder → mo_e → top_k_gate):

| gate | result |
|---|---|
| Gate 1 (native ttnn) | PASS — `host_op_selftest` n_host_ops=0 (fully on device) |
| Gate 2 (all graduated invoked) | PASS — `{image3_decoder_layer:1, mo_e:1, top_k_gate:1}` |
| Gate 3 (e2e PCC ≥ 0.95) | PASS — **0.99977** (l_aux 37.50 vs 37.15) |
| Command 3 (trace+2CQ) | PASS — prefill/decode captured host-free, trace PCC 1.0 |
| per-component sharded PCC (TP=8) | 0.99999 / 0.99970 / 1.0 |

Runnable package: `tt/pipeline.py` (shared), `demo/demo_image3_prefill.py`,
`tests/e2e/{test_e2e_prefill,test_trace_2cq}.py`. See `README.md` / `e2e_plan.json`.

## Next steps

- **All components graduated + e2e wired.** Re-run the gates:
  - `python -m pytest models/demos/vision/generative/hunyuanimage_3_0/tests/e2e -s`
