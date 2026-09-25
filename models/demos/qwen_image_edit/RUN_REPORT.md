<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 0, ungraduated: 0
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `Qwen/Qwen-Image-Edit`

_Generated: 2026-09-24 16:02:28 UTC_

## Outcome

**Converged** after bring-up.

## Placement summary

- **ON_DEVICE** (0): graduated, native ttnn, PCC verified
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
```

End-to-end / demo:
```bash
python -m pytest qwen_image_edit/tests/e2e/test_e2e_image_edit.py -svv
python -m pytest qwen_image_edit/tests/e2e/test_image_edit_perf.py -svv
python -m pytest qwen_image_edit/demo/demo.py::test_demo -svv
python -m pytest qwen_image_edit/demo/demo_image_edit.py::test_demo -svv
```

## Next steps
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `Qwen/Qwen-Image-Edit`

_Generated: 2026-09-24 16:02:28 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: (no tracked components)
- operations: (no tracked components)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `image_edit` | n/a | `qwen_image_edit/demo/demo_image_edit.py` | `qwen_image_edit/tests/e2e/test_e2e_image_edit.py` | `qwen_image_edit/tests/e2e/test_image_edit_perf.py` |

## Reproduce

### image_edit
```bash
python qwen_image_edit/demo/demo_image_edit.py
pytest qwen_image_edit/tests/e2e/test_e2e_image_edit.py -svv
pytest qwen_image_edit/tests/e2e/test_image_edit_perf.py -svv
```
<!-- END emit-e2e -->
