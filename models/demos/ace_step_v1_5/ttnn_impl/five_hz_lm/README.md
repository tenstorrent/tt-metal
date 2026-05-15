# `ttnn_impl/five_hz_lm` — TTNN LM import surface

The **canonical source** for `five_hz_llm_inference.py` and
`five_hz_constrained_logits_processor.py` remains
`models/demos/ace_step_v1_5/five_hz_lm/`, which calls into
`ttnn_impl/lm_constrained_logits_ttnn.py`, `lm_logits_ttnn.py`, etc.

This directory holds **thin re-export modules** so PCC tests can use parallel imports:

| Package | Role |
|---------|------|
| `torch_ref.five_hz_lm` | Patched PyTorch-only mirror (golden). |
| `ttnn_impl.five_hz_lm` | Same symbols as production `five_hz_lm` (TTNN assist when device is set). |

Do not duplicate large hunks of logic here; change the implementation in `five_hz_lm/` only.
