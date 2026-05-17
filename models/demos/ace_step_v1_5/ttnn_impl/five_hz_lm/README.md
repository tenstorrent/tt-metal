# `ttnn_impl/five_hz_lm` — vendored ACE-Step 5 Hz LM stack with TTNN assist

This folder is the **canonical** home for the ACE-Step 5 Hz language-model handler used by the
tt-metal demo. The implementation is fully vendored (no `acestep` package import), so the demo
runs without the upstream ACE-Step repo on `sys.path` for the LM half of the pipeline.

| File | Role |
|------|------|
| `__init__.py` | Re-exports `LocalFiveHzLMHandler`. |
| `local_five_hz_llm.py` | Public entrypoint; wraps `LocalFiveHzLMHandler` from `five_hz_llm_inference`. |
| `five_hz_llm_inference.py` | Generation loop (prefill + CoT + codes), CFG, sampling. Contains the TTNN-assist branches that call into `ttnn_impl/lm_constrained_logits_ttnn.py`, `lm_logits_ttnn.py`, and the experimental TTNN causal LM under `five_hz_causal_lm_experimental.py`. |
| `five_hz_constrained_logits_processor.py` | Metadata FSM / constrained decoder for the LM. |
| `five_hz_llm_backend_compat.py` | nano-vLLM availability preflight. |
| `five_hz_lm_constants.py` | Default LM instructions, duration limits, etc. |
| `five_hz_lm_gpu_config.py` | GPU / accelerator memory tier discovery. |
| `five_hz_lm_paths.py` | Checkpoint directory resolution (no `acestep` import). |

## Where the symmetric reference lives

For PCC / parity tests, the **PyTorch-only mirror** is at
`models/demos/ace_step_v1_5/torch_ref/five_hz_lm/`. Same module names, same public class
(`LocalFiveHzLMHandler`), but `set_ttnn_logits_device` / `experimental_ttnn_causal_lm` are no-ops
so the weights always load via `transformers.AutoModelForCausalLM`. Tests can import a parallel
pair:

```python
from models.demos.ace_step_v1_5.torch_ref.five_hz_lm   import LocalFiveHzLMHandler as RefHandler
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm   import LocalFiveHzLMHandler as TtHandler
```

## TTNN entry points

The handler dispatches into:

- `ttnn_impl/lm_constrained_logits_ttnn.py` — top-k, top-p, sampling, repetition penalty,
  EOS / pad checks, sequence concat (TTNN per-token logits assist).
- `ttnn_impl/lm_logits_ttnn.py` — CFG `uncond + s·(cond - uncond)` linear combine.
- `ttnn_impl/five_hz_causal_lm_experimental.py` → `ttnn_impl/ace_step_ds_r1_qwen.py::QwenModel`
  for `--experimental-5hz-ttnn-causal-lm` (full TTNN matmuls + host RoPE / KV / softmax).

`TTNN_LM_CONVERSION_SUMMARY.txt` is a longer-form note tracking what stays on host vs what was
moved to TTNN during bring-up.
