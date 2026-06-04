# `torch_ref/five_hz_lm` — PyTorch reference LM

This directory is a **full copy** of `../five_hz_lm/` with two behavioral changes in
`five_hz_llm_inference.py`:

1. **`set_ttnn_logits_device`** — no-op; `_ttnn_logits_device` stays `None` so logits processing
   (CFG combine, repetition, top-k/p, sampling, concat, EOS helpers, etc.) uses **PyTorch** only.
2. **`_load_pytorch_model`** — experimental TTNN causal LM (`AceStepFiveHzExperimentalTtnnCausalLM`)
   is **not** loaded; flags are ignored with a warning and **`AutoModelForCausalLM`** is always used.

Use this package for **PCC / parity tests** against `five_hz_lm` + `ttnn_impl/lm_*` when the
production handler has `set_ttnn_logits_device` and optional experimental TTNN decode enabled.

## Run the tt-metal-style demo with this handler

From the tt-metal repo root (same CLI as `models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py`):

```bash
./python_env/bin/python3 models/experimental/ace_step_v1_5/torch_ref/run_ace_step_ttmetal_demo_torch_ref_lm.py \
  --prompt "..." --variant acestep-v15-base --lm_variant acestep-5Hz-lm-1.7B \
  --ace-step-repo-root /path/to/ACE-Step-1.5 --duration_sec 15 --infer_steps 4 --out /tmp/out.wav
```

Note: `torch_ref/run_prompt_to_wav.py` is a **different** script (torch DiT / VAE pipeline). The
file above is the ACE-Step v1.5 **tt-metal** demo entry with imports switched to this package.
