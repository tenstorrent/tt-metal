# Sampling Module Overview

The `models.common.sampling` package bundles everything needed to run on-device
sampling (top-k / top-p / temperature/ seed) plus presence/frequency/repetition
penalties with optional trace capture.

## Key Components
- `SamplingGenerator`: high-level class that owns both `TTSampling` and
  `TTPenalties`, exposes helper methods to reset sampling parameters, penalties,
  prompt/output state, and to run sampling with or without trace capture.
- `format_sampling_params`: utility that pads/clamps sampling parameters to the
  hardware-friendly layout expected by `TTSampling`.

## Quick Start
```python
from models.common.sampling import SamplingGenerator, format_sampling_params

sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl)

params = format_sampling_params(user_params, max_batch_size=32)
sampling.reset_sampling_params(params)

sampling.reset_seed(seed)

sampling.reset_prompt_tokens(prompt_tokens)   # torch tensor shaped [B, S]
sampling.reset_output_state(output_tokens)

tt_tokens = sampling.sample(
    tt_logits,
    tt_out_tok=tt_out_buffer,
)
```

`SamplingGenerator.sample()` accepts `enable_trace=True` to record/replay
sampling traces. Callers typically instantiate the module with tracing enabled
and then choose per-request whether to split the trace.

## Two-Part Trace Mode
Before running decode, set `generator.enable_split_sampling = True` (or
`False`) to choose between the two behaviors:

1. **Split trace:** The model trace stops at logits and `SamplingGenerator`
   captures/executes a dedicated sampling trace immediately afterward.
2. **Single trace:** Sampling runs inline as part of the model trace for
   maximum performance.
