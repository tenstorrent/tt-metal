Moreh Clip Grad Norm – Step3 – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates input buffer address and `clip_coef_clamped` buffer address.
  - Writer: updates output buffer base address (in-place).
- Constants across cache hits: `num_tiles` per core is shape-derived and included in the default hash; core iteration matches creation.
- Hashing: Uses default hash based on tensor shapes and memory/compute configs.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_program_factory.cpp`.

Status: Reviewed – no issues identified.
