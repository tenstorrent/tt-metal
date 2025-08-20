Moreh Clip Grad Norm – Step2 – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates input buffer address and runtime scalar `decimal` (from `1.0/norm_type`).
  - Writer: updates output buffer base address.
  - Compute: updates `p` and `p_is_negative` derived from `1.0/norm_type`.
- Constants across cache hits: `num_tiles` is derived from input shape and included in the default hash; no override needed.
- Hashing: Uses default hash including `norm_type`, memory/compute configs, and input/output specs.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_program_factory.cpp`.

Status: Reviewed – no issues identified.
