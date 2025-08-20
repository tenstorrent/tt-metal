Moreh Clip Grad Norm – Step1 – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`. No program-cache issues found.

- Override updates:
  - Reader: updates per-input buffer address and runtime scalar `decimal` derived from `norm_type`.
  - Writer: updates output buffer base address.
  - Compute: updates `p` and `p_is_negative` derived from `norm_type`.
- Constants across cache hits: `num_tiles`, `origin_h/w`, and `tile_offset` are shape/attr-derived and included in the default hash; they do not need overriding.
- Hashing: Uses default hash including `norm_type`, `tile_offset_of_tmp_pow_sum`, memory/compute configs, and all input tensor shapes; prevents stale args on cache hit.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_program_factory.cpp`.

Status: Reviewed – no issues identified.
