Moreh SGD — Program Cache Review

- Status: Reviewed — issue found (failing test added)

Summary
- Reader args: param_in, grad, optional momentum_in base address followed by per-core counts and packed scalars (lr, momentum, dampening, weight_decay, one). Writer args: param_out and optional momentum_out base addresses plus tile metadata.
- Override unconditionally dereferences `tensor_return_value.at(1)->buffer()` even when the cached program was created without `momentum_buffer_out`. This can fault on cache-hit in configurations where momentum is disabled (no momentum buffers).

Key references
- Factory create: `ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/moreh_sgd_program_factory.cpp`
  - Reader arg order (addresses + scalars): around L195–L206.
  - Writer arg order (addresses + metadata): around L208–L214.
- Cache-hit override (problematic deref and updates):
  - Unconditional deref of optional momentum_out: around L238–L245.
  - Reader base addresses updated with conditional momentum: around L251–L257.
  - Writer base addresses updated with conditional momentum: around L261–L266.

Failure test
- Test: `program_cache/moreh/moreh_sgd/failures/test_moreh_sgd_cachehit_optional_momentum_out_deref.py`
- Mode: second run (cache-hit) with `momentum=0` triggers override path; expected to fault due to unconditional deref of absent `momentum_buffer_out`.

Suggested fix
- Move `auto momentum_buffer_out_buffer = tensor_return_value.at(1)->buffer();` inside the `if (has_momentum_buffer_out)` block, or guard access with a size/has_value check before dereferencing.
