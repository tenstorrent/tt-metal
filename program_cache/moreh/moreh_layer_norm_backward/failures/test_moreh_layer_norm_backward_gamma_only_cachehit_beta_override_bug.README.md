# Test: gamma-only cache-hit triggers beta override bug

- OP: `moreh/moreh_layer_norm_backward` (gamma_beta_grad factory)
- File: `test_moreh_layer_norm_backward_gamma_only_cachehit_beta_override_bug.py`

Issue:
- In override, index 1 (beta_grad addr) is guarded by `if (gamma_grad_buffer != nullptr)` instead of `if (beta_grad_buffer != nullptr)`.
- When only gamma_grad is provided, beta_grad is absent but the guard passes, causing a bad deref.

Location:
- `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp` ~298-304.

Repro:
```bash
pytest -q program_cache/moreh/moreh_layer_norm_backward/failures/test_moreh_layer_norm_backward_gamma_only_cachehit_beta_override_bug.py -s --disable-warnings
```

Suggested fix:
- Change the guard for runtime_args[1] to check `beta_grad_buffer`.
