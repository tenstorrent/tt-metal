# moreh/moreh_group_norm_backward program cache review

Reviewed two factories:
- `input_grad`: override guards optional gamma correctly; test added as regression guard.
- `gamma_beta_grad`: override guards optional outputs correctly via writer rtargs checks.

Tests:
- failures/test_moreh_group_norm_backward_input_grad_cachehit_optional_gamma_override.py
  - Purpose: ensure index 4 is only updated when gamma exists on cache-hit.

Run:
```bash
pytest -q program_cache/moreh/moreh_group_norm_backward/failures/test_moreh_group_norm_backward_input_grad_cachehit_optional_gamma_override.py -s --disable-warnings
```
