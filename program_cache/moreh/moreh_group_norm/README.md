# moreh/moreh_group_norm program cache review

- Issue: override of optional outputs (mean, rstd) on cache-hit unconditionally dereferences buffers
- Suspected root cause: Missing `has_value()` checks in `override_runtime_arguments`
- Failure mode: Crash or hang on second run when `are_required_outputs=(True, False, False)`

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp`
  - Lines 317-319: `tensor_return_value[1]->buffer()` and `tensor_return_value[2]->buffer()` without `has_value()`

Repro command:
```bash
pytest -q program_cache/moreh/moreh_group_norm/failures/test_moreh_group_norm_cachehit_optional_outputs_override.py::test_moreh_group_norm_program_cache_optional_outputs_override -s --disable-warnings
```

Suggested fix:
- In `override_runtime_arguments`, guard access to `tensor_return_value[1]` and `[2]` with `has_value()` and only override writer runtime args indices 1 and 2 when the optional outputs exist. Ensure indices match create-time ordering.
