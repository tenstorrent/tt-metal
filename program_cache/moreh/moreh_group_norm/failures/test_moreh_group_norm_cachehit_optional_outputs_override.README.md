# Test: optional outputs override on cache-hit

- Title: Override of optional outputs causes crash/hang on cache-hit
- OP: `moreh/moreh_group_norm`
- Test file: `test_moreh_group_norm_cachehit_optional_outputs_override.py`

Issue description:
- On program cache hit, `override_runtime_arguments` unconditionally dereferences `tensor_return_value[1]` and `[2]` (mean/rstd), even when the caller did not request these outputs.

Suspected root cause:
- Missing `has_value()` checks in override:
  - `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp` lines 317-319.

Failure mode:
- Second run (cache-hit) may crash/hang when `are_required_outputs=(True, False, False)`.

Reproduction:
```bash
pytest -q program_cache/moreh/moreh_group_norm/failures/test_moreh_group_norm_cachehit_optional_outputs_override.py::test_moreh_group_norm_program_cache_optional_outputs_override -s --disable-warnings
```

Suggested fix:
- Guard access to optional outputs in override and only update writer runtime args indices 1 and 2 if present.
