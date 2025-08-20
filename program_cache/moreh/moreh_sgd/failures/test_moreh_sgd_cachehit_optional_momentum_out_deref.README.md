Issue: Optional momentum_out buffer deref on cache-hit may cause fault

Summary
- When `momentum==0`, the op returns `None` for `momentum_buffer_out`. On cache-hit, the override function accesses `tensor_return_value.at(1)->buffer()` and conditionally writes `runtime_args[1]` only if `has_momentum_buffer_out` was captured from create. If create saw no momentum_out, override should not deref index 1.

References
- File: `ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/moreh_sgd_program_factory.cpp`
  - Override: lines approx L238–L265. `auto momentum_buffer_out_buffer = tensor_return_value.at(1)->buffer();` can be invalid if no momentum_out is produced for this cached program. Guarded by `has_momentum_buffer_out` for writing, but deref happens regardless.

Failure mode
- Expected: crash or fault on second run during override or kernel launch if deref yields invalid pointer.

Repro
```bash
pytest -q program_cache/moreh/moreh_sgd/failures/test_moreh_sgd_cachehit_optional_momentum_out_deref.py -s --disable-warnings
```

Suggested fix
- Delay fetching `tensor_return_value.at(1)->buffer()` until inside the `if (has_momentum_buffer_out)` block, or guard the vector size and value presence before dereferencing.
