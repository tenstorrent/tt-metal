Binary NG OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/binary_ng` program cache: hashing, program factory selection, and cache-hit overrides.
- No correctness issues found. Overrides comprehensively update buffer addresses, per-core counts/offsets, CB addresses, and scalar runtime args. Hash excludes volatile scales via `operation_attributes_t::to_hash` to avoid cache fragmentation.

Key observations

- Custom program hash
  - `BinaryNgDeviceOperation::compute_program_hash(...)` hashes attributes (with `to_hash`), input dtypes and memory configs. Shapes are not in hash; factory uses runtime args for sizes.
  - `operation_attributes_t::to_hash()` intentionally drops quant scales from post activations when `is_quant_op` to reduce key explosion.

- Override runtime arguments
  - `ProgramFactory::override_runtime_arguments(...)` delegates to `set_or_update_runtime_arguments(...)` with an updater that writes into `GetRuntimeArgs(program, kernel, core)` for reader/writer/compute.
  - Helper recomputes per-core work split, start ids, and updates CB dynamic addresses implicitly through runtime args and CB handles stored in `shared_variables`.

- Validation on cache paths
  - `validate_on_program_cache_hit/miss` enforce broadcasting rules, dtype/layout, and sharding consistency, preventing cache-hit misuse.

Conclusion

- Implementation follows the new typed infra best practices and correctly updates runtime-only values on cache hits. Marking reviewed with no issues.
