Where (experimental) — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- New infra device op with `ElementWiseMultiCoreWhereProgram` factory.
- Hash includes: program factory index, memory configs and dtypes of condition/true/false tensors, and output attributes. No runtime-only addresses are hashed.
- Override correctly re-applies runtime arguments for condition/true/false inputs and output using a shared helper, ensuring buffer base addresses and per-core counts/offsets are refreshed.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/experimental/where/device/where_device_operation.cpp`
  - `compute_program_hash(...)` uses `hash_operation<WhereDeviceOperation>(...)` over dtypes/mem_configs and program factory selection.
- `ttnn/cpp/ttnn/operations/experimental/where/device/program_factory/element_wise_multi_core_where_program.cpp`
  - `set_eltwise_ternary_runtime_args<true>` at create sets kernel runtime args in a single helper.
  - `override_runtime_arguments` calls `set_eltwise_ternary_runtime_args<false>` with the same ordering for cache-hit path, updating buffer addresses and sizes consistently.

Notes
- The helper-based approach minimizes the risk of arg-order drift between create and override paths.
