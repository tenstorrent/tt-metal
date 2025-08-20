# moreh_norm — Program Cache Review

- OP path: `ttnn/cpp/ttnn/operations/moreh/moreh_norm`
- Infra: new templated device operation (`ttnn/device_operation.hpp`)
- Result: No program-cache override issues found

## Findings

- Hashing
  - Uses default program hash (op type + `operation_attributes_t` + `tensor_args_t`).
  - Attributes include `p`, `dim`, `keepdim`, `memory_config`, and `DeviceComputeKernelConfig`.
  - Tensor args include `input` and optional `output`.

- Program factories and overrides
  - Three factories (selected by `dim`): `ProgramFactoryWOther`, `ProgramFactoryHOther`, `ProgramFactoryNCOther`.
  - Create path sets reader/writer runtime args including base addresses plus compile-time-derived counts/strides.
  - Override updates buffer base addresses at index 0 for both reader and writer kernels across all cores. Other args are stable across cache hits.

- Correctness assessment
  - Buffer addresses are correctly overridden on cache hits; arg ordering matches create path.
  - Non-overridden args are derived from hashed attributes and shapes and need not change between runs.

## Status

Reviewed for program cache correctness; no issues found.
