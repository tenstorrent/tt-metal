# moreh_nll_loss_step2 — Program Cache Review

- OP path: `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2`
- Infra: new templated device operation (`ttnn/device_operation.hpp`)
- Result: No program-cache override issues found

## Findings

- Hashing
  - No custom `compute_program_hash(...)`; default hash is used (op type + `operation_attributes_t` + `tensor_args_t`).
  - Attributes include `reduction`, `ignore_index`, `memory_config`, and `DeviceComputeKernelConfig`.
  - Tensor args include `input`, `target`, and optional `weight`/`divisor`/`output`, so presence/absence of optionals participates in the hash.

- Program factory and overrides
  - Program creation sets per-core runtime args including buffer base addresses and derived quantities:
    - 2D reader args: `[input_addr, target_addr, weight_addr, divisor_addr, ignore_index, units_per_core, tile_offset, origin_N, origin_C, input.element_size()]`; writer args: `[output_addr, units_per_core, tile_offset, origin_N]`.
    - 3D adds `origin_W` and `output.element_size()` to writer args.
    - 4D adds `Wt`, `num_inner_tile`, `weight_num_tile`, and `input.element_size()`.
  - Override updates buffer base addresses and `ignore_index` on every core; writer override updates output base address only.

- Correctness assessment
  - Buffer addresses are correctly overridden on cache hits.
  - Non-overridden args are derived from hashed determinants and remain constant across cache hits.
  - Arg ordering between create/override matches for updated indices.

## Status

Reviewed for program cache correctness; no issues found.
