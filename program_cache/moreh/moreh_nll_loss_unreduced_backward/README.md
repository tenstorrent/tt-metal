# moreh_nll_loss_unreduced_backward — Program Cache Review

- OP path: `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward`
- Infra: new templated device operation (`ttnn/device_operation.hpp`)
- Result: No program-cache override issues found

## Findings

- Hashing
  - No custom `compute_program_hash(...)`; default hash is used (op type + `operation_attributes_t` + `tensor_args_t`).
  - Attributes include `ignore_index`, `memory_config`, and `DeviceComputeKernelConfig`.
  - Tensor args include `target`, `output_grad`, optional `weight`, and optional `input_grad`. Presence/absence of optionals participates in the hash.

- Program factory and overrides
  - Program creation sets per-core runtime args including buffer base addresses and derived quantities:
    - 2D reader args: `[target_addr, output_grad_addr, weight_addr, ignore_index, units_per_core, tile_offset, Nt, channel_size, Ct]`; writer args: `[input_grad_addr, units_per_core, tile_offset]`.
    - 3D reader args: `[target_addr, output_grad_addr, weight_addr, ignore_index, units_per_core, tile_offset, channel_size, Ct, Wt]`; writer args: `[input_grad_addr, units_per_core, tile_offset]`.
    - 4D reader args: `[target_addr, output_grad_addr, weight_addr, ignore_index, units_per_core, tile_offset, num_inner_tile, channel_size, Ct]`; writer args: `[input_grad_addr, units_per_core, tile_offset]`.
  - Override updates buffer base addresses and `ignore_index` on every core; writer override updates output base address only.

- Correctness assessment
  - Buffer addresses are correctly overridden on cache hits.
  - Non-overridden args are derived from hashed determinants and remain constant across cache hits.
  - Arg ordering between create/override matches for updated indices.

## Status

Reviewed for program cache correctness; no issues found.
