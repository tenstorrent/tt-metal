# ProgramSpec validation

This folder implements `ValidateProgramSpec`.
This function peforms invariant validation for `ProgramSpec`.

The invariant tests are listed under:
`tests/tt_metal/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/`.

## Folder organization

This folder is organized by domain of validation.

| Directory / file | What it checks |
|---|---|
| `resource/` | Validate resources (e.g. dfb, scratchpad) associated with a `ProgramSpec`, e.g. binding |
| `placement/` | Validate placements of kernels and resources |
| `kernel_spec.cpp`, `hardware_config.cpp` | Validate properties of `KernelSpec` that are not listed above |
| `program_spec.cpp` | Validate properties of `ProgramSpec` that are not listed above |
