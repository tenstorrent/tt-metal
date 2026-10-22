# ProgramSpec validation folder

There are 3 stages to transform a `ProgramSpec` into a `Program`.

1. Structural lookup table build-up.
2. **Validation of `ProgramSpec`** <- you're here.
3. Construct the actual `Program`.

## Folder structure

| Directory / file | Validation target |
|---|---|
| `validate_spec.cpp` | Driver function |
| `program_spec.cpp` | Fields on `ProgramSpec` not covered by other validations |
| `placement/` | Work units, kernel placement, node capacity |
| `kernel_spec.cpp`, `hardware_config.cpp` | Fields on `KernelSpec` that are not resource related |
| `resource/` | Memory Resource & their bindings & relations |
