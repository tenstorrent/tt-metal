# ProgramSpec to Program

This folder implements the `Program(MeshDevice&, const ProgramSpec&)` constructor and the `MeshWorkload` constructors that take ProgramSpecs.
These constructors validate a `ProgramSpec` and build a `Program` from it.

## Pipeline

`BuildProgramFromSpec` (`program_spec.cpp`) runs 3 stages, in order:

1. **Collection**: index the `ProgramSpec` and generate lookup table for later stages.
2. **Validation**: performs validations for `ProgramSpec`.
3. **Construction**: builds the `Program`.

Each stage reads the output of the stages before it.

## Folder structure

| Directory / file | Overview |
|---|---|
| `program_spec.cpp` | Driver function |
| `collection/` | Stage 1 |
| `validation/` | Stage 2` |
| `construction/` | Stage 3 |

Tests live under `tests/tt_metal/tt_metal/api/metal2_host_api/`.
