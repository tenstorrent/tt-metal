# Program Construction folder

There are 3 stages to transform a `ProgramSpec` to a `Program`.

1. Structural lookup table build-up.
2. Validation of `ProgramSpec`
3. **Construct the actual Program** <- you're here.

## Program construction overview

Program construction has 3 big steps.

1. Processor assignment
2. Register resource with the `Program`.
3. Construct `Kernel`s to put into the `Program`.

## Folder structure

This folder tries to echo the 3 big steps of program construction.

| Directory / file | What it does |
|---|---|
| `construct_program.cpp` | Driver function |
| `processor_assignment/` | Performs RISC-V core allocation (step 1) |
| `resource/` | Resource oriented organization, echoing `../validation/resource`. Incl. step 2 and 3 |
| `kernel_lowering.cpp` | Lowers `KernelSpec` into a `Kernel`, handles CTA/CRTA, collects bindings. Step 3 |
