# HAL tests

Architecture-specific HAL tests live below this directory. Each architecture
owns a local LIT configuration for its target compiler, flags, and binary
inspection tools.

- `<architecture>/codegen/` contains compile-only assembly and disassembly
  checks.
- `<architecture>/diagnostics/` contains target-specific compile-time API and
  negative-compilation checks.
- `<architecture>/lit.local.cfg` defines substitutions shared by that
  architecture's tests.

Blackhole CFG runtime tests and MMIO compile-time tests place each HAL function
next to a `reference_` function using explicit `TT_*`/`TTI_*` instructions or MMIO
operations. Read the two C++ bodies to see the expected behavior.

Tests write `objdump -drz` output to a text file, then pass it to
`%{blackhole_compare_codegen}`. The script reads one adjacent function/reference
pair at a time, in either order, and compares from each section start through its
first `ret`, including instruction bytes and operands.
Local branch destinations are already encoded in these instructions, so their
`R_RISCV_BRANCH` records are ignored. Other relocation types, targets, and addends
are compared.

Runtime tests check that HAL calls inline to straight-line `TT_*`/`TTI_*`
instructions, so keep each test wrapper branch-free with a single `ret`.
Instructions after the first `ret` (for example, an early-return cold path) are
not compared. If a future test needs branches or another function shape, extend
the comparison helper when adding that test.

The comparison checks that the public API inlines and folds to the same code as
the explicit operations, including the absence of extra calls, instructions, or
memory accesses. Reference operations assume inputs satisfy the API's documented
constraints. Separate assertion and diagnostic checks cover invalid inputs;
these compile-only tests do not execute on hardware.
