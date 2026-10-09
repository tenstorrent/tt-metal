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

Blackhole CFG runtime tests place each HAL function next to a `reference_`
function using explicit `TT_*`/`TTI_*` instructions or MMIO operations. Read the
two C++ bodies to see the expected behavior.

`%{blackhole_compare_codegen}` compares the disassembled function pairs,
including instruction bytes and operands and relocation types, targets, and
addends.
