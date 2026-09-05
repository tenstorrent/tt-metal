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

Code-generation tests disassemble their target objects and use LLVM FileCheck
directives to describe the expected instruction sequence.
