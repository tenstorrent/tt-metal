# Unit tests

Run from the LLK root:

```sh
CXX=clang++ LLVM_BIN=/usr/lib/llvm-20/bin lit -v tools/tests/unit
```

No device is required. LLVM supplies `split-file` and `FileCheck`; host diagnostic
tests require Clang. Tests requiring libfmt skip when it is unavailable.

The Blackhole CFG tests under `lit/` include:

- `functional/test_cfg_write_plan.cpp`: host checks for grouping and overlap.
- `diagnostics/test_cfg_access_bounds.cpp`: array, MMIO read, and GPR bank bounds.
- `diagnostics/test_cfg_write_overlap.cpp`: field and GPR destination overlap,
  including all four words of a 128-bit transfer in both operand orders.
- `codegen/test_cfg_access.cpp`: public calls compared with handwritten hardware
  instructions and MMIO references at `-O3` for TRISC0 and TRISC2. The comparison
  ignores comments and local-label numbering; other assembly differences fail.
  A FileCheck case checks the merged MMIO mask, data, and single store directly.

CFG tests require hardware headers at `../hw/inc/internal/tt-1xx/blackhole`.
Public API tests additionally need the SFPI headers in `tests/sfpi/include` and
`tests/sfpi/compiler/bin/riscv-tt-elf-g++`. Set `SFPI_CXX` to use another SFPI
compiler. These tests are reported as unsupported when dependencies are absent.
