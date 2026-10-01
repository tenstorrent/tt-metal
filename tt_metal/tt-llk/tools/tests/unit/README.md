# Unit tests

Run from the LLK root:

```sh
CXX=clang++ LLVM_BIN=/usr/lib/llvm-20/bin lit -v tools/tests/unit
```

No device is required. LLVM supplies `split-file` and `FileCheck`; host diagnostic
tests require Clang. Tests requiring libfmt skip when it is unavailable.

The Blackhole CFG tests under `lit/` include:

- `functional/test_cfg_write_plan.cpp`: host checks for grouping and overlap.
- `diagnostics/test_cfg_access_bounds.cpp`: array, MMIO read, and GPR bank bounds,
  plus section-dependent spans of wide fields and groups without a `Raw` anchor.
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

## Additional HAL coverage


- `lit/diagnostics/` is for compile-only Clang and SFPI diagnostics.
- `lit/functional/` is for tests that compile and execute a host binary.
- `lit/hal/<architecture>/codegen/` is for target assembly and disassembly
  checks. Architecture-local `lit.local.cfg` files provide the target compiler,
  flags, and binary-tool substitutions.
- `lit/hal/<architecture>/diagnostics/` is for target-specific constexpr API
  validation and negative-compilation checks.
- `lit.cfg.py` discovers `.cpp` files below `lit/` and provides common tool and
  source-root substitutions.

## Test environment setup


Install the test requirements, Clang, and the LLVM tools package containing
`split-file`. Target HAL tests also use the SFPI compiler installed by the
normal tt-llk test-environment setup:

```bash
python3 -m pip install -r tests/requirements.txt
./tests/setup_testing_env.sh
```

Then run from the tt-llk root:

```bash
tests/.venv/bin/lit -sv tools/tests/unit
```

Use `CXX` to select Clang and `LLVM_BIN` to locate `split-file` when they are
not on `PATH`:

```bash
CXX=/path/to/clang++ LLVM_BIN=/usr/lib/llvm-20/bin \
    tests/.venv/bin/lit -sv tools/tests/unit
```

The harness defines `%clangxx`, `%split-file`, `%{tt_llk_root}`,
`%{blackhole_root}`, `%{blackhole_common_include}`, and
`%{blackhole_llk_include}` for test `RUN:` lines.

Architecture-local configurations define their own target compiler and binary
tool substitutions.
