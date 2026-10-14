# Unit tests

No device is required. LLVM supplies `split-file` and `FileCheck`; host diagnostic
tests require Clang. Tests requiring libfmt skip when it is unavailable.

The HAL tests under `lit/hal/` call the public API and check only its observable
results: the emitted instructions and MMIO accesses, or the compile-time
diagnostic. They need the hardware headers at `../hw/inc/internal/tt-1xx/` and the
SFPI toolchain in `tests/sfpi/compiler/bin`, and are reported as unsupported when
the toolchain is absent.

## Layout

- `lit/diagnostics/` is for compile-only Clang diagnostics.
  Tests can use `-verify` to check expected diagnostics and `%split-file` to keep
  related positive and negative cases in one source file.
- `lit/functional/` is for tests that compile and execute a host binary.
  Each test should use `RUN:` lines to compile its own temporary binary and
  execute it without requiring a device.
- `lit/hal/<architecture>/codegen/` is for target assembly and disassembly
  checks. Architecture-local `lit.local.cfg` files provide the target compiler,
  flags, and binary-tool substitutions.
- `lit/hal/<architecture>/diagnostics/` is for target-specific constexpr API
  validation and negative-compilation checks.
- `lit.cfg.py` discovers `.cpp` files below `lit/` and provides common tool and
  source-root substitutions.

## Setup and execution

Install Clang and the LLVM tools package containing `split-file` and `FileCheck`.
Then, from the tt-llk root, create and activate the test environment, install its
requirements, and set up SFPI:

```bash
python3 -m venv tests/.venv
source tests/.venv/bin/activate
python -m pip install -r tests/requirements.txt
./tests/setup_testing_env.sh
```

Run the suite from the same directory. Set `CXX` and `LLVM_BIN` to the installed
Clang and LLVM tools; these overrides can be omitted when the tools are on `PATH`:

```bash
CXX=clang++ LLVM_BIN=/usr/lib/llvm-20/bin \
    tests/.venv/bin/lit -sv tools/tests/unit
```

## Test commands

Host tests use `%clangxx`, `%split-file`, and `%{sanitizer_include}`. Blackhole
HAL tests use `%{blackhole_tensix_compile}`, `%{blackhole_objdump}`, and
`%{blackhole_compare_codegen}`, with `%{blackhole_unpack_thread}`,
`%{blackhole_math_thread}`, or `%{blackhole_pack_thread}` selecting the thread.
Blackhole diagnostics also provide `%{blackhole_tensix_diagnose}` for syntax-only
compilation. File-local `DEFINE` aliases can combine these substitutions.

The root configuration supplies source paths to the architecture-local
configurations, which build the target compiler and binary-tool commands.
