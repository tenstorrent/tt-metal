# Unit tests

Run from the LLK root:

```sh
CXX=clang++ LLVM_BIN=/usr/lib/llvm-20/bin lit -v tools/tests/unit
```

No device is required. LLVM supplies `split-file` and `FileCheck`; host diagnostic
tests require Clang. Tests requiring libfmt skip when it is unavailable.

The HAL tests under `lit/hal/` call the public API and check only its observable
results: the emitted instructions and MMIO accesses, or the compile-time
diagnostic. They need the hardware headers at `../hw/inc/internal/tt-1xx/` and the
SFPI toolchain in `tests/sfpi/compiler/bin`, and are reported as unsupported when
the toolchain is absent.

## Additional HAL coverage


- `lit/diagnostics/` is for compile-only Clang diagnostics.
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
