# tt-llk host unit-test scaffold

This directory contains the reusable LLVM LIT harness derived from
`llk/san/state-refactor`. It intentionally contains no test cases yet.

## Layout

- `lit/diagnostics/` is for compile-only Clang `-verify` coverage.
- `lit/functional/` is for tests that compile and execute a host binary.
- `lit.cfg.py` discovers `.cpp` files below `lit/` and provides common tool and
  source-root substitutions.

## Run locally

Install the Python `lit` package, Clang, and the LLVM tools package containing
`split-file`, then run from the tt-llk root:

```bash
python3 -m lit -sv tools/tests/unit
```

Use `CXX` to select Clang and `LLVM_BIN` to locate `split-file` when they are
not on `PATH`:

```bash
CXX=/path/to/clang++ LLVM_BIN=/usr/lib/llvm-20/bin \
    python3 -m lit -sv tools/tests/unit
```

The harness defines `%clangxx`, `%split-file`, `%{tt_llk_root}`,
`%{blackhole_root}`, `%{blackhole_common_include}`, and
`%{blackhole_llk_include}` for test `RUN:` lines.

CI wiring should be added with the first concrete test so an empty suite is
not reported as a CI failure.
