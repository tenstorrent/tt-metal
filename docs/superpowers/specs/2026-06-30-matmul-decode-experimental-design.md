# Move Matmul Decode to Experimental

## Goal

Relocate the `matmul_decode` operation from `ttnn/cpp/ttnn/operations/matmul_decode/` to
`ttnn/cpp/ttnn/operations/experimental/matmul_decode/`, and move its unit test from
`tests/ttnn/unit_tests/operations/matmul/test_matmul_decode.py` to
`tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py`. This brings
`matmul_decode` in line with `attn_matmul`, `group_attn_matmul`, and the other
"experimental" matmul variants.

## Current State

- Public API: `ttnn::matmul_decode(...)` lives in namespace `ttnn`; Python exposure is
  `ttnn.matmul_decode(...)`.
- C++ sources (16 files) under `ttnn/cpp/ttnn/operations/matmul_decode/`.
- 1 unit test: `tests/ttnn/unit_tests/operations/matmul/test_matmul_decode.py`.
- CMake target: `TTNN::Ops::MatmulDecode` (not experimental), with a single
  `add_subdirectory(cpp/ttnn/operations/matmul_decode)` in `ttnn/CMakeLists.txt`.
- Python binding registered directly in `ttnn/cpp/ttnn-nanobind/__init__.cpp`.

## Target State

- Public API: `ttnn::experimental::matmul_decode(...)`; Python exposure
  `ttnn.experimental.matmul_decode(...)`.
- C++ sources under `ttnn/cpp/ttnn/operations/experimental/matmul_decode/`,
  paralleling the existing `attn_matmul/` layout.
- 1 unit test at
  `tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py`.
- CMake target: `TTNN::Ops::Experimental::MatmulDecode` (replaces
  `TTNN::Ops::MatmulDecode`).
- Python binding registered in
  `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` via the
  `py_module(...)` entry point (i.e. under `ttnn.operations.experimental`).

## Detailed Plan

### 1. New source layout

```
ttnn/cpp/ttnn/operations/experimental/matmul_decode/
├── CMakeLists.txt
├── sources.cmake
├── matmul_decode.hpp                    # public header (ns: ttnn::experimental)
├── matmul_decode.cpp                    # wrapper -> ttnn::prim::matmul_decode
├── matmul_decode_nanobind.hpp           # bind_matmul_decode_operation decl
├── matmul_decode_nanobind.cpp           # bind_function<"matmul_decode", "ttnn.experimental.">
└── device/
    ├── matmul_decode_device_operation.hpp
    ├── matmul_decode_device_operation.cpp
    ├── full_width_sharded_program_factory.cpp
    ├── partial_width_sharded_program_factory.cpp
    ├── multi_core_program_factory.cpp
    └── kernels/
        ├── dataflow/
        │   ├── reader_full_width_sharded.cpp
        │   ├── reader_partial_width_sharded.cpp
        │   └── writer_partial_width_sharded.cpp
        └── compute/
            ├── compute_full_width_sharded.cpp
            └── compute_partial_width_sharded.cpp
```

The kernel files move with the operation; their `#include` paths and
`FILE_PATH` references in the program factories will be updated to point to
the new location.

### 2. Namespace changes

| Current namespace | New namespace |
|---|---|
| `ttnn::matmul_decode` (public API) | `ttnn::experimental::matmul_decode` |
| `ttnn::operations::matmul_decode::...` (impl) | `ttnn::operations::experimental::matmul_decode::...` |
| `ttnn::prim::matmul_decode` (device-op launch) | `ttnn::prim::matmul_decode` (unchanged) |

This matches `attn_matmul` (`ttnn::experimental::attn_matmul` +
`ttnn::prim::attn_matmul`) and the existing experimental convention.

### 3. CMake target

New `ttnn/cpp/ttnn/operations/experimental/matmul_decode/CMakeLists.txt` mirrors
`experimental/matmul/CMakeLists.txt` and `experimental/bcast_to/CMakeLists.txt`:

- Library name: `ttnn_op_experimental_matmul_decode`
- Alias: `TTNN::Ops::Experimental::MatmulDecode`
- Globs `device/kernels/*` and adds them to a `FILE_SET kernels` (the current
  `matmul_decode/CMakeLists.txt` skips this; the experimental pattern does
  include it).
- Installs both `api` and `kernels` file sets.

The new `sources.cmake` follows the `attn_matmul` pattern, listing `API_HEADERS`
and `SRCS` separately.

### 4. Build-system updates

- `ttnn/CMakeLists.txt`:
  - Remove `TTNN::Ops::MatmulDecode` from the regular link block
    (line 310).
  - Add `TTNN::Ops::Experimental::MatmulDecode` to the experimental link
    block (alphabetical position, around `TTNN::Ops::Experimental::Matmul`).
  - Replace `add_subdirectory(cpp/ttnn/operations/matmul_decode)` with
    `add_subdirectory(cpp/ttnn/operations/experimental/matmul_decode)`.
- `ttnn/sources.cmake`:
  - Remove line 162
    (`cpp/ttnn/operations/matmul_decode/matmul_decode_nanobind.cpp`).
  - Add the new path
    (`cpp/ttnn/operations/experimental/matmul_decode/matmul_decode_nanobind.cpp`)
    to the experimental group.

### 5. Python binding relocation

- `ttnn/cpp/ttnn-nanobind/__init__.cpp`:
  - Remove `#include "ttnn/operations/matmul_decode/matmul_decode_nanobind.hpp"`
    (line 65).
  - Remove the `def_submodule` + `bind_matmul_decode_operation` block
    (lines 155-156).
- `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp`:
  - Add `#include "ttnn/operations/experimental/matmul_decode/matmul_decode_nanobind.hpp"`.
  - In `py_module(...)`, add
    `matmul_decode::detail::bind_matmul_decode_operation(mod);` (alphabetical
    position, near the other matmul binds).
- `matmul_decode_nanobind.cpp`:
  - Switch to `ttnn::bind_function<"matmul_decode", "ttnn.experimental.">(...)`.
  - Bind `&ttnn::experimental::matmul_decode` instead of
    `&ttnn::matmul_decode`.
  - Rename namespace to `ttnn::operations::experimental::matmul_decode::detail`
    (matching `attn_matmul`).

### 6. Test file

- Move `tests/ttnn/unit_tests/operations/matmul/test_matmul_decode.py` →
  `tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py`.
- Update both `ttnn.matmul_decode(...)` calls in that file to
  `ttnn.experimental.matmul_decode(...)`.
- No other tests reference `matmul_decode`.

### 7. Cleanup

- Delete `ttnn/cpp/ttnn/operations/matmul_decode/` after the new tree is
  in place and the build is verified.

## Why this layout

- **Direct child of `experimental/`**: `matmul_decode` is a self-contained op
  with its own kernels and program factories, not a sibling of
  `attn_matmul`/`group_attn_matmul`. Mirroring the leaf-op pattern
  (`experimental/bcast_to/`, `experimental/minimal_matmul/`,
  `experimental/conv3d/`) is cleaner than nesting it under
  `experimental/matmul/matmul_decode/`.
- **`ttnn::experimental::` C++ namespace** + **`ttnn.experimental.` Python
  prefix**: matches the existing experimental ops and the `attn_matmul`
  precedent in particular.
- **Kernel file-set install**: brings the op in line with every other
  experimental op (its current CMake target silently omits kernel
  installation, which would break runtime kernel lookup after the move).

## Out of scope

- The `multi_core_program_factory.cpp` skeleton (marked TEMPLATE in
  source comments) is moved verbatim; no code changes.
- No changes to other ops, no namespace cleanups elsewhere, no test
  parameter changes.

## Verification

1. Configure the existing build tree:
   `cmake --build build_Release --target ttnn -j`
2. Confirm the new `ttnn_op_experimental_matmul_decode` target compiles and
   `ttnncpp` links cleanly.
3. Confirm no references to `ttnn::matmul_decode`,
   `ttnn/operations/matmul_decode`, or `ttnn.matmul_decode` remain
   (`rg "matmul_decode"` excluding the experimental subtree and tests).
4. Run the relocated test (requires a device):
   `pytest tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py -k`
   to at least confirm the test is discoverable and the import works.
