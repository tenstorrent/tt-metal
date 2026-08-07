# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

tt-metal hosts two stacked products for Tenstorrent accelerators (Grayskull, Wormhole, Blackhole):

- **TT-Metalium** (`tt_metal/`): the low-level programming model — device runtime, command-queue dispatch, allocators, JIT kernel build, and firmware. Kernels run on the RISC-V cores of each Tensix core (data-movement RISCs + 3 compute TRISCs driving the FPU/matrix and SFPU/vector engines).
- **TT-NN** (`ttnn/`): the high-level neural-network op library built on Metalium, with a C++ core and Python bindings (nanobind). This is what models are written against.

`tt-train/` is a training framework on top of ttnn; `models/` holds model implementations and demos; `tt_stl/` is a standalone support library (containers, type utilities) consumed by both layers.

`tt_metal/tt-llk/` is a vendored subtree with its own conventions and its own `.claude/CLAUDE.md`; read that file when working there.

## Build

```bash
./build_metal.sh                 # Release build (default)
./build_metal.sh --debug         # Debug build with symbols (CONFIG=Debug also works)
./build_metal.sh --development   # RelWithDebInfo
./build_metal.sh --build-tests   # build all C++ test executables
./build_metal.sh --build-tt-train
./build_metal.sh --enable-ccache # speed up rebuilds
./build_metal.sh --clean         # remove all build_* workspaces and caches
```

The build directory `build/` is a symlink to `build_<Config>/` (e.g. `build_Release/`). `ninja install` (run automatically by the script) is required before the Python environment works. Manual CMake builds use `-G Ninja` and the presets in `CMakePresets.json`.

Tracy profiler is enabled by default; pass `--disable-profiler` to turn it off. Other notable flags: `--without-python-bindings` (build `ttnncpp` standalone), `--build-programming-examples`, `--enable-fake-kernels-target` (generates `compile_commands.json` for kernels so IDEs resolve them).

### `install` is not optional

If you invoke CMake directly rather than using `build_metal.sh`, **you must build the `install` target**:

```bash
cmake --build build --target install
```

Building only a library target (e.g. `--target ttnn`) is *not* enough. Python imports `ttnn/ttnn/_ttnn.so`, and only the `install` target writes it; a plain library build leaves the new artifacts unused in `build_<Config>/ttnn/`.

This matters more than a normal stale-build problem because **compute kernels are JIT-compiled from source at runtime while program factories live in the compiled library**. Skipping the install step therefore runs *new kernels against old host code*. If the kernel change added a circular buffer, the kernel waits forever on a CB the old factory never created: the device hangs with no error message, and the hang leaves the board wedged so the *next* run fails at device open with `failed to initialize FW` — an error that points nowhere near the cause.

Before trusting any hardware result, confirm the installed library is newer than your edit:

```bash
stat -c '%y %n' ttnn/ttnn/_ttnn.so
```

Edits confined to `device/kernels/**` need no rebuild at all, since those are JIT-compiled.

If `install` fails complaining that a kernel source file does not exist, the build directory's install manifest is stale relative to your checkout (this happens after a file is renamed upstream). Re-run `cmake -S . -B build` to regenerate it.

## Python environment

```bash
./create_venv.sh                 # creates ./python_env (or $PYTHON_ENV_DIR)
source python_env/bin/activate
```

Two environment variables matter for almost everything:

- `TT_METAL_HOME` must point at the repo root.
- `PYTHONPATH` must include `$TT_METAL_HOME` (`export PYTHONPATH=$TT_METAL_HOME`), or Python op imports fail.

## Running tests

Prefer the narrowest command that exercises your change; the full suites are slow.

**C++ (Googletest, the current standard).** Tests are bundled into per-area executables under `build/test/`:

```bash
./build_metal.sh --build-tests
./build/test/tt_metal/unit_tests_api --gtest_filter="MeshDispatchFixture.TensixDRAMLoopbackSingleCore"
```

The CMake option behind `--build-tests` is `TT_METAL_BUILD_TESTS`. Tests are split by dispatch mode. For slow-dispatch tests, set `export TT_METAL_SLOW_DISPATCH_MODE=1` (otherwise fast dispatch is used). There is also a legacy suite of standalone C++ integration binaries under `${TT_METAL_HOME}/build/test/tt_metal`.

**Python (pytest).** A single test:

```bash
pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py::test_name
```

Post-commit regression entrypoints (run these before pushing — failures count as regressions):

```bash
./tests/scripts/run_python_api_unit_tests.sh
./tests/scripts/run_cpp_unit_tests.sh
```

Pytest markers (see `pytest.ini`) select hardware/perf suites, e.g. `pytest models/ -m models_performance_bare_metal`, `-m post_commit`, `-m model_perf_t3000`. Tests carry hardware requirements via markers like `requires_grid_size`.

## On-device debugging

Device code runs on hardware, so the debug tooling is environment-variable driven:

- `TT_METAL_WATCHER=10` — enable Watcher (updates every 10s). Validates NoC transactions and on-device asserts, and on a hang writes `generated/watcher/watcher.log` with per-core waypoints and `k_ids` mapping cores to the kernel source files they were running. Develop with it on; disable for perf. `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1` reduces its timing impact when a hang only reproduces with Watcher off.
- `TT_METAL_DPRINT_CORES=(0,0)-(4,4)` plus `#include "api/debug/dprint.h"` and `DPRINT("x = {}\n", x);` — print from kernels.
- `TT_LOGGER_LEVEL=Debug` — host-side debug logging.
- Host C++ / pybind debugging uses gdb on a Debug build (`gdb --args python <file>`).

After any kernel hang the board needs `tt-smi -r` before the next run will get a device.

### Reading Watcher waypoints for a compute-kernel hang

**A compute thread blocked in `cb_reserve_back` / `cb_wait_front` shows up as `K`, not `CRBW` or `CWFW`.** Those two waypoints are defined only in `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, which is the BRISC/NCRISC path; the compute-thread equivalents in `tt_metal/hw/inc/api/compute/cb_api.h` set no waypoint at all. Do not conclude "this is not a circular-buffer deadlock" from the absence of `CRBW` on TRISC0/1/2 — that inference is wrong, and it sends debugging toward the unpack/math handshake instead of the CB.

To localise a hang inside a compute kernel, add prints and re-run; kernels are JIT-compiled, so this costs no C++ rebuild:

```cpp
#include "api/debug/dprint.h"
DPRINT_PACK("reached step {}\n", step);   // also DPRINT_UNPACK / DPRINT_MATH
```

The API is printf-style; the older `DPRINT << ... << ENDL()` form is deprecated and fails to compile when it references loop variables, because the macro wraps its argument in a lambda that does not capture them.

## How a TT-NN operation is structured

This is the dominant pattern in `ttnn/cpp/ttnn/operations/<category>/<op>/` and the thing you'll most often extend. See `ttnn/cpp/ttnn/operations/examples/example/` for the canonical minimal op and `ttnn/cursor/DEVICE_OPERATION_MIGRATION_GUIDE.md` for the current device-op API.

- `<op>.hpp` / `<op>.cpp` — the user-facing op struct (the `ttnn::` entrypoint).
- `<op>_nanobind.cpp` — Python binding. Registered up the chain via the category's `*_nanobind.cpp`.
- `device/<op>_device_operation.{hpp,cpp}` — the device operation: validation, output-spec/shape inference, and program-factory selection.
- `device/*_program_factory.cpp` — builds the actual `Program` (circular buffers, kernel handles, runtime args). Ops commonly have `single_core` and `multi_core` factories chosen at runtime.
- `device/kernels/{compute,dataflow}/*.cpp` — the RISC-V kernels (compute kernels use the FPU/SFPU; dataflow kernels are the reader/writer NoC movers).

CMake wiring: each op dir has a `sources.cmake` / `CMakeLists.txt`; new source files must be added there to be compiled.

## nanobind bindings (critical pitfalls)

Bindings use **nanobind**, not pybind11, and the differences bite. Full rules in `contributing/Nanobind.md` and `.github/instructions/nanobind.instructions.md`. The ones that silently break at runtime:

- Bound signature (types, order, defaults) must exactly match the C++ declaration.
- Optional params defaulting to `None` must use `nb::arg("name") = nb::none()` — `= std::nullopt` silently fails to accept `None`.
- Returning a pointer/reference needs an explicit return-value policy (`nb::rv_policy::reference_internal`, `keep_alive`, etc.) or you get a dangling reference.
- Do **not** put holder types in the class template: `nb::class_<T>`, not `nb::class_<T, std::shared_ptr<T>>`.
- Returning `std::unique_ptr` requires `nbh::steal_rewrap_unique` (from `ttnn-nanobind/nanobind_helpers.hpp`).
- `std::reference_wrapper` is unsupported — use `std::optional<T*>` + `nbh::rewrap_optional`.
- Include the matching `<nanobind/stl/...>` header for any STL container in a signature, or you get runtime `TypeError`.
- Custom constructors need placement new: `.def("__init__", [](T* self, ...){ new (self) T(...); })`.
- Use `mod`/`m` as the module variable, not `module` (C++20 keyword).

## Conventions and review standards

The repo's PR-review expectations (`.github/copilot-instructions.md`) are good guidance for any change:

- Names must reflect actual behavior; a narrowed/widened implementation must update its symbol name.
- Hoist complex `if` conditions into a descriptively named `bool`.
- No magic numbers — derive from a named constant or comment the derivation.
- Flag/avoid duplicated logic; a constant should have one canonical home.
- Treated as merge-blockers: correctness/race bugs; ABI breaks (struct layout change or symbol removed from a public header in `tt_metal/api/` or `ttnn/api/ttnn/`); missing NoC/L1 bounds checks or broken barrier/semaphore ordering in kernels; hardcoded secrets.
- New public API needs a unit test in the nearest `tests/` target; bug fixes need a regression test.

Formatting is enforced by pre-commit (`.pre-commit-config.yaml`): clang-format for C++, clang-tidy (`.clang-tidy`), gersemi for CMake. C++ standard is C++20. Docs-only PRs should be prefixed `[skip ci]` in the title.

### When pre-commit rewrites files you did not touch

Pre-commit is configured by the **repository-root** `.pre-commit-config.yaml`, whose LLK hooks are scoped with `files: ^tt_metal/tt-llk/`. If a commit starts rewriting `uint32_t` to `std::uint32_t` across whole files, or `codespell` reports pre-existing words in files outside `tt_metal/tt-llk/`, your git hook is installed against the subtree config instead — that config needs no path filters, because inside tt-llk's own repo every file is LLK. Check and repair:

```bash
grep config .git/hooks/pre-commit     # expect --config=.pre-commit-config.yaml
pre-commit install --overwrite
```

Prefer fixing that over `--no-verify`, which also skips clang-format and the metalium validators. To check a commit already made that way:

```bash
pre-commit run --config .pre-commit-config.yaml --from-ref origin/main --to-ref <branch>
```

## Key references in-repo

- `METALIUM_GUIDE.md` — architecture deep dive: Tensix cores, dataflow vs compute kernels, the FPU/SFPU engines, dispatch, SPMD.
- `CONTRIBUTING.md` — full test/debug/git workflow.
- `tech_reports/` — design docs (allocator, data formats, flash attention, fabric/scale-out, sub-devices, mesh-of-devices).
- `.cursor/rules/*.mdc` — task-specific guides for the sweep-test / model-trace framework (validating and matching `config_hash` between model traces and sweep traces under `model_tracer/` and `tests/sweep_framework/`).

## Maintaining this file

Keep it short. Everything here is loaded into an agent's context at the start of every session, so
each line competes with the code actually being worked on. Content earns its place by being
non-obvious **and** quiet in failure — things that produce a hang, a wrong number, or a silently
skipped step rather than an error message. Do not restate `CONTRIBUTING.md`, `METALIUM_GUIDE.md` or
the architecture docs; link to them instead.

Prune as readily as you add. A stale instruction is worse than a missing one, because agents follow
it confidently: an entry naming a file, flag or command that no longer exists will send the next
session down a path that cannot work. If you find one, correct or delete it as part of the change
that revealed it.

The best source of new entries is a session that just lost time. When something non-obvious costs
an hour, the fix is not only the code — it is the sentence that saves the next agent that hour.

Subdirectories may add their own `CLAUDE.md` for guidance that only applies locally (see
`ttnn/cpp/ttnn/operations/` and `tt_metal/tt-llk/.claude/`). Those load in addition to this file,
so keep them to content specific to their area rather than repeating anything above.
