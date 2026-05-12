# Tracy profiler WebAssembly (WASM) build

This repository **optionally** builds the **Tracy profiler web viewer** (Emscripten) as part of the normal CMake graph when `emcmake` can be resolved at configure time. The native Tracy desktop GUI is separate; WASM is the browser-based UI used by `python -m tracy` and `tools/tracy/serve_wasm.py`.

## What gets built

- **CMake target:** `tracy_profiler_wasm` (defined in `cmake/tracy.cmake`), **only if** Emscripten’s `emcmake` is found at configure time.
- **Upstream project:** `tt_metal/third_party/tracy/profiler` with `-DEMSCRIPTEN=ON`.
- **Output directory:** `${CMAKE_BINARY_DIR}/profiler/build_wasm` (when the target exists).

With the usual out-of-tree build folder `build/` at the repo root, artifacts live under:

`build/profiler/build_wasm/`

Typical files there include:

- `tracy-profiler.js` / `tracy-profiler.wasm` — Emscripten output (executable target name `tracy-profiler`).
- `index.html`, `httpd.py`, `favicon.svg` — copied from `tt_metal/third_party/tracy/profiler/wasm/` by the profiler `CMakeLists.txt` when `EMSCRIPTEN` is set.

Python tooling resolves the same path via `PROFILER_WASM_DIR` in `tools/tracy/common.py` (`TT_METAL_HOME/build/profiler/build_wasm` by default).

## How `emcmake` is resolved (configure vs build)

Ninja runs recipe commands in a **minimal environment** (no shell startup files). CI also often **configures** in one step and **builds** in another (for example `./build_metal.sh --configure-only` followed by `cmake --build build --target install`), so `emcmake` is not guaranteed to be on `PATH` when Ninja runs.

For that reason, `cmake/tracy.cmake` resolves **`emcmake` once at CMake configure time** and bakes an **absolute path** into the `tracy_profiler_wasm` rule:

1. **`find_program(emcmake)`** — uses the environment seen by the `cmake` process (for example after `build_metal.sh` has sourced `<build-dir>/emsdk/emsdk_env.sh`).
2. **Unix fallback** — if step 1 fails but `${CMAKE_BINARY_DIR}/emsdk/emsdk_env.sh` exists, CMake runs
   `bash -c '. "<binary-dir>/emsdk/emsdk_env.sh" && command -v emcmake'`
   so an emsdk installed next to the build tree is still discoverable even when `PATH` was never exported for the parent of `cmake`.
3. If `emcmake` still cannot be resolved, CMake prints a **warning** and **does not** define `tracy_profiler_wasm` (configure still succeeds). That is normal for **cibuildwheel / manylinux** and other images that ship Tracy host instrumentation but not Emscripten.
4. When `emcmake` is found, the chosen path is normalized with **`get_filename_component(... REALPATH)`**.

The generated recipe is equivalent to:

```text
<absolute-path-to-emcmake> cmake -DEMSCRIPTEN=ON -B <binary_dir>/profiler/build_wasm -S <repo>/tt_metal/third_party/tracy/profiler
<same-cmake-binary> --build <binary_dir>/profiler/build_wasm
```

The second line uses the same CMake executable as the parent project (`${CMAKE_COMMAND}`), not a bare `cmake` on `PATH`.

## Prerequisites

- To **build** the WASM viewer, **Emscripten** must be resolvable using the rules in [How `emcmake` is resolved](#how-emcmake-is-resolved-configure-vs-build) **during the configure that generates your build tree**. If it is not, configure continues with a warning and the web viewer is omitted.
- **Typical local flow:** run `./build_metal.sh` (with your usual options). It installs or reuses emsdk under `$build_dir/emsdk`, sources `emsdk_env.sh`, then runs `cmake`, so `find_program(emcmake)` succeeds and Ninja later needs **no** `emcmake` on `PATH`.
- **Without `build_metal.sh`:** install [emsdk](https://github.com/emscripten-core/emsdk), put `emcmake` on `PATH` for configure, **or** ensure an activated emsdk lives at `<binary-dir>/emsdk` so the `emsdk_env.sh` fallback can run (Unix hosts only; on Windows use `PATH` at configure time).
- **Nested compile:** the Emscripten toolchain (`em++`, etc.) is still invoked from the inner CMake build; the `emcmake` wrapper is responsible for setting up that environment when the inner configure runs.

## Wheels and minimal CI images (cibuildwheel)

Profiler **wheels** (`CIBW_ENABLE_TRACY=ON`, etc.) run CMake inside manylinux without `./build_metal.sh` and usually **without** an emsdk checkout. In that case configure emits the warning above and **skips** the WASM target; the wheel still contains Tracy **host** support. The browser viewer must be produced from a full dev build that resolved `emcmake`, or from a separate Emscripten build of the Tracy profiler.

## How it is triggered

1. Top-level `CMakeLists.txt` runs `include(tracy)`, which loads `cmake/tracy.cmake`.
2. When `emcmake` was resolved, the custom target `tracy_profiler_wasm` is marked `ALL`, so a default build (`ninja`, `cmake --build …`, or `cmake --build … --target install`) runs it.
3. Emscripten-specific link flags and extra WASM assets are set in `tt_metal/third_party/tracy/profiler/CMakeLists.txt` under `if(EMSCRIPTEN)`.

## `build_metal.sh` and emsdk

`build_metal.sh` installs or reuses **emsdk** under `$build_dir/emsdk` (default `build/emsdk`), runs `./emsdk install latest` / `activate` if needed, then **sources** `$build_dir/emsdk/emsdk_env.sh` before invoking CMake. That makes `emcmake` visible to **`find_program`** during configure.

Because the Ninja rule stores the resolved absolute path, a later **`cmake --build`** in a **new shell** (no `source emsdk_env.sh`) still finds `emcmake` for the Tracy WASM step—this matches CI jobs that configure with `build_metal.sh` and compile in a separate step.

## Rebuild only the WASM viewer

After a successful configure:

```bash
cmake --build build --target tracy_profiler_wasm
```

Replace `build` with your `CMAKE_BINARY_DIR` if it differs. You do **not** need to `source` emsdk before this command for the sake of finding `emcmake`; configure already recorded its path. If configure skipped WASM (warning during CMake), this target does not exist—reconfigure after installing emsdk or using `build_metal.sh`.

## Disabling Tracy profiling

`-DENABLE_TRACY=OFF` (or `build_metal.sh --disable-tracy`) turns off Tracy **instrumentation** in Metalium. It does **not** remove the `tracy_profiler_wasm` logic from `cmake/tracy.cmake`; when `ENABLE_TRACY` is on and `emcmake` was found, the WASM viewer is still part of the default `ALL` build.

## Serving the built files locally

For HTTP ports, COOP/COEP headers, and trace handling, see `tools/tracy/serve_wasm.py`. High-level usage is described in `docs/source/tt-metalium/tools/tracy_profiler.rst` under “Web GUI (WASM)”.

## Upstream reference workflow

The vendored Tracy tree includes `tt_metal/third_party/tracy/.github/workflows/emscripten.yml` (MinSizeRel, optional compression of `.js`/`.wasm`). Metalium’s integration uses the CMake custom target and absolute `emcmake` path described here instead of that workflow file.
