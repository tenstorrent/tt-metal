# Tracy profiler WebAssembly (WASM) build

This repository builds the **Tracy profiler web viewer** (Emscripten) as part of the normal CMake graph. The native Tracy desktop GUI is separate; WASM is the browser-based UI used by `python -m tracy` and `tools/tracy/serve_wasm.py`.

## What gets built

- **CMake target:** `tracy_profiler_wasm` (defined in `cmake/tracy.cmake`).
- **Upstream project:** `tt_metal/third_party/tracy/profiler` with `-DEMSCRIPTEN=ON`.
- **Output directory:** `${CMAKE_BINARY_DIR}/profiler/build_wasm`.

With the usual out-of-tree build folder `build/` at the repo root, artifacts live under:

`build/profiler/build_wasm/`

Typical files there include:

- `tracy-profiler.js` / `tracy-profiler.wasm` — Emscripten output (executable target name `tracy-profiler`).
- `index.html`, `httpd.py`, `favicon.svg` — copied from `tt_metal/third_party/tracy/profiler/wasm/` by the profiler `CMakeLists.txt` when `EMSCRIPTEN` is set.

Python tooling resolves the same path via `PROFILER_WASM_DIR` in `tools/tracy/common.py` (`TT_METAL_HOME/build/profiler/build_wasm` by default).

## Prerequisites

- **Emscripten** on your `PATH`: `emcmake` (and the `em++` toolchain it invokes) must be available when Ninja runs the custom commands for `tracy_profiler_wasm`.

## How it is triggered

1. `cmake/tracy.cmake` is pulled in from the top-level `CMakeLists.txt` via `include(tracy)`.
2. The custom target `tracy_profiler_wasm` is marked `ALL`, so a default build (e.g. `ninja` with no explicit target) runs it after the main configure step.
3. Each build runs roughly:

   ```text
   emcmake cmake -DEMSCRIPTEN=ON -B <binary_dir>/profiler/build_wasm -S <repo>/tt_metal/third_party/tracy/profiler
   cmake --build <binary_dir>/profiler/build_wasm
   ```

   Emscripten-specific link flags and WASM helper assets are set in `tt_metal/third_party/tracy/profiler/CMakeLists.txt` under `if(EMSCRIPTEN)`.

## `build_metal.sh` and emsdk

`build_metal.sh` installs or reuses **emsdk** under `$build_dir/emsdk` (default `build/emsdk`), runs `./emsdk install latest` / `activate` if needed, then **sources** `$build_dir/emsdk/emsdk_env.sh` before invoking CMake. That puts `emcmake` on `PATH` for the Ninja build, including `tracy_profiler_wasm`.

If you configure CMake **without** going through `build_metal.sh`, install [emsdk](https://github.com/emscripten-core/emsdk) yourself and `source emsdk_env.sh` in the same environment where you run `cmake` / `ninja`.

## Rebuild only the WASM viewer

From the repo root, after a successful configure:

```bash
source build/emsdk/emsdk_env.sh   # skip if emcmake is already on PATH
cmake --build build --target tracy_profiler_wasm
```

Adjust `build` if your `CMAKE_BINARY_DIR` differs.

## Disabling Tracy profiling

`-DENABLE_TRACY=OFF` (or `build_metal.sh --disable-tracy`) turns off instrumentation in Metalium; it does **not** remove the `tracy_profiler_wasm` target from `cmake/tracy.cmake`. The WASM profiler UI is still built when the default `ALL` target runs, as long as Emscripten is available.

## Serving the built files locally

For headers and ports used by the web UI, see `tools/tracy/serve_wasm.py`. High-level usage is also described in the Sphinx doc `docs/source/tt-metalium/tools/tracy_profiler.rst` (“Web GUI (WASM)”).

## Upstream reference workflow

The vendored Tracy tree includes `.github/workflows/emscripten.yml`, which documents an alternate CI-style invocation (MinSizeRel, optional compression of `.js`/`.wasm`). The tt-metal integration uses the CMake custom target above instead of that workflow file.
