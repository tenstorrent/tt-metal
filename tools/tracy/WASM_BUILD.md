# Tracy profiler WebAssembly (WASM) build

This repository builds the **Tracy profiler web viewer** (Emscripten) as part of the normal CMake graph. The native Tracy desktop GUI is separate; WASM is the browser-based UI used by `python -m tracy` and `tools/tracy/serve_wasm.py`.

## What gets built

- **CMake target:** `tracy_profiler_wasm` (defined in `tt_metal/third_party/CMakeLists.txt`).
- **Upstream project:** `tt_metal/third_party/tracy/profiler`, configured with the Emscripten toolchain.
- **Output directory:** `${CMAKE_BINARY_DIR}/profiler/build_wasm`.

With the usual out-of-tree build folder `build/` at the repo root, artifacts live under:

`build/profiler/build_wasm/`

Typical files there include:

- `tracy-profiler.js` / `tracy-profiler.wasm` — Emscripten output (executable target name `tracy-profiler`).
- `index.html`, `httpd.py`, `favicon.svg` — copied from `tt_metal/third_party/tracy/profiler/wasm/` by the profiler `CMakeLists.txt` when `EMSCRIPTEN` is set.

Python tooling resolves the same path via `PROFILER_WASM_DIR` in `tools/tracy/common.py` (`TT_METAL_HOME/build/profiler/build_wasm` by default).

## emsdk via CPM

Emscripten is fetched through CPM in `tt_metal/third_party/CMakeLists.txt` (upstream `emscripten-core/emsdk`, pinned via `GIT_TAG`). The CPM-fetched tree contains only the installer/manager scripts (~250 KB) — no submodules, no manual `git clone`. The actual toolchain (`emcc`, `Emscripten.cmake`, node, binaryen, LLVM, …) is materialized alongside the scripts on first CMake configure, so neither the source tree nor the shared CPM cache is polluted by the install.

`CPM_SOURCE_CACHE` is shadowed (set to empty) for the single `CPMAddPackage(emsdk ...)` call so CPM falls back to its default `${CMAKE_BINARY_DIR}/_deps/<name>-src/` destination. emsdk uses `os.path.realpath(__file__)` to find its install root, so the package must live somewhere we can write hundreds of MB into; the build dir is the right place, the shared content-addressed cache is not.

On the first `cmake` configure, `tt_metal/third_party/CMakeLists.txt`:

1. Fetches the emsdk scripts via `CPMAddPackage(... DOWNLOAD_ONLY YES)` into `${CMAKE_BINARY_DIR}/_deps/emsdk-src/`.
2. Runs `./emsdk install latest && ./emsdk activate latest` in that directory, populating `upstream/`, `node/`, `downloads/`, and `.emscripten` next to the scripts.

The configure-baked toolchain path is:

`${CMAKE_BINARY_DIR}/_deps/emsdk-src/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`

`latest` is resolved against `emscripten-releases-tags.json` shipped at the pinned `GIT_TAG`, so the toolchain version is reproducible per CMake configuration. To bump the Emscripten version, update the `GIT_TAG` in `tt_metal/third_party/CMakeLists.txt` and force a re-install (delete `${CMAKE_BINARY_DIR}/_deps/emsdk-src/upstream/`, or wipe `${CMAKE_BINARY_DIR}/_deps/emsdk-src/` entirely, then reconfigure).

A clean build re-fetches the emsdk scripts (~1 MB git clone) — CPM caching is intentionally bypassed for this package because the cache hit is small relative to `./emsdk install` (~370 MB) which always runs from scratch in the build tree anyway.

## Build

After a successful configure, build everything (the WASM viewer is part of the default `ALL` build):

```bash
cmake --build build
```

Or build only the WASM viewer:

```bash
cmake --build build --target tracy_profiler_wasm
```

Replace `build` with your `CMAKE_BINARY_DIR` if it differs.

## Disabling Tracy profiling

`-DENABLE_TRACY=OFF` (or `build_metal.sh --disable-tracy`) turns off Tracy **instrumentation** in Metalium. The Tracy CLI tools and WASM viewer are still built so the profiler artifact is consistent across configurations.

## Serving the built files locally

For HTTP ports, COOP/COEP headers, and trace handling, see `tools/tracy/serve_wasm.py`. High-level usage is described in `docs/source/tt-metalium/tools/tracy_profiler.rst` under "Web GUI (WASM)".

## Upstream reference workflow

The vendored Tracy tree includes `tt_metal/third_party/tracy/.github/workflows/emscripten.yml` (MinSizeRel, optional compression of `.js`/`.wasm`). Metalium's integration uses the CMake ExternalProject described here instead of that workflow file.
