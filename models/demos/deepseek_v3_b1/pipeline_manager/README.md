# Pipeline Manager

A C++20 library that manages token scheduling between a host inference server and a hardware decode pipeline. It handles prefill queuing, decode slot management, speculative decoding, and user lifecycle — all behind a `PipelineInterface` abstraction.

Two library variants are produced:

| Library | File | Description |
|---|---|---|
| **Core** | `libpipeline_manager_core.so` | Scheduling logic only. Uses `MockPipeline` or any user-supplied `PipelineInterface`. No hardware dependencies. |
| **Full** | `libpipeline_manager.so` | Everything in Core plus `SocketPipeline`, which communicates with Tenstorrent hardware via tt-metalium H2D/D2H sockets. |

## Prerequisites

- CMake >= 3.22
- A C++20 compiler (GCC 11+, Clang 14+)
- pthreads (found automatically by CMake)
- Google Test is fetched automatically during build

For the **Full** build you also need:
- A built tt-metal tree (see [Building with tt-metal](#building-with-tt-metal))

## Directory layout

```
pipeline_manager/
├── include/pipeline_manager/   # Public headers
│   ├── pipeline_manager.hpp        # PipelineManager class
│   ├── pipeline_interface.hpp      # PipelineInterface + MockPipeline
│   ├── pipeline_manager_types.hpp  # Shared types, enums, constants
│   ├── wire_format.hpp             # Host ↔ device serialization
│   └── socket_pipeline.hpp         # SocketPipeline (requires tt-metalium)
├── src/                        # Private implementation
├── tests/                      # GTest binaries
├── tools/                      # pipeline_launcher (device kernel bootstrap)
├── cmake/                      # CMake package config template
├── build.sh                    # Convenience build script
└── CMakeLists.txt
```

## Building standalone (no tt-metal)

```bash
cd models/demos/deepseek_v3_b1/pipeline_manager
./build.sh --standalone
```

**Artifacts** (in `build-standalone/`):
- `libpipeline_manager_core.so`
- `test_pipeline_manager`

## Building with tt-metal

### Step 1 — Build tt-metal

From the tt-metal repository root:

```bash
./build_metal.sh
```

This uses the project's toolchain file (clang-20) and Ninja generator. The build output goes to `build_Release/` (symlinked as `build/`).

> **Note:** Do not run bare `cmake` against tt-metal — it requires a specific toolchain file. Always use `./build_metal.sh`. If you get a generator mismatch error ("Ninja does not match Unix Makefiles"), remove the stale build directory first: `rm -rf build_Release && ./build_metal.sh`.

### Step 2 — Build pipeline_manager

```bash
cd models/demos/deepseek_v3_b1/pipeline_manager
./build.sh --with-metal
```

The script auto-detects paths when run from inside the tt-metal tree. If you have moved the pipeline_manager directory elsewhere, set the environment variables explicitly:

```bash
export TT_METAL_SOURCE_DIR=/path/to/tt-metal
export CMAKE_PREFIX_PATH="${TT_METAL_SOURCE_DIR}/build/lib/cmake;${TT_METAL_SOURCE_DIR}/build_Release/_deps/nlohmann_json-build"
./build.sh --with-metal
```

**Artifacts** (in `build-full/`):
- `libpipeline_manager_core.so`
- `libpipeline_manager.so`
- `test_pipeline_manager`
- `test_device_pipeline`
- `pipeline_launcher`

## Running tests

### Mock tests (no hardware)

After either a standalone or full build:

```bash
# Standalone build:
LD_LIBRARY_PATH=build-standalone build-standalone/test_pipeline_manager

# Full build:
LD_LIBRARY_PATH=build-full build-full/test_pipeline_manager
```

### Device tests (requires Tenstorrent hardware)

Only available after a full build:

```bash
LD_LIBRARY_PATH=build-full build-full/test_device_pipeline
```

### Filtering tests

Both test binaries use Google Test. You can filter:

```bash
LD_LIBRARY_PATH=build-full build-full/test_pipeline_manager --gtest_filter="*Prefill*"
```

## Integrating into an upstream project (e.g. inference server)

There are two integration methods depending on how you want to consume the library.

### Method 1 — CMake `add_subdirectory` (recommended for submodule setups)

If `pipeline_manager` is a Git submodule of your project:

```
inference-server/
├── external/
│   └── pipeline_manager/   ← submodule
├── src/
└── CMakeLists.txt
```

In your top-level `CMakeLists.txt`:

```cmake
# If you also have tt-metal as a submodule and want the Full library:
# add_subdirectory(external/tt-metal)

add_subdirectory(external/pipeline_manager)

add_executable(inference_server src/main.cpp)

# Link the Core library (no hardware, uses MockPipeline or your own PipelineInterface):
target_link_libraries(inference_server PRIVATE PipelineManager::Core)

# OR link the Full library (with SocketPipeline — requires TT::Metalium in scope):
# target_link_libraries(inference_server PRIVATE PipelineManager::Full)
```

When tt-metal is added via `add_subdirectory` before pipeline_manager, the `TT::Metalium` target is already in scope and the Full library builds automatically.

### Method 2 — `find_package` (installed pipeline_manager)

First, install the pipeline_manager:

```bash
cd pipeline_manager
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/pipeline-manager
cmake --build build -j $(nproc)
cmake --install build
```

Then in your project:

```cmake
find_package(pipeline-manager REQUIRED CONFIG)

add_executable(inference_server src/main.cpp)
target_link_libraries(inference_server PRIVATE PipelineManager::Core)
# or PipelineManager::Full if you installed with tt-metal support
```

Configure with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/pipeline-manager
```

### Using the library from C++

```cpp
#include <pipeline_manager/pipeline_manager.hpp>
#include <pipeline_manager/pipeline_interface.hpp>  // MockPipeline

// For the Full library:
// #include <pipeline_manager/socket_pipeline.hpp>

using namespace models::demos::deepseek_v3_b1::pipeline_manager;

int main() {
    MockPipeline mock;
    PipelineManager pm(mock);
    pm.start();

    ISRequest req{};
    req.type = RequestType::PREFILL;
    req.user_id = 0;
    req.prompt_tokens = {100, 200, 300};
    req.max_new_tokens = 64;
    pm.push_request(req);

    pm.tick();  // process queued requests

    OutputMessage out;
    while (pm.try_pop_output(out)) {
        // handle generated tokens
    }

    pm.stop();
}
```

### Compile define: `PM_HAS_METALIUM`

When linking against `PipelineManager::Full`, the preprocessor macro `PM_HAS_METALIUM=1` is defined automatically. Use it for conditional compilation:

```cpp
#ifdef PM_HAS_METALIUM
#include <pipeline_manager/socket_pipeline.hpp>
// use SocketPipeline
#else
#include <pipeline_manager/pipeline_interface.hpp>
// use MockPipeline or custom implementation
#endif
```

## Build options

| CMake variable | Default | Description |
|---|---|---|
| `PM_BUILD_TESTS` | `ON` | Build test binaries |
| `PM_BUILD_TOOLS` | `ON` | Build `pipeline_launcher` (requires tt-metalium) |
| `PM_ENABLE_TSAN` | `OFF` | Enable ThreadSanitizer (`-fsanitize=thread`) |
| `TT_METAL_SOURCE_DIR` | `""` | Path to tt-metal source tree (needed until socket headers are part of the installed tt-metalium package) |

The `TSAN` environment variable is a shorthand when using `build.sh`:

```bash
TSAN=ON ./build.sh --standalone
```
