# Technology Stack

**Analysis Date:** 2026-03-16

## Languages

**Primary:**
- C++ 20 - Core compute framework and device drivers
- Python 3.10+ - High-level API, utilities, and model support
- C - Legacy support for JIT build system

**Secondary:**
- RISC-V assembly - Device firmware and kernel programming
- YAML - Configuration files and test specifications
- Flatbuffers schema (.fbs) - Serialized descriptor generation

## Runtime

**Environment:**
- Linux (x86_64) - Primary development and deployment target
- Requires NUMA support (libnuma library)
- Requires hwloc (hardware locality library)
- OpenMPI 5.0.7 ULFM for distributed computing support

**Package Managers:**
- Python: setuptools/setuptools_scm for wheel building, uv for dependency management
- C++: CMake 3.24+ with CPM (C++ Package Manager)
- Link mode: "copy" for uv (isolation via filesystem copy, not symlinks)

## Frameworks

**Core:**
- CMake 3.24-4.2 - Build system
- Ninja Multi-Config - Primary generator
- scikit-build-core - Python/C++ integration for TTML package
- nanobind v2.10.2 - Python C++ bindings

**Testing:**
- Google Test 1.13.0 (gtest/gmock) - C++ unit tests
- pytest - Python test runner (via pyproject.toml)

**Build/Dev Tools:**
- clang 20 / GCC 12 - Compilation
- clang-tidy 20 - Static analysis
- ccache - Build caching (enabled by default)
- Black 120-char line length - Python formatting
- isort - Python import ordering
- Ruff - Python linting

## Key Dependencies

**Critical Infrastructure:**
- Boost 1.86.0 - Algorithm, container, smart_ptr, interprocess, asio, lockfree libraries
- protobuf 21.12 - Protocol buffer support (without abseil/utf8_range)
- yaml-cpp 0.8.0 (patched) - YAML parsing
- fmt 11.1.4 - String formatting
- nlohmann/json 3.11.3 - JSON parsing
- FlatBuffers 24.3.25 - Binary serialization

**Numeric/ML Support:**
- numpy 1.24.4-<2 - Numeric computation
- xtensor 0.26.0 (patched) - C++ tensor library
- xtensor-blas 0.22.0 (patched) - BLAS operations
- xtl 0.8.0 (patched) - Tensor library utilities

**Utilities:**
- range-v3 0.12.0 (patched) - Ranges library
- magic_enum (enchantum) - Enum reflection
- Reflect (boost-ext) 1.2.6 - Struct reflection
- simde 0.8.2 - SIMD everywhere
- Tracy - Performance profiling client
- spdlog 1.15.2 - Structured logging
- tt-logger 1.1.8 - Tenstorrent logging utilities

**Benchmarking & Parallelism:**
- Google Benchmark 1.9.1 - Microbenchmark framework
- Taskflow 3.7.0 - Task-based parallelism
- Cap'n Proto 1.2.0 (patched) - RPC/serialization framework

**Python Runtime Dependencies:**
- loguru - Structured logging for Python
- networkx 3.1+ - Graph operations
- graphviz 0.20.3 - Graph visualization
- pyyaml 5.4+ - YAML support
- click 8.1.7+ - CLI framework
- pandas 2.0.3+ - Data analysis
- seaborn 0.13.2+ - Statistical visualization

## Configuration

**Build Configuration:**
- Default preset: `default` - Ninja Multi-Config generator
- Alternative presets: `gcc`, `libcpp` (libc++), `clang-tidy`, `clang-static-analyzer`
- Toolchain: `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` (default)
- Build type: RelWithDebInfo (default for single-config generators)

**CMake Cache Variables:**
- `TT_METAL_BUILD_TESTS`: TRUE - Enable metal tests
- `TTNN_BUILD_TESTS`: TRUE - Enable TTNN tests
- `BUILD_PROGRAMMING_EXAMPLES`: TRUE - Enable example programs
- `BUILD_TT_TRAIN`: TRUE - Build TTML training package
- `ENABLE_CCACHE`: TRUE - Enable compiler caching
- `TT_UNITY_BUILDS`: FALSE - Disable unity builds (one-file-per-cpp)

**Submodules (via git):**
- `tt_metal/third_party/umd` - Device management layer (https://github.com/tenstorrent/tt-umd.git)
- `tt_metal/third_party/tt_llk` - Low-level kernel API (https://github.com/tenstorrent/tt-llk.git)
- `tt_metal/third_party/tracy` - Performance profiling (https://github.com/tenstorrent-metal/tracy.git)
- `models/demos/t3000/llama2_70b/reference/llama` - LLaMA reference implementation

**Compiler Flags:**
- C++ Standard: C++20, required, no extensions
- Position Independent Code: ON
- Debug flags: `-O0 -g3 -ggnu-pubnames -DDEBUG -fno-omit-frame-pointer`
- Release flags: `-O3 -g -ggnu-pubnames -DDEBUG -fno-omit-frame-pointer`

## Platform Requirements

**Development:**
- Linux x86_64 system
- CMake 3.24 or later
- Python 3.10 or later
- libnuma and hwloc libraries (system install required)
- Clang 20 or GCC 12+ compiler
- Git with submodule support

**Production/Runtime:**
- Tenstorrent hardware device (Wormhole, Blackhole, or Quasar)
- OpenMPI 5.0.7 ULFM runtime in `/opt/openmpi-v5.0.7-ulfm/lib`
- RPATH configured to: `/opt/openmpi-v5.0.7-ulfm/lib;${PROJECT_BINARY_DIR}/lib;$ORIGIN`
- Linux kernel with NUMA support

---

*Stack analysis: 2026-03-16*
