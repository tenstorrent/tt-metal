# Technology Stack

**Analysis Date:** 2026-03-12

## Languages

**Primary:**
- C++ 20 - Core device runtime, hardware abstraction layer, kernel dispatch system
- Python 3.10+ - High-level API, TTNN operations, distributed compute, examples and testing
- RISC-V Assembly - Custom firmware for embedded cores (erisc router, packet processors)
- YAML - Hardware configuration, device descriptors, mesh graph definitions

**Secondary:**
- C - Some legacy interop code and system integration
- Protobuf - Message serialization for internal protocol buffers
- FlatBuffers - Serialization format for hardware descriptors and cached metadata

## Runtime

**Environment:**
- Linux x86_64 (primary target)
- Debian/Arch/Alpine variants supported via platform detection in `setup.py`
- PCIe-based Tenstorrent hardware (Grayskull, Wormhole, Blackhole architectures)

**Package Manager:**
- Python: setuptools 70.1.0, pip
- C++: CPM (CMake Package Manager)
- Lockfile: `pyproject.toml` (Python), CMake dependency resolution (C++)

## Frameworks

**Core:**
- CMake 3.24+ - Build system (`CMakeLists.txt`)
- Ninja - Build generator (used by `build_metal.sh`)

**Python API:**
- nanobind 2.10.2 - C++ to Python bindings (`/home/snijjar/tt-metal/ttnn/cpp/ttnn/*_nanobind.cpp`)
- setuptools - Package distribution with editable installs

**Testing:**
- Google Test (googletest v1.13.0) - C++ unit tests
- pytest - Python integration and system tests
- CTest - CMake test runner

**Build/Dev:**
- clang-20 - C++ compiler (enforced in `setup.py` line 198)
- clang-tidy - Static analysis (disabled for third-party code in `CMakeLists.txt`)
- Taskflow v3.7.0 - Task graph execution for parallel scheduling
- ccache - Build cache (optional via `ENABLE_CCACHE`)

## Key Dependencies

**Critical C++ Libraries:**

**Serialization & Configuration:**
- protobuf v21.12 - Protocol buffer support (symbol visibility configured for bundling)
- yaml-cpp 0.8.0 (patched) - YAML parsing for SOC descriptors at `tt_metal/soc_descriptors/`
- flatbuffers v24.3.25 - Hardware descriptor serialization
- nlohmann/json v3.11.3 - JSON parsing
- Cap'n Proto v1.2.0 - RPC and serialization framework

**Hardware & System:**
- Boost 1.86.0 (core, container, smart_ptr, interprocess, asio, lockfree) - System abstractions
- NUMA library (system) - Non-uniform memory access support (required, see `third_party/CMakeLists.txt` line 14)
- hwloc library (system) - Hardware topology discovery (required, line 22)

**Utilities:**
- fmt 11.1.4 - String formatting
- range-v3 0.12.0 (patched) - Functional programming ranges
- xtensor 0.26.0 (patched) - Tensor operations (with xtl 0.8.0 and xtensor-blas 0.22.0)
- spdlog 1.15.2 - Fast logging with external fmt
- tt-logger 1.1.8 - Tenstorrent logging extensions
- simde v0.8.2 - SIMD everywhere support for portable vectorization

**Data & Reflection:**
- boost-ext/reflect v1.2.6 - Compile-time reflection
- enchantum (magic_enum-like) - Enum reflection

**Performance & Monitoring:**
- benchmark v1.9.1 - Microbenchmarking framework (used in `tests/tt_metal/perf_microbenchmark/`)
- Tracy - Frame profiler (optional, enabled via `CIBW_ENABLE_TRACY=ON` in wheel builds)

**Python Runtime Dependencies:**
- numpy >=1.24.4, <2 - Array operations
- loguru >=0.6.0 - Logging framework
- networkx >=3.1 - Graph processing (for model optimization)
- graphviz >=0.20.3 - Graph visualization
- pyyaml >=5.4 - YAML parsing
- click >=8.1.7 - CLI framework
- pandas >=2.0.3 - Data manipulation
- seaborn >=0.13.2 - Data visualization

**Python Build/Infrastructure Dependencies:**
- setuptools-scm 8.1.0 - Version management from git tags
- wheel - Wheel distribution format
- ansible/ansible-lint - Infrastructure provisioning (optional, `infra/requirements-infra.txt`)
- pydantic - Data validation
- pytest - Test runner
- defusedxml - Safe XML parsing

**Python Documentation Dependencies:**
- sphinx 7.1.2 - Documentation generation
- sphinx-rtd-theme 1.3.0 - ReadTheDocs theme
- breathe 4.35.0 - Doxygen/Sphinx integration
- nbsphinx 0.9.3 - Jupyter notebook in Sphinx
- myst-parser 3.0.0 - Markdown support in Sphinx
- nbconvert 7.17.0 - Notebook conversion

## Configuration

**Build System:**
- `CMakeLists.txt` at project root - Main build configuration
- `setup.py` - Python package metadata and wheel building
- `pyproject.toml` - Python package configuration with setuptools backend
- `.github/workflows/` - GitHub Actions CI/CD pipelines (40+ workflow files)

**Build Type Selection:**
- Debug: `-O0 -g3 -DDEBUG` (CMake `CMAKE_CXX_FLAGS_DEBUG`)
- RelWithDebInfo (default): `-O3 -g -DDEBUG` (CMake `CMAKE_CXX_FLAGS_RELWITHDEBINFO`)

**Environment Variables (Notable):**
- `TT_METAL_HOME` - Project root location (required for runtime)
- `TT_FROM_PRECOMPILED_DIR` - Alternative precompiled binaries location (wheel builds)
- `CIBUILDWHEEL` - Enables wheel CI/CD specific configuration
- `CIBW_BUILD_TYPE` - Release, Debug, or RelWithDebInfo (default: Release for wheels)
- `CIBW_ENABLE_TRACY` - Enable Tracy profiling in wheel builds
- `CIBW_ENABLE_LTO` - Enable Link Time Optimization in wheel builds
- `TT_FABRIC_PROFILE_SPEEDY_PATH` - Enable fabric router profiling
- `TT_FABRIC_PROFILE_SPEEDY_TIMER_MASK` - Selective timer activation for profiling

**Platform Detection:**
- Automatic lib vs lib64 selection based on OS variant (Debian/Arch/Alpine use lib, others use lib64 for 64-bit)
- Compiler detection (clang-20 enforced) with fallback error messages

## Hardware Architecture Support

**Target Devices:**
- Grayskull (GS)
- Wormhole (WH)
- Blackhole (BH) - Primary focus for fabric routing optimizations

**Architecture Components:**
- PCIe host interface (UMD from `tt_metal/third_party/umd/`)
- Ethernet cores for fabric communication
- RISC-V cores for packet processing (erisc router)
- DRAM, L1 memory with banking
- NOC (Network-on-Chip) for intra-chip communication

## Optional Features

**Profiling & Tracing:**
- Tracy frame profiler (disabled by default, `ENABLE_TRACY=OFF` in CMake)
- LightMetal trace capture (enabled via `TT_ENABLE_LIGHT_METAL_TRACE=ON`)
- spdlog-based performance logging

**Optimization:**
- LTO (Link Time Optimization) - optional via `TT_ENABLE_LTO=ON`
- ccache - optional via `ENABLE_CCACHE=TRUE`
- Unity builds - optional via `TT_UNITY_BUILDS=ON`

**Distributed Computing:**
- Multihost support (optional, `ENABLE_DISTRIBUTED=ON`) for distributed tensor operations

---

*Stack analysis: 2026-03-12*
