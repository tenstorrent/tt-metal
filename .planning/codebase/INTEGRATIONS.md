# External Integrations

**Analysis Date:** 2026-03-12

## Hardware Integration

**PCIe/Device Communication:**
- UMD (User Mode Driver) - Located at `tt_metal/third_party/umd/`
- Device class: `Device` in `tt_metal/impl/device/device_impl.hpp`
- Hardware abstraction layer (HAL) at `tt_metal/impl/device/`
- ChipId-based device identification
- Multi-card support via DeviceManager at `tt_metal/impl/device/device_manager.hpp`

**Fabric/Ethernet Communication:**
- Fabric router kernel: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
- Fabric speedy path: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp`
- Ethernet core communication for multi-chip topologies
- Mesh graph descriptors at `tt_metal/fabric/mesh_graph_descriptors/`

**System Integration:**
- NUMA-aware memory management (libnuma required)
- hwloc for hardware topology detection
- PCIe device enumeration via UMD

## Data Storage

**On-Device Memory:**
- L1 memory banking (configurable per chip)
- DRAM channels per device
- Circular buffers for kernel communication
- Memory allocator: `tt_metal/impl/allocator/` with L1 banking support

**Configuration Storage:**
- YAML-based SOC descriptors: `tt_metal/soc_descriptors/*.yaml`
- Core descriptors: `tt_metal/core_descriptors/*.yaml`
- Hardware configuration via yaml-cpp parser
- Flatbuffer serialization for cached metadata

**No External Database:**
- All device state is in-memory
- No persistent storage integration detected
- Configuration files are read-only at runtime

## Build & Deployment Infrastructure

**Continuous Integration:**
- GitHub Actions workflows: `.github/workflows/` (40+ files)
- Key pipelines:
  - `build-and-unit-tests.yaml` - Core unit tests
  - `blackhole-post-commit.yaml` - Blackhole device tests
  - `all-static-checks.yaml` - Linting and static analysis
  - `build-all-docker-images.yaml` - Container building
  - Auto-retry mechanisms: `_auto-retry-post-commit.yaml`, `_auto-retry-nightly-workflows.yaml`

**Wheel Distribution:**
- PyPI distribution via setuptools
- Wheel building configuration in `setup.py` (CMake-based extension building)
- Multi-platform support (Linux x86_64 targets)
- Precompiled artifact bundling from `build_metal.sh`

**Docker Containerization:**
- Dockerfile-based builds for CI/CD
- Container images for testing and deployment

## Telemetry & Observability

**Logging:**
- spdlog 1.15.2 - Structured logging with external fmt
- tt-logger 1.1.8 - Tenstorrent-specific logging extensions
- Logger initialization in device startup (`tt_metal/impl/device/device.cpp` line 45)
- Loguru >=0.6.0 for Python-side logging

**Profiling & Tracing:**
- Tracy frame profiler - Optional at build time (`ENABLE_TRACY=OFF` default)
- LightMetal trace capture - Light-weight tracing via `tt_metal/impl/lightmetal/`
- NamedProfiler - Custom timer infrastructure for fabric router profiling
- DPRINT kernel debugging support (firmware-side)
- Ring buffer infrastructure for performance metrics

**Performance Monitoring:**
- Google Benchmark v1.9.1 - Microbenchmark harness
- Performance test configs: `tests/tt_metal/tt_metal/perf_microbenchmark/routing/`
- Fabric performance analysis scripts in `fabric_performance_analysis/`

## Distributed Computing

**Multi-Device Support:**
- Ethernet-based fabric for multi-chip communication
- DeviceManager handles device enumeration
- Sub-device abstractions for tensor parallel operations
- Distributed tensor operations (optional, `ENABLE_DISTRIBUTED=ON`)

**Multihost Support:**
- Optional distributed compute infrastructure
- Inter-node communication via ethernet

## Version Control & Build Metadata

**Git Integration:**
- setuptools_scm for automatic versioning from git tags
- Tag format: `v{major}.{minor}.{patch}[-rc{N}]`
- Fallback version: `0.0.0.dev0`
- License detection: SPDX-FileCopyrightText and SPDX-License-Identifier headers

**Build Artifacts:**
- Cache location: `~/.cache/tt-metal-cache/<commit_hash>/`
- Precompiled directory option: `TT_FROM_PRECOMPILED_DIR` env var
- SFPI compiler toolchain: `runtime/sfpi/compiler/bin/riscv-tt-elf-*`

## Development Tools Integration

**Static Analysis:**
- clang-tidy (configurable, disabled for third-party in build)
- Code formatting: `.clang-format` (not specified in stack, likely present)
- Copyright check via espressif/check-copyright

**Documentation Generation:**
- Sphinx with RTD theme
- Breathe for Doxygen integration
- Jupyter notebook support via nbsphinx
- Markdown support via myst-parser

**Code Analysis & Visualization:**
- networkx - Graph processing for model optimization
- graphviz - Visualization output
- Python graph APIs in TTNN operations

## No Third-Party API Integrations

**What's NOT integrated:**
- No cloud provider APIs (AWS, GCP, Azure)
- No external ML frameworks (TensorFlow, PyTorch plugins)
- No authentication services (OAuth, Cognito)
- No message queues (Kafka, RabbitMQ)
- No external monitoring (Datadog, New Relic)
- No payment processing
- No analytics platforms
- All integrations are internal or system-level (hardware, OS)

## Environment & Configuration

**Required System Libraries:**
- libnuma - NUMA memory management
- libhwloc - Hardware topology
- Standard C/C++ runtime (libc, libstdc++)

**Python Runtime:**
- Python 3.10+ as host language
- Virtual environment recommended (no specific tool mandated)
- Package installation via pip editable install: `pip install -e .`

**Build Environment Variables:**
- `TT_METAL_HOME` - Required at runtime, points to repo root
- `CIBUILDWHEEL` - GitHub Actions wheel building context
- `CMAKE_BUILD_TYPE` - Release, Debug, RelWithDebInfo
- Compiler override: `CXX=clang++-20` (hardcoded in setup.py)

**System Requirements:**
- Linux kernel with PCIe support
- Tenstorrent hardware device connected via PCIe
- 64-bit x86 architecture (lib vs lib64 detection automatic)

---

*Integration audit: 2026-03-12*
