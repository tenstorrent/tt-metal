# External Integrations

**Analysis Date:** 2026-03-16

## APIs & External Services

**Model Repositories:**
- NVIDIA Hugging Face / Model Hub - Models loaded during demo execution
  - SDK/Client: `requests` library for HTTP downloads
  - Usage: Model weights and tokenizer downloads for inference demos

**Git Repositories:**
- GitHub (Tenstorrent) - Primary code hosting
  - UMD (User Mode Driver): `tt_metal/third_party/umd`
  - TT-LLK (Kernel API): `tt_metal/third_party/tt_llk`
  - Tracy (Profiler): `tt_metal/third_party/tracy`

## Data Storage

**Databases:**
- Not detected - No persistent database integration

**File Storage:**
- Local filesystem only - All data stored in project directories
- Model weights/demos: `models/demos/` directory structure
- Kernel cache: `~/.cache/tt-metal-cache/<git-hash>/*/kernels/` (computed on demand)

**Caching:**
- Build caching: ccache for C++ incremental builds (enabled by default)
- Kernel JIT cache: Per-git-hash directory keyed on `.cpp` hash only (not `.hpp`)
- Python package cache: uv package manager with "copy" link mode for isolation

## Authentication & Identity

**Auth Provider:**
- Not detected - No API authentication or user identity management
- MPI-based distributed computing uses fault-tolerant MPI (ULFM) for node coordination

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service

**Logs:**
- Local file-based logging via spdlog 1.15.2 and tt-logger 1.1.8
- Runtime dprint parser for device-level debugging (`tt_metal/impl/debug/dprint_parser.hpp`)
- Structured logging through loguru (Python)

**Profiling:**
- Tracy profiler client (`TracyClient` target in CMake) - CPU/GPU profiling
- Google Benchmark 1.9.1 for performance microbenchmarks
- Test configuration: `tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml`

## CI/CD & Deployment

**Hosting:**
- GitHub - Primary repository and CI platform
- Tenstorrent hardware - Deployment/execution environment (Wormhole, Blackhole, Quasar)

**CI Pipeline:**
- GitHub Actions
  - Primary workflow: `.github/workflows/sanity-tests.yaml`
  - SKU configuration: `.github/sku_config.yaml`
  - Time budgets: `.github/time_budget.yaml`
  - SPDX license ignore list: `.github/spdx_ignore.yaml`

**Build Output:**
- CMake-based incremental builds with Ninja
- Wheel distribution via setuptools/scikit-build-core for Python packages
- Installed components: metalium-runtime, metalium-dev, ttml, spdlog-dev, json-dev

## Environment Configuration

**Required env vars:**
- `TT_METAL_HOME` - Metal installation directory (for tests)
- `TT_METAL_BUILD_TESTS` - Build test targets (CMake cache var)
- `BUILD_PROGRAMMING_EXAMPLES` - Enable examples (CMake cache var)

**Secrets location:**
- `.env` files (not read by this analyzer) - See `.planning/codebase/INTEGRATIONS.md` for runtime setup

## Communication Patterns

**Intra-Device Communication:**
- Device-to-Host: DMA transfers via UMD layer
- Host-to-Device: Command queue via dispatch subsystem

**Multi-Device Communication:**
- Fabric-based direct mesh interconnect - Device-to-device data paths
- Multicast inline direct write for efficient broadcast patterns
- Channel trimming and bandwidth optimization for fabric routes

**Multi-Host Communication:**
- MPI (OpenMPI 5.0.7 ULFM) for distributed compute
  - Classes: `MPIDistributedContext`, `MPIRequest` in `tt_metal/distributed/multihost/`
  - Blocking: `send()`, `recv()`, `broadcast()`, `all_reduce()`
  - Non-blocking: `isend()`, `irecv()` with async request polling
  - Single-host fallback: `SingleHostContext` for non-distributed execution
  - Socket utilities: `mesh_socket_utils.hpp` for custom network I/O

## Webhooks & Callbacks

**Incoming:**
- Not detected

**Outgoing:**
- Not detected

## Testing Infrastructure

**Test Frameworks:**
- Google Test (gtest) 1.13.0 - C++ unit test execution
- pytest - Python test discovery and execution
- Test files located in: `tests/tt_metal/` and `tests/ttnn/`

**Device Tests:**
- Hardware device tests require physical Tenstorrent hardware
- Simulation support via UMD mock device utilities
- No parallel hardware test execution (constraint: single device per test agent)

**Benchmarks:**
- Fabric bandwidth microbenchmark: `test_tt_fabric` target
  - Binary: `./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric`
  - YAML config: `tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml`
  - Kernel cache directory: `~/.cache/tt-metal-cache/<git-hash>/*/kernels/tt_fabric_test_sender/`

---

*Integration audit: 2026-03-16*
