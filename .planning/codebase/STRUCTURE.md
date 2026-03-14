# Codebase Structure: tt-metal

**Analyzed:** 2026-03-12

---

## Top-Level Directory Layout

```
tt-metal/
├── tt_metal/               # Core Metal framework (C++)
│   ├── api/tt-metalium/    # Public C++ API headers
│   ├── fabric/             # Fabric networking layer
│   │   ├── impl/kernels/edm_fabric/   # ERISC router kernels (device-side C++)
│   │   ├── hw/inc/edm_fabric/         # Device-side fabric headers
│   │   ├── builder/        # Router builder infrastructure
│   │   └── config/         # Fabric topology configs
│   ├── hw/                 # Hardware-specific code
│   ├── impl/               # Runtime implementation
│   ├── jit_build/          # JIT kernel compilation
│   ├── llrt/               # Low-level runtime (device interface)
│   ├── kernels/            # Built-in kernel library
│   └── third_party/        # Submodules (UMD, LLK)
├── ttnn/                   # TTNN Python/C++ API layer
├── tests/                  # All tests
│   ├── tt_metal/           # Metal layer tests (C++ + Python)
│   │   ├── tt_metal/perf_microbenchmark/routing/   # Fabric benchmarks
│   │   └── multihost/      # Multi-device tests
│   ├── ttnn/               # TTNN API tests
│   ├── scale_out/          # Scale-out tests
│   └── nightly/            # Nightly regression suites
├── models/                 # ML model implementations
├── docs/                   # Documentation
├── cmake/                  # CMake modules
├── scripts/                # Build and CI scripts
├── fabric_performance_analysis/  # Perf analysis scripts & results (local)
├── fabric_bisection/       # Bisection tooling (local)
├── fabric_pipeline_trace/  # Pipeline tracing (local)
├── build/                  # Build output (gitignored)
└── .planning/              # GSD planning documents
    └── codebase/           # This codebase map
```

---

## Key File Locations

### Fabric Kernel (Primary Development Area)

| File | Purpose |
|------|---------|
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Main ERISC router kernel (3,608 lines) |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp` | Mux extension kernel |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_relay_extension.cpp` | Relay extension kernel |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | Optimized hot path |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | Channel abstractions |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | Compile-time args |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` | Packet TX helpers |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp` | TID tracking |

### Fabric Host-Side

| File | Purpose |
|------|---------|
| `tt_metal/fabric/fabric_router_builder.cpp` | Router builder |
| `tt_metal/fabric/erisc_datamover_builder.cpp` | EDM builder |
| `tt_metal/fabric/control_plane.cpp` | Topology & routing tables |
| `tt_metal/fabric/fabric_context.cpp` | Fabric runtime context |
| `tt_metal/fabric/fabric_init.cpp` | Initialization entry |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | Wire format |

### Benchmark / Test

| File | Purpose |
|------|---------|
| `build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric` | Main benchmark binary |
| `tests/tt_metal/tt_metal/perf_microbenchmark/routing/*.yaml` | Benchmark config files |
| `tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden/` | Golden reference results |

### Build & Config

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Root CMake |
| `CMakePresets.json` | Preset configurations |
| `build_metal.sh` | Build script |
| `pytest.ini` | pytest configuration |
| `conftest.py` | Root pytest fixtures |

---

## Naming Conventions

### Files
- C++ headers: `snake_case.hpp`
- C++ sources: `snake_case.cpp`
- Kernel files: `fabric_*` prefix for fabric layer
- Test configs: `test_*.yaml`

### Code
- Classes: `PascalCase` (e.g., `FabricRouterBuilder`, `LineSenderState`)
- Functions: `snake_case` (e.g., `run_sender_channels_step_line_speedy`)
- Macros: `UPPER_SNAKE_CASE` (e.g., `FORCE_INLINE`, `TT_FATAL`)
- Constants: `UPPER_SNAKE_CASE` or `constexpr` with `snake_case`
- Namespaces: `tt`, `tt::tt_metal`, `tt::fabric`

### Test Files
- C++ tests: `test_*.cpp`
- Python tests: `test_*.py`
- Config YAMLs: `test_*.yaml` or descriptive names

---

## Module Boundaries

```
tt_metal/api/      ← Public surface (stable interface)
tt_metal/impl/     ← Private runtime internals
tt_metal/fabric/   ← Fabric subsystem (semi-isolated)
tt_metal/hw/       ← Hardware-specific (SoC descriptors, HAL)
tt_metal/llrt/     ← Low-level runtime (device communication)
tt_metal/kernels/  ← Reusable device kernels
ttnn/              ← High-level Python/C++ API (depends on tt_metal)
```

---

## Build Artifacts

| Path | Contents |
|------|---------|
| `build/` | All compiled artifacts |
| `build/test/` | Test binaries |
| `~/.cache/tt-metal-cache/<hash>/` | Compiled ERISC kernel cache |
| `tt_metal/pre-compiled/` | Pre-compiled artifacts (gitignored) |
| `generated/` | Generated code and test reports |

---

*Last updated: 2026-03-12 via gsd:map-codebase*
