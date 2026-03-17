# Codebase Architecture: tt-metal

**Analyzed:** 2026-03-12

---

## System Overview

tt-metal is Tenstorrent's ML hardware programming framework. It provides:
1. A hardware abstraction layer (HAL) for TT AI accelerators (Wormhole, Blackhole)
2. A kernel programming model for RISC-V compute/routing cores
3. A fabric networking layer for multi-chip scale-out
4. Python/C++ APIs (TTNN) for model execution

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────┐
│  Python Models / Applications (models/, tt-train/)  │
├─────────────────────────────────────────────────────┤
│  TTNN Python API (ttnn/)                            │
├─────────────────────────────────────────────────────┤
│  tt-Metalium C++ API (tt_metal/api/)                │
├─────────────────────────────────────────────────────┤
│  Metal Runtime (tt_metal/impl/, tt_metal/detail/)   │
│  - Program compilation (JIT)                        │
│  - Device management                                │
│  - Memory management                                │
├─────────────────────────────────────────────────────┤
│  Fabric Layer (tt_metal/fabric/)                    │
│  - EDM (Ethernet Data Mover) channels               │
│  - Router builder / control plane                   │
│  - ERISC router kernels                             │
├─────────────────────────────────────────────────────┤
│  HAL / LLRT (tt_metal/hw/, tt_metal/llrt/)          │
│  - Device-specific SoC descriptors                  │
│  - RISC-V kernel compilation & loading              │
│  - NOC interface                                    │
├─────────────────────────────────────────────────────┤
│  Hardware: TT AI Accelerators (WH / BH)             │
└─────────────────────────────────────────────────────┘
```

---

## Fabric Subsystem Architecture (Primary Focus)

### Components

| Component | Location | Role |
|-----------|----------|------|
| ERISC Router Kernel | `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | On-device packet router running on active_erisc cores |
| Speedy Path | `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | Optimized hot path for line topology |
| EDM Channels | `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | Per-VC channel state management |
| Router Builder | `tt_metal/fabric/fabric_router_builder.cpp` | Host-side builder that configures router instances |
| Control Plane | `tt_metal/fabric/control_plane.cpp` | Topology management and routing table generation |
| Fabric Context | `tt_metal/fabric/fabric_context.cpp` | Global fabric state manager |
| Packet Header | `tt_metal/fabric/fabric_edm_packet_header.hpp` | Wire format for fabric packets |

### Data Flow (Packet Transmission)

```
Sender (Tensix/ERISC)
  │
  ▼ writes packet to outbound buffer (NOC)
EDM Sender Channel (active_erisc)
  │ checks TXQ, flow control credits
  ▼ transmits via Ethernet TXQ
EDM Receiver Channel (subordinate_active_erisc)
  │ writes to local memory or forwards
  ▼
Downstream Router or Final Consumer
```

### Virtual Channels (VC)

- Multiple VCs per physical link for deadlock avoidance
- Each VC has independent: credits, epoch counters, packet buffers
- Currently refactoring from flat array indexing → per-VC template accessors

### Router Core Assignment

| Core Type | Role |
|-----------|------|
| `active_erisc` | Sender path (runs speedy path sender) |
| `subordinate_active_erisc` | Receiver path (runs speedy path receiver) |

---

## Key Abstractions

### EDM Channels
- `fabric_erisc_datamover_channels.hpp`: `EdmChannelWorkerLocationManager`, sender/receiver channel structs
- Manages handshake, credit flow, and packet buffering

### Packet Header (`fabric_edm_packet_header.hpp`)
- Fixed-size wire format
- Contains: destination coordinates, hop count, payload size, routing plane

### Router Builder (`FabricRouterBuilder`)
- Host-side: constructs router configuration (CT args, buffer addresses)
- Spawns ERISC router kernel on target device

### Speedy Path (`LineSenderState`, `LineReceiverState`)
- Stack-local structs passed by reference through `FORCE_INLINE` functions
- Allows compiler to register-promote scalar state variables
- `credit_epoch_array[]` remains file-static (volatile, accessed by hardware)

---

## Entry Points

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| `tt_metal.cpp` | `tt_metal/tt_metal.cpp` | Main Metal C++ API |
| `fabric_init.cpp` | `tt_metal/fabric/fabric_init.cpp` | Fabric initialization |
| `fabric_context.cpp` | `tt_metal/fabric/fabric_context.cpp` | Fabric runtime context |
| `test_tt_fabric` | `build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric` | Benchmark binary |
| `conftest.py` | repo root | pytest session setup |

---

## Cross-Cutting Concerns

### Logging
- `TT_FATAL(cond, msg)` — fatal errors
- `TT_ASSERT(cond)` — debug assertions
- `log_info/log_warning/log_error` — structured logging

### Kernel Compilation (JIT)
- `tt_metal/jit_build/` — JIT compilation of device kernels
- Kernel cache at `~/.cache/tt-metal-cache/<commit_hash>/`
- ERISC kernels auto-recompile on next test run (no `ninja` rebuild needed)

### Telemetry / Profiling
- `NamedProfiler` timers in speedy path (compile-time enabled via bitfield)
- `TT_FABRIC_PROFILE_SPEEDY_PATH=1` env var to enable
- `TT_FABRIC_PROFILE_SPEEDY_TIMER_MASK=<bitmask>` for one-hot timer selection
- Parser: `fabric_performance_analysis/parse_profiling.py`

---

*Last updated: 2026-03-12 via gsd:map-codebase*
