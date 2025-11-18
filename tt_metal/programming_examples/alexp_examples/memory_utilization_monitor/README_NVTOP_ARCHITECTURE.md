# Understanding nvtop and Building Monitoring Tools for Tenstorrent

This directory contains documentation and implementations related to monitoring Tenstorrent devices, inspired by nvtop's architecture.

## Quick Navigation

- **[NVTOP_ARCHITECTURE_GUIDE.md](./NVTOP_ARCHITECTURE_GUIDE.md)** - Complete guide on how nvtop works and how to build similar tools
- **[IMPLEMENTATION_COMPARISON.md](./IMPLEMENTATION_COMPARISON.md)** - Comparison of sysfs vs UMD telemetry approaches
- **[TT_SMI_UMD_TELEMETRY_GUIDE.md](./TT_SMI_UMD_TELEMETRY_GUIDE.md)** - Original guide on using UMD APIs

## Files in This Directory

### Current Implementation
- `tt_smi.cpp` - Snapshot monitoring tool (like nvidia-smi)
- `allocation_server_poc.cpp` - Memory tracking server

### Enhanced Implementation (New)
- `tt_smi_umd.cpp` - Enhanced version with UMD telemetry

### External Reference
- `nvtop/src/extract_gpuinfo_tenstorrent.c` - nvtop plugin for Tenstorrent (in nvtop repo)

## Key Concepts

### 1. nvtop Architecture

nvtop is a modular GPU monitoring tool with:
- **Plugin-based vendor support** (NVIDIA, AMD, Intel, Apple, etc.)
- **Common abstraction layer** for device info
- **Process discovery** via fdinfo
- **Interactive ncurses UI**

```
┌─────────────┐
│   ncurses   │  ← User Interface
├─────────────┤
│ Abstraction │  ← Vendor-agnostic layer
├─────────────┤
│   Plugins   │  ← NVIDIA | AMD | Intel | Tenstorrent
└─────────────┘
```

### 2. Telemetry Sources

**sysfs (Current tt-smi):**
- Pros: Fast, no dependencies
- Cons: Limited data, local only

**UMD APIs (Enhanced tt-smi-umd):**
- Pros: Complete data, remote support
- Cons: Slower, device exclusivity

### 3. Process Discovery

**Current:** Scan `/proc/*/fd/` for `/dev/tenstorrent/*`
- Tells us which processes have devices open
- No per-process GPU usage yet

**Future:** Use fdinfo (requires tt-kmd changes)
- Per-process memory usage
- Per-process GPU utilization
- Match nvtop's capabilities

## Quick Start

### Build and Run tt-smi (current)
```bash
cd /home/ttuser/aperezvicente/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Build
g++ -o tt_smi tt_smi.cpp -std=c++20 -lpthread -lstdc++fs

# Run (snapshot)
./tt_smi

# Run (watch mode)
./tt_smi -w
```

### Build and Run tt-smi-umd (enhanced)
```bash
# Set environment
export TT_METAL_HOME=/home/ttuser/aperezvicente/tt-metal-apv

# Build (requires UMD)
g++ -o tt_smi_umd tt_smi_umd.cpp \
    -std=c++20 \
    -I$TT_METAL_HOME/third_party/umd/device/api \
    -L$TT_METAL_HOME/build/lib \
    -lumd_device \
    -lpthread -lstdc++fs

# Run with UMD telemetry
./tt_smi_umd -w

# Compare with sysfs
./tt_smi_umd --sysfs -w
```

### Optional: Start Allocation Server for Memory Tracking
```bash
# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Run tt-smi
./tt_smi -w
# Now shows memory usage per device
```

## Understanding the Differences

### tt-smi vs tt-smi-umd

| Feature | tt-smi | tt-smi-umd |
|---------|--------|------------|
| Temperature | ⚠️ sysfs (may be N/A) | ✅ UMD (always available) |
| Power | ❌ | ✅ TDP/TDC |
| Clocks | ❌ | ✅ AICLK/AXICLK/ARCCLK |
| Fan Speed | ❌ | ✅ RPM |
| Remote Devices | ❌ | ✅ |
| Speed | Fast (2ms) | Slower (100ms init) |
| Dependencies | None | Requires UMD |

### When to Use Each

**Use tt-smi when:**
- Quick development checks
- Device is busy with workload
- Minimal dependencies needed
- CI/CD health checks

**Use tt-smi-umd when:**
- Need complete telemetry data
- Monitoring remote devices
- Production monitoring
- Detailed power/clock analysis

## Architecture Comparison

### nvidia-smi / nvtop Architecture
```
┌──────────────┐
│     Tool     │
│ (nvidia-smi) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     NVML     │  ← Vendor library
│   Library    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Driver     │  ← Kernel driver
│  (nvidia.ko) │
└──────┬───────┘
       │
       ▼
    Hardware
```

### tt-smi Architecture (Current)
```
┌──────────────┐
│   tt-smi     │
└───┬──────┬───┘
    │      │
    │      └─────────────┐
    │                    ▼
    │              ┌──────────────┐
    │              │ Allocation   │
    │              │   Server     │
    │              └──────────────┘
    │                (memory only)
    ▼
┌──────────────┐
│    sysfs     │  ← Limited telemetry
│   (/sys)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   tt-kmd     │  ← Kernel driver
└──────────────┘
```

### tt-smi-umd Architecture (Enhanced)
```
┌──────────────┐
│  tt-smi-umd  │
└───┬──────┬───┘
    │      │
    │      └─────────────┐
    │                    ▼
    │              ┌──────────────┐
    │              │ Allocation   │
    │              │   Server     │
    │              └──────────────┘
    │                (memory)
    ▼
┌──────────────┐
│   TT-UMD     │  ← Complete telemetry
│   Library    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Firmware    │  ← Direct access
│   (PCIe)     │
└──────────────┘
```

## Implementation Roadmap

### ✅ Phase 1: Basic Monitoring (Done)
- [x] Device enumeration
- [x] Process discovery
- [x] Memory tracking via allocation server
- [x] sysfs telemetry (limited)

### 🔄 Phase 2: Enhanced Telemetry (In Progress)
- [x] UMD telemetry integration
- [x] Complete temperature/power/clock data
- [ ] Device caching for performance
- [ ] Fallback to sysfs when device busy

### 📋 Phase 3: Advanced Process Tracking (Planned)
- [ ] fdinfo support in tt-kmd
- [ ] Per-process GPU usage
- [ ] Per-process memory tracking
- [ ] Device-to-process mapping

### 📋 Phase 4: Interactive UI (Planned)
- [ ] ncurses interface (like nvtop)
- [ ] Historical charts
- [ ] Process management (kill, sort, filter)
- [ ] Configuration system

### 📋 Phase 5: Production Features (Future)
- [ ] Remote monitoring
- [ ] Web dashboard
- [ ] Prometheus exporter
- [ ] Alert system

## nvtop Plugin for Tenstorrent

We created a plugin for nvtop to support Tenstorrent devices:
- Located in: `nvtop/src/extract_gpuinfo_tenstorrent.c`
- Uses dynamic loading of UMD library
- Integrates with nvtop's UI and process tracking

To build nvtop with Tenstorrent support:
```bash
cd /home/ttuser/aperezvicente/nvtop
mkdir -p build && cd build
cmake .. \
    -DNVIDIA_SUPPORT=ON \
    -DAMDGPU_SUPPORT=ON \
    -DINTEL_SUPPORT=ON \
    -DTENSTORRENT_SUPPORT=ON
make
sudo make install
```

## Key Insights from nvtop

### 1. Plugin Architecture
- Each vendor is a self-contained plugin
- Registers itself via constructor attribute
- Dynamic library loading (no hard dependencies)

### 2. Validity Bitmasks
- Efficient way to track optional fields
- C-compatible (no std::optional)
- Fast validity checks

### 3. fdinfo for Process Tracking
- Kernel exposes GPU usage via /proc/<pid>/fdinfo/<fd>
- Standard across AMD, Intel, NVIDIA
- Tenstorrent needs tt-kmd support

### 4. Separation of Concerns
- Data extraction (vendor plugins)
- Data aggregation (core)
- Data presentation (UI)

## Common Issues and Solutions

### Issue: "N/A" for temperature
**Cause:** sysfs not available or not populated
**Solution:** Use tt-smi-umd with UMD telemetry

### Issue: Device busy error
**Cause:** Another process owns the device (exclusivity)
**Solution:**
1. Close other processes, or
2. Use sysfs fallback, or
3. Implement telemetry server

### Issue: No per-process memory
**Cause:** Processes not connected to allocation server
**Solution:** Instrument processes to report allocations

### Issue: Can't see which device a process uses
**Cause:** No fdinfo support in tt-kmd
**Solution:** Add fdinfo to kernel driver

## Contributing

### To add a new feature:
1. Check if nvtop has similar feature
2. Adapt to Tenstorrent specifics
3. Test on multiple device types
4. Update documentation

### To improve telemetry:
1. Check TT-UMD firmware provider API
2. Add new fields to TelemetryData struct
3. Update display logic
4. Test on real hardware

## References

### External
- [nvtop GitHub](https://github.com/Syllo/nvtop)
- [Linux DRM fdinfo](https://www.kernel.org/doc/html/latest/gpu/drm-usage-stats.html)
- [NVML Documentation](https://docs.nvidia.com/deploy/nvml-api/)

### Internal
- TT-UMD API docs (in tt-metal/third_party/umd)
- TT-Metal device APIs
- tt-kmd kernel driver

## Getting Help

1. Check the guide: `NVTOP_ARCHITECTURE_GUIDE.md`
2. Compare implementations: `IMPLEMENTATION_COMPARISON.md`
3. Review UMD usage: `TT_SMI_UMD_TELEMETRY_GUIDE.md`
4. Look at nvtop code for patterns

## Future Vision: tt-nvtop

A full interactive monitoring tool for Tenstorrent:
```
┌─────────────────────────────────────────────────────────────┐
│  tt-nvtop v1.0                    Mon Nov  3 14:30:00 2025  │
├─────────────────────────────────────────────────────────────┤
│  Device 0: Blackhole                                        │
│    GPU  [████████████████████░░░░░░░░░░░░] 60% @ 1200 MHz  │
│    MEM  [████████░░░░░░░░░░░░░░░░░░░░░░░░] 25% 3.0/12 GB   │
│    TEMP [████████████████░░░░░░░░░░░░░░░░] 68°C / 95°C     │
│    PWR  [███████████████████░░░░░░░░░░░░░] 285W / 350W     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           GPU Utilization History (60s)              │ │
│  │ 100%│                     ╭─╮                         │ │
│  │  75%│         ╭─╮    ╭───╯ ╰─╮                      │ │
│  │  50%│    ╭────╯ ╰────╯        ╰─╮                   │ │
│  │  25%│╭───╯                      ╰──╮                │ │
│  │   0%│╯                             ╰────────────────│ │
│  └───────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Processes (4 running)                                      │
│  PID     User    Command         GPU%   MEM      TIME      │
│  12345   user1   model_train     45%    2.1GB    1:23:45   │
│  12346   user2   inference       15%    900MB    0:45:12   │
│  12347   user1   preprocess      5%     128MB    0:12:34   │
│  12348   user3   validation      3%     64MB     0:05:21   │
└─────────────────────────────────────────────────────────────┘
  F1:Help F2:Setup F9:Kill F12:Save q:Quit
```

## Summary

**nvtop** is an excellent reference for building GPU monitoring tools. Its plugin architecture, robust error handling, and user-friendly interface make it a great template for Tenstorrent monitoring.

**Key takeaways:**
1. Use UMD APIs for complete telemetry (not just sysfs)
2. Add fdinfo to tt-kmd for per-process tracking
3. Build modular (separate data collection from UI)
4. Leverage existing tools (nvtop) as references

**Current state:**
- ✅ Basic monitoring works (tt-smi)
- ✅ Enhanced telemetry available (tt-smi-umd)
- ⚠️ Per-process GPU usage needs kernel support
- 📋 Interactive UI (tt-nvtop) is next big step

**Next steps:**
1. Test tt-smi-umd on your 4 Blackhole devices
2. Compare telemetry with sysfs version
3. Implement device caching for better performance
4. Start work on fdinfo in tt-kmd
