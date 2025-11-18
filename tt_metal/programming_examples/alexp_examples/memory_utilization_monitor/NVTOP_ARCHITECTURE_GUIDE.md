# Building a Tenstorrent Monitoring Tool: Learning from nvtop

## Executive Summary

**nvtop** is a modular, vendor-agnostic GPU monitoring tool similar to `htop` but for GPUs. This guide explains its architecture and shows how to build similar functionality for Tenstorrent devices.

---

## How nvtop Works

### 1. Layered Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                 USER INTERFACE LAYER                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │   Device    │  │   Process    │  │   Historical   │      │
│  │   Status    │  │   List       │  │   Charts       │      │
│  └─────────────┘  └──────────────┘  └────────────────┘      │
│                    (ncurses)                                  │
└───────────────────────────────────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              ABSTRACTION/AGGREGATION LAYER                    │
│  • Device discovery & enumeration                             │
│  • Data structure management                                  │
│  • Process-to-device mapping                                  │
│  • Historical data tracking                                   │
│  • Vendor-agnostic interfaces                                 │
└───────────────────────────────────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  VENDOR PLUGIN LAYER                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────┐      │
│  │ NVIDIA  │  │   AMD   │  │  Intel  │  │   Apple    │      │
│  │ (NVML)  │  │ (libdrm)│  │ (i915)  │  │  (Metal)   │      │
│  └─────────┘  └─────────┘  └─────────┘  └────────────┘      │
│                                                                │
│  Each plugin implements:                                      │
│    - init/shutdown                                            │
│    - get_device_handles                                       │
│    - populate_static_info                                     │
│    - refresh_dynamic_info                                     │
│    - get_running_processes                                    │
└───────────────────────────────────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              HARDWARE/KERNEL INTERFACE                        │
│  • Vendor libraries (NVML, ROCm, etc.)                        │
│  • DRM ioctls                                                 │
│  • sysfs (/sys/class/...)                                     │
│  • procfs fdinfo (/proc/<pid>/fdinfo/<fd>)                    │
└───────────────────────────────────────────────────────────────┘
```

### 2. Key Design Patterns

#### A. Plugin Registration (Constructor Pattern)

```c
struct gpu_vendor {
    struct list_head list;  // Linked list of vendors

    bool (*init)(void);
    void (*shutdown)(void);
    bool (*get_device_handles)(struct list_head *devices, unsigned *count);
    void (*populate_static_info)(struct gpu_info *gpu);
    void (*refresh_dynamic_info)(struct gpu_info *gpu);
    void (*get_running_processes)(struct gpu_info *gpu);
};

// Auto-register on load
__attribute__((constructor))
static void register_nvidia_plugin(void) {
    register_gpu_vendor(&gpu_vendor_nvidia);
}
```

**Benefits:**
- No hardcoded vendor list
- Easy to add new vendors
- Compile-time vendor selection via CMake

#### B. Validity Bitmask Pattern

Instead of using `std::optional`, nvtop uses bitmasks:

```c
struct gpuinfo_dynamic_info {
    unsigned int gpu_temp;           // Value
    unsigned int power_draw;         // Value
    unsigned char valid[N];          // Bitmask for validity
};

#define SET_GPUINFO_DYNAMIC(ptr, field, value) do {  \
    (ptr)->field = (value);                          \
    SET_VALID(gpuinfo_##field##_valid, (ptr)->valid); \
} while(0)

#define GPUINFO_DYNAMIC_FIELD_VALID(ptr, field) \
    IS_VALID(gpuinfo_##field##_valid, (ptr)->valid)
```

**Why?**
- C compatibility (no C++ optional)
- Compact (1 bit per field)
- Fast validity checks

#### C. Dynamic Library Loading (dlopen)

```c
void *nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
if (!nvml_handle) {
    // NVIDIA not available, disable plugin
    return false;
}

// Load all function pointers
nvmlInit = dlsym(nvml_handle, "nvmlInit_v2");
nvmlDeviceGetCount = dlsym(nvml_handle, "nvmlDeviceGetCount_v2");
// ... hundreds more
```

**Benefits:**
- No hard dependency on vendor libraries
- Single binary works across different systems
- Graceful degradation if vendor not present

### 3. Data Flow

```
[1] Main Loop (every 100-1000ms)
     ↓
[2] gpuinfo_refresh_dynamic_info(devices)
     ↓
     For each device:
       vendor->refresh_dynamic_info(device)
         ↓
         Query hardware:
           - Temperature
           - Power usage
           - Clock speeds
           - Memory usage
           - Utilization
     ↓
[3] gpuinfo_refresh_processes(devices)
     ↓
     Scan /proc/<pid>/fdinfo/* for DRM fds
       ↓
       Parse fdinfo for GPU usage stats
       ↓
       Match to devices
       ↓
       Update process list
     ↓
[4] draw_interface(devices)
     ↓
     Render ncurses UI
```

### 4. Process Discovery (Linux fdinfo)

Modern Linux kernel (5.14+ for AMD, 5.19+ for Intel) exposes GPU usage via fdinfo:

```bash
$ cat /proc/12345/fdinfo/3
pos:    0
flags:  02100002
mnt_id: 21
ino:    1234
drm-driver:     amdgpu
drm-pdev:       0000:03:00.0
drm-client-id:  123
drm-engine-gfx: 1234567890 ns
drm-engine-compute: 9876543210 ns
drm-memory-vram:        512 MiB
drm-memory-gtt: 128 MiB
```

nvtop parses these to get per-process GPU usage.

---

## Implementing Tenstorrent Support

### Approach 1: Plugin for nvtop

Add `extract_gpuinfo_tenstorrent.c` to nvtop (already created above).

**Pros:**
- Unified interface with other GPUs
- Mature UI
- Established user base

**Cons:**
- Tenstorrent-specific features hard to expose
- Requires fdinfo support in tt-kmd
- External project dependency

### Approach 2: Standalone tt-nvtop

Build a Tenstorrent-specific tool modeled after nvtop.

**Current state:** You have `tt_smi.cpp` which is like `nvidia-smi` (snapshot tool).

**Enhancement path:**
1. Add ncurses UI (like nvtop)
2. Add historical charts
3. Add interactive process management
4. Use UMD telemetry APIs

### Approach 3: Hybrid (Recommended)

Keep both:
- **tt-smi**: Lightweight snapshot tool (current implementation)
- **tt-nvtop**: Full interactive monitor (new development)

---

## Enhancing tt-smi with UMD Telemetry

Your current `tt_smi.cpp` uses sysfs for telemetry, which shows "N/A" for many fields.
Let's fix this using TT-UMD APIs.

### Architecture Changes Needed

**Current:**
```
tt_smi.cpp
    ↓
Allocation Server (for memory)
    +
sysfs (/sys/class/tenstorrent/...) (for telemetry)
```

**Enhanced:**
```
tt_smi.cpp
    ↓
Allocation Server (for memory)
    +
TT-UMD APIs (for telemetry)
    ↓
Device firmware (via PCIe)
```

### Implementation Steps

1. **Replace sysfs with UMD in `read_telemetry_from_device()`**

Currently (line 256-260):
```cpp
TelemetryData read_telemetry_from_device(int device_id) {
    TelemetryData data;
    // For now, return empty telemetry
    return data;
}
```

Should become:
```cpp
TelemetryData read_telemetry_from_device(int device_id) {
    TelemetryData data;

    try {
        // Get device via TT-Metal
        auto device = tt::tt_metal::GetDeviceHandle(device_id);
        if (!device) {
            return data;  // Empty on failure
        }

        // Get firmware info provider
        auto fw_info = device->get_firmware_info_provider();
        if (!fw_info) {
            return data;
        }

        // Query telemetry
        data.asic_temperature = fw_info->get_asic_temperature();
        data.board_temperature = fw_info->get_board_temperature();
        data.aiclk = fw_info->get_aiclk();
        data.axiclk = fw_info->get_axiclk();
        data.arcclk = fw_info->get_arcclk();
        data.fan_speed = fw_info->get_fan_speed();
        data.tdp = fw_info->get_tdp();
        data.tdc = fw_info->get_tdc();
        data.vcore = fw_info->get_vcore();

    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to read telemetry for device "
                  << device_id << ": " << e.what() << std::endl;
    }

    return data;
}
```

2. **Add proper device initialization**

```cpp
class TTSmi {
private:
    std::map<int, tt::tt_metal::Device*> device_cache_;

    tt::tt_metal::Device* get_or_create_device(int device_id) {
        if (device_cache_.find(device_id) == device_cache_.end()) {
            // Initialize device for telemetry only (no workload)
            device_cache_[device_id] = tt::tt_metal::CreateDevice(device_id);
        }
        return device_cache_[device_id];
    }

public:
    ~TTSmi() {
        // Clean up devices
        for (auto& [id, device] : device_cache_) {
            if (device) {
                tt::tt_metal::CloseDevice(device);
            }
        }
    }
};
```

3. **Handle device exclusivity**

Problem: TT-Metal devices can only be opened by one process at a time.

Solution: Read-only telemetry access:

```cpp
TelemetryData read_telemetry_from_device(int device_id) {
    TelemetryData data;

    try {
        // Try to open device in read-only mode for telemetry
        // This requires UMD support for non-exclusive access
        auto device = tt::tt_metal::CreateDeviceReadOnly(device_id);

        // ... query telemetry ...

        tt::tt_metal::CloseDevice(device);

    } catch (const std::runtime_error& e) {
        // Device busy (another process owns it)
        // Fall back to sysfs if available
        data = read_telemetry_sysfs(device_id);
    }

    return data;
}
```

---

## Process Discovery for Tenstorrent

### Current State

Your `tt_smi.cpp` scans `/proc/*/fd/` for `/dev/tenstorrent/*` links (lines 152-189).

**Problem:** This only tells you *which* processes have devices open, not:
- GPU utilization per process
- Memory usage per process (requires allocation server)
- Which device a process is using (if multiple)

### Solution: fdinfo Support in tt-kmd

To match nvtop's process tracking, tt-kmd needs to expose fdinfo:

```c
// In tt-kmd (kernel driver)
static void tenstorrent_show_fdinfo(struct seq_file *m, struct file *f) {
    struct tt_device_ctx *ctx = f->private_data;

    seq_printf(m, "drm-driver:\ttenstorrent\n");
    seq_printf(m, "drm-pdev:\t%s\n", pci_name(ctx->pdev));
    seq_printf(m, "drm-client-id:\t%u\n", ctx->client_id);

    // Expose memory usage
    seq_printf(m, "drm-memory-dram:\t%llu KiB\n", ctx->dram_used / 1024);
    seq_printf(m, "drm-memory-l1:\t%llu KiB\n", ctx->l1_used / 1024);

    // Expose utilization (if available)
    seq_printf(m, "drm-engine-compute:\t%llu ns\n", ctx->compute_time_ns);
}

static const struct file_operations tenstorrent_fops = {
    // ...
    .show_fdinfo = tenstorrent_show_fdinfo,
};
```

Then `tt_smi` can read `/proc/<pid>/fdinfo/<fd>`:

```cpp
struct ProcessInfo parse_tt_fdinfo(pid_t pid, int fd) {
    ProcessInfo info;
    std::string path = "/proc/" + std::to_string(pid) + "/fdinfo/" + std::to_string(fd);
    std::ifstream file(path);

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("drm-pdev:") != std::string::npos) {
            // Parse PCI address to determine device_id
        }
        if (line.find("drm-memory-dram:") != std::string::npos) {
            // Parse memory usage
            info.dram_used = parse_memory(line);
        }
    }

    return info;
}
```

---

## Full Implementation Roadmap

### Phase 1: Enhanced tt-smi (Current + UMD) ✓ Mostly Done
- ✅ Process discovery via `/proc`
- ✅ Allocation server integration
- ⚠️  UMD telemetry (partially implemented)
- ❌ fdinfo parsing (needs tt-kmd support)

### Phase 2: Interactive tt-nvtop (New Tool)
- [ ] Port nvtop's ncurses UI
- [ ] Historical data tracking
- [ ] Process sorting/filtering
- [ ] Signal handling (kill processes)
- [ ] Configuration system

### Phase 3: Kernel Driver Integration
- [ ] Add fdinfo to tt-kmd
- [ ] Expose per-process stats
- [ ] Expose device utilization

### Phase 4: Advanced Features
- [ ] Remote device monitoring (UMD already supports this!)
- [ ] Multi-chip mesh visualization
- [ ] Workload profiling integration
- [ ] Alerts/notifications

---

## Comparison Matrix

| Feature | nvidia-smi | nvtop | tt-smi (current) | tt-nvtop (proposed) |
|---------|-----------|-------|------------------|---------------------|
| **UI** | Snapshot | Interactive | Snapshot | Interactive |
| **Device info** | ✅ | ✅ | ✅ | ✅ |
| **Temperature** | ✅ | ✅ | ⚠️ (sysfs) | ✅ (UMD) |
| **Power** | ✅ | ✅ | ⚠️ (sysfs) | ✅ (UMD) |
| **Memory total** | ✅ | ✅ | ✅ (server) | ✅ (server) |
| **Memory per-process** | ✅ | ✅ | ⚠️ (needs work) | ✅ |
| **Process list** | ✅ | ✅ | ✅ | ✅ |
| **GPU utilization** | ✅ | ✅ | ❌ | ✅ (needs kmd) |
| **Historical charts** | ❌ | ✅ | ❌ | ✅ |
| **Multi-vendor** | ❌ | ✅ | ❌ | ❌ |
| **Remote devices** | ❌ | ❌ | ❌ | ✅ (UMD!) |

---

## Next Steps

### Immediate (Today)

1. **Fix telemetry in tt-smi** using UMD APIs
2. **Test on your 4 Blackhole devices**
3. **Verify allocation server integration**

### Short-term (This Week)

1. **Add fdinfo to tt-kmd** (requires kernel driver changes)
2. **Implement per-process memory tracking**
3. **Test multi-device setups**

### Medium-term (This Month)

1. **Start tt-nvtop** based on nvtop's UI code
2. **Add historical data tracking**
3. **Build mesh visualization** (unique to Tenstorrent!)

### Long-term (This Quarter)

1. **Integrate with profiler** (Tracy, etc.)
2. **Add remote monitoring server**
3. **Build web dashboard** (for datacenter monitoring)

---

## Code References

### nvtop Repository Structure
```
nvtop/
├── src/
│   ├── nvtop.c                          # Main entry point
│   ├── extract_gpuinfo.c                # Core abstraction
│   ├── extract_gpuinfo_nvidia.c         # NVIDIA plugin
│   ├── extract_gpuinfo_amdgpu.c         # AMD plugin
│   ├── extract_gpuinfo_intel.c          # Intel plugin
│   ├── extract_processinfo_fdinfo.c     # Process discovery
│   ├── interface.c                      # ncurses UI
│   └── plot.c                           # Historical charts
├── include/nvtop/
│   ├── extract_gpuinfo_common.h         # Vendor interface
│   └── interface.h                      # UI interface
```

### Your Current Structure
```
tt-metal-apv/
└── tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/
    ├── tt_smi.cpp                       # Current implementation
    ├── allocation_server_poc.cpp        # Memory tracking server
    └── TT_SMI_UMD_TELEMETRY_GUIDE.md   # Your guide (very good!)
```

### Proposed Structure
```
tt-metal-apv/tools/monitoring/
├── tt_smi/
│   ├── tt_smi.cpp                       # Snapshot tool (current)
│   ├── tt_telemetry.cpp                 # UMD telemetry wrapper
│   └── CMakeLists.txt
├── tt_nvtop/
│   ├── main.cpp                         # Interactive monitor
│   ├── tt_device_plugin.cpp             # Device interface
│   ├── ui/                              # ncurses UI (port from nvtop)
│   └── CMakeLists.txt
└── common/
    ├── tt_device_info.hpp               # Shared structures
    └── process_discovery.cpp            # Shared process scanning
```

---

## Conclusion

**nvtop's key strengths:**
1. **Modularity**: Vendor plugins make it extensible
2. **Robustness**: Dynamic loading, graceful degradation
3. **Completeness**: Process tracking via fdinfo
4. **UX**: Interactive UI with history

**For Tenstorrent:**
1. **Leverage UMD**: Already supports telemetry for local+remote devices
2. **Extend tt-kmd**: Add fdinfo for per-process tracking
3. **Dual approach**: Keep tt-smi simple, build tt-nvtop for power users
4. **Unique features**: Mesh visualization, remote monitoring

**Your current tt_smi.cpp is a great foundation!** With UMD telemetry integration,
it can already provide more than sysfs-based tools. The next big unlock is
fdinfo support in tt-kmd for per-process GPU tracking.
