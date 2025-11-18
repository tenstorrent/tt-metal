# Full L1 Memory Tracking Implementation Guide

## Goal
Track **all** L1 memory usage including:
- ✅ Explicit buffer allocations (already tracked)
- ❌ Circular buffers (NOT tracked)
- ❌ Kernel code (NOT tracked)
- ❌ Runtime/Firmware (NOT tracked)

## Implementation Strategy

### Approach 1: Hook into Program Compilation (Recommended)

This intercepts kernel and CB creation to track their memory usage.

#### Step 1: Extend AllocMessage Protocol

Update `AllocMessage` struct to include CB and kernel memory:

```cpp
// In allocation_server_poc.cpp and tt_smi_umd.cpp
struct __attribute__((packed)) AllocMessage {
    // ... existing fields ...

    // NEW: Circular buffer tracking
    uint64_t cb_allocated;           // Total CB memory allocated
    uint64_t kernel_code_allocated;  // Total kernel code memory
    uint64_t firmware_allocated;     // Estimated firmware overhead

    // NEW: Per-core breakdown (optional, for detailed view)
    uint32_t num_cores;
    uint64_t cb_per_core[80];       // Max 80 cores for WH
};
```

#### Step 2: Hook Program::add_kernel()

Modify `tt_metal/impl/program/program.cpp`:

```cpp
// In Program::add_kernel()
KernelHandle Program::add_kernel(
    const std::shared_ptr<Kernel> &kernel,
    const CoreType &core_type
) {
    // ... existing code ...

    // NEW: Track kernel code size
    size_t kernel_code_size = kernel->compute_binary().size()
                            + kernel->data_movement_binary().size();

    // Report to allocation server
    if (allocation_tracking_enabled()) {
        report_kernel_allocation(
            this->device_->id(),
            kernel_code_size,
            core_type
        );
    }

    return kernel_handle;
}
```

#### Step 3: Hook CircularBuffer Creation

In `tt_metal/impl/buffers/circular_buffer.cpp`:

```cpp
// In CreateCircularBuffer()
CircularBuffer CreateCircularBuffer(
    Device *device,
    const std::unordered_map<uint8_t, tt::DataFormat> &data_format_spec,
    uint32_t size
) {
    // ... existing code ...

    // NEW: Track CB allocation
    if (allocation_tracking_enabled()) {
        report_cb_allocation(
            device->id(),
            size,
            buffer_index,
            core_range
        );
    }

    return cb;
}
```

#### Step 4: Add Tracking Functions

Create `tt_metal/impl/allocation_tracker.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <sys/socket.h>
#include <sys/un.h>

namespace tt::tt_metal::allocation_tracker {

// Enable/disable tracking
bool is_tracking_enabled();
void enable_tracking();
void disable_tracking();

// Report allocations to server
void report_kernel_allocation(
    int device_id,
    uint64_t size,
    CoreType core_type
);

void report_cb_allocation(
    int device_id,
    uint64_t size,
    uint32_t buffer_index,
    const CoreRange& cores
);

void report_cb_free(
    int device_id,
    uint64_t size,
    uint32_t buffer_index
);

} // namespace
```

Implementation in `tt_metal/impl/allocation_tracker.cpp`:

```cpp
#include "allocation_tracker.hpp"
#include <atomic>
#include <unistd.h>
#include <cstring>

namespace tt::tt_metal::allocation_tracker {

namespace {
    std::atomic<bool> tracking_enabled{false};
    int socket_fd = -1;
    const char* SOCKET_PATH = "/tmp/tt_allocation_server.sock";
}

bool is_tracking_enabled() {
    return tracking_enabled.load();
}

void enable_tracking() {
    if (tracking_enabled.load()) return;

    // Connect to allocation server
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) return;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        tracking_enabled.store(true);
    } else {
        close(socket_fd);
        socket_fd = -1;
    }
}

void disable_tracking() {
    if (!tracking_enabled.load()) return;

    if (socket_fd >= 0) {
        close(socket_fd);
        socket_fd = -1;
    }
    tracking_enabled.store(false);
}

void report_kernel_allocation(
    int device_id,
    uint64_t size,
    CoreType core_type
) {
    if (!tracking_enabled.load() || socket_fd < 0) return;

    // Create message
    struct KernelAllocMessage {
        uint8_t type = 8;  // New message type
        int32_t device_id;
        uint64_t size;
        uint8_t core_type;
    } __attribute__((packed));

    KernelAllocMessage msg;
    msg.device_id = device_id;
    msg.size = size;
    msg.core_type = static_cast<uint8_t>(core_type);

    // Send to server
    send(socket_fd, &msg, sizeof(msg), 0);
}

void report_cb_allocation(
    int device_id,
    uint64_t size,
    uint32_t buffer_index,
    const CoreRange& cores
) {
    if (!tracking_enabled.load() || socket_fd < 0) return;

    // Create message
    struct CBAllocMessage {
        uint8_t type = 9;  // New message type
        int32_t device_id;
        uint64_t size;
        uint32_t buffer_index;
        uint32_t num_cores;
    } __attribute__((packed));

    CBAllocMessage msg;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_index = buffer_index;
    msg.num_cores = cores.size();

    // Send to server
    send(socket_fd, &msg, sizeof(msg), 0);
}

void report_cb_free(
    int device_id,
    uint64_t size,
    uint32_t buffer_index
) {
    if (!tracking_enabled.load() || socket_fd < 0) return;

    struct CBFreeMessage {
        uint8_t type = 10;  // New message type
        int32_t device_id;
        uint64_t size;
        uint32_t buffer_index;
    } __attribute__((packed));

    CBFreeMessage msg;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_index = buffer_index;

    send(socket_fd, &msg, sizeof(msg), 0);
}

} // namespace
```

#### Step 5: Auto-enable in Device Creation

In `tt_metal/impl/device/device.cpp`:

```cpp
#include "allocation_tracker.hpp"

Device::Device(...) {
    // ... existing initialization ...

    // Auto-enable L1 tracking if server is available
    allocation_tracker::enable_tracking();
}

Device::~Device() {
    // Cleanup
    allocation_tracker::disable_tracking();
}
```

#### Step 6: Update Allocation Server

Modify `allocation_server_poc.cpp` to handle new message types:

```cpp
// Add tracking for CB and kernel memory
std::map<int, uint64_t> device_cb_memory;
std::map<int, uint64_t> device_kernel_memory;

void handle_client_message(int client_fd, const char* buffer, size_t len) {
    uint8_t msg_type = buffer[0];

    if (msg_type == 8) {  // Kernel allocation
        KernelAllocMessage* msg = (KernelAllocMessage*)buffer;
        device_kernel_memory[msg->device_id] += msg->size;

    } else if (msg_type == 9) {  // CB allocation
        CBAllocMessage* msg = (CBAllocMessage*)buffer;
        device_cb_memory[msg->device_id] += msg->size;

    } else if (msg_type == 10) {  // CB free
        CBFreeMessage* msg = (CBFreeMessage*)buffer;
        device_cb_memory[msg->device_id] -= msg->size;

    } else {
        // ... existing handlers ...
    }
}

// Update DEVICE_INFO_RESPONSE to include CB and kernel memory
AllocMessage create_device_info_response(int device_id) {
    AllocMessage response;
    // ... existing fields ...
    response.cb_allocated = device_cb_memory[device_id];
    response.kernel_code_allocated = device_kernel_memory[device_id];
    response.firmware_allocated = 2 * 1024 * 1024;  // Estimate 2MB
    return response;
}
```

### Approach 2: Use Memory Reporter API (Simpler, Less Accurate)

Use the existing `MemoryReporter` API which has some visibility into L1 usage:

```cpp
#include <tt-metalium/memory_reporter.hpp>

// In your application or allocation server
void report_full_l1_usage(Device* device) {
    auto reporter = tt::tt_metal::MemoryReporter(device);

    // Get memory report
    auto report = reporter.generate_report();

    // Parse report for L1 details
    // Note: This shows allocated banks but not detailed CB breakdown
    std::cout << "L1 Banks Allocated: " << report.l1_banks_allocated << std::endl;
    std::cout << "L1 Usage: " << report.l1_usage_bytes << std::endl;
}
```

**Limitation**: MemoryReporter shows high-level stats but doesn't break down CBs vs kernel code vs allocator-managed buffers.

### Approach 3: Parse Device State Dump (Quick & Dirty)

Use `DumpDeviceMemoryState()` which prints detailed per-core information:

```cpp
// In tt_metal/impl/debug/dprint.hpp
void DumpDeviceMemoryState(Device* device, std::ostream& os);

// Usage:
std::stringstream ss;
DumpDeviceMemoryState(device, ss);

// Parse output to extract CB sizes per core
// Format: "Core (x,y): CB0=1024KB, CB1=512KB, ..."
std::string output = ss.str();
// ... parse and sum ...
```

**Limitation**: Requires parsing text output, not structured data.

## Which Approach to Use?

### For Production: Approach 1 (Full Hooks)
- ✅ Accurate tracking of all L1 memory
- ✅ Real-time updates
- ✅ Integrated with allocation server
- ❌ Requires modifying TT-Metal core files
- ❌ ~500-1000 lines of code

### For Quick Analysis: Approach 3 (Dump Parsing)
- ✅ No code changes to TT-Metal
- ✅ Works today
- ❌ Not real-time
- ❌ Requires manual invocation
- ❌ Text parsing is fragile

### Hybrid Approach (Recommended for Now)
1. Use **MemoryReporter API** for high-level L1 usage
2. Call **DumpDeviceMemoryState** periodically for detailed breakdown
3. Display both in `tt_smi_umd`:
   - Allocator-tracked (existing)
   - Total L1 usage from MemoryReporter
   - Inferred CB usage = Total - Allocator

```cpp
// In tt_smi_umd.cpp
void display_full_l1_tracking(Device* device) {
    // 1. Allocator-tracked (existing)
    uint64_t allocator_l1 = get_allocator_l1_usage();

    // 2. Total L1 from MemoryReporter
    auto reporter = MemoryReporter(device);
    uint64_t total_l1 = reporter.get_l1_usage();

    // 3. Inferred CB + kernel usage
    uint64_t cb_and_kernel = total_l1 - allocator_l1;

    std::cout << "L1 Breakdown:" << std::endl;
    std::cout << "  Allocator:   " << allocator_l1 / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  CB+Kernel:   " << cb_and_kernel / 1024 / 1024 << " MB (estimated)" << std::endl;
    std::cout << "  Total:       " << total_l1 / 1024 / 1024 << " MB" << std::endl;
}
```

## Implementation Priority

1. **Phase 1** (Easy, Today): Add MemoryReporter integration to `tt_smi_umd`
2. **Phase 2** (Medium): Parse DumpDeviceMemoryState for detailed CB breakdown
3. **Phase 3** (Hard, Later): Full hooking into Program/CB creation (Approach 1)

## Files to Modify

### Phase 1 (MemoryReporter Integration)
- `tt_smi_umd.cpp`: Add MemoryReporter calls
- No TT-Metal core changes needed!

### Phase 3 (Full Tracking)
TT-Metal core files:
- `tt_metal/impl/program/program.cpp`: Hook add_kernel()
- `tt_metal/impl/buffers/circular_buffer.cpp`: Hook CreateCircularBuffer()
- `tt_metal/impl/device/device.cpp`: Auto-enable tracking
- `tt_metal/impl/allocation_tracker.hpp/cpp`: New tracking module

Monitoring tools:
- `allocation_server_poc.cpp`: Handle new message types
- `tt_smi_umd.cpp`: Display CB and kernel memory

## Next Steps

Would you like me to implement **Phase 1** (MemoryReporter integration) now? This would show you:
- Total L1 usage (including CBs and kernels)
- Allocator-tracked L1 (explicit buffers)
- Estimated CB+Kernel usage (difference)

This requires NO changes to TT-Metal core, just adding API calls in `tt_smi_umd`!
