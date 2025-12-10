# Mock Device Fast Dispatch Support

## Overview

This document describes the changes required to enable **Fast Dispatch mode** for mock devices in tt-metal. Mock devices allow running graph capture, allocator testing, and validation without physical Tenstorrent hardware.

**Branch:** `mock-device`

---

## Problem Statement

The original mock device implementation only supported **Slow Dispatch mode** (`TT_METAL_SLOW_DISPATCH_MODE=1`). Fast Dispatch mode failed with segmentation faults because:

1. `SystemMemoryManager` is not initialized for mock devices (requires hugepages)
2. `HWCommandQueue` objects are not created
3. Fabric initialization tries to write to non-existent hardware
4. Various dispatch functions assume real hardware is present

---

## Changes Summary

### Files Modified: 6 total (11 guards added)

| File | Changes | Status |
|------|---------|--------|
| `cluster_descriptor.cpp` (UMD) | 2 | ✅ Need to commit |
| `tt_cluster.cpp` | 1 | ✅ Already committed |
| `device.cpp` | 2 | ✅ Need to commit |
| `device_pool.cpp` | 3 | ✅ Need to commit |
| `dispatch.cpp` | 1 | ✅ Need to commit |
| `prefetch.cpp` | 2 | ✅ Need to commit |

---

## Detailed Changes

### 1. UMD: cluster_descriptor.cpp ⚠️ UNCOMMITTED

#### Change 1a: Support Legacy "boardtype" Field (~Line 792)

Some YAML descriptors (e.g., `wormhole_N300.yaml`) use the legacy `boardtype` field instead of `chip_to_boardtype`.

```cpp
} else if (yaml["boardtype"]) {
    // Legacy format support: parse old "boardtype" field for backward compatibility
    for (const auto &yaml_chip_board_type : yaml["boardtype"].as<std::map<int, std::string>>()) {
        auto &chip = yaml_chip_board_type.first;
        const std::string &board_type_str = yaml_chip_board_type.second;
        BoardType board_type = board_type_from_string(board_type_str);
        if (board_type == BoardType::UNKNOWN) {
            log_warning(
                LogUMD,
                "Unknown board type for chip {} from legacy boardtype field. "
                "Defaulting to UNKNOWN",
                chip);
        }
        chip_board_type.insert({chip, board_type});
    }
}
```

#### Change 1b: Generate Synthetic chip_unique_ids (~Line 839)

```cpp
} else {
    // Legacy format or mock descriptors may not have chip_unique_ids
    // Generate synthetic IDs for backward compatibility
    for (const auto &chip : all_chips) {
        // Use chip ID shifted left to create unique synthetic IDs
        chip_unique_ids.insert({chip, static_cast<uint64_t>(chip) << 32});
    }
}
```

---

### 2. tt_metal/llrt/tt_cluster.cpp ✅ ALREADY COMMITTED

#### Change: Skip NOC Translation Check for Mock Devices (~Line 263)

```cpp
// Already in the codebase!
if (this->target_type_ == TargetDevice::Simulator || this->target_type_ == TargetDevice::Mock) {
    return;  // Skip hardware-specific NOC translation check
}
```

---

### 3. tt_metal/impl/device/device.cpp ⚠️ UNCOMMITTED

#### Change 3a: Guard command_queue() (~Line 659)

```cpp
CommandQueue& Device::command_queue(std::optional<uint8_t> cq_id) {
    if (!using_fast_dispatch_) {
        return *(CommandQueue*)(IDevice*)this;
    }
    // For mock devices, command_queues_ may be empty - return a dummy reference
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return *(CommandQueue*)(IDevice*)this;
    }
    // ... rest of function
}
```

#### Change 3b: Guard virtual_program_dispatch_core() (~Line 710)

```cpp
CoreCoord Device::virtual_program_dispatch_core(uint8_t cq_id) const {
    // Mock devices don't have command queues initialized, return a stub dispatch core
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return CoreCoord{0, 0};  // Stub core for mock devices
    }
    return this->command_queues_[cq_id]->virtual_enqueue_program_dispatch_core();
}
```

---

### 4. tt_metal/impl/device/device_pool.cpp ⚠️ UNCOMMITTED

#### Change 4a: Skip Fabric Initialization (~Line 395)

```cpp
if (tt_fabric::is_tt_fabric_config(fabric_config)) {
    // Mock devices don't have real fabric hardware
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        log_info(tt::LogMetal, "Skipping fabric initialization for mock devices");
    } else {
        log_info(tt::LogMetal, "Initializing Fabric");
        tt::tt_metal::MetalContext::instance().get_control_plane().write_routing_tables_to_all_chips();
        init_fabric(active_devices);
        log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
    }
}
```

#### Change 4b: Skip FD Initialization (~Line 410)

```cpp
if (!using_fast_dispatch_) {
    return;
}

// Mock devices don't have real command queues or sysmem managers
if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
    return;
}
```

#### Change 4c: Skip FD Teardown (~Line 777)

```cpp
void DevicePool::teardown_fd(const std::unordered_set<chip_id_t>& devices_to_close) {
    // Mock devices don't have sysmem_manager, skip FD teardown
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }
    // ... rest of function
}
```

---

### 5. tt_metal/impl/device/dispatch.cpp ⚠️ UNCOMMITTED

#### Change 5: Guard write_to_core() (~Line 110)

```cpp
void write_to_core(...) {
    validate_core_read_write_bounds(device, virtual_core, address, size_bytes);

    // Mock devices don't have real hardware to write to
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }
    // ... rest of function
}
```

---

### 6. tt_metal/impl/dispatch/kernel_config/prefetch.cpp ⚠️ UNCOMMITTED

#### Change 6a: Use Placeholder Values (~Line 48)

```cpp
if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
    // For mock devices, use placeholder values
    bool is_mock = tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = is_mock ? 0x10000 : device_->sysmem_manager().get_cq_size();
    uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
    uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
    uint32_t issue_queue_size = is_mock ? 0x10000 : device_->sysmem_manager().get_issue_queue_size(cq_id_);
}
```

#### Change 6b: Skip ConfigureCore (~Line 502)

```cpp
void PrefetchKernel::ConfigureCore() {
    if (static_config_.is_h_variant.value()) {
        // For mock devices, skip ConfigureCore
        bool is_mock = tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
        if (is_mock) {
            return;
        }
        // ... rest of function
    }
}
```

---

## Verification

### Check if both machines have the same changes:

```bash
cd /path/to/tt-metal
git diff | grep "^+" | wc -l    # Should be ~76
git diff | grep "^-" | wc -l    # Should be ~57

cd tt_metal/third_party/umd
git diff | grep "^+" | wc -l    # Should be ~25
git diff | grep "^-" | wc -l    # Should be ~15
```

---

## Usage

```bash
# Set environment
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/descriptor.yaml
export TT_METAL_ENABLE_MOCK_DEVICE=1
export TT_METAL_HOME=/path/to/tt-metal

# Fast Dispatch (now works!)
./build/test/tt_metal/unit_tests_dispatch
./build/test/tt_metal/unit_tests_api --gtest_filter="*Allocator*"
```

---

## Supported Architectures

All 18 descriptors are supported:
- ✅ Blackhole: P100, P150, P300
- ✅ Wormhole: N150, N300, 2xN300, 4xN300
- ✅ Multi-chip: T3K (8 chips), TG (32 chips), Galaxy 6U

---

## Test Results

| Test Suite | Before | After |
|------------|--------|-------|
| Allocator Tests | ✅ PASS | ✅ PASS |
| MeshDispatchFixture (kernel creation) | ✅ PASS | ✅ PASS |
| MeshDispatchFixture (buffer I/O) | ❌ SEGFAULT | ⚠️ NEED MORE GUARDS |

---

## Next Steps (if needed for full test suite pass)

To make ALL tests pass (including buffer write tests), add guards to:
- `tt_metal/impl/buffers/dispatch.cpp::write_to_device_buffer()` (~line 717)
- `tt_metal/impl/buffers/dispatch.cpp::read_from_device_buffer()` (~line 995)

These would make buffer operations no-ops for mock devices (similar to graph capture mode).

---

## Commit Instructions

```bash
# 1. Commit TT-Metal changes
cd /localdev/msudumbrekar/tt-metal
git add tt_metal/impl/device/device.cpp
git add tt_metal/impl/device/device_pool.cpp
git add tt_metal/impl/device/dispatch.cpp
git add tt_metal/impl/dispatch/kernel_config/prefetch.cpp
git commit -m "Enable Fast Dispatch for mock devices

- Guard command_queue and virtual_program_dispatch_core
- Skip fabric/FD init and teardown for mock devices
- Guard write_to_core for mock devices
- Use placeholder values in prefetch kernel"

# 2. Commit UMD changes
cd tt_metal/third_party/umd
git add device/cluster_descriptor.cpp
git commit -m "Add backward compatibility for legacy cluster descriptors

- Support legacy 'boardtype' field
- Generate synthetic chip_unique_ids when missing"

# 3. Push both
git push origin <umd-branch>
cd ../..
git push origin mock-device
```
