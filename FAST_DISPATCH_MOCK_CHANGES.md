# Fast Dispatch Mock Device Support - Changes Checklist

Use this to verify both machines have the same changes before committing.

---

## ✅ FILE 1: tt_metal/third_party/umd/device/cluster_descriptor.cpp

### Change 1a: Support legacy "boardtype" field (~Line 792)
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

### Change 1b: Generate synthetic chip_unique_ids (~Line 839)
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

## ✅ FILE 2: tt_metal/impl/device/device.cpp

### Change 2a: Guard command_queue() (~Line 659)
```cpp
CommandQueue& Device::command_queue(std::optional<uint8_t> cq_id) {
    if (!using_fast_dispatch_) {
        return *(CommandQueue*)(IDevice*)this;
    }
    // For mock devices, command_queues_ may be empty - return a dummy reference
    // Mock devices don't actually dispatch to hardware
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return *(CommandQueue*)(IDevice*)this;
    }
    // ... rest of function
}
```

### Change 2b: Guard virtual_program_dispatch_core() (~Line 710)
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

## ✅ FILE 3: tt_metal/impl/device/device_pool.cpp

### Change 3a: Skip fabric initialization for mock devices (~Line 395)
```cpp
if (tt_fabric::is_tt_fabric_config(fabric_config)) {
    // Mock devices don't have real fabric hardware, skip fabric initialization
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

### Change 3b: Skip FD initialization for mock devices (~Line 410)
```cpp
if (!using_fast_dispatch_) {
    return;
}

// Mock devices don't have real command queues or sysmem managers, skip FD kernel setup
if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
    return;
}
```

### Change 3c: Skip FD teardown for mock devices (~Line 777)
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

## ✅ FILE 4: tt_metal/impl/device/dispatch.cpp

### Change 4: Guard write_to_core() (~Line 110)
```cpp
void write_to_core(
    IDevice* device,
    const CoreCoord& virtual_core,
    uint64_t address,
    const void* data,
    size_t size_bytes,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    validate_core_read_write_bounds(device, virtual_core, address, size_bytes);

    // Mock devices don't have real hardware to write to, skip actual dispatch
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }
    // ... rest of function
}
```

---

## ✅ FILE 5: tt_metal/impl/dispatch/kernel_config/prefetch.cpp

### Change 5a: Use placeholder values for mock devices (~Line 48)
```cpp
if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
    // For mock devices, sysmem_manager is not initialized, so use placeholder values
    bool is_mock = tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
    uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    uint32_t cq_size = is_mock ? 0x10000 : device_->sysmem_manager().get_cq_size();
    uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id_, cq_size);
    uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
    uint32_t issue_queue_size = is_mock ? 0x10000 : device_->sysmem_manager().get_issue_queue_size(cq_id_);
    // ... rest of function
}
```

### Change 5b: Skip ConfigureCore for mock devices (~Line 502)
```cpp
void PrefetchKernel::ConfigureCore() {
    if (static_config_.is_h_variant.value()) {
        // For mock devices, skip ConfigureCore as it writes to device L1 which isn't available
        bool is_mock = tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
        if (is_mock) {
            return;
        }
        // ... rest of function
    }
}
```

---

## 📊 Quick Verification

Run this on BOTH machines:

```bash
cd /path/to/tt-metal
git diff | grep "^+" | wc -l    # Should be ~76
git diff | grep "^-" | wc -l    # Should be ~57

cd tt_metal/third_party/umd
git diff | grep "^+" | wc -l    # Should be ~25
git diff | grep "^-" | wc -l    # Should be ~15
```

If numbers match = changes are identical! ✅

---

## 📝 Summary

**5 files modified:**
1. cluster_descriptor.cpp (UMD) - 2 changes
2. device.cpp - 2 changes
3. device_pool.cpp - 3 changes
4. dispatch.cpp - 1 change
5. prefetch.cpp - 2 changes

**Total: 10 mock device guards added**
