// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_stats_shm.hpp"
#include <tt-logger/tt-logger.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>

namespace tt::tt_metal {

// Implementation of SharedMemoryStatsProvider

SharedMemoryStatsProvider::SharedMemoryStatsProvider(uint64_t asic_id, int device_id) :
    asic_id_(asic_id),
    device_id_(device_id),
    shm_fd_(-1),
    region_(nullptr),
    per_pid_tracking_enabled_(false)  // Disabled by default for performance
    ,
    is_creator_(false) {
    // Create shared memory name using composite asic_id
    // Format: /tt_device_<asic_id>_memory
    // asic_id = (board_id << 8) | asic_location is globally unique and never changes
    std::string shm_name = "/tt_device_" + std::to_string(asic_id) + "_memory";

    // Try exclusive create first to see if we're the first
    shm_fd_ = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd_ != -1) {
        is_creator_ = true;
    } else if (errno == EEXIST) {
        // Already exists, just open it
        shm_fd_ = shm_open(shm_name.c_str(), O_RDWR, 0666);
        is_creator_ = false;
    }

    if (shm_fd_ == -1) {
        std::cerr << "Warning: Failed to create shared memory " << shm_name << ": " << strerror(errno) << std::endl;
        return;
    }

    // Set size (this is idempotent - won't shrink if already larger)
    if (ftruncate(shm_fd_, sizeof(DeviceMemoryRegion)) == -1) {
        std::cerr << "Warning: Failed to set shared memory size: " << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return;
    }

    // Map into address space
    region_ = static_cast<DeviceMemoryRegion*>(
        mmap(nullptr, sizeof(DeviceMemoryRegion), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0));

    if (region_ == MAP_FAILED) {
        std::cerr << "Warning: Failed to mmap shared memory: " << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        region_ = nullptr;
        return;
    }

    // Initialize if we're the creator
    if (is_creator_) {
        initialize_region();
    }

    // Increment reference count (this process is now attached)
    region_->reference_count.fetch_add(1, std::memory_order_relaxed);

    // ALWAYS update identifiers, even when reattaching to existing SHM
    // Store composite asic_id for identification
    region_->board_serial = asic_id_ >> 8;  // Extract board_id
    region_->asic_id = asic_id_ & 0xFF;     // Extract asic_location
    region_->device_id = device_id_;

    // Check if per-PID tracking should be enabled
    const char* per_pid_env = std::getenv("TT_METAL_SHM_STATS_PER_PID");
    if (per_pid_env && std::string(per_pid_env) == "1") {
        per_pid_tracking_enabled_ = true;
    }
}

SharedMemoryStatsProvider::~SharedMemoryStatsProvider() {
    if (region_ != nullptr && region_ != MAP_FAILED) {
        pid_t my_pid = getpid();

        // Remove our PID entry if per-PID tracking was enabled
        if (per_pid_tracking_enabled_) {
            for (size_t i = 0; i < MAX_PROCESSES; i++) {
                if (region_->processes[i].pid == my_pid) {
                    // IMPORTANT: Subtract this process's allocations from aggregated totals
                    // before clearing the entry (to properly track memory when process exits)
                    uint64_t dram = region_->processes[i].dram_allocated;
                    uint64_t l1 = region_->processes[i].l1_allocated;
                    uint64_t l1_small = region_->processes[i].l1_small_allocated;
                    uint64_t trace = region_->processes[i].trace_allocated;
                    uint64_t cb = region_->processes[i].cb_allocated;
                    uint64_t kernel = region_->processes[i].kernel_allocated;

                    // Subtract from aggregated totals with underflow protection
                    if (region_->total_dram_allocated >= dram) {
                        region_->total_dram_allocated.fetch_sub(dram, std::memory_order_relaxed);
                    } else {
                        region_->total_dram_allocated.store(0, std::memory_order_relaxed);
                    }

                    if (region_->total_l1_allocated >= l1) {
                        region_->total_l1_allocated.fetch_sub(l1, std::memory_order_relaxed);
                    } else {
                        region_->total_l1_allocated.store(0, std::memory_order_relaxed);
                    }

                    if (region_->total_l1_small_allocated >= l1_small) {
                        region_->total_l1_small_allocated.fetch_sub(l1_small, std::memory_order_relaxed);
                    } else {
                        region_->total_l1_small_allocated.store(0, std::memory_order_relaxed);
                    }

                    if (region_->total_trace_allocated >= trace) {
                        region_->total_trace_allocated.fetch_sub(trace, std::memory_order_relaxed);
                    } else {
                        region_->total_trace_allocated.store(0, std::memory_order_relaxed);
                    }

                    if (region_->total_cb_allocated >= cb) {
                        region_->total_cb_allocated.fetch_sub(cb, std::memory_order_relaxed);
                    } else {
                        region_->total_cb_allocated.store(0, std::memory_order_relaxed);
                    }

                    if (region_->total_kernel_allocated >= kernel) {
                        region_->total_kernel_allocated.fetch_sub(kernel, std::memory_order_relaxed);
                    } else {
                        region_->total_kernel_allocated.store(0, std::memory_order_relaxed);
                    }

                    // Now clear this PID's entry
                    region_->processes[i].pid = 0;
                    region_->processes[i].dram_allocated = 0;
                    region_->processes[i].l1_allocated = 0;
                    region_->processes[i].l1_small_allocated = 0;
                    region_->processes[i].trace_allocated = 0;
                    region_->processes[i].cb_allocated = 0;
                    region_->processes[i].kernel_allocated = 0;
                    region_->num_active_processes--;
                    break;
                }
            }
        }

        // Decrement reference count (but DON'T delete SHM - keep it persistent like UMD locks)
        uint32_t prev_refcount = region_->reference_count.fetch_sub(1, std::memory_order_acq_rel);

        // If this was the last process, reset all stats to ensure cleanup
        // This handles cases where explicit deallocation wasn't called
        if (prev_refcount == 1) {  // We just decremented to 0
            log_debug(
                tt::LogMetal,
                "Device {}: Last process exiting, cleaning up all stats (refcount {} → 0)",
                device_id_,
                prev_refcount);

            // Reset aggregated totals
            region_->total_dram_allocated.store(0, std::memory_order_relaxed);
            region_->total_l1_allocated.store(0, std::memory_order_relaxed);
            region_->total_l1_small_allocated.store(0, std::memory_order_relaxed);
            region_->total_trace_allocated.store(0, std::memory_order_relaxed);
            region_->total_cb_allocated.store(0, std::memory_order_relaxed);
            region_->total_kernel_allocated.store(0, std::memory_order_relaxed);

            // Reset per-chip stats
            for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
                if (region_->chip_stats[i].chip_id != 0) {
                    region_->chip_stats[i].dram_allocated.store(0, std::memory_order_relaxed);
                    region_->chip_stats[i].l1_allocated.store(0, std::memory_order_relaxed);
                    region_->chip_stats[i].l1_small_allocated.store(0, std::memory_order_relaxed);
                    region_->chip_stats[i].trace_allocated.store(0, std::memory_order_relaxed);
                    region_->chip_stats[i].cb_allocated.store(0, std::memory_order_relaxed);
                    region_->chip_stats[i].kernel_allocated.store(0, std::memory_order_relaxed);
                }
            }
        } else {
            log_debug(
                tt::LogMetal,
                "Device {}: Process exiting, refcount {} → {}",
                device_id_,
                prev_refcount,
                prev_refcount - 1);
        }

        munmap(region_, sizeof(DeviceMemoryRegion));
        region_ = nullptr;
    }

    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }

    // NOTE: We intentionally do NOT call shm_unlink() here
    // SHM files persist like UMD lock files (/dev/shm/TT_UMD_LOCK.*)
    // Benefits:
    //   1. Monitoring tools see device state even when idle (0 allocations is meaningful!)
    //   2. Consistent with UMD semantics
    //   3. Files show "last known state" until next run
    // Cleanup: Manually remove with: rm /dev/shm/tt_device_*
}

void SharedMemoryStatsProvider::initialize_region() {
    if (!region_) {
        return;
    }

    // Set version
    region_->version = 1;
    region_->num_active_processes = 0;
    region_->last_update_timestamp = current_timestamp_ns();
    region_->reference_count.store(0, std::memory_order_relaxed);

    // Set physical chip identification (for proper device correlation)
    region_->board_serial = asic_id_ >> 8;  // Extract board_id
    region_->asic_id = asic_id_ & 0xFF;     // Extract asic_location
    region_->device_id = device_id_;

    // Initialize atomic counters to zero
    region_->total_dram_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_allocated.store(0, std::memory_order_relaxed);
    region_->total_l1_small_allocated.store(0, std::memory_order_relaxed);
    region_->total_trace_allocated.store(0, std::memory_order_relaxed);
    region_->total_cb_allocated.store(0, std::memory_order_relaxed);
    region_->total_kernel_allocated.store(0, std::memory_order_relaxed);

    // Initialize per-chip entries (for remote device tracking)
    for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
        region_->chip_stats[i].chip_id = 0;  // 0 = unused
        region_->chip_stats[i].is_remote = 0;
        region_->chip_stats[i].dram_allocated.store(0, std::memory_order_relaxed);
        region_->chip_stats[i].l1_allocated.store(0, std::memory_order_relaxed);
        region_->chip_stats[i].l1_small_allocated.store(0, std::memory_order_relaxed);
        region_->chip_stats[i].trace_allocated.store(0, std::memory_order_relaxed);
        region_->chip_stats[i].cb_allocated.store(0, std::memory_order_relaxed);
        region_->chip_stats[i].kernel_allocated.store(0, std::memory_order_relaxed);
    }

    // Register the gateway chip itself (chip_id = device_id, is_remote = false)
    region_->chip_stats[0].chip_id = device_id_;
    region_->chip_stats[0].is_remote = 0;

    // Clear per-process entries
    for (size_t i = 0; i < MAX_PROCESSES; i++) {
        region_->processes[i].pid = 0;  // 0 = unused
        region_->processes[i].dram_allocated = 0;
        region_->processes[i].l1_allocated = 0;
        region_->processes[i].l1_small_allocated = 0;
        region_->processes[i].trace_allocated = 0;
        region_->processes[i].cb_allocated = 0;
        region_->processes[i].kernel_allocated = 0;
        region_->processes[i].last_update_timestamp = 0;
        std::memset(region_->processes[i].process_name, 0, 64);
    }
}

void SharedMemoryStatsProvider::record_allocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id) {
    if (!region_) {
        return;
    }

    // Update aggregated counters (always - this is the fast path)
    switch (type) {
        case ShmBufferType::DRAM: region_->total_dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::L1: region_->total_l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::L1_SMALL:
            region_->total_l1_small_allocated.fetch_add(size, std::memory_order_relaxed);
            break;
        case ShmBufferType::TRACE: region_->total_trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::CB: region_->total_cb_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case ShmBufferType::KERNEL: region_->total_kernel_allocated.fetch_add(size, std::memory_order_relaxed); break;
        default: break;
    }

    // Update per-chip counters (for remote device tracking)
    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        switch (type) {
            case ShmBufferType::DRAM: chip_entry->dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::L1: chip_entry->l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::L1_SMALL:
                chip_entry->l1_small_allocated.fetch_add(size, std::memory_order_relaxed);
                break;
            case ShmBufferType::TRACE: chip_entry->trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::CB: chip_entry->cb_allocated.fetch_add(size, std::memory_order_relaxed); break;
            case ShmBufferType::KERNEL: chip_entry->kernel_allocated.fetch_add(size, std::memory_order_relaxed); break;
            default: break;
        }
    }

    // Update timestamp
    region_->last_update_timestamp = current_timestamp_ns();

    // Update per-PID stats if enabled
    if (per_pid_tracking_enabled_) {
        auto* pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry) {
            switch (type) {
                case ShmBufferType::DRAM: pid_entry->dram_allocated += size; break;
                case ShmBufferType::L1: pid_entry->l1_allocated += size; break;
                case ShmBufferType::L1_SMALL: pid_entry->l1_small_allocated += size; break;
                case ShmBufferType::TRACE: pid_entry->trace_allocated += size; break;
                case ShmBufferType::CB: pid_entry->cb_allocated += size; break;
                case ShmBufferType::KERNEL: pid_entry->kernel_allocated += size; break;
                default: break;
            }
            pid_entry->last_update_timestamp = current_timestamp_ns();
        }
    }
}

void SharedMemoryStatsProvider::record_deallocation(pid_t pid, uint64_t size, ShmBufferType type, uint32_t chip_id) {
    if (!region_) {
        return;
    }

    // Update aggregated counters with underflow protection
    // Note: We use compare-and-swap loop to prevent underflow
    auto safe_sub = [](std::atomic<uint64_t>& counter, uint64_t size) {
        uint64_t current = counter.load(std::memory_order_relaxed);
        uint64_t new_val;
        do {
            if (current < size) {
                // Underflow would occur - clamp to 0
                new_val = 0;
            } else {
                new_val = current - size;
            }
        } while (
            !counter.compare_exchange_weak(current, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
    };

    switch (type) {
        case ShmBufferType::DRAM: safe_sub(region_->total_dram_allocated, size); break;
        case ShmBufferType::L1: safe_sub(region_->total_l1_allocated, size); break;
        case ShmBufferType::L1_SMALL: safe_sub(region_->total_l1_small_allocated, size); break;
        case ShmBufferType::TRACE: safe_sub(region_->total_trace_allocated, size); break;
        case ShmBufferType::CB: safe_sub(region_->total_cb_allocated, size); break;
        case ShmBufferType::KERNEL: safe_sub(region_->total_kernel_allocated, size); break;
        default: break;
    }

    // Update per-chip counters (with underflow protection)
    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        switch (type) {
            case ShmBufferType::DRAM: safe_sub(chip_entry->dram_allocated, size); break;
            case ShmBufferType::L1: safe_sub(chip_entry->l1_allocated, size); break;
            case ShmBufferType::L1_SMALL: safe_sub(chip_entry->l1_small_allocated, size); break;
            case ShmBufferType::TRACE: safe_sub(chip_entry->trace_allocated, size); break;
            case ShmBufferType::CB: safe_sub(chip_entry->cb_allocated, size); break;
            case ShmBufferType::KERNEL: safe_sub(chip_entry->kernel_allocated, size); break;
            default: break;
        }
    }

    // Update timestamp
    region_->last_update_timestamp = current_timestamp_ns();

    // Update per-PID stats if enabled (with underflow protection)
    if (per_pid_tracking_enabled_) {
        auto* pid_entry = find_or_create_pid_entry(pid);
        if (pid_entry) {
            switch (type) {
                case ShmBufferType::DRAM:
                    if (pid_entry->dram_allocated >= size) {
                        pid_entry->dram_allocated -= size;
                    } else {
                        pid_entry->dram_allocated = 0;
                    }
                    break;
                case ShmBufferType::L1:
                    if (pid_entry->l1_allocated >= size) {
                        pid_entry->l1_allocated -= size;
                    } else {
                        pid_entry->l1_allocated = 0;
                    }
                    break;
                case ShmBufferType::L1_SMALL:
                    if (pid_entry->l1_small_allocated >= size) {
                        pid_entry->l1_small_allocated -= size;
                    } else {
                        pid_entry->l1_small_allocated = 0;
                    }
                    break;
                case ShmBufferType::TRACE:
                    if (pid_entry->trace_allocated >= size) {
                        pid_entry->trace_allocated -= size;
                    } else {
                        pid_entry->trace_allocated = 0;
                    }
                    break;
                case ShmBufferType::CB:
                    if (pid_entry->cb_allocated >= size) {
                        pid_entry->cb_allocated -= size;
                    } else {
                        pid_entry->cb_allocated = 0;
                    }
                    break;
                case ShmBufferType::KERNEL:
                    if (pid_entry->kernel_allocated >= size) {
                        pid_entry->kernel_allocated -= size;
                    } else {
                        pid_entry->kernel_allocated = 0;
                    }
                    break;
                default: break;
            }
            pid_entry->last_update_timestamp = current_timestamp_ns();
        }
    }
}

SharedMemoryStatsProvider::DeviceStats SharedMemoryStatsProvider::get_device_stats() const {
    if (!region_) {
        return {0, 0, 0, 0, 0, 0, 0};
    }

    return {
        region_->total_dram_allocated.load(std::memory_order_relaxed),
        region_->total_l1_allocated.load(std::memory_order_relaxed),
        region_->total_l1_small_allocated.load(std::memory_order_relaxed),
        region_->total_trace_allocated.load(std::memory_order_relaxed),
        region_->total_cb_allocated.load(std::memory_order_relaxed),
        region_->total_kernel_allocated.load(std::memory_order_relaxed),
        region_->last_update_timestamp};
}

std::vector<SharedMemoryStatsProvider::ProcessInfo> SharedMemoryStatsProvider::get_process_stats() const {
    std::vector<ProcessInfo> result;

    if (!region_) {
        return result;
    }

    for (size_t i = 0; i < MAX_PROCESSES; i++) {
        if (region_->processes[i].pid != 0) {
            ProcessInfo info;
            info.pid = region_->processes[i].pid;
            info.dram_allocated = region_->processes[i].dram_allocated;
            info.l1_allocated = region_->processes[i].l1_allocated;
            info.l1_small_allocated = region_->processes[i].l1_small_allocated;
            info.trace_allocated = region_->processes[i].trace_allocated;
            info.cb_allocated = region_->processes[i].cb_allocated;
            info.kernel_allocated = region_->processes[i].kernel_allocated;
            info.timestamp = region_->processes[i].last_update_timestamp;
            info.process_name = std::string(region_->processes[i].process_name);
            result.push_back(info);
        }
    }

    return result;
}

DeviceMemoryRegion::ProcessStats* SharedMemoryStatsProvider::find_or_create_pid_entry(pid_t pid) {
    if (!region_) {
        return nullptr;
    }

    // First, try to find existing entry
    for (size_t i = 0; i < MAX_PROCESSES; i++) {
        if (region_->processes[i].pid == pid) {
            return &region_->processes[i];
        }
    }

    // Not found, create new entry
    for (size_t i = 0; i < MAX_PROCESSES; i++) {
        if (region_->processes[i].pid == 0) {
            region_->processes[i].pid = pid;
            region_->processes[i].dram_allocated = 0;
            region_->processes[i].l1_allocated = 0;
            region_->processes[i].l1_small_allocated = 0;
            region_->processes[i].trace_allocated = 0;
            region_->processes[i].cb_allocated = 0;
            region_->processes[i].kernel_allocated = 0;
            region_->processes[i].last_update_timestamp = current_timestamp_ns();

            // Get process name
            std::string proc_name = get_process_name(pid);
            strncpy(region_->processes[i].process_name, proc_name.c_str(), 63);
            region_->processes[i].process_name[63] = '\0';

            region_->num_active_processes++;
            return &region_->processes[i];
        }
    }

    // No free slots
    return nullptr;
}

uint64_t SharedMemoryStatsProvider::current_timestamp_ns() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

std::string SharedMemoryStatsProvider::get_process_name(pid_t pid) {
    std::string path = "/proc/" + std::to_string(pid) + "/comm";
    std::ifstream file(path);
    if (!file) {
        return "unknown";
    }

    std::string name;
    std::getline(file, name);
    return name;
}

DeviceMemoryRegion::ChipStats* SharedMemoryStatsProvider::find_or_create_chip_entry(uint32_t chip_id) {
    if (!region_) {
        return nullptr;
    }

    // First, try to find existing entry
    for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
        if (region_->chip_stats[i].chip_id == chip_id) {
            return &region_->chip_stats[i];
        }
    }

    // Not found, create new entry
    for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
        if (region_->chip_stats[i].chip_id == 0) {
            region_->chip_stats[i].chip_id = chip_id;
            region_->chip_stats[i].is_remote = 0;  // Will be set by register_chip if needed
            region_->chip_stats[i].dram_allocated.store(0, std::memory_order_relaxed);
            region_->chip_stats[i].l1_allocated.store(0, std::memory_order_relaxed);
            region_->chip_stats[i].l1_small_allocated.store(0, std::memory_order_relaxed);
            region_->chip_stats[i].trace_allocated.store(0, std::memory_order_relaxed);
            region_->chip_stats[i].cb_allocated.store(0, std::memory_order_relaxed);
            region_->chip_stats[i].kernel_allocated.store(0, std::memory_order_relaxed);
            return &region_->chip_stats[i];
        }
    }

    // No free slots
    return nullptr;
}

void SharedMemoryStatsProvider::register_chip(uint32_t chip_id, bool is_remote) {
    if (!region_) {
        return;
    }

    auto* chip_entry = find_or_create_chip_entry(chip_id);
    if (chip_entry) {
        chip_entry->is_remote = is_remote ? 1 : 0;
    }
}

std::vector<SharedMemoryStatsProvider::ChipInfo> SharedMemoryStatsProvider::get_chip_stats() const {
    std::vector<ChipInfo> result;
    if (!region_) {
        return result;
    }

    for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
        if (region_->chip_stats[i].chip_id != 0) {
            ChipInfo info;
            info.chip_id = region_->chip_stats[i].chip_id;
            info.is_remote = (region_->chip_stats[i].is_remote != 0);
            info.dram_allocated = region_->chip_stats[i].dram_allocated.load(std::memory_order_relaxed);
            info.l1_allocated = region_->chip_stats[i].l1_allocated.load(std::memory_order_relaxed);
            info.l1_small_allocated = region_->chip_stats[i].l1_small_allocated.load(std::memory_order_relaxed);
            info.trace_allocated = region_->chip_stats[i].trace_allocated.load(std::memory_order_relaxed);
            info.cb_allocated = region_->chip_stats[i].cb_allocated.load(std::memory_order_relaxed);
            info.kernel_allocated = region_->chip_stats[i].kernel_allocated.load(std::memory_order_relaxed);
            result.push_back(info);
        }
    }

    return result;
}

}  // namespace tt::tt_metal
