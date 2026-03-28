// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "impl/context/metal_context.hpp"
#include "system_memory_manager.hpp"
#include <tt-metalium/tt_align.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <thread>
#include <string>
#include <tuple>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "memcpy.hpp"
#include "allocator/allocator.hpp"
#include "command_queue_common.hpp"
#include "system_memory_cq_interface.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/tt_io.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>

// Doorbell delay injection for KMD overhead simulation.
// Set TT_DOORBELL_DELAY_NS to inject a spin delay before each MMIO doorbell
// write, simulating ioctl round-trip overhead.
static uint64_t get_doorbell_delay_ns() {
    const char* env = std::getenv("TT_DOORBELL_DELAY_NS");
    return env ? std::strtoull(env, nullptr, 10) : 0;
}
static const uint64_t doorbell_delay_ns = get_doorbell_delay_ns();

static inline void spin_delay_ns(uint64_t ns) {
    if (ns == 0) return;
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start).count() < (int64_t)ns) {}
}
#include <tracy/Tracy.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include <impl/debug/inspector/inspector.hpp>
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <impl/device/device_manager.hpp>

namespace tt::tt_metal {

void on_dispatch_timeout_detected();

namespace {

bool wrap_ge(uint32_t a, uint32_t b) {
    // Signed Diff uses 2's Complement to handle wrap
    // Works as long as a and b are 2^31 apart
    int32_t diff = a - b;
    return diff >= 0;
}

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
// Please note that the FuncBody is going to loop until the FuncWait returns false.
// GetProgress is optional - if provided, timeout only triggers if BOTH wait_condition is true AND no progress made
template <typename FuncBody, typename FuncWait, typename OnTimeout, typename GetProgress>
void loop_and_wait_with_timeout(
    const FuncBody& func_body,
    const FuncWait& wait_condition,
    const OnTimeout& on_timeout,
    std::chrono::duration<float> timeout_duration,
    const GetProgress& get_progress) {
    if (timeout_duration.count() > 0.0f) {
        auto last_progress_time = std::chrono::high_resolution_clock::now();
        uint32_t last_progress_value = 0;
        // We won't read progress value initially as most of the waits are expected to be shorter than progress update
        // interval. Only long running operations will read progress value updates.
        auto last_progress_update_time = std::chrono::high_resolution_clock::now();
        auto progress_update_interval = std::chrono::milliseconds(
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_progress_update_ms());

        while (true) {
            func_body();

            // Check if operation is finished
            if (!wait_condition()) {
                break;
            }

            // Check if progress should be updated
            if (std::chrono::high_resolution_clock::now() - last_progress_update_time >= progress_update_interval) {
                uint32_t current_progress = get_progress();

                last_progress_update_time = std::chrono::high_resolution_clock::now();
                if (current_progress != last_progress_value) {
                    last_progress_value = current_progress;
                    last_progress_time = std::chrono::high_resolution_clock::now();
                }
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - last_progress_time).count();

            if (elapsed >= timeout_duration.count()) {
                on_timeout();
                break;
            }

            // Sleep briefly to avoid busy-waiting
            std::this_thread::yield();
        }
    } else {
        do {
            func_body();
        } while (wait_condition());
    }
}
}  // namespace

SystemMemoryManager::SystemMemoryManager(ContextId context_id, ChipId device_id, uint8_t num_hw_cqs) :
    context_id(context_id),
    device_id(device_id),
    completion_byte_addrs(num_hw_cqs),
    cq_to_event_locks(num_hw_cqs),
    prefetcher_cores(num_hw_cqs),
    prefetch_q_dev_ptrs(num_hw_cqs),
    prefetch_q_dev_fences(num_hw_cqs) {
    this->prefetch_q_writers.reserve(num_hw_cqs);
    this->completion_q_writers.reserve(num_hw_cqs);

    if (is_mock_device()) {
        this->cq_size = 65536;
        this->cq_sysmem_start = nullptr;
        this->channel_offset = 0;
        this->cq_to_event.resize(num_hw_cqs, 0);
        this->cq_to_last_completed_event.resize(num_hw_cqs, 0);
        const uint32_t alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::HOST);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            this->cq_interfaces.emplace_back(0, cq_id, this->cq_size, 0, alignment);
        }
        log_debug(tt::LogMetal, "SystemMemoryManager: Initialized with stubs for mock device");
        return;
    }

    auto& ctx = tt::tt_metal::MetalContext::instance(context_id);

    if (is_dram_backed()) {
        log_warning(
            tt::LogMetal,
            "DRAM-backed CQs are enabled; this feature is intended for niche use-cases such as simulator "
            "environments, may not work in all configurations, and will result in significantly slower fast dispatch "
            "performance due to DRAM read/write latency replacing direct host memory access.");
        const uint32_t dram_backed_command_queues_size =
            ctx.hal().get_dev_size(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
        TT_ASSERT(dram_backed_command_queues_size > 0);
        TT_FATAL(
            (dram_backed_command_queues_size % num_hw_cqs) == 0,
            "Size of DRAM region reserved for command queues {}B is not divisible by number of command queues {}",
            dram_backed_command_queues_size,
            num_hw_cqs);
        this->cq_size = dram_backed_command_queues_size / num_hw_cqs;
        TT_ASSERT((this->cq_size % ctx.hal().get_alignment(tt::tt_metal::HalMemType::DRAM)) == 0);
        const IDevice* device = ctx.device_manager()->get_active_device(this->device_id);
        TT_FATAL(device->is_mmio_capable(), "Device {} is not an MMIO device", this->device_id);
        this->dram_region_staging_buffer = std::make_unique<char[]>(dram_backed_command_queues_size);
        this->cq_sysmem_start = this->dram_region_staging_buffer.get();
        this->channel_offset = 0;
        this->init_dispatch_core_interfaces(num_hw_cqs, 0);
        return;
    }

    // Real hardware initialization below
    ChipId mmio_device_id = ctx.get_cluster().get_associated_mmio_device(device_id);
    uint16_t channel = ctx.get_cluster().get_assigned_channel_for_device(device_id);
    char* hugepage_start = static_cast<char*>(ctx.get_cluster().host_dma_address(0, mmio_device_id, channel));
    hugepage_start += (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    this->cq_sysmem_start = hugepage_start;

    // TODO(abhullar): Remove env var and expose sizing at the API level
    char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
    if (cq_size_override_env != nullptr) {
        uint32_t cq_size_override = std::stoi(std::string(cq_size_override_env));
        this->cq_size = cq_size_override;
    } else {
        this->cq_size = ctx.get_cluster().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
        if (ctx.get_cluster().is_galaxy_cluster()) {
            // We put 4 galaxy devices per huge page since number of hugepages available is less than number of
            // devices.
            this->cq_size = this->cq_size / DispatchSettings::DEVICES_PER_UMD_CHANNEL;
        }
    }
    this->channel_offset = DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel) +
                           (channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

    this->init_dispatch_core_interfaces(num_hw_cqs, channel);
}

void SystemMemoryManager::init_dispatch_core_interfaces(uint8_t num_hw_cqs, uint16_t channel) {
    auto& ctx = tt::tt_metal::MetalContext::instance(context_id);
    const CoreType core_type =
        ctx.get_dispatch_core_manager().get_dispatch_core_type();
    const uint32_t completion_q_rd_ptr = ctx.dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::COMPLETION_Q_RD);
    const uint32_t prefetch_q_base = ctx.dispatch_mem_map().get_device_command_queue_addr(
        CommandQueueDeviceAddrType::UNRESERVED);
    const uint32_t cq_start =
        ctx.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetcher_core =
            ctx.get_dispatch_core_manager().prefetcher_core(device_id, channel, cq_id);
        auto prefetcher_virtual = ctx.get_cluster().get_virtual_coordinate_from_logical_coordinates(
            prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
        this->prefetcher_cores[cq_id] = tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
        this->prefetch_q_writers.emplace_back(ctx.get_cluster().get_static_tlb_writer(this->prefetcher_cores[cq_id]));

        tt_cxy_pair completion_queue_writer_core =
            ctx.get_dispatch_core_manager().completion_queue_writer_core(this->device_id, channel, cq_id);
        auto completion_queue_writer_virtual = ctx.get_cluster().get_virtual_coordinate_from_logical_coordinates(
            completion_queue_writer_core.chip,
            CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
            core_type);

        const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = ctx.get_cluster()
                                                                                 .get_tlb_data(tt_cxy_pair(
                                                                                     completion_queue_writer_core.chip,
                                                                                     completion_queue_writer_virtual.x,
                                                                                     completion_queue_writer_virtual.y))
                                                                                 .value();
        auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;

        this->completion_byte_addrs[cq_id] = completion_q_rd_ptr % completion_tlb_size;
        this->completion_q_writers.emplace_back(ctx.get_cluster().get_static_tlb_writer(tt_cxy_pair(
            completion_queue_writer_core.chip, completion_queue_writer_virtual.x, completion_queue_writer_virtual.y)));

        const uint32_t alignment =
            is_dram_backed() ? ctx.hal().get_alignment(HalMemType::DRAM) : ctx.hal().get_alignment(HalMemType::HOST);
        const uint32_t base = is_dram_backed() ? this->get_dram_region_base_addr() : 0;
        this->cq_interfaces.emplace_back(channel, cq_id, this->cq_size, cq_start, alignment, base);
        // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
        // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping plus
        // PREFETCH_MAX_OUTSTANDING_PCIE_READS to allow us to start writing to issue queue
        // before we reserve space in the prefetch queue
        TT_FATAL(
            ctx.dispatch_mem_map().max_prefetch_command_size() *
                    (ctx.dispatch_mem_map().prefetch_q_entries() + 1U +
                     PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS) <=
                this->get_issue_queue_size(cq_id),
            "Issue queue for cq_id {} has size of {} which is too small",
            cq_id,
            this->get_issue_queue_size(cq_id));
        this->cq_to_event.push_back(0);
        this->cq_to_last_completed_event.push_back(0);
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        this->prefetch_q_dev_fences[cq_id] = prefetch_q_base + ctx.dispatch_mem_map().prefetch_q_entries() *
                                                                   sizeof(DispatchSettings::prefetch_q_entry_type);
    }
}

bool SystemMemoryManager::is_mock_device() const {
    return tt::tt_metal::MetalContext::instance(this->context_id).get_cluster().get_target_device_type() ==
           tt::TargetDevice::Mock;
}

uint32_t SystemMemoryManager::get_next_event(const uint8_t cq_id) {
    if (is_mock_device()) {
        return ++this->cq_to_event[cq_id];
    }
    cq_to_event_locks[cq_id].lock();
    uint32_t next_event = ++this->cq_to_event[cq_id];  // Event ids start at 1

    cq_to_event_locks[cq_id].unlock();
    return next_event;
}

// Get last issued event to Command Queue
uint32_t SystemMemoryManager::get_last_event(const uint8_t cq_id) {
    std::lock_guard<std::mutex> lock(cq_to_event_locks[cq_id]);
    return this->cq_to_event[cq_id];
}

void SystemMemoryManager::set_current_and_last_completed_event(
    const uint8_t cq_id, const uint32_t current_event_id, const uint32_t last_completed_event_id) {
    cq_to_event_locks[cq_id].lock();

    this->cq_to_event[cq_id] = current_event_id;
    this->cq_to_last_completed_event[cq_id] = last_completed_event_id;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::reset_event_id(const uint8_t cq_id) {
    if (is_mock_device()) {
        this->cq_to_event[cq_id] = 0;
        return;
    }
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] = 0;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::increment_event_id(const uint8_t cq_id, const uint32_t val) {
    if (is_mock_device()) {
        this->cq_to_event[cq_id] += val;
        return;
    }
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] += val;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
    if (is_mock_device()) {
        this->cq_to_last_completed_event[cq_id] = event_id;
        return;
    }
    TT_ASSERT(
        wrap_ge(event_id, this->cq_to_last_completed_event[cq_id]),
        "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
        "completed event is {}, manager {}",
        event_id,
        this->cq_to_last_completed_event[cq_id],
        fmt::ptr(this));
    cq_to_event_locks[cq_id].lock();

    this->cq_to_last_completed_event[cq_id] = event_id;
    cq_to_event_locks[cq_id].unlock();
}

uint32_t SystemMemoryManager::get_current_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t current_event = this->cq_to_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return current_event;
}

uint32_t SystemMemoryManager::get_last_completed_event(const uint8_t cq_id) {
    if (is_mock_device()) {
        return this->cq_to_last_completed_event[cq_id];
    }
    cq_to_event_locks[cq_id].lock();
    uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return last_completed_event;
}

void SystemMemoryManager::reset(const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
    cq_interface.issue_fifo_wr_toggle = false;
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = false;
}

void SystemMemoryManager::set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
    if (is_mock_device()) {
        return;
    }

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_size = (issue_queue_size >> 4);
    cq_interface.issue_fifo_limit = (cq_interface.cq_start + cq_interface.offset + issue_queue_size) >> 4;
}

void SystemMemoryManager::set_bypass_mode(const bool enable, const bool clear) {
    this->bypass_enable = enable;
    if (clear) {
        this->bypass_buffer.clear();
        this->bypass_buffer_write_offset = 0;
    }
}

bool SystemMemoryManager::get_bypass_mode() const { return this->bypass_enable; }

std::vector<uint32_t>& SystemMemoryManager::get_bypass_data() { return this->bypass_buffer; }

uint32_t SystemMemoryManager::get_issue_queue_size(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 65536;
    }
    return this->cq_interfaces[cq_id].issue_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_limit(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 65536;
    }
    return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_size(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 65536;
    }
    return this->cq_interfaces[cq_id].completion_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_limit(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 65536;
    }
    return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_write_ptr(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 0;
    }
    if (this->bypass_enable) {
        return this->bypass_buffer_write_offset;
    }
    return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_read_ptr(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 0;
    }
    return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
}

void* SystemMemoryManager::get_completion_queue_ptr(uint8_t cq_id) const {
    if (is_dram_backed()) {
        auto& ctx = tt::tt_metal::MetalContext::instance(this->context_id);
        const IDevice* device = ctx.device_manager()->get_active_device(this->device_id);
        const uint32_t dram_channel =
            device->allocator_impl()->get_dram_channel_from_bank_id(this->get_dram_region_bank_id());
        ctx.get_cluster().read_dram_vec(
            this->cq_sysmem_start +
                (this->get_issue_queue_limit(cq_id) - this->get_dram_region_base_addr() - this->channel_offset),
            this->get_completion_queue_size(cq_id),
            this->device_id,
            dram_channel,
            this->get_dram_region_base_addr() + get_relative_cq_offset(cq_id, this->cq_size) +
                cq_interfaces[cq_id].cq_start + cq_interfaces[cq_id].command_issue_region_size);
        return (void*)(this->cq_sysmem_start +
                       (this->get_issue_queue_limit(cq_id) - this->get_dram_region_base_addr() - this->channel_offset));
    }
    // The completion queue follows issue queue in contiguous memory
    // get_issue_queue_limit() returns absolute device address where the issue queue ends.
    // We subtract channel_offset (absolute device channel base) to get relative offset,
    // then add it to cq_sysmem_start (host channel base) to get host virtual address
    return (void*)(this->cq_sysmem_start + (this->get_issue_queue_limit(cq_id) - this->channel_offset));
}

uint32_t SystemMemoryManager::get_completion_queue_read_toggle(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return 0;
    }
    return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
}

uint32_t SystemMemoryManager::get_cq_size() const {
    if (is_mock_device()) {
        return 65536;
    }
    return this->cq_size;
}

ChipId SystemMemoryManager::get_device_id() const { return this->device_id; }

ContextId SystemMemoryManager::get_context_id() const { return this->context_id; }

std::vector<SystemMemoryCQInterface>& SystemMemoryManager::get_cq_interfaces() { return this->cq_interfaces; }

void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
    if (is_mock_device()) {
        thread_local std::array<char, 65536> dummy_buffer{};
        return dummy_buffer.data();
    }

    if (this->bypass_enable) {
        uint32_t curr_size = this->bypass_buffer.size();
        uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
        this->bypass_buffer.resize(new_size);
        return (void*)((char*)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
    }

    uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

    const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
    const uint32_t alignment =
        is_dram_backed()
            ? tt::tt_metal::MetalContext::instance(this->context_id).hal().get_alignment(tt::tt_metal::HalMemType::DRAM)
            : tt::tt_metal::MetalContext::instance(this->context_id)
                  .hal()
                  .get_alignment(tt::tt_metal::HalMemType::HOST);
    if (issue_q_write_ptr + align(cmd_size_B, alignment) > command_issue_limit) {
        this->wrap_issue_queue_wr_ptr(cq_id);
        issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
    }

    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757
    void* issue_q_region = nullptr;
    if (is_dram_backed()) {
        issue_q_region =
            this->cq_sysmem_start + (issue_q_write_ptr - this->get_dram_region_base_addr() - this->channel_offset);
    } else {
        issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);
    }

    return issue_q_region;
}

void SystemMemoryManager::cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
    if (is_mock_device()) {
        return;
    }

    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
    // queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on
    // https://github.com/tenstorrent/tt-metal/issues/4757

    if (this->bypass_enable) {
        std::copy((uint8_t*)data, (uint8_t*)data + size_in_bytes, (uint8_t*)this->bypass_buffer.data() + write_ptr);
    } else if (is_dram_backed()) {
        void* user_scratchspace =
            this->cq_sysmem_start + (write_ptr - this->get_dram_region_base_addr() - this->channel_offset);
        memcpy(user_scratchspace, data, size_in_bytes);
    } else {
        void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);
        memcpy_to_device(user_scratchspace, data, size_in_bytes);
    }
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    if (this->bypass_enable) {
        this->bypass_buffer_write_offset += push_size_B;
        return;
    }

    auto& ctx = tt::tt_metal::MetalContext::instance(context_id);
    const uint32_t alignment =
        is_dram_backed()
            ? tt::tt_metal::MetalContext::instance(this->context_id).hal().get_alignment(tt::tt_metal::HalMemType::DRAM)
            : tt::tt_metal::MetalContext::instance(this->context_id)
                  .hal()
                  .get_alignment(tt::tt_metal::HalMemType::HOST);
    const uint32_t push_size_16B = align(push_size_B, alignment) >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    uint32_t issue_q_wr_ptr = ctx.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR);

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;            // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }

    if (is_dram_backed()) {
        const IDevice* device = ctx.device_manager()->get_active_device(this->device_id);
        const uint32_t dram_channel =
            device->allocator_impl()->get_dram_channel_from_bank_id(this->get_dram_region_bank_id());
        ctx.get_cluster().write_dram_vec(
            this->cq_sysmem_start + (cq_interface.offset - this->get_dram_region_base_addr() - this->channel_offset) +
                cq_interface.cq_start,
            this->get_issue_queue_size(cq_id),
            this->device_id,
            dram_channel,
            this->get_dram_region_base_addr() + get_relative_cq_offset(cq_id, this->cq_size) + cq_interface.cq_start);
        ctx.get_cluster().write_dram_vec(
            &cq_interface.issue_fifo_wr_ptr,
            sizeof(uint32_t),
            this->device_id,
            dram_channel,
            this->get_dram_region_base_addr() + get_relative_cq_offset(cq_id, this->cq_size) + issue_q_wr_ptr);
        return;
    }

    // Also store this data in hugepages, so if a hang happens we can see what was written by host.
    ChipId mmio_device_id = ctx.get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel = ctx.get_cluster().get_assigned_channel_for_device(this->device_id);
    ctx.get_cluster().write_sysmem(
        &cq_interface.issue_fifo_wr_ptr,
        sizeof(uint32_t),
        issue_q_wr_ptr + get_relative_cq_offset(cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::send_completion_queue_read_ptr(const uint8_t cq_id) const {
    if (is_mock_device()) {
        return;
    }

    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    uint32_t read_ptr_and_toggle = cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
    spin_delay_ns(doorbell_delay_ns);
    this->completion_q_writers[cq_id].write(this->completion_byte_addrs[cq_id], read_ptr_and_toggle);
    auto& ctx = tt::tt_metal::MetalContext::instance(this->context_id);
    const uint32_t completion_q_rd_ptr =
        ctx.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD);

    if (is_dram_backed()) {
        const IDevice* device = ctx.device_manager()->get_active_device(this->device_id);
        ctx.get_cluster().write_dram_vec(
            &read_ptr_and_toggle,
            sizeof(uint32_t),
            this->device_id,
            device->allocator_impl()->get_dram_channel_from_bank_id(this->get_dram_region_bank_id()),
            this->get_dram_region_base_addr() + get_relative_cq_offset(cq_id, this->cq_size) + completion_q_rd_ptr);
        return;
    }

    // Also store this data in hugepages in case we hang and can't get it from the device.
    ChipId mmio_device_id = ctx.get_cluster().get_associated_mmio_device(this->device_id);
    uint16_t channel = ctx.get_cluster().get_assigned_channel_for_device(this->device_id);
    ctx.get_cluster().write_sysmem(
        &read_ptr_and_toggle,
        sizeof(uint32_t),
        completion_q_rd_ptr + get_relative_cq_offset(cq_id, this->cq_size),
        mmio_device_id,
        channel);
}

void SystemMemoryManager::fetch_queue_reserve_back(const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    if (this->bypass_enable) {
        return;
    }

    auto& ctx = tt::tt_metal::MetalContext::instance(context_id);
    const uint32_t prefetch_q_rd_ptr =
        ctx.dispatch_mem_map().get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD);

    // Helper to wait for fetch queue space, if needed
    uint32_t fence;
    auto wait_for_fetch_q_space = [&]() {
        if (this->prefetch_q_dev_ptrs[cq_id] != this->prefetch_q_dev_fences[cq_id]) {
            return;
        }
        ZoneScopedN("wait_for_fetch_q_space");

        // Body of the operation
        auto fetch_operation_body = [&]() {
            ctx.get_cluster().read_core(&fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
            this->prefetch_q_dev_fences[cq_id] = fence;
        };

        // Condition to check if should continue waiting
        auto fetch_wait_condition = [&]() -> bool {
            return this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id];
        };

        // Handler for timeout
        auto fetch_on_timeout = [this]() {
            tt::tt_metal::MetalContext::instance(this->context_id).on_dispatch_timeout_detected();
            TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
        };

        // Get dispatch progress for timeout detection
        auto get_dispatch_progress = [&]() -> uint32_t { return get_cq_dispatch_progress(this->device_id, cq_id); };

        auto timeout_duration = ctx.rtoptions().get_timeout_duration_for_operations();

        loop_and_wait_with_timeout(
            fetch_operation_body, fetch_wait_condition, fetch_on_timeout, timeout_duration, get_dispatch_progress);
    };

    wait_for_fetch_q_space();
    // Wrap FetchQ if possible
    uint32_t prefetch_q_base =
        ctx.dispatch_mem_map().get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
    uint32_t prefetch_q_limit = prefetch_q_base + (ctx.dispatch_mem_map().prefetch_q_entries() *
                                                   sizeof(DispatchSettings::prefetch_q_entry_type));
    if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
        wait_for_fetch_q_space();
    }
}

uint32_t SystemMemoryManager::completion_queue_wait_front(
    const uint8_t cq_id, std::atomic<bool>& exit_condition) const {
    if (is_mock_device()) {
        return 0;
    }

    uint32_t write_ptr_and_toggle;
    uint32_t write_ptr;
    uint32_t write_toggle;
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    // Body of the operation to be timed out
    auto wait_operation_body = [this, cq_id, &write_ptr_and_toggle, &write_ptr, &write_toggle]() {
        write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
        write_ptr = write_ptr_and_toggle & 0x7fffffff;
        write_toggle = write_ptr_and_toggle >> 31;
    };

    // Condition to check if the operation should continue
    auto wait_condition = [&cq_interface, &write_ptr, &write_toggle]() -> bool {
        return cq_interface.completion_fifo_rd_ptr == write_ptr and
               cq_interface.completion_fifo_rd_toggle == write_toggle;
    };

    // Handler for the timeout
    auto on_timeout = [this, &exit_condition]() {
        exit_condition.store(true);

        tt::tt_metal::MetalContext::instance(this->context_id).on_dispatch_timeout_detected();

        TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable");
    };

    // Get dispatch progress for timeout detection
    auto get_dispatch_progress = [this, cq_id]() -> uint32_t {
        return get_cq_dispatch_progress(this->device_id, cq_id);
    };

    loop_and_wait_with_timeout(
        wait_operation_body,
        wait_condition,
        on_timeout,
        tt::tt_metal::MetalContext::instance().rtoptions().get_timeout_duration_for_operations(),
        get_dispatch_progress);

    return write_ptr_and_toggle;
}

void SystemMemoryManager::wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    if (this->bypass_enable) {
        return;
    }
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (cq_interface.cq_start + cq_interface.offset) >> 4;
    cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
}

void SystemMemoryManager::wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
}

void SystemMemoryManager::completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
    if (is_mock_device()) {
        return;
    }

    uint32_t data_read_B = num_pages_read * DispatchSettings::TRANSFER_PAGE_SIZE;
    uint32_t data_read_16B = data_read_B >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr += data_read_16B;
    if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    // Notify dispatch core
    this->send_completion_queue_read_ptr(cq_id);
}

void SystemMemoryManager::fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher) {
    if (is_mock_device()) {
        return;
    }

    uint32_t max_command_size_B = MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    TT_ASSERT(
        command_size_B <= max_command_size_B,
        "Generated prefetcher command of size {} B exceeds max command size {} B",
        command_size_B,
        max_command_size_B);
    TT_ASSERT(
        (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");
    TT_ASSERT(command_size_B > 0, "Command size must be greater than 0");
    if (this->bypass_enable) {
        return;
    }
    tt_driver_atomics::sfence();
    DispatchSettings::prefetch_q_entry_type command_size_16B =
        command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

    // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
    // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
    // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
    // prefetcher is MSB of FetchQ entry.
    if (stall_prefetcher) {
        command_size_16B |= (1 << ((sizeof(DispatchSettings::prefetch_q_entry_type) * 8) - 1));
    }
    spin_delay_ns(doorbell_delay_ns);
    this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
    this->prefetch_q_dev_ptrs[cq_id] += sizeof(DispatchSettings::prefetch_q_entry_type);
}

bool SystemMemoryManager::is_dram_backed() const {
    return tt::tt_metal::MetalContext::instance(this->context_id).rtoptions().get_dram_backed_cq();
}

uint32_t SystemMemoryManager::get_dram_region_base_addr() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return tt::tt_metal::MetalContext::instance(this->context_id)
        .hal()
        .get_dev_addr(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
}

uint32_t SystemMemoryManager::get_dram_region_bank_id() const {
    TT_FATAL(this->is_dram_backed(), "CQs are not DRAM backed");
    return 0;
}

}  // namespace tt::tt_metal
