// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cq_backing.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include <internal/l1_ecc_halt.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt_stl/assert.hpp>

#include "allocator/allocator.hpp"
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include "memcpy.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include <impl/device/device_manager.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

namespace {

// The timeout counts from the last change in get_progress(); a non-positive timeout_duration waits forever.
template <typename StillWaiting, typename OnTimeout, typename GetProgress>
void loop_and_wait_with_timeout(
    MetalContext& ctx,
    const StillWaiting& still_waiting,
    const OnTimeout& on_timeout,
    std::chrono::duration<float> timeout_duration,
    const GetProgress& get_progress,
    std::chrono::milliseconds progress_update_interval,
    ChipId device_id,
    std::atomic<bool>* exit_condition = nullptr) {
    tt::tt_metal::internal::L1EccHaltWatch ecc_watch(ctx, device_id);
    if (timeout_duration.count() > 0.0f) {
        auto last_progress_time = std::chrono::high_resolution_clock::now();
        uint32_t last_progress_value = 0;
        // First sampled after one interval; most waits end sooner.
        auto last_progress_update_time = std::chrono::high_resolution_clock::now();

        while (true) {
            if (exit_condition != nullptr && exit_condition->load(std::memory_order_acquire)) {
                break;
            }

            if (!still_waiting()) {
                break;
            }

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
                ecc_watch.check_now();
                on_timeout();
                break;
            }
            ecc_watch.poll();

            std::this_thread::yield();
        }
    } else {
        while (exit_condition == nullptr || !exit_condition->load(std::memory_order_acquire)) {
            ecc_watch.poll();
            if (!still_waiting()) {
                break;
            }
        }
    }
}

class MockCqBacking final : public CqBacking {
public:
    // Issue and completion queue size: room for a max-size prefetch command with slack.
    static constexpr uint32_t kMockCqSize = 256 * 1024;
    static constexpr uint32_t kMockCqRegionSize = 2 * kMockCqSize;

    MockCqBacking(MetalEnvImpl& env, MetalContext& ctx, uint8_t num_hw_cqs) :
        buffer_(std::make_unique<char[]>(static_cast<size_t>(num_hw_cqs) * kMockCqRegionSize)) {
        const uint32_t max_command_size_B = ctx.dispatch_mem_map().max_prefetch_command_size();
        TT_FATAL(
            kMockCqSize >= max_command_size_B,
            "Mock issue queue of {} B can't hold a {} B prefetch command",
            kMockCqSize,
            max_command_size_B);
        set_layout(Layout{
            .host_base = buffer_.get(),
            .cq_addr_base = 0,
            .cq_size = kMockCqRegionSize,
            .issue_alignment = env.get_hal().get_alignment(HalMemType::HOST),
        });
        log_debug(tt::LogMetal, "SystemMemoryManager: Initialized with a host buffer for mock device");
    }

    std::vector<SystemMemoryCQInterface> make_cq_interfaces(uint8_t num_hw_cqs) const override {
        std::vector<SystemMemoryCQInterface> cq_interfaces;
        cq_interfaces.reserve(num_hw_cqs);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            cq_interfaces.emplace_back(cq_id, cq_id * kMockCqRegionSize, kMockCqSize, kMockCqSize);
        }
        return cq_interfaces;
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        std::memcpy(dst, src, size_B);
    }

    void issue_queue_pushed(uint8_t, uint32_t, uint32_t, uint32_t) override {}

    void sync_completion_queue(uint8_t, const SystemMemoryCQInterface&) const override {}

    void send_completion_queue_read_ptr(uint8_t, uint32_t) const override {}

    void fetch_queue_reserve_back(uint8_t) override {}

    void fetch_queue_write(uint32_t, uint8_t, bool) override {}

    uint32_t completion_queue_wait_front(uint8_t, const SystemMemoryCQInterface&, std::atomic<bool>&) const override {
        return 0;
    }

private:
    std::unique_ptr<char[]> buffer_;
};

// Shared by the hugepage and DRAM backings: the IO windows onto the prefetcher's fetch queue and the completion queue
// writer's read pointer.
class DeviceCqBacking : public CqBacking {
public:
    std::vector<SystemMemoryCQInterface> make_cq_interfaces(uint8_t num_hw_cqs) const override {
        const auto& mem_map = ctx_.dispatch_mem_map();
        const uint32_t cq_start = mem_map.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        std::vector<SystemMemoryCQInterface> cq_interfaces;
        cq_interfaces.reserve(num_hw_cqs);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            const SystemMemoryCQInterface& cq_interface =
                cq_interfaces.emplace_back(channel_, cq_id, cq_size(), cq_start, issue_alignment(), interface_base_);
            const uint32_t issue_queue_size = cq_interface.issue_fifo_size << 4;
            // The fetch queue keeps the host from overwriting unread issue queue data, so the issue queue holds a full
            // fetch queue of max-size commands, plus one for wrapping, plus PREFETCH_MAX_OUTSTANDING_PCIE_READS so the
            // host can write before it reserves a fetch queue entry.
            TT_FATAL(
                mem_map.max_prefetch_command_size() *
                        (mem_map.prefetch_q_entries() + 1U + PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS) <=
                    issue_queue_size,
                "Issue queue for cq_id {} has size of {} which is too small",
                cq_id,
                issue_queue_size);
        }
        return cq_interfaces;
    }

    void send_completion_queue_read_ptr(uint8_t cq_id, uint32_t read_ptr_and_toggle) const override {
        this->completion_q_windows_[cq_id]->write32(this->completion_byte_addrs_[cq_id], read_ptr_and_toggle);
        write_host_cq_word(
            cq_id,
            ctx_.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD),
            read_ptr_and_toggle);
    }

    void issue_queue_pushed(uint8_t cq_id, uint32_t /*start_addr*/, uint32_t /*size_B*/, uint32_t wr_ptr_16B) override {
        write_host_cq_word(
            cq_id,
            ctx_.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR),
            wr_ptr_16B);
    }

    void fetch_queue_reserve_back(uint8_t cq_id) override {
        const auto& mem_map = ctx_.dispatch_mem_map();
        const uint32_t prefetch_q_rd_ptr =
            mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD, cq_id);

        auto wait_for_fetch_q_space = [&]() {
            if (this->prefetch_q_dev_ptrs_[cq_id] != this->prefetch_q_dev_fences_[cq_id]) {
                return;
            }
            TTZoneScopedDN(DISPATCH, "wait_for_fetch_q_space");
            wait_with_timeout(
                cq_id,
                [&]() {
                    uint32_t fence;
                    env_.get_cluster().read_core(
                        &fence, sizeof(uint32_t), this->prefetcher_cores_[cq_id], prefetch_q_rd_ptr);
                    this->prefetch_q_dev_fences_[cq_id] = fence;
                    return this->prefetch_q_dev_ptrs_[cq_id] == fence;
                },
                [this]() {
                    ctx_.on_dispatch_timeout_detected();
                    TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
                });
        };

        wait_for_fetch_q_space();
        // Wrap at the end of the fetch queue.
        if (this->prefetch_q_dev_ptrs_[cq_id] == prefetch_q_limit(cq_id)) {
            this->prefetch_q_dev_ptrs_[cq_id] = prefetch_q_base(cq_id);
            if (mem_map.prefetch_q_cached()) {
                this->prefetch_q_phases_[cq_id] ^= 1U;
            }
            wait_for_fetch_q_space();
        }
    }

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher) override {
        const DispatchMemMap& dispatch_mem_map = ctx_.dispatch_mem_map();
        const uint32_t max_command_size_B = dispatch_mem_map.max_prefetch_command_size();
        TT_ASSERT(
            command_size_B <= max_command_size_B,
            "Generated prefetcher command of size {} B exceeds max command size {} B",
            command_size_B,
            max_command_size_B);

        const uint32_t entry_bytes = dispatch_mem_map.prefetch_q_entry_size_bytes();
        const uint32_t shift_for_msb = entry_bytes * 8 - 1;
        const uint32_t max_encodable = 1u << shift_for_msb;
        TT_ASSERT(
            (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < max_encodable,
            "FetchQ command too large to represent");
        TT_ASSERT(command_size_B > 0, "Command size must be greater than 0");
        tt_driver_atomics::sfence();
        uint32_t entry_val = command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

        // Trace replay takes over the prefetcher's cmd_data_q, so commands fetched along with a trace command would be
        // corrupted. The stall flag, the entry's MSB, stops the prefetcher from fetching past this entry.
        if (stall_prefetcher) {
            entry_val |= 1u << shift_for_msb;
        }

        if (dispatch_mem_map.prefetch_q_cached()) {
            constexpr uint32_t phase_bit = PrefetchConstants::PREFETCH_Q_PHASE_BIT;
            TT_ASSERT(entry_bytes == 4, "Cached fetch queue entries must be 4 bytes");
            static_assert(
                sizeof(command_size_B) * 8 - DispatchSettings::PREFETCH_Q_LOG_MINSIZE <= phase_bit,
                "FetchQ entry size field would overlap the phase bit");
            entry_val |= this->prefetch_q_phases_[cq_id] << phase_bit;
        }

        if (entry_bytes == 2) {
            this->prefetch_q_windows_[cq_id]->write16(
                this->prefetch_q_dev_ptrs_[cq_id], static_cast<uint16_t>(entry_val));
        } else {
            TT_ASSERT(entry_bytes == 4);
            this->prefetch_q_windows_[cq_id]->write32(this->prefetch_q_dev_ptrs_[cq_id], entry_val);
        }
        this->prefetch_q_dev_ptrs_[cq_id] += entry_bytes;
    }

    uint32_t completion_queue_wait_front(
        uint8_t cq_id, const SystemMemoryCQInterface& cq_interface, std::atomic<bool>& exit_condition) const override {
        uint32_t write_ptr_and_toggle = 0;
        wait_with_timeout(
            cq_id,
            [&]() {
                write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(ctx_, this->device_id_, cq_id, this->cq_size());
                return cq_interface.completion_fifo_rd_ptr == (write_ptr_and_toggle & 0x7fffffff) and
                       cq_interface.completion_fifo_rd_toggle == (write_ptr_and_toggle >> 31);
            },
            [this, &exit_condition]() {
                exit_condition.store(true);
                ctx_.on_dispatch_timeout_detected();
                TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable");
            },
            &exit_condition);
        return write_ptr_and_toggle;
    }

protected:
    DeviceCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs, uint16_t channel) :
        env_(env),
        ctx_(ctx),
        device_id_(device_id),
        channel_(channel),
        prefetcher_cores_(num_hw_cqs),
        completion_byte_addrs_(num_hw_cqs),
        prefetch_q_dev_ptrs_(num_hw_cqs),
        prefetch_q_dev_fences_(num_hw_cqs),
        // The ring starts zeroed, which reads as phase 0, so the first lap writes phase 1.
        prefetch_q_phases_(num_hw_cqs, 1) {
        prefetch_q_windows_.reserve(num_hw_cqs);
        completion_q_windows_.reserve(num_hw_cqs);
    }

    // Takes each CQ's aux share out of layout.cq_size, puts the aux region after the CQs, sets the layout, and opens
    // the IO windows onto each CQ's dispatch cores.
    void init_layout(Layout layout, bool with_d2h_fallback_reserve, uint8_t num_hw_cqs) {
        const uint32_t per_cq_reduction =
            AuxRegionAllocator::reserved_bytes_per_cq(num_hw_cqs, with_d2h_fallback_reserve);
        TT_FATAL(
            layout.cq_size > per_cq_reduction,
            "Command queue size {} B is too small for the {} B aux reservation",
            layout.cq_size,
            per_cq_reduction);
        layout.cq_size -= per_cq_reduction;
        const uint32_t total_cq_space = static_cast<uint32_t>(num_hw_cqs) * layout.cq_size;
        layout.aux_region_allocator = AuxRegionAllocator(
            layout.host_base + total_cq_space,
            layout.cq_addr_base + total_cq_space,
            static_cast<uint32_t>(num_hw_cqs) * per_cq_reduction,
            with_d2h_fallback_reserve);
        set_layout(std::move(layout));

        auto& core_manager = ctx_.get_dispatch_core_manager();
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            // Per CQ, since CQs can share a dispatch core.
            const uint32_t completion_q_rd_ptr = ctx_.dispatch_mem_map().get_device_command_queue_addr(
                CommandQueueDeviceAddrType::COMPLETION_Q_RD, cq_id);
            auto [prefetcher_core, prefetch_q_window] = open_dispatch_core_window(
                core_manager.prefetcher_core(device_id_, channel_, cq_id), prefetch_q_limit(cq_id));
            this->prefetcher_cores_[cq_id] = prefetcher_core;
            this->prefetch_q_windows_.push_back(std::move(prefetch_q_window));
            this->completion_q_windows_.push_back(
                open_dispatch_core_window(
                    core_manager.completion_queue_writer_core(device_id_, channel_, cq_id),
                    completion_q_rd_ptr + sizeof(uint32_t))
                    .second);
            this->completion_byte_addrs_[cq_id] = completion_q_rd_ptr;

            this->prefetch_q_dev_ptrs_[cq_id] = prefetch_q_base(cq_id);
            this->prefetch_q_dev_fences_[cq_id] = prefetch_q_limit(cq_id);
        }
    }

    // Stores a host-side CQ pointer (a CommandQueueHostAddrType address) in the device-visible CQ, for hang debugging.
    virtual void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const = 0;

    MetalEnvImpl& env_;
    MetalContext& ctx_;
    const ChipId device_id_;
    const uint16_t channel_;
    // Base address passed to SystemMemoryCQInterface: the DRAM region for DRAM-backed CQs, 0 for the hugepage.
    uint32_t interface_base_ = 0;

private:
    uint32_t prefetch_q_base(uint8_t cq_id) const {
        return ctx_.dispatch_mem_map().get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, cq_id);
    }

    uint32_t prefetch_q_limit(uint8_t cq_id) const {
        const auto& mem_map = ctx_.dispatch_mem_map();
        return prefetch_q_base(cq_id) + mem_map.prefetch_q_entries() * mem_map.prefetch_q_entry_size_bytes();
    }

    // Returns the dispatch core's virtual coordinates and an IO window onto its L1 [0, size), so L1 addresses are
    // window offsets.
    std::pair<tt_cxy_pair, std::unique_ptr<tt::umd::IoWindow>> open_dispatch_core_window(
        const tt_cxy_pair& logical_core, uint32_t size) const {
        const auto virtual_core = env_.get_cluster().get_virtual_coordinate_from_logical_coordinates(
            logical_core.chip,
            CoreCoord(logical_core.x, logical_core.y),
            ctx_.get_dispatch_core_manager().get_dispatch_core_type());
        const tt::umd::CoreCoord translated =
            env_.get_cluster()
                .get_soc_desc(logical_core.chip)
                .get_coord_at(tt_xy_pair(virtual_core.x, virtual_core.y), tt::CoordSystem::TRANSLATED);
        return {
            tt_cxy_pair(logical_core.chip, virtual_core.x, virtual_core.y),
            env_.get_cluster().get_driver()->create_io_window(
                logical_core.chip, translated, /*addr=*/0, {.size = size})};
    }

    template <typename StillWaiting, typename OnTimeout>
    void wait_with_timeout(
        uint8_t cq_id,
        const StillWaiting& still_waiting,
        const OnTimeout& on_timeout,
        std::atomic<bool>* exit_condition = nullptr) const {
        loop_and_wait_with_timeout(
            ctx_,
            [&]() {
                const bool waiting = still_waiting();
                // Yield to clock the simulator when running on TTSim; no-op on real hardware.
                env_.get_cluster().advance_device_execution(device_id_);
                return waiting;
            },
            on_timeout,
            env_.get_rtoptions().get_timeout_duration_for_operations(),
            [&]() -> uint32_t { return get_cq_dispatch_progress(ctx_, device_id_, cq_id); },
            std::chrono::milliseconds(env_.get_rtoptions().get_dispatch_progress_update_ms()),
            device_id_,
            exit_condition);
    }

    std::vector<tt_cxy_pair> prefetcher_cores_;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> prefetch_q_windows_;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> completion_q_windows_;
    std::vector<uint32_t> completion_byte_addrs_;
    std::vector<uint32_t> prefetch_q_dev_ptrs_;
    std::vector<uint32_t> prefetch_q_dev_fences_;
    // Phase bit value for the current lap of each fetch queue, used when the prefetcher polls it through its cache.
    std::vector<uint32_t> prefetch_q_phases_;
};

class HugepageCqBacking final : public DeviceCqBacking {
public:
    HugepageCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
        DeviceCqBacking(env, ctx, device_id, num_hw_cqs, env.get_cluster().get_assigned_channel_for_device(device_id)),
        mmio_device_id_(env.get_cluster().get_associated_mmio_device(device_id)) {
        char* hugepage_start = static_cast<char*>(env_.get_cluster().host_dma_address(0, mmio_device_id_, channel_));
        hugepage_start += (channel_ >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

        // TODO(abhullar): Remove env var and expose sizing at the API level
        uint32_t cq_size_B = 0;
        char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
        if (cq_size_override_env != nullptr) {
            cq_size_B = std::stoi(std::string(cq_size_override_env));
        } else {
            cq_size_B = env_.get_cluster().get_host_channel_size(mmio_device_id_, channel_) / num_hw_cqs;
            if (env_.get_cluster().is_galaxy_cluster()) {
                // Galaxy has fewer hugepages than devices, so DEVICES_PER_UMD_CHANNEL devices share one.
                cq_size_B = cq_size_B / DispatchSettings::DEVICES_PER_UMD_CHANNEL;
            }
        }
        // CQ addresses include the channel offset, matching the device's view of the hugepage.
        const uint32_t channel_offset = DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel_) +
                                        (channel_ >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

        init_layout(
            Layout{
                .host_base = hugepage_start,
                .cq_addr_base = channel_offset,
                .cq_size = cq_size_B,
                .issue_alignment = env_.get_hal().get_alignment(HalMemType::HOST),
            },
            env_.get_cluster().needs_d2h_hugepage_fallback(),
            num_hw_cqs);
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        memcpy_to_device(dst, src, size_B);
    }

    void sync_completion_queue(uint8_t, const SystemMemoryCQInterface&) const override {}

protected:
    void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const override {
        env_.get_cluster().write_sysmem(
            &value,
            sizeof(uint32_t),
            host_cq_addr + get_relative_cq_offset(cq_id, cq_size()),
            mmio_device_id_,
            channel_);
    }

private:
    const ChipId mmio_device_id_;
};

class DramCqBacking final : public DeviceCqBacking {
public:
    DramCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
        DeviceCqBacking(env, ctx, device_id, num_hw_cqs, /*channel=*/0) {
        log_warning(
            tt::LogMetal,
            "DRAM-backed CQs are enabled; this feature is intended for niche use-cases such as simulator "
            "environments, may not work in all configurations, and will result in significantly slower fast dispatch "
            "performance due to DRAM read/write latency replacing direct host memory access.");
        const uint32_t dram_backed_command_queues_size =
            env_.get_hal().get_dev_size(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
        TT_ASSERT(dram_backed_command_queues_size > 0);
        TT_FATAL(
            (dram_backed_command_queues_size % num_hw_cqs) == 0,
            "Size of DRAM region reserved for command queues {}B is not divisible by number of command queues {}",
            dram_backed_command_queues_size,
            num_hw_cqs);
        const uint32_t cq_size_B = dram_backed_command_queues_size / num_hw_cqs;
        TT_ASSERT((cq_size_B % env_.get_hal().get_alignment(tt::tt_metal::HalMemType::DRAM)) == 0);
        const IDevice* device = ctx_.device_manager()->get_active_device(device_id_);
        TT_FATAL(
            device->is_mmio_capable() || env_.get_cluster().get_target_device_type() == tt::TargetDevice::Simulator,
            "Device {} is not an MMIO device",
            device_id_);
        // The CQ region is in DRAM bank 0.
        dram_channel_ = device->allocator_impl()->get_dram_channel_from_bank_id(0);
        staging_buffer_ = std::make_unique<char[]>(dram_backed_command_queues_size);
        dram_base_ = env_.get_hal().get_dev_addr(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);

        interface_base_ = dram_base_;
        // No D2H fallback reserve: that fallback needs a hugepage, and DRAM-backed CQs have none.
        init_layout(
            Layout{
                .host_base = staging_buffer_.get(),
                .cq_addr_base = dram_base_,
                .cq_size = cq_size_B,
                .issue_alignment = env_.get_hal().get_alignment(HalMemType::DRAM),
                .is_dram_backed = true,
            },
            /*with_d2h_fallback_reserve=*/false,
            num_hw_cqs);
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        std::memcpy(dst, src, size_B);
    }

    void issue_queue_pushed(uint8_t cq_id, uint32_t start_addr, uint32_t size_B, uint32_t wr_ptr_16B) override {
        env_.get_cluster().write_dram_vec(host_ptr(start_addr), size_B, device_id_, dram_channel_, start_addr);
        DeviceCqBacking::issue_queue_pushed(cq_id, start_addr, size_B, wr_ptr_16B);
    }

    void sync_completion_queue(uint8_t cq_id, const SystemMemoryCQInterface& cq_interface) const override {
        const uint32_t completion_queue_addr = cq_interface.issue_fifo_limit << 4;
        env_.get_cluster().read_dram_vec(
            host_ptr(completion_queue_addr),
            cq_interface.completion_fifo_size << 4,
            device_id_,
            dram_channel_,
            dram_base_ + get_relative_cq_offset(cq_id, cq_size()) + cq_interface.cq_start +
                cq_interface.command_issue_region_size);
    }

protected:
    void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const override {
        env_.get_cluster().write_dram_vec(
            &value,
            sizeof(uint32_t),
            device_id_,
            dram_channel_,
            dram_base_ + get_relative_cq_offset(cq_id, cq_size()) + host_cq_addr);
    }

private:
    uint32_t dram_channel_ = 0;
    std::unique_ptr<char[]> staging_buffer_;
    uint32_t dram_base_ = 0;
};

}  // namespace

void CqBacking::set_layout(Layout layout) {
    TT_ASSERT(cq_size_ == 0, "CQ backing layout is set once");
    TT_ASSERT(layout.host_base != nullptr, "CQ backing has no host memory");
    TT_ASSERT(layout.cq_size > 0, "CQ backing has no CQ space");
    TT_ASSERT(layout.issue_alignment > 0, "CQ backing has no issue queue alignment");
    host_base_ = layout.host_base;
    cq_addr_base_ = layout.cq_addr_base;
    cq_size_ = layout.cq_size;
    issue_alignment_ = layout.issue_alignment;
    is_dram_backed_ = layout.is_dram_backed;
    aux_region_allocator_ = std::move(layout.aux_region_allocator);
}

std::unique_ptr<CqBacking> CqBacking::create(
    MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) {
    std::unique_ptr<CqBacking> backing;
    if (env.get_cluster().is_mock_or_emulated()) {
        backing = std::make_unique<MockCqBacking>(env, ctx, num_hw_cqs);
    } else if (env.get_rtoptions().get_dram_backed_cq()) {
        backing = std::make_unique<DramCqBacking>(env, ctx, device_id, num_hw_cqs);
    } else {
        backing = std::make_unique<HugepageCqBacking>(env, ctx, device_id, num_hw_cqs);
    }
    TT_ASSERT(backing->cq_size_ > 0, "CQ backing constructor did not set its layout");
    return backing;
}

}  // namespace tt::tt_metal
