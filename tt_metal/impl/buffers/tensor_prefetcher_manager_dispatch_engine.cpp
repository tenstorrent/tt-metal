// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorPrefetcherManager when its DRAM-sender endpoints live on the dispatch engine
// (DramSenderPlacement::DispatchEngine). Fast dispatch launches the reader and pusher that serve them
// when the device opens, so starting the prefetcher only takes over their request FIFO; requests are
// the same pages a DRAM-core sender receives, grouped one slot per request so that the kernels serve
// all of its receivers together.

#include "impl/buffers/tensor_prefetcher_manager.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/kernel_config/tensor_prefetcher.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::distributed {

namespace {

namespace fifo = tensor_prefetcher_fifo;

const TensorPrefetcherRequestHeader& header_of(const std::vector<uint8_t>& page) {
    return *reinterpret_cast<const TensorPrefetcherRequestHeader*>(page.data());
}

}  // namespace

void TensorPrefetcherManager::start_on_dispatch_engine() {
    auto& metal = mesh_device_->impl().metal_context();
    const DispatchMemMap& mem_map = metal.dispatch_mem_map();
    fifo_region_base_ = mem_map.unused_l1_base();

    // The request serializer sizes layouts from the stage ring as the DRAM-core kernel splits it (halves
    // and thirds). The dispatch-engine kernels split it per receiver instead, which queue time checks.
    fifo_layout_ = tensor_prefetcher_region_layout(mesh_device_->impl().metal_env().get_cluster());
    set_stage_ring_size(
        fifo_layout_.stage_ring_bytes(mem_map.unused_l1_size()), metal.hal().get_alignment(HalMemType::L1));

    // The kernels run for as long as fast dispatch does and keep their counters across prefetcher runs,
    // so a run resumes at the device's count. An earlier run's stop waited for them to go idle.
    fifos_.clear();
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        DeviceFifo& device_fifo = fifos_.emplace_back();
        device_fifo.dispatch_engine =
            dram_sender_dispatch_engine(mesh_device_->impl().get_context_id(), devices_[d]->id());
        uint32_t written = 0;
        ::tt::tt_metal::detail::ReadFromDeviceL1(
            devices_[d],
            device_fifo.dispatch_engine,
            fifo_region_base_ + fifo::kFifoWrOffset,
            std::span(reinterpret_cast<uint8_t*>(&written), sizeof(written)),
            CoreType::DISPATCH);
        device_fifo.written = written;
        TT_FATAL(
            fifo_slots_retired(d) == written,
            "Device {}: the Tensor prefetcher kernels have {} unserved requests from an earlier run",
            devices_[d]->id(),
            written - fifo_slots_retired(d));
    }

    stop_requested_.store(false);
    host_worker_ = std::thread(&TensorPrefetcherManager::worker_loop_dispatch_engine, this);
    ++run_id_;
    active_ = true;
}

void TensorPrefetcherManager::validate_dispatch_engine_pages(
    const std::vector<std::vector<uint8_t>>& sender_pages, const RequestTarget& target) const {
    TT_FATAL(
        target.mapping.size() <= fifo_layout_.lanes,
        "Tensor prefetcher request names {} DRAM-sender endpoints; fast dispatch's prefetcher serves at most {}",
        target.mapping.size(),
        fifo_layout_.lanes);
    TT_FATAL(
        target.total_receivers <= fifo::kMaxReceivers,
        "Tensor prefetcher request has {} receivers; fast dispatch's prefetcher serves at most {}",
        target.total_receivers,
        fifo::kMaxReceivers);
    // The stage slot each of this target's blocks lands in, exactly as the kernels size it.
    const uint32_t stage_slot_bytes = fifo::stage_slot_bytes(
        fifo_layout_.stage_ring_bytes(mesh_device_->impl().metal_context().dispatch_mem_map().unused_l1_size()),
        target.total_receivers);
    for (const auto& page : sender_pages) {
        const TensorPrefetcherRequestHeader& header = header_of(page);
        if (header.base.cmd_id != DRAM_PREFETCHER_CMD_PREFETCH) {
            continue;
        }
        const auto* entries =
            reinterpret_cast<const TensorPrefetcherEntry*>(page.data() + sizeof(TensorPrefetcherRequestHeader));
        for (uint32_t e = 0; e < header.prefetch.num_entries; ++e) {
            const TensorPrefetcherEntry& entry = entries[e];
            TensorPrefetcherTensorLayout layout;
            std::memcpy(&layout, page.data() + entry.layout_offset, sizeof(layout));
            TT_FATAL(
                layout.layout_mode == static_cast<uint32_t>(TensorPrefetcherLayoutMode::ReceiverContiguous),
                "Fast dispatch's Tensor prefetcher serves receiver-contiguous weights only; tensor {} is K-row-major "
                "(#65545)",
                e);
            TT_FATAL(
                layout.streaming == 0,
                "Fast dispatch's Tensor prefetcher does not stream (tensor {} is streaming; #65545)",
                e);
            TT_FATAL(
                entry.num_groups == 1 && entry.selector_mode == TENSOR_PREFETCHER_SELECTOR_NONE,
                "Fast dispatch's Tensor prefetcher does not serve grouped or selected weights (tensor {}; #65545)",
                e);
            TT_FATAL(
                layout.page_bytes_per_recv <= stage_slot_bytes,
                "Tensor {} has {} B blocks; the dispatch engine stages {} receivers' blocks in {} slots of {} B each",
                e,
                layout.page_bytes_per_recv,
                target.total_receivers,
                fifo::kStageSlotsPerReceiver,
                stage_slot_bytes);
        }
    }
}

uint32_t TensorPrefetcherManager::fifo_slots_retired(uint32_t d) const {
    // The reader's and the pusher's counts, one cache line apart, in one read.
    constexpr uint32_t kRdWords = (fifo::kPusherRdOffset - fifo::kReaderRdOffset) / sizeof(uint32_t) + 1;
    std::array<uint32_t, kRdWords> rd{};
    ::tt::tt_metal::detail::ReadFromDeviceL1(
        devices_[d],
        fifos_[d].dispatch_engine,
        fifo_region_base_ + fifo::kReaderRdOffset,
        std::span(reinterpret_cast<uint8_t*>(rd.data()), sizeof(rd)),
        CoreType::DISPATCH);
    return std::min(rd.front(), rd.back());
}

void TensorPrefetcherManager::wait_for_fifo_retired(uint32_t d, uint32_t target) const {
    const auto timeout = mesh_device_->impl().metal_env().get_rtoptions().get_timeout_duration_for_operations();
    uint32_t retired = fifo_slots_retired(d);
    auto last_progress = std::chrono::steady_clock::now();
    // The counts only grow and may wrap, so compare their difference.
    while (static_cast<int32_t>(target - retired) > 0) {
        std::this_thread::yield();
        const uint32_t now_retired = fifo_slots_retired(d);
        const auto now = std::chrono::steady_clock::now();
        if (now_retired != retired) {
            retired = now_retired;
            last_progress = now;
        } else if (timeout.count() > 0 && now - last_progress > timeout) {
            mesh_device_->impl().metal_context().on_dispatch_timeout_detected();
            TT_THROW(
                "TIMEOUT: the Tensor prefetcher kernels on device {} retired no request for {} s (waiting for {}, at "
                "{}); potential hang detected",
                devices_[d]->id(),
                timeout.count(),
                target,
                retired);
        }
    }
}

void TensorPrefetcherManager::write_fifo_slot(uint32_t d, std::span<const uint8_t> slot) {
    IDevice* device = devices_[d];
    DeviceFifo& device_fifo = fifos_[d];
    auto& cluster = mesh_device_->impl().metal_env().get_cluster();
    // A slot is free once both kernels have retired it; they may still be pushing to receivers that
    // only drain after the caller's next workload launches, so wait without holding any API lock.
    wait_for_fifo_retired(d, device_fifo.written - fifo::kFifoSlots + 1);
    ::tt::tt_metal::detail::WriteToDeviceL1(
        device,
        device_fifo.dispatch_engine,
        fifo_region_base_ + fifo_layout_.slot_offset(device_fifo.written),
        slot,
        CoreType::DISPATCH);
    cluster.l1_barrier(device->id());
    // The slot is visible before the count that publishes it.
    const uint32_t count = ++device_fifo.written;
    ::tt::tt_metal::detail::WriteToDeviceL1(
        device,
        device_fifo.dispatch_engine,
        fifo_region_base_ + fifo::kFifoWrOffset,
        std::span(reinterpret_cast<const uint8_t*>(&count), sizeof(count)),
        CoreType::DISPATCH);
    cluster.l1_barrier(device->id());
}

void TensorPrefetcherManager::wait_for_fifo_idle() const {
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        wait_for_fifo_retired(d, fifos_[d].written);
    }
}

void TensorPrefetcherManager::worker_loop_dispatch_engine() {
    // One slot per request and device: the header, then the request page of each lane in use. A PREFETCH
    // request becomes one slot whose lanes are its senders' pages, so the kernels serve all of its
    // receivers together. A DRAIN request's lanes are independent, so it may take several slots.
    const auto slots_for = [this](const Request& request) {
        std::vector<std::vector<uint8_t>> slots;
        for (size_t first = 0; first < request.target_sender_indices.size(); first += fifo_layout_.lanes) {
            const size_t num_lanes = std::min<size_t>(fifo_layout_.lanes, request.target_sender_indices.size() - first);
            std::vector<uint8_t> slot(fifo::kSlotHeaderBytes + num_lanes * fifo::kLanePageBytes, 0);
            fifo::SlotHeader header{};
            header.command = fifo::kSlotRequest;
            header.num_lanes = static_cast<uint32_t>(num_lanes);
            for (size_t l = 0; l < num_lanes; ++l) {
                const uint32_t sender = request.target_sender_indices[first + l];
                header.lane_bank_id[l] = static_cast<uint8_t>(sender_logical_cores_[sender].x);
                const auto& page = request.sender_pages[request.sender_pages.size() == 1 ? 0 : first + l];
                // validate_request_pages holds every page to exactly one lane.
                std::memcpy(
                    slot.data() + fifo::kSlotHeaderBytes + l * fifo::kLanePageBytes, page.data(), fifo::kLanePageBytes);
            }
            std::memcpy(slot.data(), &header, sizeof(header));
            slots.push_back(std::move(slot));
        }
        return slots;
    };

    std::deque<std::shared_ptr<const Request>> incoming;
    while (true) {
        {
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait(lk, [&] { return !pending_.empty() || stop_requested_.load(); });
            if (pending_.empty()) {
                return;
            }
            incoming.swap(pending_);
            has_pending_.store(false);
        }
        for (const auto& request : incoming) {
            const auto slots = slots_for(*request);
            for (const auto& coord : request->target_devices) {
                const auto it = device_index_by_coord_.find(coord);
                TT_FATAL(
                    it != device_index_by_coord_.end(),
                    "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                    "started on",
                    coord);
                for (const auto& slot : slots) {
                    write_fifo_slot(it->second, slot);
                }
            }
        }
        incoming.clear();
    }
}

}  // namespace tt::tt_metal::distributed
