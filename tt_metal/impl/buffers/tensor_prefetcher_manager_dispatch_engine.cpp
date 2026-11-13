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
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::distributed {

namespace {

namespace fifo = tensor_prefetcher_fifo;

uint32_t read_region_word(IDevice* device, const CoreCoord& dispatch_engine, uint32_t addr) {
    std::vector<uint32_t> word;
    ::tt::tt_metal::detail::ReadFromDeviceL1(device, dispatch_engine, addr, sizeof(uint32_t), word, CoreType::DISPATCH);
    return word.at(0);
}

const TensorPrefetcherRequestHeader& header_of(const std::vector<uint8_t>& page) {
    return *reinterpret_cast<const TensorPrefetcherRequestHeader*>(page.data());
}

}  // namespace

void TensorPrefetcherManager::start_on_dispatch_engine() {
    auto& metal = MetalContext::instance(mesh_device_->impl().get_context_id());
    const DispatchMemMap& mem_map = metal.dispatch_mem_map();
    fifo_region_base_ = mem_map.tensor_prefetcher_region_base();

    // The request serializer sizes layouts from the stage ring as the DRAM-core kernel splits it (halves
    // and thirds). The dispatch-engine kernels split it per receiver instead, which queue time checks.
    const uint32_t l1_alignment = metal.hal().get_alignment(HalMemType::L1);
    const uint32_t ring_size_align = 6u * l1_alignment;
    stage_ring_base_ = fifo_region_base_ + fifo::kStageRingOffset;
    stage_ring_size_ =
        fifo::stage_ring_bytes(mem_map.tensor_prefetcher_region_size()) / ring_size_align * ring_size_align;
    ring_half_ = stage_ring_size_ / 2;
    stage_third_ = stage_ring_size_ / 3;

    // The kernels run for as long as fast dispatch does and keep their counters across prefetcher runs,
    // so a run resumes at the device's count. An earlier run's stop waited for them to go idle.
    fifo_slots_written_.assign(devices_.size(), 0);
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        const CoreCoord dispatch_engine =
            dram_sender_dispatch_engine(mesh_device_->impl().get_context_id(), devices_[d]->id());
        fifo_slots_written_[d] =
            read_region_word(devices_[d], dispatch_engine, fifo_region_base_ + fifo::kFifoWrOffset);
        TT_FATAL(
            fifo_slots_retired(d) == fifo_slots_written_[d],
            "Device {}: the Tensor prefetcher kernels have {} unserved requests from an earlier run",
            devices_[d]->id(),
            fifo_slots_written_[d] - fifo_slots_retired(d));
    }

    stop_requested_.store(false);
    host_worker_ = std::thread(&TensorPrefetcherManager::worker_loop_dispatch_engine, this);
    ++run_id_;
    active_ = true;
}

void TensorPrefetcherManager::validate_dispatch_engine_request(const Request& request, uint32_t total_receivers) const {
    TT_FATAL(
        request.target_sender_indices.size() <= fifo::kMaxLanes,
        "Tensor prefetcher request names {} DRAM-sender endpoints; fast dispatch's prefetcher serves at most {}",
        request.target_sender_indices.size(),
        fifo::kMaxLanes);
    TT_FATAL(
        total_receivers <= fifo::kMaxReceivers,
        "Tensor prefetcher request has {} receivers; fast dispatch's prefetcher serves at most {}",
        total_receivers,
        fifo::kMaxReceivers);
    for (const auto& page : request.sender_pages) {
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
                layout.layout_mode == fifo::kReceiverContiguousLayout,
                "Fast dispatch's Tensor prefetcher serves receiver-contiguous weights only; tensor {} is K-row-major",
                e);
            TT_FATAL(
                layout.streaming == 0, "Fast dispatch's Tensor prefetcher does not stream (tensor {} is streaming)", e);
            TT_FATAL(
                entry.num_groups == 1 && entry.selector_mode == TENSOR_PREFETCHER_SELECTOR_NONE,
                "Fast dispatch's Tensor prefetcher does not serve grouped or selected weights (tensor {})",
                e);
            const uint64_t stage_bytes =
                static_cast<uint64_t>(total_receivers) * fifo::kStageSlotsPerReceiver * layout.page_bytes_per_recv;
            TT_FATAL(
                stage_bytes <= stage_ring_size_,
                "Tensor {} needs {} B of stage ring ({} receivers x {} blocks of {} B); the dispatch engine has {} B",
                e,
                stage_bytes,
                total_receivers,
                fifo::kStageSlotsPerReceiver,
                layout.page_bytes_per_recv,
                stage_ring_size_);
        }
    }
}

uint32_t TensorPrefetcherManager::fifo_slots_retired(uint32_t d) const {
    const CoreCoord dispatch_engine =
        dram_sender_dispatch_engine(mesh_device_->impl().get_context_id(), devices_[d]->id());
    return std::min(
        read_region_word(devices_[d], dispatch_engine, fifo_region_base_ + fifo::kReaderRdOffset),
        read_region_word(devices_[d], dispatch_engine, fifo_region_base_ + fifo::kPusherRdOffset));
}

void TensorPrefetcherManager::write_fifo_slot(uint32_t d, const std::vector<uint32_t>& slot) {
    IDevice* device = devices_[d];
    const CoreCoord dispatch_engine = dram_sender_dispatch_engine(mesh_device_->impl().get_context_id(), device->id());
    auto& cluster = MetalContext::instance(mesh_device_->impl().get_context_id()).get_cluster();
    uint32_t& written = fifo_slots_written_[d];
    // A slot is free once both kernels have retired it; they may still be pushing to receivers that
    // only drain after the caller's next workload launches, so wait without holding any API lock.
    while (written - fifo_slots_retired(d) >= fifo::kFifoSlots) {
        std::this_thread::yield();
    }
    std::vector<uint32_t> words(slot);
    ::tt::tt_metal::detail::WriteToDeviceL1(
        device, dispatch_engine, fifo_region_base_ + fifo::slot_offset(written), words, CoreType::DISPATCH);
    cluster.l1_barrier(device->id());
    // The slot is visible before the count that publishes it.
    std::vector<uint32_t> count = {++written};
    ::tt::tt_metal::detail::WriteToDeviceL1(
        device, dispatch_engine, fifo_region_base_ + fifo::kFifoWrOffset, count, CoreType::DISPATCH);
    cluster.l1_barrier(device->id());
}

void TensorPrefetcherManager::wait_for_fifo_idle() const {
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        while (fifo_slots_retired(d) != fifo_slots_written_[d]) {
            std::this_thread::yield();
        }
    }
}

void TensorPrefetcherManager::worker_loop_dispatch_engine() {
    constexpr uint32_t kSlotWords = fifo::kSlotBytes / sizeof(uint32_t);
    constexpr uint32_t kLaneWords = fifo::kLanePageBytes / sizeof(uint32_t);
    // One slot per request and device. A PREFETCH request becomes one slot whose lanes are its senders'
    // pages, so the kernels serve all of its receivers together. A DRAIN request's lanes are independent,
    // so it may take several slots. STOP pages are dropped: the kernels outlive the prefetcher run.
    const auto slots_for = [&](const Request& request) {
        std::vector<std::vector<uint32_t>> slots;
        if (request.target_sender_indices.empty()) {
            TT_FATAL(
                header_of(request.sender_pages.front()).base.cmd_id == DRAM_PREFETCHER_CMD_STOP,
                "Tensor prefetcher: a request for every sender must be STOP when fast dispatch runs the prefetcher");
            return slots;
        }
        for (size_t first = 0; first < request.target_sender_indices.size(); first += fifo::kMaxLanes) {
            const size_t num_lanes = std::min<size_t>(fifo::kMaxLanes, request.target_sender_indices.size() - first);
            std::vector<uint32_t> slot(kSlotWords, 0);
            fifo::SlotHeader header{};
            header.command = fifo::kSlotRequest;
            header.num_lanes = static_cast<uint32_t>(num_lanes);
            for (size_t l = 0; l < num_lanes; ++l) {
                const uint32_t sender = request.target_sender_indices[first + l];
                header.lane_bank_id[l] = static_cast<uint32_t>(sender_logical_cores_[sender].x);
                const auto& page = request.sender_pages[request.sender_pages.size() == 1 ? 0 : first + l];
                TT_FATAL(page.size() >= fifo::kLanePageBytes, "Tensor prefetcher request page is {} B", page.size());
                std::memcpy(
                    slot.data() + fifo::kSlotHeaderBytes / sizeof(uint32_t) + l * kLaneWords,
                    page.data(),
                    fifo::kLanePageBytes);
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
