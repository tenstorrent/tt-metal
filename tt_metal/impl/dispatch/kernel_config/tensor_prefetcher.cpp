// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher.hpp"

#include <map>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"

namespace tt::tt_metal {

namespace fifo = tensor_prefetcher_fifo;

bool dispatch_engine_hosts_tensor_prefetcher(
    CoreType dispatch_core_type, uint32_t num_hw_cqs, const DispatchMemMap& dispatch_mem_map) {
    return dispatch_core_type == CoreType::DISPATCH && num_hw_cqs == 1 &&
           dispatch_mem_map.unused_l1_size() >= fifo::kMinRegionBytes;
}

TensorPrefetcherKernel::TensorPrefetcherKernel(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    DispatchWorkerType type,
    const ContextDescriptor& descriptor,
    dispatch_core_manager& dispatch_core_manager,
    const GetControlPlaneFn& get_control_plane,
    const GetDispatchQueryManagerFn& get_dispatch_query_manager,
    const GetReadsDispatchCoresFn& get_reads_dispatch_cores) :
    FDKernel(
        node_id,
        device_id,
        servicing_device_id,
        cq_id,
        noc_selection,
        descriptor,
        dispatch_core_manager,
        get_control_plane,
        get_dispatch_query_manager,
        get_reads_dispatch_cores),
    type_(type) {
    TT_ASSERT(type_ == TENSOR_PREFETCHER_READER || type_ == TENSOR_PREFETCHER_PUSHER);
    TT_FATAL(
        GetCoreType() == CoreType::DISPATCH,
        "Tensor prefetcher FD kernels run only on a dispatch engine (device {})",
        device_id);
    const uint16_t channel = descriptor.cluster().get_assigned_channel_for_device(device_id);
    // Shares the dispatch engine (and its L1) with this CQ's prefetch, dispatch and dispatch_s kernels.
    logical_core_ = dispatch_core_manager.dispatcher_core(device_id, channel, cq_id_);
    kernel_type_ = FDKernelType::DISPATCH;
}

void TensorPrefetcherKernel::GenerateStaticConfigs() {
    const DispatchMemMap& mem_map = get_dispatch_mem_map();
    region_base_ = mem_map.unused_l1_base();
    // The topology adds these kernels only where dispatch_engine_hosts_tensor_prefetcher holds.
    TT_ASSERT(mem_map.unused_l1_size() >= fifo::kMinRegionBytes);
}

void TensorPrefetcherKernel::CreateKernel() {
    const std::map<std::string, std::string> defines = {
        {"TENSOR_PREFETCHER_REGION_BASE", std::to_string(region_base_)},
        {"TENSOR_PREFETCHER_STAGE_RING_SIZE",
         std::to_string(fifo::stage_ring_bytes(get_dispatch_mem_map().unused_l1_size()))},
    };
    configure_kernel_variant(dispatch_kernel_file_names[type_], {}, defines);
}

void TensorPrefetcherKernel::ConfigureCore() {
    // Both kernels share the region; the reader's node clears it once per launch. Nothing is in flight:
    // fast dispatch has not launched either kernel yet.
    if (type_ != TENSOR_PREFETCHER_READER) {
        return;
    }
    const CoreCoord core(logical_core_.x, logical_core_.y);
    std::vector<uint32_t> control(fifo::kControlBytes / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device_, core, region_base_, control, CoreType::DISPATCH);
    std::vector<uint32_t> counters((fifo::kSenderStateZoneOffset - fifo::kStagedOffset) / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device_, core, region_base_ + fifo::kStagedOffset, counters, CoreType::DISPATCH);
}

std::optional<TerminationInfo> TensorPrefetcherKernel::GetTerminationInfo() const {
    // Both kernels watch the same word, so both nodes report the same signal and teardown writes it once.
    return TerminationInfo{
        .logical_core = CoreCoord(logical_core_.x, logical_core_.y),
        .core_type = CoreType::DISPATCH,
        .address = region_base_ + fifo::kTerminateOffset,
        .val = 1,
        .timing = TerminationTiming::BeforeDispatchWait,
    };
}

}  // namespace tt::tt_metal
