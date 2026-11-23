// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "hostdev/remote_dfb_config_layout.h"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {
namespace fifo = tensor_prefetcher_fifo;

// tensor_prefetcher_fifo::kSenderStatePageBytes is derived from these.
static_assert(PREFETCHER_PIPE_CONFIG_HEADER_WORDS == 10 && PREFETCHER_PIPE_MAX_CREDIT_LANES == 4);
static_assert(
    fifo::kSenderStatePageBytes >= prefetcher_pipe_noc_xy_byte_offset() + 2 * fifo::kMaxReceivers * sizeof(uint32_t) +
                                       2 * fifo::kMaxReceivers * PREFETCHER_PIPE_MAX_CREDIT_LANES * 16,
    "A sender config page for kMaxReceivers receivers must fit kSenderStatePageBytes");

// A kernel's state rounded up to whole cache lines, which is what the region gives it.
uint32_t kernel_state_zone_bytes(const TensorPrefetcherKernelInfo& kernel) {
    return (kernel.state_bytes + fifo::kLineBytes - 1) / fifo::kLineBytes * fifo::kLineBytes;
}
}  // namespace

fifo::RegionLayout tensor_prefetcher_region_layout(const Cluster& cluster) {
    uint32_t num_dram_banks = 1;
    for (const ChipId chip : cluster.all_chip_ids()) {
        num_dram_banks = std::max<uint32_t>(num_dram_banks, cluster.get_soc_desc(chip).get_num_dram_views());
    }
    const auto kernels = tensor_prefetcher_kernels();
    TT_FATAL(
        !kernels.empty() && kernels.size() <= fifo::kMaxKernels,
        "The Tensor prefetcher runs 1 to {} kernels on a dispatch engine, not {}",
        fifo::kMaxKernels,
        kernels.size());
    uint32_t kernel_state_bytes = 0;
    for (const TensorPrefetcherKernelInfo& kernel : kernels) {
        kernel_state_bytes += kernel_state_zone_bytes(kernel);
    }
    return fifo::RegionLayout{.lanes = num_dram_banks, .kernel_state_bytes = kernel_state_bytes};
}

bool dispatch_engine_hosts_tensor_prefetcher(
    const Cluster& cluster, CoreType dispatch_core_type, uint32_t num_hw_cqs, const DispatchMemMap& dispatch_mem_map) {
    if (dispatch_core_type != CoreType::DISPATCH || num_hw_cqs != 1) {
        return false;
    }
    const fifo::RegionLayout layout = tensor_prefetcher_region_layout(cluster);
    return layout.lanes <= fifo::kMaxLanes && dispatch_mem_map.unused_l1_size() >= layout.min_region_bytes();
}

TensorPrefetcherKernel::TensorPrefetcherKernel(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
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
        get_reads_dispatch_cores) {
    TT_FATAL(
        GetCoreType() == CoreType::DISPATCH,
        "Tensor prefetcher FD kernels run only on a dispatch engine (device {})",
        device_id);
    const uint16_t channel = descriptor.cluster().get_assigned_channel_for_device(device_id);
    // Shares the dispatch engine (and its L1) with this CQ's prefetch, dispatch and dispatch_s kernels.
    logical_core_ = dispatch_core_manager.dispatcher_core(device_id, channel, cq_id_);
    const CoreCoord named = tensor_prefetcher_dispatch_engine(dispatch_core_manager, device_id);
    TT_FATAL(
        CoreCoord(logical_core_.x, logical_core_.y) == named,
        "Device {}: fast dispatch placed the Tensor prefetcher kernels on dispatch engine {}, but device init named "
        "{} as theirs",
        device_id,
        CoreCoord(logical_core_.x, logical_core_.y),
        named);
    kernel_type_ = FDKernelType::DISPATCH;
}

CoreCoord tensor_prefetcher_dispatch_engine(dispatch_core_manager& dispatch_core_manager, ChipId device_id) {
    const std::vector<CoreCoord> pool = dispatch_core_manager.get_all_logical_dispatch_cores(device_id);
    TT_FATAL(!pool.empty(), "Device {} has no dispatch cores to run the Tensor prefetcher on", device_id);
    return pool.front();
}

uint32_t TensorPrefetcherKernel::index() const {
    if (upstream_kernels_.empty()) {
        return 0;
    }
    TT_ASSERT(upstream_kernels_.size() == 1, "A Tensor prefetcher kernel has at most one upstream kernel");
    const auto* upstream = dynamic_cast<const TensorPrefetcherKernel*>(upstream_kernels_.front());
    TT_ASSERT(upstream != nullptr, "A Tensor prefetcher kernel's upstream kernel is another Tensor prefetcher kernel");
    return upstream->index() + 1;
}

void TensorPrefetcherKernel::GenerateStaticConfigs() {
    const DispatchMemMap& mem_map = get_dispatch_mem_map();
    region_base_ = mem_map.unused_l1_base();
    layout_ = tensor_prefetcher_region_layout(descriptor_.cluster());
    // The topology adds these kernels only where dispatch_engine_hosts_tensor_prefetcher holds.
    TT_ASSERT(mem_map.unused_l1_size() >= layout_.min_region_bytes());
}

void TensorPrefetcherKernel::CreateKernel() {
    const auto kernels = tensor_prefetcher_kernels();
    const uint32_t k = index();
    TT_ASSERT(k < kernels.size());
    uint32_t state_offset = layout_.kernel_state_zone_offset();
    for (uint32_t j = 0; j < k; ++j) {
        state_offset += kernel_state_zone_bytes(kernels[j]);
    }
    const std::map<std::string, std::string> defines = {
        {"TENSOR_PREFETCHER_REGION_BASE", std::to_string(region_base_)},
        {"TENSOR_PREFETCHER_LANES", std::to_string(layout_.lanes)},
        {"TENSOR_PREFETCHER_KERNEL_STATE_BYTES", std::to_string(layout_.kernel_state_bytes)},
        {"TENSOR_PREFETCHER_STAGE_RING_SIZE",
         std::to_string(layout_.stage_ring_bytes(get_dispatch_mem_map().unused_l1_size()))},
        {"TENSOR_PREFETCHER_KERNEL_INDEX", std::to_string(k)},
        {"TENSOR_PREFETCHER_STATE_BASE", std::to_string(region_base_ + state_offset)},
        {"TENSOR_PREFETCHER_STATE_SIZE", std::to_string(kernels[k].state_bytes)},
    };
    configure_kernel_variant(kernels[k].file, {}, defines);
}

void TensorPrefetcherKernel::ConfigureCore() {
    // The kernels share the region; the first kernel's node clears it once per launch: the control block, the
    // reader/pusher counters and every kernel's state. Nothing is in flight: fast dispatch has not launched any
    // of them yet.
    if (index() != 0) {
        return;
    }
    const CoreCoord core(logical_core_.x, logical_core_.y);
    std::vector<uint32_t> control(fifo::kControlBytes / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device_, core, region_base_, control, CoreType::DISPATCH);
    std::vector<uint32_t> counters(
        (layout_.sender_state_zone_offset() - layout_.staged_offset()) / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device_, core, region_base_ + layout_.staged_offset(), counters, CoreType::DISPATCH);
    if (layout_.kernel_state_bytes > 0) {
        std::vector<uint32_t> state(layout_.kernel_state_bytes / sizeof(uint32_t), 0);
        detail::WriteToDeviceL1(
            device_, core, region_base_ + layout_.kernel_state_zone_offset(), state, CoreType::DISPATCH);
    }
}

std::optional<TerminationInfo> TensorPrefetcherKernel::GetTerminationInfo() const {
    // Every kernel watches the same word, so every node reports the same signal and teardown writes it once.
    return TerminationInfo{
        .logical_core = CoreCoord(logical_core_.x, logical_core_.y),
        .core_type = CoreType::DISPATCH,
        .address = region_base_ + fifo::kTerminateOffset,
        .val = 1,
    };
}

}  // namespace tt::tt_metal
