// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

class DispatchMemMap;

// The layout of the dispatch engine's Tensor prefetcher region on `cluster`: one lane per DRAM bank of its
// devices, each bank's one sender endpoint.
tensor_prefetcher_fifo::RegionLayout tensor_prefetcher_region_layout(const Cluster& cluster);

// Whether fast dispatch runs the Tensor prefetcher kernels beside the CQ kernels: only on a dispatch
// engine (not the Tensix dispatch cores) serving a single CQ, with the region its layout needs unused above
// the CQ zones, on devices with no more DRAM banks than a slot can name (tensor_prefetcher_fifo::kMaxLanes).
bool dispatch_engine_hosts_tensor_prefetcher(
    const Cluster& cluster, CoreType dispatch_core_type, uint32_t num_hw_cqs, const DispatchMemMap& dispatch_mem_map);

// One of the two Tensor prefetcher kernels that fast dispatch launches on a dispatch engine beside the CQ
// kernels, and terminates with them. The reader stages weight blocks from DRAM into the engine's L1; the
// pusher sends staged blocks on to the receivers. Both serve a request FIFO the host writes into the L1
// region DispatchMemMap::unused_l1_base() names (layout: impl/buffers/tensor_prefetcher_fifo.hpp).
class TensorPrefetcherKernel : public FDKernel {
public:
    // `type` is TENSOR_PREFETCHER_READER or TENSOR_PREFETCHER_PUSHER.
    TensorPrefetcherKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        DispatchWorkerType type,
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override {}
    void CreateKernel() override;
    void ConfigureCore() override;
    std::optional<TerminationInfo> GetTerminationInfo() const override;

private:
    DispatchWorkerType type_;
    uint32_t region_base_ = 0;
    tensor_prefetcher_fifo::RegionLayout layout_;
};

}  // namespace tt::tt_metal
