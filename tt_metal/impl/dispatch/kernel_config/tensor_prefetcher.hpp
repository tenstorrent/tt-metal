// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

// One of the two Tensor prefetcher kernels that fast dispatch launches on a dispatch engine beside the CQ
// kernels, and terminates with them. The reader stages weight blocks from DRAM into the engine's L1; the
// pusher sends staged blocks on to the receivers. Both serve a request FIFO the host writes into the L1
// region DispatchMemMap::tensor_prefetcher_region_base() names (layout: impl/buffers/tensor_prefetcher_fifo.hpp).
class TensorPrefetcherKernel : public FDKernel {
public:
    enum class Role : uint8_t { Reader, Pusher };

    TensorPrefetcherKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        Role role,
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
    Role role_;
    uint32_t region_base_ = 0;
};

}  // namespace tt::tt_metal
