// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "fd_kernel.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

class DispatchMemMap;

// One Tensor prefetcher kernel that fast dispatch launches on a dispatch engine beside the CQ kernels, and
// terminates with them.
struct TensorPrefetcherKernelInfo {
    // Kernel source file.
    const char* file = nullptr;
    // L1 the kernel keeps for itself in the region, zeroed before fast dispatch launches it.
    uint32_t state_bytes = 0;
    // NoC transaction ids the kernel tags its transactions with, at least one: one for each group of transactions it
    // keeps in flight and waits on by itself. All the kernels' together fit in tensor_prefetcher_fifo::kMaxTrids.
    uint32_t trids = 1;
};

// The Tensor prefetcher kernels in the order blocks flow through them. Kernel `k` retires request slots as
// FIFO consumer `k` (tensor_prefetcher_fifo::fifo_rd_offset). The first kernel plans how many blocks of each
// tensor flow, and the rest follow its plan (kernels/tensor_prefetcher_fifo_device.h). Defined in
// tensor_prefetcher_kernels.cpp.
std::span<const TensorPrefetcherKernelInfo> tensor_prefetcher_kernels();

// The layout of the dispatch engine's Tensor prefetcher region on `cluster`: one lane per DRAM bank of its
// devices, each bank's one sender endpoint, and the state of every kernel in tensor_prefetcher_kernels().
tensor_prefetcher_fifo::RegionLayout tensor_prefetcher_region_layout(const Cluster& cluster);

// Whether fast dispatch runs the Tensor prefetcher kernels beside the CQ kernels: only on a dispatch
// engine (not the Tensix dispatch cores) serving a single CQ, with the region its layout needs unused above
// the CQ zones, on devices with no more DRAM banks than a slot can name (tensor_prefetcher_fifo::kMaxLanes).
bool dispatch_engine_hosts_tensor_prefetcher(
    const Cluster& cluster, CoreType dispatch_core_type, uint32_t num_hw_cqs, const DispatchMemMap& dispatch_mem_map);

// The logical dispatch engine of device `device_id` that runs the Tensor prefetcher kernels: CQ 0's, which is
// the first core of the pool fast dispatch assigns CQ 0's kernels from. Read from that pool, not assigned, so
// device init can name it in the firmware's tables before fast dispatch assigns any core.
CoreCoord tensor_prefetcher_dispatch_engine(dispatch_core_manager& dispatch_core_manager, ChipId device_id);

// One of the kernels in tensor_prefetcher_kernels(). The topology chains their nodes in that order, so a
// node's index is the number of nodes upstream of it. They serve a request FIFO the host writes into the L1
// region DispatchMemMap::unused_l1_base() names (layout: impl/buffers/tensor_prefetcher_fifo.hpp).
class TensorPrefetcherKernel : public FDKernel {
public:
    TensorPrefetcherKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
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
    // This kernel's index in tensor_prefetcher_kernels().
    uint32_t index() const;

    uint32_t region_base_ = 0;
    tensor_prefetcher_fifo::RegionLayout layout_;
};

}  // namespace tt::tt_metal
