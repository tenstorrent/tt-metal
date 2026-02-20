// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefetch_writer.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>
#include <map>
#include <string>

#include <tt_stl/assert.hpp>
#include "device.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "prefetch.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

using namespace tt::tt_metal;

PrefetchWriterKernel::PrefetchWriterKernel(
    int node_id, ChipId device_id, ChipId servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
    FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
    uint16_t channel = MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
    // Runs on the same physical core as the upstream PrefetchKernel (BRISC). Core is set in
    // GenerateStaticConfigs() once the upstream kernel is known.
    this->logical_core_ =
        MetalContext::instance().get_dispatch_core_manager().prefetcher_core(device_id, channel, cq_id);
    this->kernel_type_ = FDKernelType::DISPATCH;
}

void PrefetchWriterKernel::GenerateStaticConfigs() {
    // The writer stub has no independent static config: it shares the core and all defines
    // with the upstream PrefetchKernel. Nothing to do here.
}

void PrefetchWriterKernel::GenerateDependentConfigs() {
    TT_ASSERT(upstream_kernels_.size() == 1);
    auto* prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
    TT_ASSERT(prefetch_kernel, "PrefetchWriterKernel upstream must be a PrefetchKernel");
    // Confirm we are on the same core.
    TT_ASSERT(
        logical_core_ == prefetch_kernel->GetLogicalCore(),
        "PrefetchWriterKernel must be on the same core as its upstream PrefetchKernel");
}

void PrefetchWriterKernel::CreateKernel() {
    TT_ASSERT(upstream_kernels_.size() == 1);
    auto* prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
    TT_ASSERT(prefetch_kernel, "PrefetchWriterKernel upstream must be a PrefetchKernel");

    // Use the same defines as the reader so both kernels compile with an identical config set.
    auto defines = prefetch_kernel->GetDefines();

    auto optimization_level = (GetCoreType() == CoreType::WORKER) ? KernelBuildOptLevel::O2 : KernelBuildOptLevel::Os;
    configure_kernel_variant(
        dispatch_kernel_file_names[PREFETCH_HD_WRITER],
        {},
        defines,
        false,
        false,  // send_to_brisc=false => NCRISC (RISCV_1)
        false,
        optimization_level);
}
