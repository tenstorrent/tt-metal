// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "impl/context/metal_env_accessor.hpp"

#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"

namespace tt::tt_metal::experimental {

namespace {

auto get_max_slots_per_core(const Hal& hal) {
    return hal.has_tile_counter_registers() ? static_cast<uint32_t>(::dfb::NUM_DFBS) : hal.get_num_dataflow_buffers();
}

auto get_arch(const Hal& hal) { return hal.get_arch(); }

CoreCoord get_compute_grid_size(MetalContext& metal_ctx) {
    MetalEnvImpl& env_impl = MetalEnvAccessor(metal_ctx.get_env()).impl();

    // Handle the mock device case (for cheap unit testing)
    const bool is_mock = metal_ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Mock;

    // A default DispatchCoreConfig and 1 CQ is sufficient to look up the compute grid size
    // from the YAML descriptor, and both are available in mock mode.
    DispatchCoreConfig dispatch_core_config{};
    uint8_t num_hw_cqs = 1;
    constexpr ChipId chip_id = 0;

    // But, best get the real dispatch_core_config and num_hw_cqs
    // (Makes no difference now, but hardbaking that assumption could be brittle)
    if (!is_mock) {
        auto& dispatch_mgr = metal_ctx.get_dispatch_core_manager();
        dispatch_core_config = dispatch_mgr.get_dispatch_core_config();
        num_hw_cqs = dispatch_mgr.get_num_hw_cqs();
    }

    // The compute_grid already accounts for the dispatch row/col
    // No need for dispatch-specific checks (and dispatch-specific error messages confuse users)
    return tt::get_compute_grid_size(env_impl, chip_id, num_hw_cqs, dispatch_core_config);
}

auto get_l1_alignment(const Hal& hal) { return hal.get_alignment(HalMemType::L1); }

}  // namespace

void ValidateProgramSpec(
    const ProgramSpec& spec, const CollectedSpecData& collected, MetalContext& metal_ctx, const Allocator& allocator) {
    const Hal& hal = metal_ctx.hal();
    // Sanity check for supported architecture.
    TT_FATAL(is_gen1_arch(hal) || is_gen2_arch(hal), "Unsupported architecture.");

    auto max_slots_per_core = get_max_slots_per_core(hal);
    auto arch = get_arch(hal);

    const ValidationContext ctx{.spec = spec, .collected = collected};

    // Order matters: later checks rely on earlier ones. Every local "> 0" / name-resolution check
    // passes before the structural checks that divide by or look up those values.
    ValidateProgramMisc(ctx);
    ValidateWorkUnitFields(ctx, get_compute_grid_size(metal_ctx));

    for (const auto& kernel : spec.kernels) {
        ValidateKernelSpec(kernel, ctx, arch);
    }
    ValidateResourceSpecs(ctx, arch, get_l1_alignment(hal), make_num_banks_from_buffer_type_fun(allocator));
    ValidateResourceUsage(ctx, arch);

    for (const auto& work_unit : spec.work_units) {
        ValidateWorkUnitSpec(work_unit, ctx, arch);
        ValidateDFBSlotsPerNode(work_unit, ctx, max_slots_per_core, arch);
    }
    // Checked over every node at once rather than per WorkUnitSpec: WorkUnitSpecs with the same name
    // are exempt from the overlap check, so a node's kernels can come from more than one WorkUnitSpec.
    ValidateGen1DMPlacement(ctx, arch);
    ValidateScratchpadBindersPerNode(ctx);

    // NOTE:
    // Placement consistency between kernels, DFBs, and WorkUnitSpecs is now structural,
    // not validated:
    //  - Kernels' effective node sets ARE the union of their containing WorkUnitSpecs' target_nodes
    //  - DFBs' allocation node sets are the union of their binding kernels' node sets
}

}  // namespace tt::tt_metal::experimental
