// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// Create endpoint kernels that bind each DFB on node (0, 0).
inline ProgramSpec MakeDFBPlacementSpec(tt::ARCH arch, uint32_t count = 1) {
    auto spec = arch == tt::ARCH::QUASAR ? MakeMinimalValidProgramSpec() : MakeMinimalGen1ValidProgramSpec();
    spec.dataflow_buffers.clear();
    spec.kernels[0].dfb_bindings.clear();
    spec.kernels[1].dfb_bindings.clear();
    for (uint32_t i = 0; i < count; ++i) {
        const std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
        spec.kernels[0].dfb_bindings.push_back(ProducerOf(DFBSpecName{name}, name));
        spec.kernels[1].dfb_bindings.push_back(ConsumerOf(DFBSpecName{name}, name));
    }
    return spec;
}

inline ProgramSpec MakeDisjointDFBPlacementsForTest(tt::ARCH arch) {
    auto spec = MakeDFBPlacementSpec(arch, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    return spec;
}

inline ProgramSpec MakeOverlappingDFBPlacementsForTest() {
    auto spec = MakeDFBPlacementSpec(tt::ARCH::WORMHOLE_B0, 3);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {3, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}, {3, 0}));
    spec.dataflow_buffers[2].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    spec.dataflow_buffers[0].entry_size = 128;
    spec.dataflow_buffers[1].entry_size = 384;
    spec.dataflow_buffers[2].entry_size = 64;
    return spec;
}

// Create two DFBs whose endpoint pairs run on separate nodes.
inline ProgramSpec MakeSeparateDFBEndpointPairs(tt::ARCH arch) {
    auto spec = MakeDFBPlacementSpec(arch);
    auto other = MakeDFBPlacementSpec(arch);
    other.kernels[0].unique_id = KernelSpecName{"other_producer"};
    other.kernels[1].unique_id = KernelSpecName{"other_consumer"};
    other.kernels[0].dfb_bindings[0].dfb_spec_name = DFBSpecName{"dfb_1"};
    other.kernels[1].dfb_bindings[0].dfb_spec_name = DFBSpecName{"dfb_1"};
    other.dataflow_buffers[0].unique_id = DFBSpecName{"dfb_1"};
    spec.kernels.push_back(other.kernels[0]);
    spec.kernels.push_back(other.kernels[1]);
    spec.dataflow_buffers.push_back(other.dataflow_buffers[0]);
    spec.work_units.push_back(MakeMinimalWorkUnit("other", NodeCoord{1, 0}, {"other_producer", "other_consumer"}));
    return spec;
}

// Create a Gen1 DFB with two producer kernels on the same node.
inline ProgramSpec MakeDFBWithTwoGen1Producers() {
    auto spec = MakeDFBPlacementSpec(tt::ARCH::WORMHOLE_B0);
    auto second = MakeMinimalGen1DMKernel("second", DataMovementProcessor::RISCV_1);
    second.dfb_bindings = {ProducerOf(DFBSpecName{"dfb_0"}, "out")};
    spec.kernels.push_back(second);
    spec.work_units[0].kernels.push_back(KernelSpecName{"second"});
    return spec;
}

inline uint32_t TensixIndexForTest(const Program& program) {
    return MetalContext::instance(program.impl().get_context_id())
        .hal()
        .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

// Finalize DFB configuration sizes and launch masks without requiring mock kernel binaries.
inline void FinalizeMockDFBConfig(Program& program) {
    auto& impl = program.impl();
    const auto& context = MetalContext::instance(impl.get_context_id());
    const uint32_t index = context.hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    auto& groups = impl.get_kernel_groups(index);
    auto& config = impl.get_program_config(index);
    const uint32_t offset = tt::tt_metal::program_dispatch::finalize_cbs(
        context, index, groups, 0, config.cb_offset, config.cb_size, config.local_cb_size);
    dfb::detail::finalize_dfbs(index, groups, impl.dataflow_buffers(), offset, config.dfb_offset, config.dfb_size);
    tt::tt_metal::program_dispatch::finalize_dfb_masks(context, groups, impl.dataflow_buffers());
}

}  // namespace tt::tt_metal::experimental::test_helpers
