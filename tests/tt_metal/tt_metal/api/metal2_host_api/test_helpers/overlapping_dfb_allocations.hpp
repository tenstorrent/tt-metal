// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// The #62451 layout: A/1024, B/2048, then A-B/1024. Each DFB is
// self-looped by a no-op Gen1 DM kernel, so placement needs no overrides.
inline ProgramSpec MakeOverlappingDFBAllocationSpec() {
    ProgramSpec spec;
    spec.name = "overlapping_dfb_allocations";
    spec.dataflow_buffers = {
        MakeMinimalDFB("d_a", 256, 4), MakeMinimalDFB("d_b", 256, 8), MakeMinimalDFB("d_ab", 256, 4)};
    for (size_t i = 0; i < spec.dataflow_buffers.size(); ++i) {
        const auto& dfb = spec.dataflow_buffers[i];
        auto kernel = MakeMinimalGen1DMKernel(
            "k" + std::to_string(i), i == 2 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1);
        kernel.dfb_bindings = {
            {.dfb_spec_name = dfb.unique_id, .accessor_name = "buf", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = dfb.unique_id, .accessor_name = "buf", .endpoint_type = DFBEndpointType::CONSUMER}};
        spec.kernels.push_back(std::move(kernel));
    }
    spec.work_units = {
        MakeMinimalWorkUnit("left", NodeCoord{0, 0}, {"k0", "k2"}),
        MakeMinimalWorkUnit("right", NodeCoord{1, 0}, {"k1", "k2"})};
    return spec;
}

}  // namespace tt::tt_metal::experimental::test_helpers
