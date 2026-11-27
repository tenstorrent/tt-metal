// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

// `rand_tile` is documented as inclusive of `from + scale`. Shrink the scale by one ULP if rounding
// would push the top of the range past the intended upper bound.
inline uint32_t compute_rand_scale_bits(float lower, float upper) {
    float scale = upper - lower;
    uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (lower + scale > upper && scale_bits != 0U) {
        --scale_bits;
    }
    return scale_bits;
}

// Builds one program PER MESH COORDINATE rather than a single broadcast program: the RNG stream
// must differ per device on data-parallel axes, so create_descriptor() takes the device's
// coordinate and the framework calls it once per device. On a cache hit only
// override_runtime_arguments() runs: seed, the temperature value, the mask batch stride and, in
// position mode, the token count are left out of the program hash, so it re-applies them together
// with every buffer address, on every core.
struct GumbelSampleProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

}  // namespace ttml::metal::ops::gumbel_sample::device
