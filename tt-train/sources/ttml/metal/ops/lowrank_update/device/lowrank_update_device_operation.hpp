// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lowrank_update_device_operation_types.hpp"
#include "lowrank_update_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::lowrank_update::device {

struct LowrankUpdateDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<LowrankUpdateProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Tile geometry shared by the host code.
struct LowrankGeometry {
    uint32_t images = 0;  // B * C
    uint32_t Mt = 0;
    uint32_t Nt = 0;
};
LowrankGeometry lowrank_geometry(const ttnn::Tensor& acc);

}  // namespace ttml::metal::ops::lowrank_update::device

namespace ttnn::prim {

ttnn::Tensor ttml_lowrank_update(
    const ttnn::Tensor& acc, const ttnn::Tensor& a, const ttnn::Tensor& w, float scale, bool transpose_b);

}  // namespace ttnn::prim
