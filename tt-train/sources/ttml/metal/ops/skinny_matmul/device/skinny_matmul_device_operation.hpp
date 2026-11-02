// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "skinny_matmul_device_operation_types.hpp"
#include "skinny_matmul_program_factory.hpp"

namespace ttml::metal::ops::skinny_matmul::device {

struct SkinnyMatmulPartialDeviceOperation {
    using operation_attributes_t = partial::operation_attributes_t;
    using tensor_args_t = partial::tensor_args_t;
    using spec_return_value_t = partial::spec_return_value_t;
    using tensor_return_value_t = partial::tensor_return_value_t;
    using program_factory_t = std::variant<SkinnyMatmulPartialProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

struct SkinnyMatmulReduceDeviceOperation {
    using operation_attributes_t = reduce::operation_attributes_t;
    using tensor_args_t = reduce::tensor_args_t;
    using spec_return_value_t = reduce::spec_return_value_t;
    using tensor_return_value_t = reduce::tensor_return_value_t;
    using program_factory_t = std::variant<SkinnyMatmulReduceProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Geometry shared by the host code: N tiles of the output and K tiles.
struct SkinnyGeometry {
    uint32_t rows = 0;  // B * C * Mt
    uint32_t Kt = 0;    // K tiles of the product (w's K)
    uint32_t Kx = 0;    // x row stride in tiles (x may be wider than K: the wide LoRA input buffer)
    uint32_t Nt = 0;
    uint32_t n_logical = 0;
};
SkinnyGeometry skinny_geometry(
    const ttnn::Tensor& x, const ttnn::Tensor& w, bool transpose_b, bool x_transposed = false);

}  // namespace ttml::metal::ops::skinny_matmul::device

namespace ttnn::prim {

ttnn::Tensor ttml_skinny_matmul_partial(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    const ttml::metal::ops::skinny_matmul::device::SkinnySplit& split,
    bool transpose_b,
    bool x_transposed = false);

ttnn::Tensor ttml_skinny_matmul_reduce(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    const ttnn::Tensor& partials,
    const ttml::metal::ops::skinny_matmul::device::SkinnySplit& split,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& tail_out = std::nullopt,
    uint32_t tail_col = 0,
    uint32_t tail_tiles = 0,
    bool x_transposed = false,
    bool transpose_out = false);

}  // namespace ttnn::prim
