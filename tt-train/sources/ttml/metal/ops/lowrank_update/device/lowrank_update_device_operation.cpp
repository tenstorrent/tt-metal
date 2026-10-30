// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lowrank_update_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::lowrank_update::device {

LowrankGeometry lowrank_geometry(const ttnn::Tensor& acc) {
    const auto& p = acc.padded_shape();
    return LowrankGeometry{
        .images = p[0] * p[1], .Mt = p[2] / tt::constants::TILE_HEIGHT, .Nt = p[3] / tt::constants::TILE_WIDTH};
}

void LowrankUpdateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& acc = tensor_args.acc;
    const auto& a = tensor_args.a;
    const auto& w = tensor_args.w;
    const DeviceTensorRequirements dram_bf16{.buffer_type = tt::tt_metal::BufferType::DRAM};
    check_device_tensor(acc, "LowrankUpdate", "acc", dram_bf16);
    check_device_tensor(a, "LowrankUpdate", "a", dram_bf16);
    check_device_tensor(w, "LowrankUpdate", "w", dram_bf16);
    const auto& as = acc.logical_shape();
    const auto& xs = a.logical_shape();
    const auto& ws = w.logical_shape();
    TT_FATAL(
        as.rank() == 4 && xs.rank() == 4 && ws.rank() == 4, "LowrankUpdate: acc {}, a {}, w {} must be 4D", as, xs, ws);
    TT_FATAL(
        xs[0] == as[0] && xs[1] == as[1] && xs[2] == as[2], "LowrankUpdate: a {} must match acc {} in B, C, M", xs, as);
    TT_FATAL(ws[0] == 1 && ws[1] == 1, "LowrankUpdate: w must be [1, 1, ., .], got {}", ws);
    const uint32_t K = xs[3];
    const uint32_t wK = args.transpose_b ? ws[3] : ws[2];
    const uint32_t wN = args.transpose_b ? ws[2] : ws[3];
    TT_FATAL(K >= 1 && K <= tt::constants::TILE_WIDTH, "LowrankUpdate: K must be <= 32, got {}", K);
    TT_FATAL(wK == K, "LowrankUpdate: K of a ({}) and w ({}) differ (transpose_b={})", K, wK, args.transpose_b);
    TT_FATAL(wN == as[3], "LowrankUpdate: N of w ({}) and acc ({}) differ", wN, as[3]);
    TT_FATAL(
        a.padded_shape()[3] == tt::constants::TILE_WIDTH &&
            (args.transpose_b ? w.padded_shape()[3] : w.padded_shape()[2]) == tt::constants::TILE_WIDTH,
        "LowrankUpdate: the K dimension must pad to one tile");
    const auto geo = lowrank_geometry(acc);
    TT_FATAL(geo.Mt <= 128, "LowrankUpdate: M must be <= 4096 rows (a column of a is kept in L1), got {}", as[2]);
}

spec_return_value_t LowrankUpdateDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& acc = tensor_args.acc;
    return tt::tt_metal::TensorSpec(
        acc.logical_shape(),
        tt::tt_metal::TensorLayout(
            acc.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), acc.memory_config()));
}

tensor_return_value_t LowrankUpdateDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.acc.device());
}

ttsl::hash::hash_t LowrankUpdateDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<LowrankUpdateDeviceOperation>(
        args.scale,
        args.transpose_b,
        tensor_args.acc.logical_shape(),
        tensor_args.a.logical_shape(),
        tensor_args.w.logical_shape());
}

}  // namespace ttml::metal::ops::lowrank_update::device

namespace ttnn::prim {

ttnn::Tensor ttml_lowrank_update(
    const ttnn::Tensor& acc, const ttnn::Tensor& a, const ttnn::Tensor& w, float scale, bool transpose_b) {
    using OperationType = ttml::metal::ops::lowrank_update::device::LowrankUpdateDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{.scale = scale, .transpose_b = transpose_b};
    auto tensor_args = OperationType::tensor_args_t{.acc = acc, .a = a, .w = w};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
