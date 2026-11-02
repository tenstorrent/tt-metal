// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_contract_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::skinny_contract::device {

void SkinnyContractDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& st = tensor_args.small_t;
    const auto& big = tensor_args.big;
    const DeviceTensorRequirements dram_bf16{.buffer_type = tt::tt_metal::BufferType::DRAM};
    check_device_tensor(st, "SkinnyContract", "small_t", dram_bf16);
    check_device_tensor(big, "SkinnyContract", "big", dram_bf16);
    const auto& ss = st.logical_shape();
    const auto& bs = big.logical_shape();
    TT_FATAL(ss.rank() == 4 && bs.rank() == 4, "SkinnyContract: small_t {} and big {} must be 4D", ss, bs);
    TT_FATAL(
        ss[0] == 1 && ss[1] == 1 && bs[0] == 1 && bs[1] == 1,
        "SkinnyContract: one image only, got small_t {} and big {}",
        ss,
        bs);
    TT_FATAL(ss[2] >= 1 && ss[2] <= tt::constants::TILE_HEIGHT, "SkinnyContract: r must be <= 32, got {}", ss[2]);
    const uint32_t T = args.big_transposed ? bs[3] : bs[2];
    const uint32_t N = args.big_transposed ? bs[2] : bs[3];
    TT_FATAL(ss[3] == T, "SkinnyContract: T of small_t ({}) and big ({}) differ", ss[3], T);
    TT_FATAL(T % tt::constants::TILE_HEIGHT == 0, "SkinnyContract: T must be a multiple of 32, got {}", T);
    TT_FATAL(N % tt::constants::TILE_WIDTH == 0, "SkinnyContract: N must be a multiple of 32, got {}", N);
    TT_FATAL(T / tt::constants::TILE_HEIGHT <= 128, "SkinnyContract: T must be <= 4096 (small_t resident in L1)");
}

spec_return_value_t SkinnyContractDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t r = tensor_args.small_t.logical_shape()[2];
    const uint32_t N = args.big_transposed ? tensor_args.big.logical_shape()[2] : tensor_args.big.logical_shape()[3];
    const auto shape = args.transpose_out ? ttnn::Shape{1, 1, N, r} : ttnn::Shape{1, 1, r, N};
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.big.memory_config()));
}

tensor_return_value_t SkinnyContractDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.big.device());
}

ttsl::hash::hash_t SkinnyContractDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SkinnyContractDeviceOperation>(
        args.scale,
        args.transpose_out,
        args.big_transposed,
        tensor_args.small_t.logical_shape(),
        tensor_args.big.logical_shape());
}

}  // namespace ttml::metal::ops::skinny_contract::device

namespace ttnn::prim {

ttnn::Tensor ttml_skinny_contract(
    const ttnn::Tensor& small_t, const ttnn::Tensor& big, float scale, bool transpose_out, bool big_transposed) {
    using OperationType = ttml::metal::ops::skinny_contract::device::SkinnyContractDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .scale = scale, .transpose_out = transpose_out, .big_transposed = big_transposed};
    auto tensor_args = OperationType::tensor_args_t{.small_t = small_t, .big = big};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
