// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include "common/constants.hpp"

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"


namespace tt {

namespace tt_metal {

using unary_tensor_op_t = Tensor (const Tensor& a);
using binary_tensor_op_t = Tensor (const Tensor& a, const Tensor& b);

//addalpha(input, other, alpha) = input + (alpha * other)
std::vector<Tensor> addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

} //namespace tt_metal

} //namespace tt
