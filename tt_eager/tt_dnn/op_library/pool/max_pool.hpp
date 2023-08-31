/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace tt_metal {

struct MaxPool {
    uint32_t in_h_, in_w_;
    uint32_t out_h_, out_w_;
    uint32_t kernel_size_h_, kernel_size_w_;
    uint32_t stride_h_, stride_w_;
    uint32_t pad_h_, pad_w_;
    uint32_t dilation_h_, dilation_w_;
    MemoryConfig out_mem_config_;
    uint32_t nblocks_;
    bool use_multicore_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor max_pool2d(const Tensor &input,
                  uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h = 1, uint32_t stride_w = 1,
                  uint32_t pad_h = 0, uint32_t pad_w = 0,               // default: no padding
                  uint32_t dilation_h = 1, uint32_t dilation_w = 1,
                  const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  uint32_t nblocks = 1, bool use_multicore = false);

}  // namespace tt_metal
}  // namespace tt
