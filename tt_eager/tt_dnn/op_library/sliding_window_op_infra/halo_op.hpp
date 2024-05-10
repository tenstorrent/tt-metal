// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>

#include "tensor/tensor.hpp"
#include "run_operation.hpp"
#include "sliding_window_op_infra/sliding_window.hpp"
#include "untilize/untilize_op.hpp"


namespace ttnn::operations {
namespace halo {

struct Halo {
    SlidingWindowConfig config_;
    uint32_t pad_val_;
    bool remote_read_;
    bool transpose_mcast_;
    uint32_t reshard_num_cores_nhw_;
    MemoryConfig output_memory_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("config_", "pad_val_", "remote_read_", "transpose_mcast_", "reshard_num_cores_nhw_", "output_memory_config_");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(config_),
            std::cref(pad_val_),
            std::cref(remote_read_),
            std::cref(transpose_mcast_),
            std::cref(reshard_num_cores_nhw_),
            std::cref(output_memory_config_)
        );
    }
};


Tensor construct_on_device_config_tensor(const std::vector<std::vector<uint16_t>>& config, const Device* device) {
    // TODO: ...
}


Tensor halo_op(const Tensor& a,
                const SlidingWindowConfig& config,
                uint32_t pad_val = 0x0,
                bool remote_read = false,
                bool transpose_mcast = true,
                uint32_t reshard_num_cores_nhw = 0,
                MemoryConfig output_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along width)

    auto halo_op = [&config, pad_val, remote_read, transpose_mcast, reshard_num_cores_nhw, &output_memory_config]
        (const std::vector<Tensor>& input_tensors) -> std::vector<Tensor> {
        return operation::run(
            Halo{
                .config_ = config,
                .pad_val_ = pad_val,
                .remote_read_ = remote_read,
                .transpose_mcast_ = transpose_mcast,
                .reshard_num_cores_nhw_ = reshard_num_cores_nhw,
                .output_memory_config_ = output_memory_config
            },
            input_tensors)
            .at(0);
    };
    std::vector<Tensor> output_tensors = { Tensor(tt::tt_metal::operation::get_workers_for_op_output({a}, {})) };
    operation::launch_op(halo_op, {a}, {output_tensors});

    return output_tensors.at(0);
}

} // namespace halo

} // namespace ttnn::operations
