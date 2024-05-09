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
    uint32_t max_out_nsticks_per_core_;
    MemoryConfig output_memory_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("config_", "pad_val_", "remote_read_", "transpose_mcast_", "reshard_num_cores_nhw_", "max_out_nsticks_per_core_", "output_memory_config_");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(config_),
            std::cref(pad_val_),
            std::cref(remote_read_),
            std::cref(transpose_mcast_),
            std::cref(reshard_num_cores_nhw_),
            std::cref(max_out_nsticks_per_core_),
            std::cref(output_memory_config_)
        );
    }
};


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
        const auto& a = input_tensors.at(0);

        auto device = a.device();

        auto pad_metadata = sliding_window::generate_pad_metadata(config);
        auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config);
        auto shard_boundaries = sliding_window::generate_shard_boundaries(config, op_trace_metadata);
        auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, config, reshard_num_cores_nhw);
        auto kernel_config_tensors = sliding_window::generate_halo_kernel_config_tensors(tensor_metadata, shard_boundaries, remote_read, device);

        const auto& padding_config = std::get<0>(kernel_config_tensors);
        const auto& local_config = std::get<1>(kernel_config_tensors);
        const auto& remote_config = std::get<2>(kernel_config_tensors);
        uint32_t max_out_nsticks_per_core = std::get<3>(kernel_config_tensors);

        return operation::run(
            Halo{
                .config_ = config,
                .pad_val_ = pad_val,
                .remote_read_ = remote_read,
                .transpose_mcast_ = transpose_mcast,
                .reshard_num_cores_nhw_ = reshard_num_cores_nhw,
                .max_out_nsticks_per_core_ = max_out_nsticks_per_core,
                .output_memory_config_ = output_memory_config
            },
            {
                a,
                padding_config,
                local_config,
                remote_config
            })
            .at(0);
    };
    std::vector<Tensor> output_tensors = { Tensor(tt::tt_metal::operation::get_workers_for_op_output({a})) };
    operation::launch_op(halo_op, {a}, {output_tensors});

    return output_tensors.at(0);
}

} // namespace halo

} // namespace ttnn::operations
