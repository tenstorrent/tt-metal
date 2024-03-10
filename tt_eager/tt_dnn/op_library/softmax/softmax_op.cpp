// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tt_dnn/op_library/math.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void Softmax::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 1, "Must have 1 or 2 input tensors");
    auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);
    if (optional_input_tensors.size() == 1) {
        if (optional_input_tensors.at(0).has_value()) {
            auto& mask = optional_input_tensors.at(0).value();
            TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_FATAL(input_tensor.device() == mask.device());
            if (mask.is_sharded()) { // sharded mask
                TT_FATAL(mask.get_layout() == Layout::TILE);
                TT_FATAL(mask.get_legacy_shape() == input_tensor.get_legacy_shape());
            } else {
                if (mask.get_layout() == Layout::ROW_MAJOR) {
                    Shape expected_shape = {mask.get_legacy_shape()[0], 1, input_tensor.get_legacy_shape()[-1] / TILE_WIDTH, TILE_WIDTH};
                    TT_FATAL(mask.get_legacy_shape() == expected_shape);
                }
                for (uint32_t i = 1; i < input_tensor.get_legacy_shape().rank() - 2; i++) {
                    TT_FATAL(mask.get_legacy_shape()[i] == 1);
                }
            }

            std::visit(
                [&](const auto& program_config) {
                    using ProgramConfigType = std::decay_t<decltype(program_config)>;
                    if constexpr (
                        std::is_same_v<ProgramConfigType, tt::operations::primary::transformers::SoftmaxDefaultProgramConfig>
                    ) {
                        TT_FATAL(input_tensor.get_legacy_shape()[0] == mask.get_legacy_shape()[0]);
                    } else if constexpr (
                        std::is_same_v<ProgramConfigType, tt::operations::primary::transformers::SoftmaxShardedMultiCoreProgramConfig>
                    ) {
                        const auto shape = input_tensor.get_legacy_shape();
                        uint32_t M = input_tensor.volume() / shape[-1];
                        uint32_t K = shape[-1];
                        // block
                        uint32_t block_w = program_config.block_w * TILE_WIDTH;
                        uint32_t block_h = program_config.block_h * TILE_HEIGHT;
                        uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;
                        // grid
                        auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                        auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                        // check dims
                        TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
                        TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                        TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                        TT_FATAL(M * K / (block_w * block_h) == num_cores_r * num_cores_c, "number of shards must equal to number of cores");
                        TT_FATAL(this->inplace);
                        // check sharding dim
                        TT_FATAL(block_w == shape[3], "shard width must equal to input tensor shape[3]!");
                    }
                },
                this->program_config
            );
        } else {
            TT_FATAL(not this->scale.has_value());
        }
    } else {
        TT_FATAL(not this->scale.has_value());
    }
}

std::vector<Shape> Softmax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // Do nothing because it's an in-place operation
    if (this->inplace) {
        return {};
    } else {
        return {input_tensors.at(0).get_legacy_shape()};
    }
}

std::vector<Tensor> Softmax::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    // Do nothing because it's an in-place operation
    if (this->inplace) {
        return {};
    }  else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Softmax::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = this->inplace ? input_tensors.at(0) : output_tensors.at(0);
    const auto& mask = optional_input_tensors.at(0);
    // bool causal_mask = mask.has_value() ? mask.value().get_legacy_shape()[-2] == mask.value().get_legacy_shape()[-1] : false;
    bool causal_mask = this->is_causal_mask;

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, tt::operations::primary::transformers::SoftmaxShardedMultiCoreProgramConfig>
            ) {
                MathFidelity fidelity = program_config.math_fidelity;
                return scale_mask_softmax_sharded_multi_core(
                                            input_tensor, output_tensor, mask, this->scale,
                                            fidelity,
                                            program_config.im_data_format,
                                            causal_mask,
                                            program_config.compute_with_storage_grid_size,
                                            program_config.subblock_w,
                                            program_config.block_h,
                                            program_config.block_w
                                            );
            } else if constexpr (
                std::is_same_v<ProgramConfigType, tt::operations::primary::transformers::SoftmaxInterleavedMultiCoreProgramConfig>
            ) {
                MathFidelity fidelity = program_config.math_fidelity;
                return scale_mask_softmax_multi_core(input_tensor, output_tensor, mask, this->scale, fidelity, program_config.im_data_format, causal_mask);
            } else {
                return scale_mask_softmax_multi_core(input_tensor, output_tensor, mask, this->scale, MathFidelity::HiFi4, DataType::BFLOAT16, causal_mask);
            }
        },
        this->program_config
    );
}

tt::stl::reflection::Attributes Softmax::attributes() const {
    return {
        {"scale", this->scale},
        {"inplace", this->inplace},
        {"output_mem_config", this->output_mem_config},
    };
}


const operation::Hash Softmax::compute_program_hash(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<Softmax>(
        input_tensors.at(0).memory_config(),
        input_tensors.at(0).get_dtype(),
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().memory_config()}
                                                 : std::nullopt,
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().get_dtype()}
                                                 : std::nullopt,
        this->output_mem_config);
}

Tensor softmax_in_place(Tensor& input_tensor, const transformers::SoftmaxProgramConfig& program_config) {
    return transformers::scale_mask_softmax_in_place(input_tensor, std::nullopt, std::nullopt, program_config);
}

namespace transformers {
Tensor scale_mask_softmax_in_place(Tensor& input_tensor, std::optional<float> scale, std::optional<const Tensor> mask, const SoftmaxProgramConfig& program_config, const bool is_causal_mask) {
    operation::run(Softmax{.scale=scale, .inplace=true, .output_mem_config=input_tensor.memory_config(), .program_config=program_config, .is_causal_mask=is_causal_mask}, {input_tensor}, {mask});
    return input_tensor;
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations

namespace tt_metal {
Tensor softmax(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    return transformers::scale_mask_softmax(input_tensor, std::nullopt, std::nullopt, output_mem_config);
}

namespace transformers {
Tensor scale_mask_softmax(const Tensor& input_tensor, std::optional<float> scale, std::optional<const Tensor> mask, const MemoryConfig& output_mem_config, const bool is_causal_mask) {
    Shape input_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    FormatParams input_format_params = {.pad_shape=input_pad_shape, .pad_value=-std::numeric_limits<float>::infinity(), .target_layout=Layout::TILE};
    std::optional<FormatParams> mask_format_params = std::nullopt;
    if (mask.has_value()) {
        TT_FATAL(input_tensor.get_legacy_shape()[-1] == mask.value().get_legacy_shape()[-1]);
        TT_FATAL(input_tensor.get_legacy_shape()[0] == mask.value().get_legacy_shape()[0]);
        TT_FATAL(mask.value().get_legacy_shape()[-2] == 1 or mask.value().get_legacy_shape()[-2] == TILE_HEIGHT);
        for (uint32_t i = 1; i < input_tensor.get_legacy_shape().rank() - 2; i++) {
            TT_FATAL(mask.value().get_legacy_shape()[i] == 1);
        }
        Shape mask_pad_shape = AutoFormat::pad_to_tile_shape(mask.value().get_legacy_shape());
        mask_format_params = {.pad_shape=mask_pad_shape, .pad_value=-std::numeric_limits<float>::infinity(), .target_layout=Layout::TILE};
    }
    return operation::run_with_autoformat(tt::operations::primary::Softmax{.scale=scale, .inplace=false, .output_mem_config=output_mem_config, .is_causal_mask=is_causal_mask}, {input_tensor}, {input_format_params}, {Layout::TILE}, {mask}, {mask_format_params}).at(0);
}
}  // namespace transformers
}  // namespace tt_metal
}  // namespace tt
