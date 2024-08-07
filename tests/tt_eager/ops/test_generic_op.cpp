// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "core_coord.h"
#include "logger.hpp"
#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;

using tt::tt_metal::Layout;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::Shape;
using tt::tt_metal::Tensor;

namespace detail {
float sqrt(float x) { return std::sqrt(x); }
float exp(float x) { return std::exp(x); }
float recip(float x) { return 1 / x; }
float gelu(float x) { return x * (0.5 * (1 + std::erf(x / std::sqrt(2)))); }
float relu(float x) { return std::max(0.0f, x); }
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
float log(float x) { return std::log(x); }
float tanh(float x) { return std::tanh(x); }
}  // namespace detail

Tensor gelu_fast(const Tensor& t) { return ttnn::gelu(t, true); }

Tensor gelu_slow(const Tensor& t) { return ttnn::gelu(t, false); }

template <auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor);

    auto output_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(input_tensor.volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = UnaryFunction(input_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }

    return Tensor(
        OwnedStorage{output_buffer},
        input_tensor.get_legacy_shape(),
        input_tensor.get_dtype(),
        input_tensor.get_layout());
}

template <typename BinaryFunction>
Tensor host_function(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    auto input_a_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor_a);
    auto input_b_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor_b);

    auto output_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(input_tensor_a.volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = BinaryFunction{}(input_a_buffer[index].to_float(), input_b_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }
    return Tensor(OwnedStorage{output_buffer}, input_tensor_a.get_legacy_shape(), input_tensor_a.get_dtype(), input_tensor_a.get_layout());
}

template <ttnn::operations::unary::UnaryOpType unary_op_type, typename... Args>
bool run_test(Device* device, const Shape& shape, float low, float high, Args... args) {
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    if constexpr (unary_op_type == UnaryOpType::SQRT) {
        auto host_output = host_function<::detail::sqrt>(input_tensor);
        auto device_output = ttnn::sqrt(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::EXP) {
        auto host_output = host_function<::detail::exp>(input_tensor);
        auto device_output = ttnn::exp(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RECIP) {
        auto host_output = host_function<::detail::recip>(input_tensor);
        auto device_output = ttnn::reciprocal(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::GELU) {
        auto host_output = host_function<::detail::gelu>(input_tensor);
        auto device_output = ttnn::gelu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RELU) {
        auto host_output = host_function<::detail::relu>(input_tensor);
        auto device_output = ttnn::relu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::SIGMOID) {
        auto host_output = host_function<::detail::sigmoid>(input_tensor);
        auto device_output = ttnn::sigmoid(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::LOG) {
        auto host_output = host_function<::detail::log>(input_tensor);
        auto device_output = ttnn::log(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::TANH) {
        auto host_output = host_function<::detail::tanh>(input_tensor);
        auto device_output =ttnn::tanh(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    }
    TT_ASSERT(false, "Unsupported function");
    return false;
}

// void test_operation_infrastructure() {
//     tt::log_info(tt::LogTest, "Running {}", __func__);

//     using ttnn::operations::unary::UnaryWithParam;
//     using ttnn::operations::unary::UnaryOpType;

//     int device_id = 0;
//     auto device = tt::tt_metal::CreateDevice(device_id);

//     auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
//     auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), shape).to(Layout::TILE).to(device);

//     auto op = tt::tt_metal::operation::DeviceOperation(ttnn::operations::unary::Unary{
//         {UnaryWithParam{UnaryOpType::SQRT}},
//         MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}});

//     auto program_hash = op.compute_program_hash({input_tensor}, {});
//     TT_FATAL(program_hash == 3018574135764717736ULL, fmt::format("Actual value is {}", program_hash));

//     auto profiler_info = op.create_profiler_info({input_tensor});
//     TT_FATAL(
//         profiler_info.preferred_name.value() == "ttnn::operations::unary::Unary",
//         fmt::format("Actual value is {}", profiler_info.preferred_name.value()));
//     TT_FATAL(
//         profiler_info.parallelization_strategy.value() == "UnaryOpParallelizationStrategy::MULTI_CORE",
//         fmt::format("Actual value is {}", profiler_info.parallelization_strategy.value()));

//     TT_FATAL(tt::tt_metal::CloseDevice(device));
// }

// void test_shape_padding() {
//     tt::log_info(tt::LogTest, "Running {}", __func__);

//     using ttnn::operations::unary::UnaryWithParam;
//     using ttnn::operations::unary::UnaryOpType;

//     int device_id = 0;
//     auto device = tt::tt_metal::CreateDevice(device_id);
//     tt::tt_metal::AutoFormat::SetDefaultDevice(device);

//     tt::tt_metal::Array4D input_shape = {1, 1, 13, 18};
//     tt::tt_metal::Array4D padded_input_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
//     auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);

//     auto padded_input_tensor = ttnn::pad(input_tensor, padded_input_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

//     padded_input_tensor = padded_input_tensor.to(Layout::TILE);
//     padded_input_tensor = padded_input_tensor.to(device);
//     auto output_tensor =
//         tt::tt_metal::operation::run(
//             ttnn::operations::unary::Unary{
//                 {UnaryWithParam{UnaryOpType::SQRT}},
//                 tt::tt_metal::MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}},
//             {padded_input_tensor})
//             .at(0);
//     output_tensor = output_tensor.cpu();

//     auto output_shape = output_tensor.get_legacy_shape();
//     TT_FATAL(output_shape == tt::tt_metal::Shape(padded_input_shape));
//     TT_FATAL(output_shape.without_padding() == tt::tt_metal::Shape(input_shape));

//     TT_FATAL(tt::tt_metal::CloseDevice(device));
// }


namespace test {
    namespace detail {
        std::unordered_map<CoreCoord, std::vector<uint32_t>> cast_args_to_core_coords(const CoreRangeSet& core_spec, std::vector<uint32_t> args) {
            std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core;
            for (const auto& core_range : core_spec.ranges()) {
                for (const auto& core : core_range)
                {
                    runtime_args_per_core[core] = args; // there is a good reason why it should be shared pointer
                }

            }
            return runtime_args_per_core;
        }
    }
}

namespace tt {
namespace tt_metal {
template <bool approx_value = false>
struct exp_with_param {
    static Tensor fn(const tt::tt_metal::Tensor& t) {
        return ttnn::exp(t, approx_value, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    }
};
}  // namespace tt_metal
}  // namespace tt

void test_numerically() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);


    // // ttnn::sqrt
    // {
    //     auto shape = Shape{13, 6, TILE_HEIGHT, TILE_WIDTH}
    //     auto allclose = run_test<UnaryOpType::SQRT>(device, shape, 0.0f, 1.0f, 1e-1f, 1e-5f);
    //     TT_FATAL(allclose);
    // }

    // // ttnn::relu
    // {
    //     auto shape = Shape{13, 6, TILE_HEIGHT, TILE_WIDTH};
    //     auto allclose = run_test<UnaryOpType::RELU>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
    //     TT_FATAL(allclose);
    // }


    const std::map<std::string, std::string> defines_sqrt = {
        {"SFPU_OP_CHAIN_0", "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"},
        {"SFPU_OP_CHAIN_0_FUNC_0", "sqrt_tile(0);"},
        {"SFPU_OP_CHAIN_0_INIT_0", "sqrt_tile_init();"},
        {"SFPU_OP_SQRT_INCLUDE", "1"}
    };

    const std::map<std::string, std::string> defines_relu = {
        {"SFPU_OP_CHAIN_0", "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"},
        {"SFPU_OP_CHAIN_0_FUNC_0", "relu_tile(0);"},
        {"SFPU_OP_CHAIN_0_INIT_0", "relu_tile_init();"},
        {"SFPU_OP_RELU_FAMILY_INCLUDE", "1"}
    };

    const std::map<std::string, std::string> defines_eltwise_add = {
        {"ELTWISE_OP", "add_tiles"},
        {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"},
        // {"SFPU_OP_CHAIN_0", ""},
    };

    // ttnn::generic_op
    // run unary sqrt interleaved
    {
        auto shape = Shape{13, 6, TILE_HEIGHT, TILE_WIDTH};

        tt::log_info(tt::LogTest, "generic_op sqrt");
        auto input_tensor = tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(1.0f), shape).to(Layout::TILE);

        // "sqrt golden"
        auto host_output = host_function<::detail::sqrt>(input_tensor);

        auto device_input_tensor = input_tensor.to(device);

        auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.get_dtype());
        bool is_dram_input = device_input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

        // split_work_to_cores
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_worker_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, input_tensor.volume() / tt::constants::TILE_HW);

        ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    tt::CB::c_in0,
                    {
                        .core_spec = all_cores,
                        .total_size = 2 * tt::tt_metal::detail::TileSize(input_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_cb_data_format),
                        .data_format = input_cb_data_format,
                    }
                },
                {
                    tt::CB::c_out0,
                    {
                        .core_spec = all_cores,
                        .total_size = 2 * tt::tt_metal::detail::TileSize(input_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_cb_data_format),
                        .data_format = input_cb_data_format,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig({(uint32_t)is_dram_input})
                },
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::WriterDataMovementConfig({(uint32_t)tt::CB::c_out0, (uint32_t)is_dram_input})
                }
            },
                // per_core_block_cnt; per_core_block_size
            .compute_attributes =
            {
                {
                    .core_spec = core_group_1,
                    .kernel_path = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                    .config = {
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .preserve_fp32_precision = false,
                        .math_approx_mode = false,
                        .compile_args = {num_tiles_per_core_group_1, 1},
                        .defines = defines_sqrt,
                    },
                },
                {
                    .core_spec = core_group_2,
                    .kernel_path = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                    .config = {
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .preserve_fp32_precision = false,
                        .math_approx_mode = false,
                        .compile_args = {num_tiles_per_core_group_2, 1},
                        .defines = defines_sqrt,
                    },
                },
            },
        };

        // Data movement kernel needs output tensor address to be passed as a runtime argument.
        auto device_output_tensor = tt::tt_metal::create_device_tensor(
            device_input_tensor.tensor_attributes->shape,
            device_input_tensor.tensor_attributes->dtype,
            device_input_tensor.tensor_attributes->layout,
            device_input_tensor.device(),
            device_input_tensor.memory_config());

        // calculate data movement runtime arguments
        uint32_t num_tiles_written = 0;
        for (uint32_t i = 0; i < num_worker_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {device_input_tensor.buffer()->address(), num_tiles_per_core, num_tiles_written};
            program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {device_output_tensor.buffer()->address(), num_tiles_per_core, num_tiles_written};

            num_tiles_written += num_tiles_per_core;
        }
        // end of data movement runtime arguments calculus

        ttnn::generic_op(device_input_tensor, device_output_tensor, program_attributes);
        auto device_output = device_output_tensor.cpu();
        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }

    // =================
    // run unary relu sharded
    {
        tt::log_info(tt::LogTest, "unary relu");

        auto shape = Shape{64, 16, TILE_HEIGHT, TILE_WIDTH};

        CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        CoreRange all_cores_range = {CoreCoord(0,0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
        CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

        MemoryConfig mem_config = MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .shard_spec = std::make_optional<tt::tt_metal::ShardSpec>(ShardSpec(all_cores, { 16 * TILE_HEIGHT, TILE_WIDTH}, tt::tt_metal::ShardOrientation::ROW_MAJOR)),
        };

        auto input_tensor = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), shape).to(Layout::TILE);

        // "relu golden"
        auto host_output = host_function<::detail::relu>(input_tensor);

        // unary relu with sharding
        auto device_input_tensor = input_tensor.to(device, mem_config);
        auto device_output_ref = ttnn::relu(device_input_tensor).cpu();
        auto allclose_ref = tt::numpy::allclose<bfloat16>(host_output, device_output_ref, 1e-1f, 1e-5f);
        TT_FATAL(allclose_ref);

        tt::log_info(tt::LogTest, "generic_op relu");


        auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.get_dtype());
        uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);

        uint32_t num_tile_per_core = 0;
        if (input_tensor.get_dtype() == DataType::BFLOAT8_B) {
            uint32_t ntiles_along_width = ceil(mem_config.shard_spec->shape[1] / (float) tt::constants::TILE_WIDTH);
            uint32_t ntiles_along_height = ceil(mem_config.shard_spec->shape[0] / (float) tt::constants::TILE_HEIGHT);
            num_tile_per_core = ntiles_along_width * ntiles_along_height;
        } else {
            TT_FATAL(
                (mem_config.shard_spec->shape[1] * datum_size(input_cb_data_format)) % L1_ALIGNMENT == 0,
                "Shard width should be multiple of L1_ADRESS_ALIGNMENT");
            size_t shard_height = mem_config.shard_spec->shape[0];
            size_t shard_width = round_up_to_mul16(
                mem_config.shard_spec->shape[1]);  // rounding up is done to aligned with  --> tt-metal/tt_metal/detail/util.hpp:31
            size_t shard_size_in_bytes = shard_height * shard_width * datum_size(input_cb_data_format);
            TT_FATAL(shard_size_in_bytes % input_tile_size == 0, "Shard Size must be multiple of input_tile_size");
            num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size;  // ceil value
        }

        uint32_t aligned_input_tile_nbytes = round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32

        ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    tt::CB::c_in0,
                    {
                        .core_spec = all_cores,
                        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
                        .page_size = aligned_input_tile_nbytes,
                        .data_format = input_cb_data_format,
                        .set_globally_allocated_address = 0,
                    }
                },
                {
                    tt::CB::c_out0,
                    {
                        .core_spec = all_cores,
                        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
                        .page_size = aligned_input_tile_nbytes,
                        .data_format = input_cb_data_format,
                        .set_globally_allocated_address = 1,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig({(uint32_t)tt::CB::c_in0})
                }
            },
            .compute_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                    .config = {
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .preserve_fp32_precision = false,
                        .math_approx_mode = false,
                        .compile_args = {1, num_tile_per_core},
                        .defines = defines_relu,
                    },
                    .runtime_args_per_core = test::detail::cast_args_to_core_coords(all_cores, {num_tile_per_core}),
                },
            }
        };

        // // calculate data movement runtime arguments
        for (uint32_t i = 0; i < compute_with_storage_grid_size.x * compute_with_storage_grid_size.y; i++) {
            CoreCoord core = {i / compute_with_storage_grid_size.y, i % compute_with_storage_grid_size.y};
            program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {num_tile_per_core};
        }
        // // end of data movement runtime arguments calculus

        // Data movement kernel needs output tensor address to be passed as a runtime argument./
        auto device_output_tensor = tt::tt_metal::create_device_tensor(
            device_input_tensor.tensor_attributes->shape,
            device_input_tensor.tensor_attributes->dtype,
            device_input_tensor.tensor_attributes->layout,
            device_input_tensor.device(),
            device_input_tensor.memory_config());


        ttnn::generic_op(device_input_tensor, device_output_tensor, program_attributes);
        auto device_output = device_output_tensor.cpu();
        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }

    // =================
    // run binary add interleaved original
    {
        tt::log_info(tt::LogTest, "binary add interleaved");
        Shape shape = {11, 9, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        // Shape shape = {8, 8, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);

        auto host_output = host_function<std::plus<float>>(input_tensor_a, input_tensor_b);
        auto device_output = ttnn::add(
            input_tensor_a.to(Layout::TILE).to(device),
            input_tensor_b.to(Layout::TILE).to(device)
            ).cpu().to(Layout::ROW_MAJOR);

        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output);
        TT_FATAL(allclose);
    }

    // run binary add interleaved generic
    {
        tt::log_info(tt::LogTest, "generic add interleaved");
        Shape shape = {11, 9, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        // Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

        auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto host_output = host_function<std::plus<float>>(input_tensor_a, input_tensor_b);


        auto device_input_tensor_a = input_tensor_a.to(Layout::TILE).to(device);
        auto device_input_tensor_b = input_tensor_b.to(Layout::TILE).to(device);


        // Data movement kernel needs output tensor address to be passed as a runtime argument.
        auto device_output_tensor = tt::tt_metal::create_device_tensor(
            device_input_tensor_a.tensor_attributes->shape,
            device_input_tensor_a.tensor_attributes->dtype,
            device_input_tensor_a.tensor_attributes->layout,
            device_input_tensor_a.device(),
            device_input_tensor_a.memory_config());

        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        CoreRange all_cores_range = {CoreCoord(0,0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
        CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

        auto input_a_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_a.get_dtype());
        auto input_b_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_b.get_dtype());
        auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.get_dtype());


        bool src0_is_dram = device_input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool src1_is_dram = device_input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram};

        bool dst_is_dram = device_output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CB::c_out0, (std::uint32_t)dst_is_dram};

        ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    tt::CB::c_in0,
                    {
                        .core_spec = all_cores,
                        .total_size = 2 * tt::tt_metal::detail::TileSize(input_a_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_a_cb_data_format),
                        .data_format = input_a_cb_data_format,
                    }
                },
                {
                    tt::CB::c_in1,
                    {
                        .core_spec = all_cores,
                        .total_size = 2 * tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .data_format = input_b_cb_data_format,
                    }
                },
                {
                    tt::CB::c_out0,
                    {
                        .core_spec = all_cores,
                        .total_size = 2 * tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .data_format = input_b_cb_data_format,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args)
                },
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args)
                }
            },
            .compute_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
                    .config = {
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .preserve_fp32_precision = false,
                        .math_approx_mode = false,
                        .compile_args = {},
                        .defines = defines_eltwise_add,
                    },
                },
            },
        };

        // setup runtime parameters - replicating element_wise_multi_core_program_factory.cpp
        {
            uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

            uint32_t num_tiles = device_input_tensor_a.volume() / TILE_HW;
            bool row_major = true;

            auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major);

            uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1, max_block_size = 1;
            uint32_t block_cnt_per_core_group_1, block_cnt_per_core_group_2;
            block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
            block_cnt_per_core_group_2 = num_tiles_per_core_group_2;

            auto cores = grid_to_cores(num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);

            uint32_t g1_numcores = core_group_1.num_cores();
            uint32_t g2_numcores = core_group_2.num_cores();


            // read cached? .. element_wise_multi_core_program_factory.cpp

            for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
                const CoreCoord& core = cores.at(i);
                uint32_t num_tiles_per_core = 0;
                uint32_t block_cnt_per_core = 0;
                uint32_t block_size_per_core = 0;
                if (i < g1_numcores) {
                    num_tiles_per_core = num_tiles_per_core_group_1;
                    block_cnt_per_core = block_cnt_per_core_group_1;
                    block_size_per_core = block_size_per_core_group_1;
                } else if (i < num_cores) {
                    num_tiles_per_core = num_tiles_per_core_group_2;
                    block_cnt_per_core = block_cnt_per_core_group_2;
                    block_size_per_core = block_size_per_core_group_2;
                } else {
                    continue;
                }

                program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {device_input_tensor_a.buffer()->address(), device_input_tensor_b.buffer()->address(), num_tiles_per_core, num_tiles_read};
                program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {device_output_tensor.buffer()->address(), num_tiles_per_core, num_tiles_read};
                program_attributes.compute_attributes[0].runtime_args_per_core[core] = {block_cnt_per_core, block_size_per_core};

                num_tiles_read += num_tiles_per_core;
            }

        }
        ttnn::generic_op(std::vector<Tensor>{device_input_tensor_a, device_input_tensor_b}, device_output_tensor, program_attributes);
        auto device_output = device_output_tensor.cpu().to(Layout::ROW_MAJOR);

        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output);
        TT_FATAL(allclose);
    }

    // =================
    // run binary add sharded original
    {
        tt::log_info(tt::LogTest, "binary add sharded");

        auto shape = Shape{64, 16, TILE_HEIGHT, TILE_WIDTH};

        CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        CoreRange all_cores_range = {CoreCoord(0,0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
        CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

        MemoryConfig mem_config = MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .shard_spec = std::make_optional<tt::tt_metal::ShardSpec>(ShardSpec(all_cores, { 16 * TILE_HEIGHT, TILE_WIDTH}, tt::tt_metal::ShardOrientation::ROW_MAJOR)),
        };


        auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto host_output = host_function<std::plus<float>>(input_tensor_a, input_tensor_b);

        auto device_input_tensor_a = input_tensor_a.to(Layout::TILE).to(device, mem_config);
        auto device_input_tensor_b = input_tensor_b.to(Layout::TILE).to(device, mem_config);

        auto device_output = ttnn::add(
            device_input_tensor_a,
            device_input_tensor_b
            ).cpu().to(Layout::ROW_MAJOR);

        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output);
        TT_FATAL(allclose);
    }

    // run binary add sharded generic
    {
        tt::log_info(tt::LogTest, "generic add sharded");

        auto shape = Shape{64, 16, TILE_HEIGHT, TILE_WIDTH};

        CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        CoreRange all_cores_range = {CoreCoord(0,0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
        CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

        MemoryConfig mem_config = MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .shard_spec = std::make_optional<tt::tt_metal::ShardSpec>(ShardSpec(all_cores, { 16 * TILE_HEIGHT, TILE_WIDTH}, tt::tt_metal::ShardOrientation::ROW_MAJOR)),
        };

        auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);
        auto host_output = host_function<std::plus<float>>(input_tensor_a, input_tensor_b);

        auto device_input_tensor_a = input_tensor_a.to(Layout::TILE).to(device, mem_config);
        auto device_input_tensor_b = input_tensor_b.to(Layout::TILE).to(device, mem_config);

        // Data movement kernel needs output tensor address to be passed as a runtime argument.
        auto device_output_tensor = tt::tt_metal::create_device_tensor(
            device_input_tensor_a.tensor_attributes->shape,
            device_input_tensor_a.tensor_attributes->dtype,
            device_input_tensor_a.tensor_attributes->layout,
            device_input_tensor_a.device(),
            device_input_tensor_a.memory_config());

        auto input_a_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_a.get_dtype());
        auto input_b_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_b.get_dtype());
        auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.get_dtype());

        bool src0_is_dram = device_input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool src1_is_dram = device_input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram};

        bool dst_is_dram = device_output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CB::c_out0, (std::uint32_t)dst_is_dram};


        bool block_sharded = mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        uint32_t num_tiles_per_shard = mem_config.shard_spec.value().shape[0] * mem_config.shard_spec.value().shape[1] / tt::constants::TILE_HW;
        uint32_t max_block_size = find_max_block_size(num_tiles_per_shard);

        ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    tt::CB::c_in0,
                    {
                        .core_spec = all_cores,
                        .total_size = num_tiles_per_shard * tt::tt_metal::detail::TileSize(input_a_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_a_cb_data_format),
                        .data_format = input_a_cb_data_format,
                        .set_globally_allocated_address = 0,
                    }
                },
                {
                    tt::CB::c_in1,
                    {
                        .core_spec = all_cores,
                        .total_size = num_tiles_per_shard * tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .data_format = input_b_cb_data_format,
                        .set_globally_allocated_address = 1,
                    }
                },
                {
                    tt::CB::c_out0,
                    {
                        .core_spec = all_cores,
                        .total_size = num_tiles_per_shard * tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .page_size = tt::tt_metal::detail::TileSize(input_b_cb_data_format),
                        .data_format = input_b_cb_data_format,
                        .set_globally_allocated_address = 2,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args,
                    {
                        {"IN0_SHARDED","1"},
                        {"IN1_SHARDED","1"},
                    })
                },
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args,
                    {
                        {"OUT_SHARDED","1"},
                    })
                }
            },
            .compute_attributes =
            {
                {
                    .core_spec = all_cores,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
                    .config = {
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .preserve_fp32_precision = false,
                        .math_approx_mode = false,
                        .compile_args = {},
                        .defines = defines_eltwise_add,
                    },
                },
            },
        };

        // setup runtime parameters!
        {
            // uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

            uint32_t num_tiles = device_input_tensor_a.volume() / TILE_HW;
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;
            uint32_t num_cores_total = num_cores_x * num_cores_y;

            all_cores = mem_config.shard_spec.value().grid;
            uint32_t num_cores = all_cores.num_cores();
            auto core_group_1 = all_cores;
            auto core_group_2 = CoreRangeSet({});
            uint32_t num_tiles_per_core_group_1 = num_tiles_per_shard;
            uint32_t num_tiles_per_core_group_2 = 0;
            uint32_t block_size_per_core_group_1 = find_max_block_size(num_tiles_per_core_group_1);
            uint32_t block_size_per_core_group_2 = 1;
            uint32_t max_block_size = block_size_per_core_group_1;

            uint32_t block_cnt_per_core_group_1 = num_tiles_per_core_group_1 / block_size_per_core_group_1;
            uint32_t block_cnt_per_core_group_2 = num_tiles_per_core_group_2 / block_size_per_core_group_2;

            bool row_major = mem_config.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;

            uint32_t block_height = 0, block_width = 0, block_size = 0, output_width = 0, last_unpadded_block_height = 0, last_unpadded_block_width = 0;
            CoreCoord end_core;

            if (block_sharded) {
                block_height = mem_config.shard_spec.value().shape[0] / TILE_HEIGHT;
                block_width = mem_config.shard_spec.value().shape[1] / TILE_WIDTH;
                block_size = block_width * block_height;
                end_core = (*mem_config.shard_spec.value().grid.ranges().begin()).end_coord;
                output_width = device_output_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
                uint32_t output_height = device_output_tensor.volume() / device_output_tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
                last_unpadded_block_height = block_height - (tt::round_up(output_height, block_height) - output_height);
                last_unpadded_block_width = block_width - (tt::round_up(output_width, block_width) - output_width);
            }
            auto bbox = core_group_1.bounding_box();

            vector<CoreCoord> cores = grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

            uint32_t g1_numcores = core_group_1.num_cores();

            // read cached? .. element_wise_multi_core_program_factory.cpp

            for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
                const CoreCoord& core = cores.at(i);
                uint32_t num_tiles_per_core = 0;
                uint32_t block_cnt_per_core = 0;
                uint32_t block_size_per_core = 0;
                if (i < g1_numcores) {
                    num_tiles_per_core = num_tiles_per_core_group_1;
                    block_cnt_per_core = block_cnt_per_core_group_1;
                    block_size_per_core = block_size_per_core_group_1;
                } else if (i < num_cores) {
                    num_tiles_per_core = num_tiles_per_core_group_2;
                    block_cnt_per_core = block_cnt_per_core_group_2;
                    block_size_per_core = block_size_per_core_group_2;
                } else {
                    continue;
                }

                program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {device_input_tensor_a.buffer()->address(), device_input_tensor_b.buffer()->address(), num_tiles_per_core, num_tiles_read};
                program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {device_output_tensor.buffer()->address(), num_tiles_per_core, num_tiles_read};
                program_attributes.compute_attributes[0].runtime_args_per_core[core] = {block_cnt_per_core, block_size_per_core};

                num_tiles_read += num_tiles_per_core;
            }

            // if (block_sharded and not out_sharded) {
            //     TT_FATAL(false, "Block sharded but output not sharded");
            // }

        }
        ttnn::generic_op(std::vector<Tensor>{device_input_tensor_a, device_input_tensor_b}, device_output_tensor, program_attributes);
        auto device_output = device_output_tensor.cpu().to(Layout::ROW_MAJOR);

        auto allclose = tt::numpy::allclose<bfloat16>(host_output, device_output);
        TT_FATAL(allclose);
    }

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

void test_program_cache() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor =
            tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

        // Program Cache Hit
        run_test<UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    device->enable_program_cache();
    run_tests();

    TT_FATAL(device->num_program_cache_entries() == 4, "There are {} entries", device->num_program_cache_entries());

    device->disable_and_clear_program_cache();
    TT_FATAL(device->num_program_cache_entries() == 0);
    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

int main(int argc, char** argv) {
    // test_operation_infrastructure();
    // test_shape_padding();
    test_numerically();
    // test_program_cache();
    return 0;
}
