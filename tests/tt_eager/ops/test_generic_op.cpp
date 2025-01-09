// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "base_types.hpp"
#include "common/constants.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "logger.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
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
                        .data_format = output_cb_data_format,
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

    // =================
    // softmax original and generic test
    // from test_softmax_op.cpp
    {

        // Softmax original test

        tt::log_info(tt::LogTest, "Running original softmax test");
        const Shape softmax_shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
        Tensor input_tensor_generic = tt::numpy::random::random(softmax_shape);
        Tensor input_tensor_original = input_tensor_generic;
        input_tensor_original = input_tensor_original.to(Layout::TILE).to(device);
        Tensor device_output_tensor_original = ttnn::softmax_in_place(input_tensor_original);
        Tensor output_tensor_original = device_output_tensor_original.cpu();

        // Softmax generic test

        tt::log_info(tt::LogTest, "Running generic softmax test");

        // Copy paste arguments from original softmax call so we can compare the results.
        std::optional<const Tensor> mask = nullopt;
        std::optional<float> scale = std::nullopt;
        bool causal_mask = false;

        // Compute kernel configuration parameters copied from original test.
        MathFidelity math_fidelity = MathFidelity::HiFi4;
        bool math_approx_mode = true;
        bool fp32_dest_acc_en = false;

        input_tensor_generic = input_tensor_generic.to(Layout::TILE).to(device);

        auto device_output_tensor = tt::tt_metal::create_device_tensor(
            input_tensor_generic.tensor_attributes->shape,
            input_tensor_generic.tensor_attributes->dtype,
            input_tensor_generic.tensor_attributes->layout,
            input_tensor_generic.device(),
            input_tensor_generic.memory_config());

        auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_generic);

        const auto shape = input_tensor.get_legacy_shape();
        uint32_t W = shape[-1], H = (input_tensor.volume() / (shape[0] * shape[-1])), NC = shape[0];
        uint32_t HW = H*W;

        bool mask_padded_data = false;
        uint32_t num_datum_padded = 0;
        const auto shape_unpadded = input_tensor.get_shape();
        uint32_t W_unpadded = shape_unpadded[-1];
        if (W > W_unpadded) {
            mask_padded_data = true;
            num_datum_padded = W - W_unpadded;
        }

        uint32_t Wt = W/TILE_WIDTH;
        uint32_t Ht = H/TILE_HEIGHT;

        uint32_t mask_H = H;
        if (mask.has_value()) {
            mask_H = mask.value().get_legacy_shape()[2];
        }
        uint32_t mask_Ht = mask_H/TILE_HEIGHT;

        // This should allocate input_tensor DRAM buffer on the device
        Device *device = input_tensor.device();

        tt::DataFormat in0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
        uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(in0_cb_data_format);

        tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
        uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_cb_data_format);

        tt::DataFormat out0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.get_dtype());
        uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(out0_cb_data_format);

        tt::DataFormat mask_cb_data_format = mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(mask.value().get_dtype()) : tt::DataFormat::Float16_b;
        uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_cb_data_format);

        tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_cb_data_format);

        tt::log_debug("in0_cb_data_format: {}", in0_cb_data_format);
        tt::log_debug("out0_cb_data_format: {}", out0_cb_data_format);
        tt::log_debug("mask_cb_data_format: {}", mask_cb_data_format);
        tt::log_debug("im_cb_data_format: {}", im_cb_data_format);
        tt::log_debug("math_fidelity: {}", math_fidelity);
        tt::log_debug("math_approx_mode: {}", math_approx_mode);
        tt::log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

        auto src0_buffer = input_tensor.buffer();
        auto out0_buffer = device_output_tensor.buffer();

        uint32_t num_tiles = input_tensor.volume()/TILE_HW;

        uint32_t block_size = fp32_dest_acc_en ? find_max_divisor(Wt, 4) : find_max_divisor(Wt, 8);

        // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
        uint32_t in0_t  = block_size*2;
        uint32_t out0_t = block_size*2;
        uint32_t im1_t  = 1; // 1/sum(exp(x))
        uint32_t in2_t  = 1; // scaler for reduce coming from reader
        uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
        uint32_t in4_t  = tt::div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
        uint32_t in5_t = 1;

        // cb_exps - keeps exps in tt::CB in L1 to avoid recomputing
        uint32_t im0_t  = block_size*tt::div_up(Wt, block_size);
        TT_ASSERT(im0_t == Wt);

        // used for buffering scale-mask
        // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
        uint32_t im3_t  = block_size*(tt::div_up(Wt, block_size)+1);
        TT_ASSERT(im3_t == Wt+block_size);

        TT_ASSERT(Wt % block_size == 0);
        TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
        TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        TT_ASSERT(in4_t % block_size == 0);
        TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

        uint32_t num_tile_rows = NC * Ht;
        auto grid_size = device->compute_with_storage_grid_size();
        auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
        auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

        auto all_device_cores_set = CoreRangeSet({all_device_cores});

        bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool out0_is_dram = out0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            src0_is_dram
        };
        if (mask.has_value()) {
            bool mask_is_dram = mask.value().buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
            reader_compile_time_args.push_back(mask_is_dram);
        }
        if (causal_mask) {
            uint32_t num_tiles_causal_mask = mask.value().get_legacy_shape()[-1] * mask.value().get_legacy_shape()[-2] / TILE_WIDTH / TILE_HEIGHT;
            reader_compile_time_args.push_back(num_tiles_causal_mask);
        }

        std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                        out0_is_dram};
        std::map<string, string> softmax_defines;
        if (mask.has_value()) {
            softmax_defines["FUSED_SCALE_MASK"] = "1";
        }
        if (causal_mask) {
            softmax_defines["CAUSAL_MASK"] = "1";
        }

       ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    tt::CB::c_in0,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = in0_t * in0_tile_size,
                        .page_size = in0_tile_size,
                        .data_format = in0_cb_data_format,
                    }
                },
                {
                    tt::CB::c_out0,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = out0_t * out0_tile_size,
                        .page_size = out0_tile_size,
                        .data_format = out0_cb_data_format,
                    }
                },
                {
                    tt::CB::c_intermed1,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = im1_t * im_tile_size,
                        .page_size = im_tile_size,
                        .data_format = im_cb_data_format,
                    }
                },
                {
                    tt::CB::c_in2,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = in2_t * scalar_tile_size,
                        .page_size = scalar_tile_size,
                        .data_format = scalar_cb_data_format,
                    }
                },
                {
                    tt::CB::c_intermed0,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = im0_t * im_tile_size,
                        .page_size = im_tile_size,
                        .data_format = im_cb_data_format,
                    }
                },
                {
                    tt::CB::c_in5,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = in5_t * mask_tile_size,
                        .page_size = mask_tile_size,
                        .data_format = mask_cb_data_format,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_device_cores_set,
                    .kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/reader_unary_interleaved_sm.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, softmax_defines)
                },
                {
                    .core_spec = all_device_cores_set,
                    .kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp",
                    .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, softmax_defines)
                }
            },
        };

        softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";

        program_attributes.compute_attributes = {
            {
                .core_spec = all_device_cores_set,
                .kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax.cpp",
                .config = {
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .math_approx_mode = math_approx_mode,
                    .compile_args = {},
                    .defines = softmax_defines,
                },
            },
        };

        uint32_t src_addr = src0_buffer->address();
        uint32_t mask_addr = mask.has_value() ? mask.value().buffer()->address() : 0;
        uint32_t out_addr = out0_buffer->address();

        uint32_t curr_row = 0;
        union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
        for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};
            if (i >= num_cores) {
                program_attributes.data_movement_attributes[0].runtime_args_per_core[core] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // [8]=1.0f is scaler
                program_attributes.compute_attributes[0].runtime_args_per_core[core] =  { 0, 0, 0, 0, 0, 0 };
                program_attributes.data_movement_attributes[1].runtime_args_per_core[core] =  { 0, 0, 0, 0, 0, 0, 0};

                // SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
                // SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0, 0 });
                // SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0});
                continue;
            }
            uint32_t num_tile_rows_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tile_rows_per_core = num_tile_rows_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tile_rows_per_core = num_tile_rows_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }

            uint32_t tile_offset = curr_row * Wt;
            uint32_t curr_ht = curr_row % Ht;
            uint32_t mask_curr_ht = curr_ht % mask_Ht;   // the start offset for causal mask
            uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt; // causal mask batch offset
            uint32_t mask_id = causal_mask ? (mask_curr_ht * Wt + mask_offset) : (curr_row / Ht * Wt); // causal mask start offset + causal mask batch offset

            if (causal_mask) {
                program_attributes.data_movement_attributes[0].runtime_args_per_core[core] =  { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }; // [8]=1.0f is scaler
                // SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }); // [10]=1.0f is scaler
            } else {
                program_attributes.data_movement_attributes[0].runtime_args_per_core[core] =  { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80 }; // [8]=1.0f is scaler
                // SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
            }

            program_attributes.compute_attributes[0].runtime_args_per_core[core] = { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, mask_padded_data };
            // SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht, mask_padded_data });

            program_attributes.data_movement_attributes[1].runtime_args_per_core[core] =  { out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size, mask_padded_data, num_datum_padded, 0xFF00FF00};
            // SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size, mask_padded_data, num_datum_padded, 0xFF00FF00});

            curr_row += num_tile_rows_per_core;
        }

        ttnn::generic_op(std::vector<Tensor>{input_tensor}, device_output_tensor, program_attributes);

        auto reshaped_device_output_tensor = ttnn::reshape(device_output_tensor, input_tensor_generic.get_shape());

        auto output_tensor = reshaped_device_output_tensor.cpu();

        auto allclose = tt::numpy::allclose<bfloat16>(output_tensor_original, output_tensor);

        TT_FATAL(allclose);
    }

    // =================
    // Matmul original and generic test
    {
        // Matmul original
        tt::log_info(tt::LogTest, "Running matmul original test");
        uint32_t Mt_original = 10;
        uint32_t Kt_original = 2;
        uint32_t Nt_original = 4;
        uint32_t B_original = 3;

        Shape shapea = {B_original, 1, Mt_original*TILE_HEIGHT, Kt_original*TILE_WIDTH};
        Shape shapeb = {B_original, 1, Kt_original*TILE_HEIGHT, Nt_original*TILE_WIDTH};
        Tensor a_original = tt::numpy::random::random(shapea);
        Tensor b_original = tt::numpy::random::random(shapeb);

        Tensor a = a_original;
        Tensor b = b_original;

        a_original = a_original.to(Layout::TILE).to(device);
        b_original = b_original.to(Layout::TILE).to(device);

        Tensor mm = tt::operations::primary::matmul(a_original, b_original);

        // Matmul generic

        tt::log_info(tt::LogTest, "Running matmul generic test");

        // Parameters for matmul call - copy paste from matmul_multi_core in
        // bmm_op_multi_core.cpp
        bool bcast_batch = false;

        a = a.to(Layout::TILE).to(device);
        b = b.to(Layout::TILE).to(device);

        const auto& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

        Shape output_shape = Shape{B_original, 1, Mt_original*TILE_HEIGHT, Nt_original*TILE_WIDTH};
        auto output = tt::tt_metal::create_device_tensor(
            output_shape,
            a.tensor_attributes->dtype,
            a.tensor_attributes->layout,
            a.device(),
            a.memory_config());

        tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
        tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.get_dtype());
        tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
        uint32_t in0_single_tile_size = tt::tt_metal::detail::TileSize(in0_data_format);
        uint32_t in1_single_tile_size = tt::tt_metal::detail::TileSize(in1_data_format);
        uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_data_format);
        MathFidelity math_fidelity = MathFidelity::HiFi4;

        tt::tt_metal::Buffer *src0_buffer = a.buffer();
        tt::tt_metal::Buffer *src1_buffer = b.buffer();

        // This should allocate a DRAM buffer on the device
        tt::tt_metal::Device *device = a.device();
        Shape cshape = output.get_legacy_shape(); // C=A*B, N1MK*11KN->N1MN

        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t c_batch_size = get_batch_size(cshape);
        auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

        tt::tt_metal::Buffer *dst_buffer = output.buffer();
        TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

        // C = A*B*...
        // MN = MK*KN
        uint32_t B = get_batch_size(ashape);
        uint32_t Mt = ashape[-2]/TILE_HEIGHT;
        uint32_t Kt = ashape[-1]/TILE_WIDTH;
        uint32_t Nt = bshape[-1]/TILE_WIDTH;
        uint32_t KtNt = Kt * Nt;
        uint32_t MtKt = Mt * Kt;
        uint32_t MtNt = Mt * Nt;

        uint32_t src0_addr = src0_buffer->address();
        uint32_t src1_addr = src1_buffer->address();
        uint32_t dst_addr = dst_buffer->address();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;

        uint32_t src1_cb_index = 1;

        uint32_t output_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 2;

        bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

        bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram
        };

        auto all_device_cores_set = CoreRangeSet({all_cores});

        ttnn::operations::generic::GenericOpDeviceOperation::operation_attributes_t program_attributes =
        {
            .circular_buffer_attributes =
            {
                {
                    src0_cb_index,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = num_input_tiles * in0_single_tile_size,
                        .page_size = in0_single_tile_size,
                        .data_format = in0_data_format,
                    }
                },
                {
                    src1_cb_index,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = num_input_tiles * in1_single_tile_size,
                        .page_size = in1_single_tile_size,
                        .data_format = in1_data_format,
                    }
                },
                {
                    output_cb_index,
                    {
                        .core_spec = all_device_cores_set,
                        .total_size = num_output_tiles * output_single_tile_size,
                        .page_size = output_single_tile_size,
                        .data_format = output_data_format,
                    }
                }
            },
            .data_movement_attributes =
            {
                {
                    .core_spec = all_device_cores_set,
                    .kernel_path = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
                    .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args)
                },
                {
                    .core_spec = all_device_cores_set,
                    .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                    .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args)
                }
            },
        };

        vector<uint32_t> compute_args_group_1 = {
            1, // B
            1, // Mt
            Kt, // Kt
            num_output_tiles_per_core_group_1 // Nt
        }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

        if (!core_group_2.ranges().empty()) {
            vector<uint32_t> compute_args_group_2 = {
                1, // B
                1, // Mt
                Kt, // Kt
                num_output_tiles_per_core_group_2 // Nt
            }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

            program_attributes.compute_attributes = {
                {
                    .core_spec = core_group_1,
                    .kernel_path = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
                    .config = {
                        .math_fidelity = math_fidelity,
                        .compile_args = compute_args_group_1,
                    },
                },
                {
                    .core_spec = all_device_cores_set,
                    .kernel_path ="ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
                    .config = {
                        .math_fidelity = math_fidelity,
                        .compile_args = compute_args_group_2,
                    },
                }
            };

        } else {
            TT_FATAL(false,
                     "Core group 2 for matmul generic test is empty. Purpose of the test is to test generic op "
                     "with multiple core groups, so we should never hit this case.");
        }


        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){

            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            uint32_t num_output_tiles_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            } else {
                TT_FATAL(false, "Core not in specified core ranges");
            }

            program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {
                src0_addr,
                src1_addr,
                Mt,
                Kt,
                Nt,
                MtKt,
                KtNt,
                B,
                uint32_t(bcast_batch),
                num_tiles_written,
                num_output_tiles_per_core,
                MtNt };

            program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {
                dst_addr,
                num_output_tiles_per_core,
                num_tiles_written };

            // tt_metal::SetRuntimeArgs(
            //     program, reader_id, core,
            //     {src0_addr,
            //     src1_addr,
            //     Mt,
            //     Kt,
            //     Nt,
            //     MtKt,
            //     KtNt,
            //     B,
            //     uint32_t(bcast_batch),
            //     num_tiles_written,
            //     num_output_tiles_per_core,
            //     MtNt }
            // );
            // tt_metal::SetRuntimeArgs(
            //     program,
            //     writer_id,
            //     core,
            //     {dst_addr,
            //     num_output_tiles_per_core,
            //     num_tiles_written }
            // );

            num_tiles_written += num_output_tiles_per_core;
        }

        ttnn::generic_op(std::vector<Tensor>{a, b}, output, program_attributes);

        auto output_tensor = output.cpu();

        auto allclose = tt::numpy::allclose<bfloat16>(mm.cpu(), output_tensor);

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
