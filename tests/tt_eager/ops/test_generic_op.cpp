// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "base_types.hpp"
#include <tt_metal/api/tt-metalium/core_coord.hpp>
#include <tt_metal/api/tt-metalium/work_split.hpp>
#include <tt_metal/api/tt-metalium/host_api.hpp>
#include <tt_metal/api/tt-metalium/assert.hpp>

#include "kernel_types.hpp"
#include "ttnn/operations/generic/generic_op/generic_op.hpp"

#include "logger.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::IDevice;

using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_HW;
using tt::constants::TILE_WIDTH;

using namespace ttnn::operations::generic;

void test_matmul(tt::tt_metal::IDevice* device) {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    // =================
    // Matmul original and generic test
    tt::log_info(tt::LogTest, "Running matmul original test");
    uint32_t Mt_original = 10;
    uint32_t Kt_original = 2;
    uint32_t Nt_original = 4;
    uint32_t B_original = 3;

    ttnn::Shape shapea({B_original, 1, Mt_original * TILE_HEIGHT, Kt_original * TILE_WIDTH});
    ttnn::Shape shapeb({B_original, 1, Kt_original * TILE_HEIGHT, Nt_original * TILE_WIDTH});
    Tensor input_tensor_a = ttnn::random::random(shapea).to_layout(Layout::TILE).to_device(device);
    ;
    Tensor input_tensor_b = ttnn::random::random(shapeb).to_layout(Layout::TILE).to_device(device);
    ;

    Tensor golden = ttnn::matmul(input_tensor_a, input_tensor_b);

    tt::log_info(tt::LogTest, "Running matmul generic test");

    // Parameters for matmul call - copy paste from matmul_multi_core in bmm_op_multi_core.cpp
    bool bcast_batch = false;

    ttnn::Shape output_shape = ttnn::Shape{B_original, 1, Mt_original * TILE_HEIGHT, Nt_original * TILE_WIDTH};
    auto output = tt::tt_metal::create_device_tensor(
        output_shape,
        input_tensor_a.get_dtype(),
        input_tensor_a.get_layout(),
        input_tensor_a.device(),
        input_tensor_a.memory_config());

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in0_single_tile_size = tt::tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt::tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt::tt_metal::Buffer* src0_buffer = input_tensor_a.buffer();
    tt::tt_metal::Buffer* src1_buffer = input_tensor_b.buffer();

    ttnn::Shape cshape = output.get_logical_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    const auto &ashape = input_tensor_a.get_logical_shape(), bshape = input_tensor_b.get_logical_shape();
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    auto src0_cb_index = tt::CBIndex::c_0;
    auto src1_cb_index = tt::CBIndex::c_1;
    auto output_cb_index = tt::CBIndex::c_16;
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    auto all_device_cores_set = CoreRangeSet({all_cores});

    ttnn::operations::generic::circular_buffer_attributes_t src0_cb_attributes = {
        .core_spec = all_device_cores_set,
        .total_size = num_input_tiles * in0_single_tile_size,
        .page_size = in0_single_tile_size,
        .data_format = in0_data_format,
    };

    ttnn::operations::generic::circular_buffer_attributes_t src1_cb_attributes = {
        .core_spec = all_device_cores_set,
        .total_size = num_input_tiles * in1_single_tile_size,
        .page_size = in1_single_tile_size,
        .data_format = in1_data_format,
    };

    ttnn::operations::generic::circular_buffer_attributes_t output_cb_attributes = {
        .core_spec = all_device_cores_set,
        .total_size = num_output_tiles * output_single_tile_size,
        .page_size = output_single_tile_size,
        .data_format = output_data_format,
    };

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes =
            {{src0_cb_index, src0_cb_attributes},
             {src1_cb_index, src1_cb_attributes},
             {output_cb_index, output_cb_attributes}},
        .data_movement_attributes =
            {{.core_spec = all_device_cores_set,
              .kernel_path = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                             "reader_bmm_8bank_output_tiles_partitioned.cpp",
              .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args)},
             {.core_spec = all_device_cores_set,
              .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                             "writer_unary_interleaved_start_id.cpp",
              .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args)}},
    };

    std::vector<uint32_t> compute_args_group_1 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
        // for simplicity

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {
            1,                                 // B
            1,                                 // Mt
            Kt,                                // Kt
            num_output_tiles_per_core_group_2  // Nt
        };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set
            // Nt for simplicity

        program_attributes.compute_attributes = {
            {
                .core_spec = core_group_1,
                .kernel_path = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
                .config =
                    {
                        .math_fidelity = math_fidelity,
                        .compile_args = compute_args_group_1,
                    },
            },
            {
                .core_spec = all_device_cores_set,
                .kernel_path = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
                .config =
                    {
                        .math_fidelity = math_fidelity,
                        .compile_args = compute_args_group_2,
                    },
            }};

    } else {
        TT_FATAL(
            false,
            "Core group 2 for matmul generic test is empty. Purpose of the test is to test generic op "
            "with multiple core groups, so we should never hit this case.");
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
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
            MtNt};

        program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {
            dst_addr, num_output_tiles_per_core, num_tiles_written};

        num_tiles_written += num_output_tiles_per_core;
    }

    ttnn::generic_op(std::vector<Tensor>{input_tensor_a, input_tensor_b, output}, program_attributes);

    auto output_tensor = output.cpu();

    auto allclose = ttnn::allclose<bfloat16>(golden.cpu(), output_tensor, 1e-1f, 1e-5f);

    TT_FATAL(allclose, "Error");
}

void test_eltwise_sfpu(tt::tt_metal::IDevice* device) {
    const std::map<std::string, std::string> sfpu_defines = {
        {"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}};

    tt::log_info(tt::LogTest, "Running {}", __func__);

    uint32_t num_tiles = 4;
    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    auto shape = ttnn::Shape{1, num_tiles, TILE_HEIGHT, TILE_WIDTH};
    Tensor input_tensor = ttnn::random::random(shape, DataType::BFLOAT16);
    ttnn::MemoryConfig dram_memory_config = ttnn::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};

    Tensor device_input_tensor = input_tensor.to_layout(Layout::TILE).to_device(device, dram_memory_config);
    Tensor device_output_tensor = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(
            device_input_tensor.get_logical_shape(),
            ttnn::TensorLayout(
                device_input_tensor.get_dtype(), ttnn::PageConfig(device_input_tensor.get_layout()), device_input_tensor.memory_config())),
        device_input_tensor.device()
    );

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.get_dtype());
    bool is_dram_input = device_input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    CoreCoord core = {0, 0};
    CoreRange core_range = {core, core};
    CoreRangeSet device_cores = std::set<CoreRange>({core_range});

    ttnn::operations::generic::circular_buffer_attributes_t input_cb_attributes = {
        .core_spec = device_cores,
        .total_size = 2 * tt::tt_metal::detail::TileSize(input_cb_data_format),
        .page_size = tt::tt_metal::detail::TileSize(input_cb_data_format),
        .data_format = input_cb_data_format,
    };

    ttnn::operations::generic::circular_buffer_attributes_t output_cb_attributes = {
        .core_spec = device_cores,
        .total_size = 2 * tt::tt_metal::detail::TileSize(input_cb_data_format),
        .page_size = tt::tt_metal::detail::TileSize(input_cb_data_format),
        .data_format = input_cb_data_format,
    };

    const std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)is_dram_input};
    const std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CBIndex::c_16, (std::uint32_t)is_dram_input};
    const std::vector<uint32_t> read_rt_args = {device_input_tensor.buffer()->address(), num_tiles, src_bank_id};
    const std::vector<uint32_t> write_rt_args = {device_output_tensor.buffer()->address(), num_tiles, dst_bank_id};

    ttnn::operations::generic::data_movement_attributes_t reader_attributes = {
        .core_spec = device_cores,
        .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args),
        .runtime_args_per_core = {{core, read_rt_args}},
    };

    ttnn::operations::generic::data_movement_attributes_t writer_attributes = {
        .core_spec = device_cores,
        .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args),
        .runtime_args_per_core = {{core, write_rt_args}},
    };

    ttnn::operations::generic::compute_attributes_t compute_attributes = {
        .core_spec = device_cores,
        .kernel_path = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        .config = {
            .math_approx_mode = false,
            .compile_args = {num_tiles, 1},
            .defines = sfpu_defines,
        },
    };

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes = {
            {tt::CBIndex::c_0, input_cb_attributes},
            {tt::CBIndex::c_16, output_cb_attributes},
        },
        .data_movement_attributes = {reader_attributes, writer_attributes},
        .compute_attributes = {compute_attributes},
    };

    Tensor device_output = ttnn::generic_op(std::vector{device_input_tensor, device_output_tensor}, program_attributes);
    Tensor golden = ttnn::exp(device_input_tensor);

    auto allclose = ttnn::allclose<bfloat16>(golden.cpu(), device_output.cpu());

    TT_FATAL(allclose, "Error");

}

int main(int argc, char** argv) {
    constexpr int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    test_eltwise_sfpu(device);
    test_matmul(device);

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Failed to close device");
    return 0;
}
