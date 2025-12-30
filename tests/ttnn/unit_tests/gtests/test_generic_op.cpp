// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_metal/api/tt-metalium/core_coord.hpp>
#include <tt_metal/api/tt-metalium/work_split.hpp>
#include <tt_metal/api/tt-metalium/host_api.hpp>
#include <tt_metal/impl/buffers/semaphore.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

#include <tt-logger/tt-logger.hpp>
#include "ttnn_test_fixtures.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/to_string.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/point_to_point/point_to_point.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/distributed/api.hpp"
#include <ttnn/distributed/distributed_tensor.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "ttnn/tensor/shape/shape.hpp"
#include <llrt/tt_cluster.hpp>

namespace ttnn::operations::generic::test {

TEST_F(TTNNFixtureWithDevice, TestGenericOpArgmaxSingleCore) {
    uint32_t batch = 1;
    uint32_t channels = 4;
    ttnn::Shape shape{batch, channels, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor input_tensor = ttnn::random::random(shape, DataType::BFLOAT16);
    Tensor device_input_tensor = input_tensor.to_device(this->device_);
    Tensor golden = ttnn::argmax(device_input_tensor).cpu();

    Tensor device_output_tensor = tt::tt_metal::create_device_tensor(golden.tensor_spec(), this->device_);

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t unit_size = input_tensor.element_size();

    const CoreCoord core_coord(0, 0);
    const CoreRangeSet core = std::set<CoreRange>({core_coord, core_coord});

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t rank = input_shape.size();
    const bool reduce_all = true;
    const uint32_t red_dim_units = input_shape[rank - 1];  // Last dimension in input i.e. reduction dimension
    const auto output_last_dim = 1;                        // Last dimension in output i.e. the dim left after reduction

    // Create input CB to read reduction dim worth of data at once
    // Create output CB based on the output shape's last dimension
    const tt::CBIndex src_cb_idx = tt::CBIndex::c_0;
    const tt::CBIndex dst_cb_idx = tt::CBIndex::c_1;
    const uint32_t src_page_size = round_up_to_mul32(red_dim_units * unit_size);
    const uint32_t dst_page_size = round_up_to_mul32(output_last_dim * unit_size);

    CBFormatDescriptor input_format_descriptor = {
        .buffer_index = src_cb_idx,
        .data_format = cb_data_format,
        .page_size = src_page_size,
    };
    CBFormatDescriptor output_format_descriptor = {
        .buffer_index = dst_cb_idx,
        .data_format = cb_data_format,
        .page_size = dst_page_size,
    };
    CBDescriptor input_cb_descriptor = {
        .total_size = src_page_size,
        .core_ranges = core,
        .format_descriptors = {input_format_descriptor},
    };
    CBDescriptor output_cb_descriptor = {
        .total_size = dst_page_size,
        .core_ranges = core,
        .format_descriptors = {output_format_descriptor},
    };

    auto* const src_buffer = device_input_tensor.buffer();
    auto* const dst_buffer = device_output_tensor.buffer();

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input_tensor.logical_volume() / inner_dim_units / red_dim_units;

    KernelDescriptor::CompileTimeArgs compile_time_args = {
        (uint32_t)src_cb_idx,
        (uint32_t)dst_cb_idx,
        src_page_size,
        dst_page_size,
        outer_dim_units,
        inner_dim_units,
        red_dim_units,
        (uint32_t)(reduce_all),
    };
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    const KernelDescriptor::CoreRuntimeArgs runtime_args = {
        src_buffer->address(),
        dst_buffer->address(),
    };
    const KernelDescriptor::RuntimeArgs runtime_args_per_cores = {{{0, 0}, runtime_args}};  // single-core

    KernelDescriptor kernel_descriptor = {
        .kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp",
        .core_ranges = core,
        .compile_time_args = compile_time_args,
        .runtime_args = runtime_args_per_cores,
        .common_runtime_args = {},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {kernel_descriptor},
        .semaphores = {},
        .cbs = {input_cb_descriptor, output_cb_descriptor},
    };

    ttnn::generic_op(std::vector<Tensor>{device_input_tensor, device_output_tensor}, program_descriptor);
    Tensor output_tensor = device_output_tensor.cpu();
    auto allclose = ttnn::allclose<uint32_t>(golden, output_tensor);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpUnaryReluSharded) {
    const std::vector<std::pair<std::string, std::string>> defines_relu = {
        {"SFPU_OP_CHAIN_0", "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"},
        {"SFPU_OP_CHAIN_0_FUNC_0", "relu_tile(0);"},
        {"SFPU_OP_CHAIN_0_INIT_0", "relu_tile_init();"},
        {"SFPU_OP_RELU_FAMILY_INCLUDE", "1"}};

    auto shape = ttnn::Shape{64, 16, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    CoreCoord compute_with_storage_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange all_cores_range = {
        CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    ttnn::MemoryConfig mem_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            all_cores,
            {(num_cores_x * num_cores_y) * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto input_tensor = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), shape, Layout::TILE);
    auto device_input_tensor = input_tensor.to_device(this->device_, mem_config);
    auto device_output_tensor = tt::tt_metal::create_device_tensor(device_input_tensor.tensor_spec(), this->device_);

    auto shard_spec = device_input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.grid == all_cores, "shard spec grid should be same as all_cores");

    log_info(tt::LogTest, "Running ttnn unary relu sharded");
    auto golden = ttnn::relu(device_input_tensor).cpu();

    log_info(tt::LogTest, "Running generic_op unary relu sharded");
    auto act_df = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.dtype());
    auto out_df = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.dtype());
    uint32_t input_tile_size = tt::tile_size(act_df);
    uint32_t output_tile_size = tt::tile_size(out_df);
    TT_FATAL(input_tile_size == output_tile_size, "input and output tile size should be the same");

    uint32_t num_tile_per_core = 0;
    size_t shard_height = shard_spec.shape[0];
    size_t shard_width = shard_spec.shape[1];
    size_t shard_size_in_bytes = shard_height * shard_width * datum_size(act_df);
    TT_FATAL(shard_size_in_bytes % input_tile_size == 0, "Shard Size must be multiple of input_tile_size");
    num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size;  // ceil value

    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;

    tt::CBIndex in_cb_id = tt::CBIndex::c_0;
    tt::CBIndex out_cb_id = tt::CBIndex::c_2;
    CBFormatDescriptor in_format_descriptor = {
        .buffer_index = in_cb_id,
        .data_format = act_df,
        .page_size = in_cb_pagesize,
    };
    CBFormatDescriptor out_format_descriptor = {
        .buffer_index = out_cb_id,
        .data_format = out_df,
        .page_size = in_cb_pagesize,
    };
    CBDescriptor input_cb_descriptor = {
        .total_size = in_cb_npages * in_cb_pagesize,
        .core_ranges = all_cores,
        .format_descriptors = {in_format_descriptor},
        .buffer = device_input_tensor.buffer(),
    };
    CBDescriptor output_cb_descriptor = {
        .total_size = in_cb_npages * in_cb_pagesize,
        .core_ranges = all_cores,
        .format_descriptors = {out_format_descriptor},
        .buffer = device_output_tensor.buffer(),
    };

    const KernelDescriptor::CompileTimeArgs reader_ct_args = {(std::uint32_t)in_cb_id};
    const KernelDescriptor::CompileTimeArgs compute_ct_args = {1, num_tile_per_core};

    // calculate data movement runtime arguments: every core has the same runtime args
    KernelDescriptor::RuntimeArgs reader_rt_args_per_core;
    for (uint32_t i = 0; i < num_cores_x * num_cores_y; i++) {
        uint32_t core_x = i / num_cores_y;
        uint32_t core_y = i % num_cores_y;
        reader_rt_args_per_core.push_back({{core_x, core_y}, {num_tile_per_core}});
    }

    KernelDescriptor reader_kernel_descriptor = {
        .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        .core_ranges = all_cores,
        .compile_time_args = reader_ct_args,
        .runtime_args = reader_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    KernelDescriptor compute_kernel_descriptor = {
        .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        .core_ranges = all_cores,
        .compile_time_args = compute_ct_args,
        .defines = defines_relu,
        .common_runtime_args = {},
        .config = tt::tt_metal::ComputeConfigDescriptor{},
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {reader_kernel_descriptor, compute_kernel_descriptor},
        .semaphores = {},
        .cbs = {input_cb_descriptor, output_cb_descriptor},
    };
    ttnn::generic_op(std::vector{device_input_tensor, device_output_tensor}, program_descriptor);
    auto device_output = device_output_tensor.cpu();
    auto allclose = ttnn::allclose<bfloat16>(golden, device_output, 1e-1f, 1e-5f);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpBinaryEltwiseAdd) {
    const std::vector<std::pair<std::string, std::string>> defines_eltwise_add = {
        {"ELTWISE_OP", "add_tiles"},
        {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"},
    };

    log_info(tt::LogTest, "Running ttnn binary add interleaved");
    ttnn::Shape shape{11, 9, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);
    auto device_input_tensor_a = input_tensor_a.to_layout(Layout::TILE).to_device(this->device_);
    auto device_input_tensor_b = input_tensor_b.to_layout(Layout::TILE).to_device(this->device_);

    auto golden = ttnn::add(device_input_tensor_a, device_input_tensor_b).cpu().to_layout(Layout::ROW_MAJOR);

    log_info(tt::LogTest, "Running generic add interleaved");

    // Data movement kernel needs output tensor address to be passed as a runtime argument.
    auto device_output_tensor = tt::tt_metal::create_device_tensor(device_input_tensor_a.tensor_spec(), this->device_);

    auto compute_with_storage_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange all_cores_range = {
        CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

    auto input_a_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_a.dtype());
    auto input_b_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_b.dtype());
    auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.dtype());

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    tt::CBIndex dst_cb_index = tt::CBIndex::c_2;

    CBFormatDescriptor in0_cb_format_descriptor = {
        .buffer_index = src0_cb_index,
        .data_format = input_a_cb_data_format,
        .page_size = tt::tile_size(input_a_cb_data_format),
    };
    CBFormatDescriptor in1_cb_format_descriptor = {
        .buffer_index = src1_cb_index,
        .data_format = input_b_cb_data_format,
        .page_size = tt::tile_size(input_b_cb_data_format),
    };
    CBFormatDescriptor out_cb_format_descriptor = {
        .buffer_index = dst_cb_index,
        .data_format = output_cb_data_format,
        .page_size = tt::tile_size(output_cb_data_format),
    };
    CBDescriptor in0_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_a_cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {in0_cb_format_descriptor},
    };
    CBDescriptor in1_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_b_cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {in1_cb_format_descriptor},
    };
    CBDescriptor output_cb_descriptor = {
        .total_size = 2 * tt::tile_size(output_cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {out_cb_format_descriptor},
    };

    bool block_or_width_sharded = false;
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {(uint32_t)block_or_width_sharded};
    TensorAccessorArgs(*device_input_tensor_a.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*device_input_tensor_b.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {dst_cb_index};
    TensorAccessorArgs(*device_output_tensor.buffer()).append_to(writer_compile_time_args);

    // setup runtime arguments for data movement kernels
    uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t num_tiles = device_input_tensor_a.physical_volume() / tt::constants::TILE_HW;
    bool row_major = true;
    auto [num_cores, _, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major);

    uint32_t block_size_per_core_group_1 = 1;
    uint32_t block_size_per_core_group_2 = 1;
    uint32_t block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
    uint32_t block_cnt_per_core_group_2 = num_tiles_per_core_group_2;
    auto cores =
        grid_to_cores(num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    KernelDescriptor::RuntimeArgs reader_rt_args_per_core;
    KernelDescriptor::RuntimeArgs writer_rt_args_per_core;
    KernelDescriptor::RuntimeArgs compute_rt_args_per_core;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t core_x = core.x;
        uint32_t core_y = core.y;
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
        }

        reader_rt_args_per_core.push_back(
            {{core_x, core_y},
             {
                 device_input_tensor_a.buffer()->address(),
                 device_input_tensor_b.buffer()->address(),
                 num_tiles_per_core,
                 num_tiles_read,  // start_id
                 0,               // block_height = 0 when not sharded
                 0,               // block_width = 0 when not sharded
                 0,               // num_cores_y = 0 when not sharded
             }});
        writer_rt_args_per_core.push_back(
            {{core_x, core_y},
             {
                 device_output_tensor.buffer()->address(),
                 num_tiles_per_core,
                 num_tiles_read  // start_id
             }});
        compute_rt_args_per_core.push_back({{core_x, core_y}, {block_cnt_per_core, block_size_per_core}});

        num_tiles_read += num_tiles_per_core;
    }

    KernelDescriptor reader_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        .core_ranges = all_cores,
        .compile_time_args = reader_compile_time_args,
        .runtime_args = reader_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    KernelDescriptor writer_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .core_ranges = all_cores,
        .compile_time_args = writer_compile_time_args,
        .runtime_args = writer_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    KernelDescriptor compute_kernel_descriptor = {
        .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
        .core_ranges = all_cores,
        .compile_time_args = {},
        .defines = defines_eltwise_add,
        .runtime_args = compute_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::ComputeConfigDescriptor{},
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor},
        .semaphores = {},
        .cbs = {in0_cb_descriptor, in1_cb_descriptor, output_cb_descriptor},
    };

    ttnn::generic_op(
        std::vector<Tensor>{device_input_tensor_a, device_input_tensor_b, device_output_tensor}, program_descriptor);

    auto device_output = device_output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto allclose = ttnn::allclose<bfloat16>(golden, device_output);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpMatmul) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == tt::BoardType::P150) {
        GTEST_SKIP();
    }
    log_info(tt::LogTest, "Running ttnn matmul");
    uint32_t Mt_original = 10;
    uint32_t Kt_original = 2;
    uint32_t Nt_original = 4;
    uint32_t B_original = 3;

    ttnn::Shape shapea(
        {B_original, 1, Mt_original * tt::constants::TILE_HEIGHT, Kt_original * tt::constants::TILE_WIDTH});
    ttnn::Shape shapeb(
        {B_original, 1, Kt_original * tt::constants::TILE_HEIGHT, Nt_original * tt::constants::TILE_WIDTH});
    Tensor input_tensor_a = ttnn::random::random(shapea).to_layout(Layout::TILE).to_device(this->device_);
    Tensor input_tensor_b = ttnn::random::random(shapeb).to_layout(Layout::TILE).to_device(this->device_);

    Tensor golden = ttnn::matmul(
        input_tensor_a,
        input_tensor_b,
        false,                                  // transpose_a
        false,                                  // transpose_b
        std::nullopt,                           // memory_config
        std::nullopt,                           // dtype
        matmul::MatmulMultiCoreProgramConfig{}  // program_config to indicate we want multi-core
    );

    log_info(tt::LogTest, "Running matmul generic test");

    // Parameters for matmul call - copy paste from matmul_multi_core in bmm_op_multi_core.cpp
    bool bcast_batch = false;

    ttnn::Shape output_shape =
        ttnn::Shape{B_original, 1, Mt_original * tt::constants::TILE_HEIGHT, Nt_original * tt::constants::TILE_WIDTH};
    auto output = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor_a.dtype(),
                tt::tt_metal::PageConfig(input_tensor_a.layout()),
                input_tensor_a.memory_config())),
        input_tensor_a.device());

    tt::tt_metal::Buffer* src0_buffer = input_tensor_a.buffer();
    tt::tt_metal::Buffer* src1_buffer = input_tensor_b.buffer();

    ttnn::Shape cshape = output.logical_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = this->device_->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / tt::constants::TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    TT_FATAL(
        !core_group_2.ranges().empty(),
        "Core group 2 for matmul generic test is empty. We should never hit this case.");

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto &ashape = input_tensor_a.logical_shape(), bshape = input_tensor_b.logical_shape();
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Nt = bshape[-1] / tt::constants::TILE_WIDTH;
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

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    auto all_device_cores_set = CoreRangeSet({all_cores});

    tt::tt_metal::CBFormatDescriptor in0_format_descriptor = {
        .buffer_index = src0_cb_index,
        .data_format = in0_data_format,
        .page_size = in0_single_tile_size,
    };
    tt::tt_metal::CBFormatDescriptor in1_format_descriptor = {
        .buffer_index = src1_cb_index,
        .data_format = in1_data_format,
        .page_size = in1_single_tile_size,
    };
    tt::tt_metal::CBFormatDescriptor output_format_descriptor = {
        .buffer_index = output_cb_index,
        .data_format = output_data_format,
        .page_size = output_single_tile_size,
    };

    tt::tt_metal::CBDescriptor in0_cb_descriptor = {
        .total_size = num_input_tiles * in0_single_tile_size,
        .core_ranges = all_device_cores_set,
        .format_descriptors = {in0_format_descriptor},
    };
    tt::tt_metal::CBDescriptor in1_cb_descriptor = {
        .total_size = num_input_tiles * in1_single_tile_size,
        .core_ranges = all_device_cores_set,
        .format_descriptors = {in1_format_descriptor},
    };
    tt::tt_metal::CBDescriptor output_cb_descriptor = {
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_device_cores_set,
        .format_descriptors = {output_format_descriptor},
    };

    uint32_t last_ktile_w = input_tensor_a.logical_shape()[-1] % tt::constants::TILE_WIDTH;
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {last_ktile_w};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {(uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    log_info(tt::LogTest, "num_cores: {}, num_core_x: {}, num_core_y: {}", num_cores, num_cores_x, num_cores_y);
    KernelDescriptor::RuntimeArgs reader_rt_args_per_core;
    KernelDescriptor::RuntimeArgs writer_rt_args_per_core;

    // setup reader/writer runtime args
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        uint32_t core_x = i / num_cores_y;
        uint32_t core_y = i % num_cores_y;
        CoreCoord core = {core_x, core_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        reader_rt_args_per_core.push_back(
            {{core_x, core_y},
             {src0_addr,
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
              MtNt}});

        writer_rt_args_per_core.push_back({{core_x, core_y}, {dst_addr, num_output_tiles_per_core, num_tiles_written}});

        log_info(
            tt::LogTest,
            "core: {}, reader_rt_args {}, writer_rt_args {}",
            core,
            reader_rt_args_per_core.back().second,
            writer_rt_args_per_core.back().second);

        num_tiles_written += num_output_tiles_per_core;
    }
    tt::tt_metal::KernelDescriptor reader_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
        .core_ranges = all_device_cores_set,
        .compile_time_args = reader_compile_time_args,
        .runtime_args = reader_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor writer_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .core_ranges = all_device_cores_set,
        .compile_time_args = writer_compile_time_args,
        .runtime_args = writer_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };

    const KernelDescriptor::CompileTimeArgs compute_ct_args_group_1 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
        // for simplicity
    const KernelDescriptor::CompileTimeArgs compute_ct_args_group_2 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_2  // Nt
    };
    log_info(tt::LogTest, "core_group_1: {}, core_group_2: {}", core_group_1.ranges(), core_group_2.ranges());
    tt::tt_metal::KernelDescriptor compute_kernel_descriptor_1 = {
        .kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        .core_ranges = core_group_1,
        .compile_time_args = compute_ct_args_group_1,
        .defines = {},
        .runtime_args = {{{}}},
        .common_runtime_args = {},
        .config = tt::tt_metal::ComputeConfigDescriptor{.dst_full_sync_en = true},
    };
    tt::tt_metal::KernelDescriptor compute_kernel_descriptor_2 = {
        .kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        .core_ranges = core_group_2,
        .compile_time_args = compute_ct_args_group_2,
        .defines = {},
        .runtime_args = {{{}}},
        .common_runtime_args = {},
        .config = tt::tt_metal::ComputeConfigDescriptor{.dst_full_sync_en = true},
    };

    tt::tt_metal::ProgramDescriptor program_descriptor = {
        .kernels =
            {reader_kernel_descriptor,
             writer_kernel_descriptor,
             compute_kernel_descriptor_1,
             compute_kernel_descriptor_2},
        .semaphores = {},
        .cbs = {in0_cb_descriptor, in1_cb_descriptor, output_cb_descriptor},
    };

    ttnn::generic_op(std::vector<Tensor>{input_tensor_a, input_tensor_b, output}, program_descriptor);
    auto output_tensor = output.cpu();
    auto allclose = ttnn::allclose<bfloat16>(golden.cpu(), output_tensor, 1e-1f, 1e-5f);

    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpEltwiseSFPU) {
    const std::vector<std::pair<std::string, std::string>> sfpu_defines = {
        {"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}};

    uint32_t num_tiles = 4;
    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    auto shape = ttnn::Shape{1, num_tiles, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor input_tensor = ttnn::random::random(shape, DataType::BFLOAT16);
    ttnn::MemoryConfig dram_memory_config =
        ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    Tensor device_input_tensor = input_tensor.to_layout(Layout::TILE).to_device(this->device_, dram_memory_config);
    Tensor device_output_tensor = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(
            device_input_tensor.logical_shape(),
            ttnn::TensorLayout(
                device_input_tensor.dtype(),
                ttnn::PageConfig(device_input_tensor.layout()),
                device_input_tensor.memory_config())),
        device_input_tensor.device());

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.dtype());

    CoreCoord core = {0, 0};
    CoreRange core_range = {core, core};
    CoreRangeSet device_cores = std::set<CoreRange>({core_range});
    tt::CBIndex cb_in_id = tt::CBIndex::c_0;
    tt::CBIndex cb_out_id = tt::CBIndex::c_16;

    CBFormatDescriptor input_cb_format_descriptor = {
        .buffer_index = cb_in_id,
        .data_format = input_cb_data_format,
        .page_size = tt::tile_size(input_cb_data_format),
    };
    CBFormatDescriptor output_cb_format_descriptor = {
        .buffer_index = cb_out_id,
        .data_format = input_cb_data_format,
        .page_size = tt::tile_size(input_cb_data_format),
    };
    CBDescriptor input_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_cb_data_format),
        .core_ranges = device_cores,
        .format_descriptors = {input_cb_format_descriptor},
    };
    CBDescriptor output_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_cb_data_format),
        .core_ranges = device_cores,
        .format_descriptors = {output_cb_format_descriptor},
    };

    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    TensorAccessorArgs(*device_input_tensor.buffer()).append_to(reader_compile_time_args);
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {(std::uint32_t)cb_out_id};
    TensorAccessorArgs(*device_output_tensor.buffer()).append_to(writer_compile_time_args);

    // only core (0, 0) is used
    const KernelDescriptor::CoreRuntimeArgs reader_rt_args = {
        device_input_tensor.buffer()->address(), num_tiles, src_bank_id};
    const KernelDescriptor::RuntimeArgs reader_rt_args_per_core = {{{0, 0}, reader_rt_args}};

    const KernelDescriptor::CoreRuntimeArgs writer_rt_args = {
        device_output_tensor.buffer()->address(), num_tiles, dst_bank_id};
    const KernelDescriptor::RuntimeArgs writer_rt_args_per_core = {{{0, 0}, writer_rt_args}};

    KernelDescriptor reader_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        .core_ranges = device_cores,
        .compile_time_args = reader_compile_time_args,
        .runtime_args = reader_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    KernelDescriptor writer_kernel_descriptor = {
        .kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .core_ranges = device_cores,
        .compile_time_args = writer_compile_time_args,
        .runtime_args = writer_rt_args_per_core,
        .common_runtime_args = {},
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    KernelDescriptor compute_kernel_descriptor = {
        .kernel_source = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        .core_ranges = device_cores,
        .compile_time_args = {num_tiles, 1},
        .defines = sfpu_defines,
        .runtime_args = {{{0, 0}, {}}},
        .common_runtime_args = {},
        .config = tt::tt_metal::ComputeConfigDescriptor{},
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor},
        .semaphores = {},
        .cbs = {input_cb_descriptor, output_cb_descriptor},
    };

    log_info(tt::LogTest, "Running ttnn unary exp");
    Tensor golden = ttnn::exp(device_input_tensor);
    log_info(tt::LogTest, "Running generic_op unary exp");
    Tensor device_output = ttnn::generic_op(std::vector{device_input_tensor, device_output_tensor}, program_descriptor);

    auto allclose = ttnn::allclose<bfloat16>(golden.cpu(), device_output.cpu());

    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpProgramCache) {
    log_info(tt::LogTest, "Running {}", __func__);

    const std::vector<std::pair<std::string, std::string>> sfpu_defines = {
        {"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}};

    ttnn::Shape shape{1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    // Setup initial tensors
    Tensor input_tensor_1 = ttnn::random::random(shape, DataType::BFLOAT16);
    Tensor device_input_tensor_1 = input_tensor_1.to_layout(Layout::TILE).to_device(this->device_);
    Tensor device_output_tensor_1 =
        tt::tt_metal::create_device_tensor(device_input_tensor_1.tensor_spec(), this->device_);

    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_1.dtype());
    uint32_t num_tiles = device_input_tensor_1.physical_volume() / tt::constants::TILE_HW;

    CoreCoord core = {0, 0};
    CoreRange core_range = {core, core};
    CoreRangeSet device_cores = CoreRangeSet(core_range);
    tt::CBIndex cb_in_id = tt::CBIndex::c_0;
    tt::CBIndex cb_out_id = tt::CBIndex::c_16;

    CBDescriptor input_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_cb_data_format),
        .core_ranges = device_cores,
        .format_descriptors = {{cb_in_id, input_cb_data_format, tt::tile_size(input_cb_data_format)}},
    };
    CBDescriptor output_cb_descriptor = {
        .total_size = 2 * tt::tile_size(input_cb_data_format),
        .core_ranges = device_cores,
        .format_descriptors = {{cb_out_id, input_cb_data_format, tt::tile_size(input_cb_data_format)}},
    };

    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*device_input_tensor_1.buffer()).append_to(reader_ct_args);
    KernelDescriptor::CompileTimeArgs writer_ct_args = {(uint32_t)cb_out_id};
    TensorAccessorArgs(*device_output_tensor_1.buffer()).append_to(writer_ct_args);

    ProgramDescriptor program_descriptor = {
        .kernels =
            {{
                 .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                  "reader_unary_interleaved_start_id.cpp",
                 .core_ranges = device_cores,
                 .compile_time_args = reader_ct_args,
                 .runtime_args = {{{0, 0}, {device_input_tensor_1.buffer()->address(), num_tiles, 0u}}},
                 .config = tt::tt_metal::ReaderConfigDescriptor{},
             },
             {
                 .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                  "writer_unary_interleaved_start_id.cpp",
                 .core_ranges = device_cores,
                 .compile_time_args = writer_ct_args,
                 .runtime_args = {{{0, 0}, {device_output_tensor_1.buffer()->address(), num_tiles, 0u}}},
                 .config = tt::tt_metal::WriterConfigDescriptor{},
             },
             {
                 .kernel_source = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                 .core_ranges = device_cores,
                 .compile_time_args = {num_tiles, 1},
                 .defines = sfpu_defines,
                 .runtime_args = {{{0, 0}, {}}},
                 .config = tt::tt_metal::ComputeConfigDescriptor{},
             }},
        .semaphores = {},
        .cbs = {input_cb_descriptor, output_cb_descriptor},
    };

    // Test 1: Program Cache Miss - first run
    log_info(tt::LogTest, "Test 1: Program Cache Miss");
    ttnn::generic_op(std::vector{device_input_tensor_1, device_output_tensor_1}, program_descriptor);
    Tensor golden_1 = ttnn::exp(device_input_tensor_1);
    TT_FATAL(ttnn::allclose<bfloat16>(golden_1.cpu(), device_output_tensor_1.cpu()), "First run correctness failed");
    TT_FATAL(
        this->device_->num_program_cache_entries() == 2,
        "Expected 2 cache entries, got {}",
        this->device_->num_program_cache_entries());

    // Test 2: Program Cache Hit - same tensors
    log_info(tt::LogTest, "Test 2: Program Cache Hit - same tensors");
    ttnn::generic_op(std::vector{device_input_tensor_1, device_output_tensor_1}, program_descriptor);
    TT_FATAL(ttnn::allclose<bfloat16>(golden_1.cpu(), device_output_tensor_1.cpu()), "Second run correctness failed");
    TT_FATAL(
        this->device_->num_program_cache_entries() == 2,
        "Expected 2 cache entries after cache hit, got {}",
        this->device_->num_program_cache_entries());

    // Test 3: Program Cache Hit with different tensors (different addresses)
    log_info(tt::LogTest, "Test 3: Program Cache Hit - different tensor addresses");
    auto dummy_tensor = ttnn::random::uniform(bfloat16(0.0f), bfloat16(0.0f), ttnn::Shape({1, 1, 32, 32}))
                            .to_layout(Layout::TILE)
                            .to_device(this->device_);

    Tensor input_tensor_2 = ttnn::random::random(shape, DataType::BFLOAT16);
    Tensor device_input_tensor_2 = input_tensor_2.to_layout(Layout::TILE).to_device(this->device_);
    Tensor device_output_tensor_2 =
        tt::tt_metal::create_device_tensor(device_input_tensor_2.tensor_spec(), this->device_);

    program_descriptor.kernels[0].runtime_args[0].first = {0, 0};
    program_descriptor.kernels[0].runtime_args[0].second = {device_input_tensor_2.buffer()->address(), num_tiles, 0};
    program_descriptor.kernels[1].runtime_args[0].first = {0, 0};
    program_descriptor.kernels[1].runtime_args[0].second = {device_output_tensor_2.buffer()->address(), num_tiles, 0};

    ttnn::generic_op(std::vector{device_input_tensor_2, device_output_tensor_2}, program_descriptor);
    Tensor golden_2 = ttnn::exp(device_input_tensor_2);
    TT_FATAL(
        ttnn::allclose<bfloat16>(golden_2.cpu(), device_output_tensor_2.cpu()),
        "Third run with different addresses failed - override_runtime_arguments not working correctly!");
    TT_FATAL(
        this->device_->num_program_cache_entries() == 2,
        "Expected 2 cache entries after cache hit with new addresses, got {}",
        this->device_->num_program_cache_entries());
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpSemaphoreDescriptorValidId) {
    // Test that valid semaphore IDs work correctly
    log_info(tt::LogTest, "Running {}", __func__);

    CoreCoord core = {0, 0};
    CoreRange core_range = {core, core};
    CoreRangeSet device_cores = CoreRangeSet(core_range);

    SemaphoreDescriptor sem_descriptor_1 = {
        .id = 0,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = device_cores,
        .initial_value = 0,
    };
    SemaphoreDescriptor sem_descriptor_2 = {
        .id = 1,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = device_cores,
        .initial_value = 1,
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {},
        .semaphores = {sem_descriptor_1, sem_descriptor_2},
        .cbs = {},
    };

    EXPECT_NO_THROW({ tt::tt_metal::Program program(program_descriptor); });
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpSemaphoreDescriptorInvalidIdExceedsMax) {
    // Test that semaphore ID exceeding NUM_SEMAPHORES (16) throws an error
    log_info(tt::LogTest, "Running {}", __func__);

    CoreCoord core = {0, 0};
    CoreRange core_range = {core, core};
    CoreRangeSet device_cores = CoreRangeSet(core_range);

    SemaphoreDescriptor invalid_sem_descriptor = {
        .id = NUM_SEMAPHORES,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = device_cores,
        .initial_value = 0,
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {},
        .semaphores = {invalid_sem_descriptor},
        .cbs = {},
    };

    EXPECT_THROW({ tt::tt_metal::Program program(program_descriptor); }, std::exception);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpSemaphoreDescriptorDuplicateIdOnOverlappingCores) {
    // Test that duplicate semaphore IDs on overlapping cores throw an error
    log_info(tt::LogTest, "Running {}", __func__);

    // Overlap on core (0, 0)
    CoreRange core_range_1 = {CoreCoord(0, 0), CoreCoord(0, 1)};
    CoreRange core_range_2 = {CoreCoord(0, 0), CoreCoord(1, 0)};

    SemaphoreDescriptor sem_descriptor_1 = {
        .id = 0,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(core_range_1),
        .initial_value = 0,
    };
    SemaphoreDescriptor sem_descriptor_2 = {
        .id = 0,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(core_range_2),
        .initial_value = 1,
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {},
        .semaphores = {sem_descriptor_1, sem_descriptor_2},
        .cbs = {},
    };

    EXPECT_THROW({ tt::tt_metal::Program program(program_descriptor); }, std::exception);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpSemaphoreDescriptorSameIdNonOverlappingCores) {
    // Test that same semaphore ID on non-overlapping cores is allowed
    log_info(tt::LogTest, "Running {}", __func__);

    CoreRangeSet cores_0 = CoreRangeSet(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet cores_1 = CoreRangeSet(CoreRange({1, 0}, {1, 0}));

    SemaphoreDescriptor sem_on_core_0 = {
        .id = 0,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = cores_0,
        .initial_value = 0,
    };
    SemaphoreDescriptor sem_on_core_1 = {
        .id = 0,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = cores_1,
        .initial_value = 1,
    };

    ProgramDescriptor program_descriptor = {
        .kernels = {},
        .semaphores = {sem_on_core_0, sem_on_core_1},
        .cbs = {},
    };

    EXPECT_NO_THROW({ tt::tt_metal::Program program(program_descriptor); });
}

TEST_F(MeshDevice2x4Fabric1DFixture, TestGenericOpPointToPoint) {
    log_info(tt::LogTest, "Running {}: sending data from device (0,0) to device (0,1)", __func__);
    auto mesh_device = get_mesh_device();
    auto mesh_shape = mesh_device->shape();
    const size_t num_devices = mesh_shape[0] * mesh_shape[1];  // 2x4 = 8

    auto sender_coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    auto receiver_coord = tt::tt_metal::distributed::MeshCoordinate(0, 1);

    auto full_shape = ttnn::Shape({1, static_cast<uint32_t>(num_devices), 32, 64});
    auto dtype = tt::tt_metal::DataType::BFLOAT16;
    auto layout = tt::tt_metal::Layout::TILE;
    auto memory_config =
        tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM);

    Tensor input_full = ttnn::random::random(full_shape, dtype).to_layout(layout);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device, 1);
    Tensor input_sharded = ttnn::distributed::distribute_tensor(input_full, *mapper);
    Tensor input_tensor = input_sharded.to_device(mesh_device.get(), memory_config);
    tt::tt_metal::distributed::Synchronize(mesh_device.get(), std::nullopt, {});

    // Get sender's data for verification - this is device 0's unique shard
    auto input_shards = ttnn::distributed::get_device_tensors(input_tensor);
    auto sender_data = input_shards[0].cpu();

    // Create output tensor (same spec as input, allocated on mesh)
    Tensor output_tensor = tt::tt_metal::allocate_tensor_on_device(input_tensor.tensor_spec(), mesh_device.get());

    const uint32_t input_num_pages = ttnn::operations::data_movement::get_num_pages(input_tensor);
    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);

    // Figure out packets - single packet
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes = std::bit_floor(fabric_max_packet_size_bytes);
    const uint32_t num_pages_per_packet =
        std::min(max_packet_size_bytes / aligned_input_page_size_bytes, input_num_pages);
    const uint32_t packet_size_bytes = aligned_input_page_size_bytes * num_pages_per_packet;
    const uint32_t num_page_segments = 1;  // pages fit in single packet

    // Create intermediate tensor using the same logic as point_to_point operation
    TensorSpec intermediate_spec = ttnn::operations::point_to_point::p2p_compute_intermediate_tensor_spec(
        input_tensor, receiver_coord, sender_coord, ttnn::ccl::Topology::Linear);
    Tensor intermediate_tensor = tt::tt_metal::allocate_tensor_on_device(intermediate_spec, mesh_device.get());

    // Use single core for simplicity
    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {0, 0};
    CoreRangeSet sender_core_set(std::set<CoreRange>{CoreRange(sender_core)});
    CoreRangeSet receiver_core_set(std::set<CoreRange>{CoreRange(receiver_core)});

    // Determine routing direction (sender -> receiver along dim 1)
    const int dim = 1;  // transmit along row
    const int hops = receiver_coord[dim] - sender_coord[dim];
    const bool is_forward = (hops > 0);
    const uint32_t num_hops = std::abs(hops);

    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device.get(), available_cores, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device.get(), std::nullopt, {});

    // =========================================================================
    // Build MeshProgramDescriptor
    // =========================================================================
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;

    // ----- SENDER PROGRAM -----
    {
        constexpr auto sender_cb_id = tt::CBIndex::c_0;
        constexpr auto packet_header_cb_id = tt::CBIndex::c_1;
        constexpr auto packet_cb_id = tt::CBIndex::c_2;
        constexpr auto cb_num_pages = 2;

        CBFormatDescriptor sender_cb_format = {
            .buffer_index = sender_cb_id,
            .data_format = input_dataformat,
            .page_size = aligned_input_page_size_bytes,
        };
        CBDescriptor sender_cb_desc = {
            .total_size = cb_num_pages * aligned_input_page_size_bytes,
            .core_ranges = sender_core_set,
            .format_descriptors = {sender_cb_format},
        };

        CBFormatDescriptor packet_header_cb_format = {
            .buffer_index = packet_header_cb_id,
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        };
        CBDescriptor packet_header_cb_desc = {
            .total_size = 2 * 2 * packet_header_size_bytes,  // 2 headers * 2 buffering
            .core_ranges = sender_core_set,
            .format_descriptors = {packet_header_cb_format},
        };

        CBFormatDescriptor packet_cb_format = {
            .buffer_index = packet_cb_id,
            .data_format = input_dataformat,
            .page_size = packet_size_bytes,
        };
        CBDescriptor packet_cb_desc = {
            .total_size = packet_size_bytes,
            .core_ranges = sender_core_set,
            .format_descriptors = {packet_cb_format},
        };

        std::vector<uint32_t> reader_ct_args;
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct_args);

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),
            input_num_pages,
            0,  // page_idx_start
            input_page_size_bytes,
        };

        std::vector<uint32_t> writer_ct_args = {
            sender_cb_id,
            packet_header_cb_id,
            packet_cb_id,
            l1_alignment,
        };
        tt::tt_metal::TensorAccessorArgs(*intermediate_tensor.buffer()).append_to(writer_ct_args);

        std::vector<uint32_t> writer_rt_args = {
            intermediate_tensor.buffer()->address(),
            0,                // page_idx_start
            input_num_pages,  // page_idx_end
            num_hops,
            input_page_size_bytes,
            packet_size_bytes,
            num_pages_per_packet,
            num_page_segments,
            semaphore.address(),
            (uint32_t)is_forward,
        };

        KernelDescriptor reader_kernel = {
            .kernel_source =
                "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/"
                "reader_unary_interleaved_start_id_gen.cpp",
            .core_ranges = sender_core_set,
            .compile_time_args = reader_ct_args,
            .runtime_args = {{sender_core, reader_rt_args}},
            .config = tt::tt_metal::ReaderConfigDescriptor{},
        };

        KernelDescriptor writer_kernel = {
            .kernel_source = "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
            .core_ranges = sender_core_set,
            .compile_time_args = writer_ct_args,
            .runtime_args = {{sender_core, writer_rt_args}},
            .config = tt::tt_metal::WriterConfigDescriptor{},
        };

        ProgramDescriptor sender_program = {
            .kernels = {reader_kernel, writer_kernel},
            .cbs = {sender_cb_desc, packet_header_cb_desc, packet_cb_desc},
        };

        mesh_program_descriptor.mesh_programs.emplace(ttnn::MeshCoordinateRange(sender_coord), sender_program);

        mesh_program_descriptor.fabric_connections.push_back(tt::tt_metal::experimental::FabricConnectionDescriptor{
            .src_coord = sender_coord,
            .dst_coord = receiver_coord,
            .kernel_index = 1,  // writer kernel
            .worker_core = sender_core,
            .link_idx = 0,
            .core_type = tt::CoreType::WORKER,
        });
    }

    // ----- RECEIVER PROGRAM -----
    {
        constexpr auto packet_header_cb_id = tt::CBIndex::c_0;
        constexpr auto packet_cb_id = tt::CBIndex::c_1;
        constexpr auto receiver_cb_id = tt::CBIndex::c_2;

        CBFormatDescriptor packet_header_cb_format = {
            .buffer_index = packet_header_cb_id,
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        };
        CBDescriptor packet_header_cb_desc = {
            .total_size = 2 * 2 * packet_header_size_bytes,
            .core_ranges = receiver_core_set,
            .format_descriptors = {packet_header_cb_format},
        };

        CBFormatDescriptor packet_cb_format = {
            .buffer_index = packet_cb_id,
            .data_format = input_dataformat,
            .page_size = packet_size_bytes,
        };
        CBDescriptor packet_cb_desc = {
            .total_size = packet_size_bytes,
            .core_ranges = receiver_core_set,
            .format_descriptors = {packet_cb_format},
        };

        const uint32_t receiver_cb_num_pages = 3 * num_pages_per_packet;
        CBFormatDescriptor receiver_cb_format = {
            .buffer_index = receiver_cb_id,
            .data_format = input_dataformat,
            .page_size = input_page_size_bytes,
        };
        CBDescriptor receiver_cb_desc = {
            .total_size = receiver_cb_num_pages * input_page_size_bytes,
            .core_ranges = receiver_core_set,
            .format_descriptors = {receiver_cb_format},
        };

        std::vector<uint32_t> reader_ct_args = {
            (uint32_t)packet_header_cb_id,
            (uint32_t)packet_cb_id,
            (uint32_t)receiver_cb_id,
            l1_alignment,
        };
        tt::tt_metal::TensorAccessorArgs(*intermediate_tensor.buffer()).append_to(reader_ct_args);

        const bool sender_is_forward = is_forward;
        std::vector<uint32_t> reader_rt_args = {
            0,                // page_idx_start
            input_num_pages,  // page_idx_end
            num_pages_per_packet,
            intermediate_tensor.buffer()->address(),
            packet_size_bytes,
            input_page_size_bytes,
            num_page_segments,
            semaphore.address(),
            num_hops,
            (uint32_t)sender_is_forward,
        };

        std::vector<uint32_t> writer_ct_args = {(uint32_t)receiver_cb_id};
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),
            input_num_pages,
            0,  // page_idx_start
            input_page_size_bytes,
        };

        KernelDescriptor reader_kernel = {
            .kernel_source = "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive.cpp",
            .core_ranges = receiver_core_set,
            .compile_time_args = reader_ct_args,
            .runtime_args = {{receiver_core, reader_rt_args}},
            .config = tt::tt_metal::ReaderConfigDescriptor{},
        };

        KernelDescriptor writer_kernel = {
            .kernel_source =
                "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id_gen.cpp",
            .core_ranges = receiver_core_set,
            .compile_time_args = writer_ct_args,
            .runtime_args = {{receiver_core, writer_rt_args}},
            .config = tt::tt_metal::WriterConfigDescriptor{},
        };

        ProgramDescriptor receiver_program = {
            .kernels = {reader_kernel, writer_kernel},
            .cbs = {packet_header_cb_desc, packet_cb_desc, receiver_cb_desc},
        };

        mesh_program_descriptor.mesh_programs.emplace(ttnn::MeshCoordinateRange(receiver_coord), receiver_program);

        mesh_program_descriptor.fabric_connections.push_back(tt::tt_metal::experimental::FabricConnectionDescriptor{
            .src_coord = receiver_coord,
            .dst_coord = sender_coord,
            .kernel_index = 0,  // reader kernel
            .worker_core = receiver_core,
            .link_idx = 0,
            .core_type = tt::CoreType::WORKER,
        });
    }

    ttnn::generic_op(std::vector<Tensor>{input_tensor, intermediate_tensor, output_tensor}, mesh_program_descriptor);
    tt::tt_metal::distributed::Synchronize(mesh_device.get(), std::nullopt, {});

    size_t output_receiver_idx = 1;
    auto output_data = ttnn::distributed::get_device_tensors(output_tensor);
    auto output_tensor_cpu = output_data[output_receiver_idx].cpu();

    log_info(tt::LogTest, "Sender data: {}", ttnn::to_string(sender_data));
    log_info(tt::LogTest, "Output at receiver: {}", ttnn::to_string(output_tensor_cpu));

    EXPECT_EQ(output_tensor_cpu.to_vector<bfloat16>(), sender_data.to_vector<bfloat16>())
        << "Receiver should have sender's data after transfer";
}

}  // namespace ttnn::operations::generic::test
