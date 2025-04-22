// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "base_types.hpp"
#include <tt_metal/api/tt-metalium/core_coord.hpp>
#include <tt_metal/api/tt-metalium/work_split.hpp>
#include <tt_metal/api/tt-metalium/host_api.hpp>
#include <tt_metal/api/tt-metalium/assert.hpp>
#include <tt-metalium/constants.hpp>

#include "logger.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/generic/generic_op/generic_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

namespace ttnn::operations::generic::test {
TEST_F(TTNNFixtureWithDevice, TestGenericOpArgmax) {
    uint32_t batch = 1;
    uint32_t channels = 4;
    ttnn::Shape shape{batch, channels, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor input_tensor = ttnn::random::random(shape, DataType::BFLOAT16);
    Tensor device_input_tensor = input_tensor.to_device(this->device_);
    Tensor golden = ttnn::argmax(device_input_tensor).cpu();

    Tensor device_output_tensor = tt::tt_metal::create_device_tensor(golden.get_tensor_spec(), this->device_);

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    const uint32_t unit_size = input_tensor.element_size();

    const CoreCoord core_coord(0, 0);
    const CoreRangeSet core = std::set<CoreRange>({core_coord, core_coord});

    const auto& input_shape = input_tensor.get_padded_shape();
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
    ttnn::operations::generic::circular_buffer_attributes_t input_cb_attributes = {
        .core_spec = core,
        .total_size = src_page_size,
        .page_size = src_page_size,
        .data_format = cb_data_format,
    };
    ttnn::operations::generic::circular_buffer_attributes_t output_cb_attributes = {
        .core_spec = core,
        .total_size = dst_page_size,
        .page_size = dst_page_size,
        .data_format = cb_data_format,
    };

    const auto src_buffer = device_input_tensor.buffer();
    const auto dst_buffer = device_output_tensor.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input_tensor.get_logical_volume() / inner_dim_units / red_dim_units;

    const std::vector<uint32_t> compile_time_args = {
        (uint32_t)src_cb_idx,
        (uint32_t)dst_cb_idx,
        src_is_dram,
        dst_is_dram,
        src_page_size,
        dst_page_size,
        outer_dim_units,
        inner_dim_units,
        red_dim_units,
        (uint32_t)(reduce_all),
    };
    const std::vector<uint32_t> runtime_args = {
        src_buffer->address(),
        dst_buffer->address(),
    };
    ttnn::operations::generic::data_movement_attributes_t data_movement_attributes = {
        .core_spec = core,
        .kernel_path = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp",
        .config = tt::tt_metal::ReaderDataMovementConfig(compile_time_args),
        .runtime_args_per_core = {
            {core_coord, runtime_args},
        }};

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes =
            {
                {src_cb_idx, input_cb_attributes},
                {dst_cb_idx, output_cb_attributes},
            },
        .data_movement_attributes = {data_movement_attributes},
    };

    ttnn::generic_op(std::vector<Tensor>{device_input_tensor, device_output_tensor}, program_attributes);
    Tensor output_tensor = device_output_tensor.cpu();
    auto dtype = golden.get_dtype();
    auto allclose = ttnn::allclose<uint32_t>(golden, output_tensor);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpUnaryReluSharded) {
    const std::map<std::string, std::string> defines_relu = {
        {"SFPU_OP_CHAIN_0", "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"},
        {"SFPU_OP_CHAIN_0_FUNC_0", "relu_tile(0);"},
        {"SFPU_OP_CHAIN_0_INIT_0", "relu_tile_init();"},
        {"SFPU_OP_RELU_FAMILY_INCLUDE", "1"}};
    auto shape = ttnn::Shape{64, 16, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    CoreCoord compute_with_storage_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange all_cores_range = {
        CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

    ttnn::MemoryConfig mem_config = ttnn::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .shard_spec = tt::tt_metal::ShardSpec(
            all_cores,
            {16 * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto input_tensor = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), shape, Layout::TILE);
    auto device_input_tensor = input_tensor.to_device(this->device_, mem_config);
    auto device_output_tensor =
        tt::tt_metal::create_device_tensor(device_input_tensor.get_tensor_spec(), this->device_);

    auto shard_spec = device_input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.grid == all_cores, "shard spec grid should be same as all_cores");

    tt::log_info(tt::LogTest, "Running ttnn unary relu sharded");
    auto golden = ttnn::relu(device_input_tensor).cpu();

    tt::log_info(tt::LogTest, "Running generic_op unary relu sharded");
    auto act_df = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor.get_dtype());
    auto out_df = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.get_dtype());
    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);
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
    ttnn::operations::generic::circular_buffer_attributes_t input_cb_attributes = {
        .core_spec = all_cores,
        .total_size = in_cb_npages * in_cb_pagesize,
        .page_size = in_cb_pagesize,
        .data_format = act_df,
        .set_globally_allocated_address = 0,
    };

    ttnn::operations::generic::circular_buffer_attributes_t output_cb_attributes = {
        .core_spec = all_cores,
        .total_size = in_cb_npages * in_cb_pagesize,
        .page_size = in_cb_pagesize,
        .data_format = out_df,
        .set_globally_allocated_address = 1,
    };

    const std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)in_cb_id};
    ttnn::operations::generic::data_movement_attributes_t reader_attributes = {
        .core_spec = all_cores,
        .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args),
    };

    ttnn::operations::generic::compute_attributes_t compute_attributes = {
        .core_spec = all_cores,
        .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        .config =
            {
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = {1, num_tile_per_core},
                .defines = defines_relu,
            },
    };

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes = {{in_cb_id, input_cb_attributes}, {out_cb_id, output_cb_attributes}},
        .data_movement_attributes = {{reader_attributes}},
        .compute_attributes = {compute_attributes},
    };

    // calculate data movement runtime arguments: every core has the same runtime args
    for (uint32_t i = 0; i < compute_with_storage_grid_size.x * compute_with_storage_grid_size.y; i++) {
        CoreCoord core = {i / compute_with_storage_grid_size.y, i % compute_with_storage_grid_size.y};
        program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {num_tile_per_core};
    }

    ttnn::generic_op(std::vector{device_input_tensor, device_output_tensor}, program_attributes);
    auto device_output = device_output_tensor.cpu();
    auto allclose = ttnn::allclose<bfloat16>(golden, device_output, 1e-1f, 1e-5f);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpBinaryEltwiseAdd) {
    const std::map<std::string, std::string> defines_eltwise_add = {
        {"ELTWISE_OP", "add_tiles"},
        {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"},
    };

    tt::log_info(tt::LogTest, "Running ttnn binary add interleaved");
    ttnn::Shape shape{11, 9, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);
    auto device_input_tensor_a = input_tensor_a.to_layout(Layout::TILE).to_device(this->device_);
    auto device_input_tensor_b = input_tensor_b.to_layout(Layout::TILE).to_device(this->device_);

    auto golden = ttnn::add(device_input_tensor_a, device_input_tensor_b).cpu().to_layout(Layout::ROW_MAJOR);

    tt::log_info(tt::LogTest, "Running generic add interleaved");

    // Data movement kernel needs output tensor address to be passed as a runtime argument.
    auto device_output_tensor =
        tt::tt_metal::create_device_tensor(device_input_tensor_a.get_tensor_spec(), this->device_);

    auto compute_with_storage_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange all_cores_range = {
        CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

    auto input_a_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_a.get_dtype());
    auto input_b_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_input_tensor_b.get_dtype());
    auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(device_output_tensor.get_dtype());

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    tt::CBIndex dst_cb_index = tt::CBIndex::c_2;

    bool block_or_width_sharded = false;
    bool src0_is_dram = device_input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = device_input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)block_or_width_sharded};

    bool dst_is_dram = device_output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)dst_cb_index, (std::uint32_t)dst_is_dram};

    ttnn::operations::generic::circular_buffer_attributes_t input_cb0_atrributes = {
        .core_spec = all_cores,
        .total_size = 2 * tt::tt_metal::detail::TileSize(input_a_cb_data_format),
        .page_size = tt::tt_metal::detail::TileSize(input_a_cb_data_format),
        .data_format = input_a_cb_data_format,
    };
    ttnn::operations::generic::circular_buffer_attributes_t input_cb1_attributes = {
        .core_spec = all_cores,
        .total_size = 2 * tt::tt_metal::detail::TileSize(input_b_cb_data_format),
        .page_size = tt::tt_metal::detail::TileSize(input_b_cb_data_format),
        .data_format = input_b_cb_data_format,
    };
    ttnn::operations::generic::circular_buffer_attributes_t output_cb_attributes = {
        .core_spec = all_cores,
        .total_size = 2 * tt::tt_metal::detail::TileSize(output_cb_data_format),
        .page_size = tt::tt_metal::detail::TileSize(output_cb_data_format),
        .data_format = output_cb_data_format,
    };
    ttnn::operations::generic::data_movement_attributes_t reader_attributes = {
        .core_spec = all_cores,
        .kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args),
    };
    ttnn::operations::generic::data_movement_attributes_t writer_attributes = {
        .core_spec = all_cores,
        .kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args),
    };
    ttnn::operations::generic::compute_attributes_t compute_attributes = {
        .core_spec = all_cores,
        .kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
        .config =
            {
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                // .preserve_fp32_precision = false,
                .math_approx_mode = false,
                .compile_args = {},
                .defines = defines_eltwise_add,
            },
    };

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes =
            {
                {src0_cb_index, input_cb0_atrributes},
                {src1_cb_index, input_cb1_attributes},
                {dst_cb_index, output_cb_attributes},
            },
        .data_movement_attributes = {reader_attributes, writer_attributes},
        .compute_attributes = {compute_attributes},
    };

    // setup runtime arguments for data movement kernels
    uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t num_tiles = device_input_tensor_a.volume() / tt::constants::TILE_HW;
    bool row_major = true;

    auto [num_cores, _, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major);

    uint32_t block_size_per_core_group_1 = 1;
    uint32_t block_size_per_core_group_2 = 1;
    uint32_t max_block_size = 1;
    uint32_t block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
    uint32_t block_cnt_per_core_group_2 = num_tiles_per_core_group_2;

    auto cores =
        grid_to_cores(num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

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

        program_attributes.data_movement_attributes[0].runtime_args_per_core[core] = {
            device_input_tensor_a.buffer()->address(),
            device_input_tensor_b.buffer()->address(),
            num_tiles_per_core,
            num_tiles_read,  // start_id
            0,               // block_height = 0 when not sharded
            0,               // block_width = 0 when not sharded
            0,               // num_cores_y = 0 when not sharded
        };
        program_attributes.data_movement_attributes[1].runtime_args_per_core[core] = {
            device_output_tensor.buffer()->address(),
            num_tiles_per_core,
            num_tiles_read  // start_id
        };
        program_attributes.compute_attributes[0].runtime_args_per_core[core] = {
            block_cnt_per_core, block_size_per_core};

        num_tiles_read += num_tiles_per_core;
    }

    ttnn::generic_op(
        std::vector<Tensor>{device_input_tensor_a, device_input_tensor_b, device_output_tensor}, program_attributes);

    auto device_output = device_output_tensor.cpu().to_layout(Layout::ROW_MAJOR);
    auto allclose = ttnn::allclose<bfloat16>(golden, device_output);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpMatmul) {
    // =================
    // Matmul original and generic test
    tt::log_info(tt::LogTest, "Running ttnn matmul");
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

    Tensor golden = ttnn::matmul(input_tensor_a, input_tensor_b);

    tt::log_info(tt::LogTest, "Running matmul generic test");

    // Parameters for matmul call - copy paste from matmul_multi_core in bmm_op_multi_core.cpp
    bool bcast_batch = false;

    ttnn::Shape output_shape =
        ttnn::Shape{B_original, 1, Mt_original * tt::constants::TILE_HEIGHT, Nt_original * tt::constants::TILE_WIDTH};
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

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    const auto &ashape = input_tensor_a.get_logical_shape(), bshape = input_tensor_b.get_logical_shape();
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

    TT_FATAL(
        !core_group_2.ranges().empty(),
        "Core group 2 for matmul generic test is empty. We should never hit this case.");

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

    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestGenericOpEltwiseSFPU) {
    const std::map<std::string, std::string> sfpu_defines = {
        {"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}};

    uint32_t num_tiles = 4;
    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    auto shape = ttnn::Shape{1, num_tiles, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    Tensor input_tensor = ttnn::random::random(shape, DataType::BFLOAT16);
    ttnn::MemoryConfig dram_memory_config = ttnn::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};

    Tensor device_input_tensor = input_tensor.to_layout(Layout::TILE).to_device(this->device_, dram_memory_config);
    Tensor device_output_tensor = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(
            device_input_tensor.get_logical_shape(),
            ttnn::TensorLayout(
                device_input_tensor.get_dtype(),
                ttnn::PageConfig(device_input_tensor.get_layout()),
                device_input_tensor.memory_config())),
        device_input_tensor.device());

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
    const std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_16, (std::uint32_t)is_dram_input};
    const std::vector<uint32_t> read_rt_args = {device_input_tensor.buffer()->address(), num_tiles, src_bank_id};
    const std::vector<uint32_t> write_rt_args = {device_output_tensor.buffer()->address(), num_tiles, dst_bank_id};

    ttnn::operations::generic::data_movement_attributes_t reader_attributes = {
        .core_spec = device_cores,
        .kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        .config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args),
        .runtime_args_per_core = {{core, read_rt_args}},
    };

    ttnn::operations::generic::data_movement_attributes_t writer_attributes = {
        .core_spec = device_cores,
        .kernel_path =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args),
        .runtime_args_per_core = {{core, write_rt_args}},
    };

    ttnn::operations::generic::compute_attributes_t compute_attributes = {
        .core_spec = device_cores,
        .kernel_path = "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        .config =
            {
                .math_approx_mode = false,
                .compile_args = {num_tiles, 1},
                .defines = sfpu_defines,
            },
    };

    ttnn::operations::generic::program_attributes_t program_attributes = {
        .circular_buffer_attributes =
            {
                {tt::CBIndex::c_0, input_cb_attributes},
                {tt::CBIndex::c_16, output_cb_attributes},
            },
        .data_movement_attributes = {reader_attributes, writer_attributes},
        .compute_attributes = {compute_attributes},
    };

    Tensor device_output = ttnn::generic_op(std::vector{device_input_tensor, device_output_tensor}, program_attributes);
    Tensor golden = ttnn::exp(device_input_tensor);

    auto allclose = ttnn::allclose<bfloat16>(golden.cpu(), device_output.cpu());

    ASSERT_TRUE(allclose);
}

}  // namespace ttnn::operations::generic::test
