// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tt_stl/reflection.hpp"

#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/auto_format.hpp"

#include "tensor/tensor_utils.hpp"

using namespace tt::constants;
namespace tt {

namespace tt_metal {

const uint32_t act_cb                                 = CB::c_in0;
const uint32_t weight_cb                              = CB::c_in1;
const uint32_t bias_cb                                = CB::c_in2;
const uint32_t sharded_act_cb                         = CB::c_in3;
const uint32_t matmul_partials_cb                     = CB::c_intermed0;
const uint32_t tilize_mode_tilized_act_cb             = CB::c_intermed1;
const uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
const uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
const uint32_t cb_for_reader_indices                  = CB::c_intermed4;
const uint32_t out0_cb                                = CB::c_out0;

pair<uint32_t, uint32_t> compute_opt_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t padding_for_32B_alignment=0) {
    uint32_t conv_output_h = ((conv_activation_h - filter_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_w = ((conv_activation_w - filter_w + (2 * pad_w) - padding_for_32B_alignment) / stride_w) + 1;
    return {conv_output_h, conv_output_w};
}
pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(Shape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t padding_for_32B_alignment) {
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = compute_opt_conv_output_face_shape(conv_activation_shape[1], conv_activation_shape[2], filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, padding_for_32B_alignment);
    uint32_t batch_size = conv_activation_shape[0];
    // pad height
    uint32_t num_rows = (uint32_t) batch_size * conv_output_h * conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t num_cols_padded = conv_activation_shape[3] * filter_w * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}


CircularBufferID create_CBs(tt_metal::Program &program,
                                const Tensor& input,
                                CoreRange core,
                                uint32_t num_cb0_tiles,
                                uint32_t num_cb1_tiles,
                                uint32_t num_cb0_tilized_tiles,
                                uint32_t num_output_tiles,
                                uint32_t num_reblock_cb_tiles,
                                uint32_t num_writer_output_tiles,
                                uint32_t num_bytes_for_df,
                                bool untilize_out,
                                uint32_t bias_ntiles = 0,
                                bool with_bias = false,
                                std::optional<uint32_t> output_cb_address = std::nullopt
) {

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    // Invariants
    CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * single_tile_size, {{act_cb, tt::DataFormat::Float16_b}})
		.set_page_size(act_cb, single_tile_size);
    auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);

    if (input.memory_config().is_sharded()) {
        auto shard_shape = input.shard_spec().value().shard_shape;
        CircularBufferConfig cb_sharded_act_config = CircularBufferConfig(shard_shape[0] * shard_shape[1] * num_bytes_for_df, {{sharded_act_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(sharded_act_cb, shard_shape[1] * num_bytes_for_df);
        // incoming data is the input cb instead of raw l1/dram addr
        cb_sharded_act_config.set_globally_allocated_address(input.buffer()->address());
        auto cb_sharded_act = tt_metal::CreateCircularBuffer(program, core, cb_sharded_act_config);
    }

    CircularBufferConfig cb_weight_config = CircularBufferConfig(num_cb1_tiles * single_tile_size, {{weight_cb, tt::DataFormat::Float16_b}})
		.set_page_size(weight_cb, single_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, core, cb_weight_config);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config = CircularBufferConfig(num_cb0_tilized_tiles * single_tile_size, {{tilize_mode_tilized_act_cb, tt::DataFormat::Float16_b}})
		.set_page_size(tilize_mode_tilized_act_cb, single_tile_size);
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

    CircularBufferID cb_output;
    if (untilize_out) {
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        CircularBufferConfig cb_final_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        CircularBufferConfig cb_reblock_config = CircularBufferConfig(num_reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        CircularBufferConfig cb_output_config = CircularBufferConfig(num_writer_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(out0_cb, single_tile_size);
        if (output_cb_address.has_value()) {
            cb_output_config = cb_output_config.set_globally_allocated_address(output_cb_address.value());
        }
        cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    } else {
        CoreRangeSet cores(std::set<CoreRange>({core}));
        std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
            {out0_cb, tt::DataFormat::Float16_b},
            {matmul_partials_cb, tt::DataFormat::Float16_b}
        };
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, cb_output_data_format_spec)
		    .set_page_size(out0_cb, single_tile_size)
            .set_page_size(matmul_partials_cb, single_tile_size);
        if (output_cb_address.has_value()) {
            cb_matmul_partials_config = cb_matmul_partials_config.set_globally_allocated_address(output_cb_address.value());
        }
        cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_matmul_partials_config);
    }

    if (with_bias) {
        // bias input
        uint32_t bias_pagesize = single_tile_size;
        CircularBufferConfig cb_bias_config = CircularBufferConfig(bias_ntiles * bias_pagesize, {{bias_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(bias_cb, bias_pagesize);
        auto cb_bias = tt_metal::CreateCircularBuffer(program, core, cb_bias_config);

        log_debug("BIAS CBs: {} {} {}", bias_cb, bias_ntiles, bias_pagesize);
    }

    return cb_output;
}

operation::ProgramWithCallbacks optimized_conv_(const Tensor& a, const Tensor &b, const Shape& ashape, std::optional<const Tensor> bias, vector<int> conv_params, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity,
                                       const OptimizedConvParallelizationConfig& parallelization_config,
                                       const OptimizedConvBlockConfig& block_config,
                                       uint32_t extra_padding_for_32B_alignment, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_ASSERT(output_channels <= b.shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = block_config.weight_block_w_ntiles;
    uint32_t out_block_h_ntiles = block_config.out_block_h_ntiles;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;
    //assert(out_block_h_ntiles == act_block_h_ntiles); // TODO: fix output block sizing
    TT_ASSERT(out_block_h_ntiles >= act_block_h_ntiles, "Output block height (in # of tiles) should be greater than or equal to activation block height (in # of tiles)");

    uint32_t conv_act_size_h = ashape[1];
    uint32_t conv_act_size_w = ashape[2];
    uint32_t conv_act_size_c = ashape[3];
    uint32_t weight_size_h = (uint32_t) conv_params[0];
    uint32_t weight_size_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];

    bool rn50_first_conv = (conv_act_size_h == 230 && conv_act_size_w == (231 + extra_padding_for_32B_alignment) &&
                        weight_size_h == 7 && weight_size_w == 8 &&
                        stride_h == 2 && stride_w == 2);
    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] = compute_opt_conv_activation_as_mm_shape(ashape, conv_params, out_block_h_ntiles, extra_padding_for_32B_alignment);
    assert(act_matrix_shape.size() == 3);
    assert(act_matrix_shape[0] == 1);
    uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t) act_matrix_shape[2];
    uint32_t act_matrix_height_unpadded = (uint32_t) act_matrix_shape_unpadded[1];
    uint32_t act_matrix_width_unpadded = (uint32_t) act_matrix_shape_unpadded[2];

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_ASSERT(b.layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_ASSERT(b.shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_ASSERT(b.shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.shape()[2];
    uint32_t weight_matrix_width = b.shape()[3];

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_ASSERT(bias.has_value());
        TT_ASSERT(bias.value().buffer() != nullptr);
        auto bias_shape_without_padding = bias.value().shape().without_padding();
        TT_ASSERT(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
        TT_ASSERT(bias_shape_without_padding[1] == 1 && bias_shape_without_padding[2] == 1, "Bias should have H == W == 1");
        TT_ASSERT(bias_shape_without_padding[3] == output_channels, "Bias should have output_channels");
    }

    // Normal matrix shape check
    TT_ASSERT(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(act_matrix_height % TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_ASSERT(act_matrix_width % TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_height % TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_width % TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(a.storage_type() == StorageType::DEVICE &&
              b.storage_type() == StorageType::DEVICE &&
              "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_ASSERT(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_ASSERT(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    // Convert tensor dims to tile dims
    uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
    uint32_t act_matrix_width_ntiles = act_matrix_width / TILE_WIDTH;
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    assert(act_matrix_height_ntiles % act_block_h_ntiles == 0);
    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(weight_matrix_width_ntiles % weight_block_w_ntiles == 0);
    assert(act_matrix_height_ntiles % out_block_h_ntiles == 0);

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    if (rn50_first_conv) {
        assert(num_blocks_weight_w == 1);
    }

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;
    TT_ASSERT((act_block_w_datums == conv_act_size_c * weight_size_w) || ((act_block_w_datums <= conv_act_size_c) && (conv_act_size_c % act_block_w_datums == 0)));


    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    assert(weight_block_w_ntiles % out_subblock_w_ntiles == 0);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
    assert(output_channels_padded_to_tile_width <= weight_matrix_width);
    uint32_t output_width_num_tiles = output_channels_padded_to_tile_width / TILE_WIDTH;
    uint32_t num_blocks_output_w = (uint32_t) ceil((double) output_channels_padded_to_tile_width / (double) weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0) ? weight_block_w_datums : (output_channels_padded_to_tile_width % weight_block_w_datums);
    assert(last_block_width_datums % TILE_WIDTH == 0);
    uint32_t output_row_size_bytes = output_channels_padded_to_tile_width * num_bytes_of_df;
    uint32_t last_block_row_size_bytes = last_block_width_datums * num_bytes_of_df;
    // sanity check
    assert(num_blocks_output_w == num_blocks_weight_w);
    tt_metal::Program program = tt_metal::Program();
    //CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    //CoreRange core = {.start={0, 0}, .end={0, 0}};
    //tt_start_debug_print_server();

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_row_size = weight_matrix_width * num_bytes_of_df;
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(act_block_h_ntiles % out_subblock_h_ntiles == 0);
    assert(out_block_h_ntiles % out_subblock_h_ntiles == 0);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    uint32_t weight_dram_addr = src1_dram_buffer->address();
    auto weight_dram_noc_xy = src1_dram_buffer->noc_coordinates();
    uint32_t weight_noc_x = weight_dram_noc_xy.x;
    uint32_t weight_noc_y = weight_dram_noc_xy.y;

    // bias
    Buffer *bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0, bias_tile_nbytes = 0, bias_log2_of_pagesize = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles = bias.value().shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
        bias_tile_nbytes = single_tile_size;
        bias_log2_of_pagesize = (uint32_t) std::log2((float) bias_tile_nbytes);
    }

    //uint32_t conv_output_size_h = ((conv_act_size_h - weight_size_h + (2 * pad_h)) / stride_h) + 1;
    //uint32_t conv_output_size_w = ((conv_act_size_w - weight_size_w + (2 * pad_w)) / stride_w) + 1;

    auto [conv_output_size_h, conv_output_size_w] = compute_opt_conv_output_face_shape(conv_act_size_h, conv_act_size_w, weight_size_h, weight_size_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    std::map<string, string> reader_defines;

    if (act_matrix_height_unpadded < act_block_h_datums * num_blocks_act_h) {
        reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
    }

    uint32_t output_height_padded_to_tile_height = round_up(act_matrix_height_unpadded, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    assert(output_height_num_tiles <= act_matrix_height_ntiles);

    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes = act_block_h_ntiles * act_block_w_ntiles * single_tile_size;
    uint32_t dst_l1_weight_buffer_size_bytes = weight_block_h_ntiles * weight_block_w_ntiles * single_tile_size;

    // more args for writer
    uint32_t out_block_row_size_bytes = weight_block_w_ntiles*TILE_WIDTH*num_bytes_of_df;
    uint32_t out_row_size_bytes = output_channels_padded_to_tile_width*num_bytes_of_df;

    // output data format
    const auto out_df = datatype_to_dataformat_converter(a.dtype());

    // For debug
    {
        log_debug(tt::LogOp, "conv_act_size_c: {}", conv_act_size_c);
        log_debug(tt::LogOp, "conv_act_size_h: {}", conv_act_size_h);
        log_debug(tt::LogOp, "conv_act_size_w: {}", conv_act_size_w);
        log_debug(tt::LogOp, "act_matrix_height: {}", act_matrix_height);
        log_debug(tt::LogOp, "act_matrix_width: {}", act_matrix_width);
        log_debug(tt::LogOp, "act_matrix_height_unpadded: {}", act_matrix_height_unpadded);
        log_debug(tt::LogOp, "act_matrix_width_unpadded: {}", act_matrix_width_unpadded);
        log_debug(tt::LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(tt::LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(tt::LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(tt::LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(tt::LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(tt::LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(tt::LogOp, "num_blocks_out_h: {}", num_blocks_out_h);
        log_debug(tt::LogOp, "act_dram_addr: {}", act_dram_addr);
        log_debug(tt::LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(tt::LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(tt::LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(tt::LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(tt::LogOp, "out_block_h_ntiles: {}", out_block_h_ntiles);
        log_debug(tt::LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(tt::LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(tt::LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(tt::LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(tt::LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(tt::LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(tt::LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(tt::LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(tt::LogOp, "has_bias: {}", has_bias);
        log_debug(tt::LogOp, "bias_dram_addr: {}", bias_dram_addr);
        log_debug(tt::LogOp, "bias_ntiles: {}", bias_ntiles);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_groups: {}", num_groups);
    }
    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = num_cores_x * num_cores_y;
    assert(num_cores_x < 13);
    assert(num_cores_y < 10);
    uint32_t per_core_out_matrix_height_ntiles = p_config.per_core_out_matrix_height_ntiles;
    uint32_t per_core_weight_matrix_width_ntiles = p_config.per_core_weight_matrix_width_ntiles;
    //cout << "per_core_weight_matrix_width_ntiles=" << per_core_weight_matrix_width_ntiles << endl;
    // cout << "total_num_cores=" << total_num_cores << endl;
    // cout << "per_core_out_matrix_height_ntiles=" << per_core_out_matrix_height_ntiles << endl;
    // cout << "act_matrix_height_ntiles=" << act_matrix_height_ntiles << endl;
    // cout << "act_block_h_datums=" << act_block_h_datums << endl;
    // cout << "num_blocks_act_h=" << num_blocks_act_h << endl;
    bool weight_width_sliced = per_core_weight_matrix_width_ntiles < weight_matrix_width_ntiles;
    assert(weight_matrix_width_ntiles % per_core_weight_matrix_width_ntiles == 0);
    assert(per_core_weight_matrix_width_ntiles % weight_block_w_ntiles == 0);
    uint32_t num_blocks_weight_w_per_core = per_core_weight_matrix_width_ntiles / weight_block_w_ntiles;
    if (not weight_width_sliced) {
        assert(num_blocks_weight_w_per_core == num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_weight_matrix_width_ntiles;
    assert(num_cores_y % num_weight_slices_width == 0);
    uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
    uint32_t total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
    if (weight_width_sliced) {
        assert(total_num_cores_per_weight_slice * per_core_out_matrix_height_ntiles == act_matrix_height_ntiles);
    }
    else {
        assert(total_num_cores * per_core_out_matrix_height_ntiles >= act_matrix_height_ntiles);
    }
    assert(per_core_out_matrix_height_ntiles % act_block_h_ntiles == 0);
    uint32_t num_blocks_act_h_per_core = per_core_out_matrix_height_ntiles / act_block_h_ntiles;
    assert(per_core_out_matrix_height_ntiles % out_block_h_ntiles == 0);
    uint32_t num_blocks_out_h_per_core = per_core_out_matrix_height_ntiles / out_block_h_ntiles;
    bool act_height_sliced = per_core_out_matrix_height_ntiles < act_matrix_height_ntiles;
    if (not act_height_sliced) {
        assert(num_blocks_act_h_per_core == num_blocks_act_h);
        assert(num_blocks_out_h_per_core == num_blocks_out_h);
        assert(num_cores_x == 1);
    }
    // cout << "num_blocks_act_h_per_core=" << num_blocks_act_h_per_core << endl;
    assert(act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0);
    uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / per_core_out_matrix_height_ntiles;
    assert(total_active_num_cores_per_weight_slice <= total_num_cores_per_weight_slice);
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - total_active_num_cores_per_weight_slice;
    uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
    if (weight_width_sliced) {
        assert(total_noop_cores == 0);
        assert(total_active_num_cores == total_num_cores);
    }
    // cout << "act_matrix_height_ntiles=" << act_matrix_height_ntiles << endl;
    // cout << "per_core_out_matrix_height_ntiles=" << per_core_out_matrix_height_ntiles << endl;
    // cout << "total_active_num_cores_per_weight_slice="<< total_active_num_cores_per_weight_slice <<  endl;
    // cout << "num weight slices = " <<  num_weight_slices_width << endl;
    // cout << "total num active cores" << total_active_num_cores << endl;
    if (has_bias) {
    assert(bias_ntiles % num_weight_slices_width == 0);
    assert(bias_ntiles == weight_matrix_width_ntiles);
    }
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    bool act_block_w_equals_input_channels_x_filter_width = (act_block_w_datums == (conv_act_size_c * weight_size_w));
    if (rn50_first_conv) {
        assert(not weight_width_sliced); // weight width slicing not supported for rn50 first conv
        assert(act_block_w_equals_input_channels_x_filter_width);
    }

    vector<CoreCoord> debug_cores;
    for(uint32_t core_i = 0; core_i < total_num_cores; core_i++) {
        uint32_t core_x_i = core_i % num_cores_x;
        uint32_t core_y_i = core_i / num_cores_x;
        debug_cores.push_back({core_x_i+1, core_y_i+1});
    }

    CoreRange all_cores = {.start = CoreCoord(0, 0), .end = CoreCoord(num_cores_x - 1, num_cores_y - 1)};
    assert(total_active_num_cores >= num_cores_x);
    uint32_t num_active_cores_x = num_cores_x;
    uint32_t num_active_cores_y_with_full_x = total_active_num_cores / num_cores_x;
    uint32_t num_active_cores_x_last_y =  total_active_num_cores % num_cores_x;
    assert((num_active_cores_x * num_active_cores_y_with_full_x) + num_active_cores_x_last_y == total_active_num_cores);

    // cout << "All active cores. Core Ranges:" << endl;
    // cout << "Core range 1 - (0,0) to (" << num_active_cores_x - 1 << "," << num_active_cores_y_with_full_x - 1 << ")" << endl;

    std::set<CoreRange> all_active_cores_set;
    all_active_cores_set.insert((CoreRange) {.start = CoreCoord(0, 0), .end = CoreCoord(num_active_cores_x - 1, num_active_cores_y_with_full_x - 1)});
    if (num_active_cores_x_last_y > 0) {
        all_active_cores_set.insert((CoreRange) {.start = CoreCoord(0, num_active_cores_y_with_full_x), .end = CoreCoord(num_active_cores_x_last_y - 1, num_active_cores_y_with_full_x)});
        // cout << "Core range 2 - (0," << num_active_cores_y_with_full_x << ") to (" << num_active_cores_x_last_y - 1 << "," << num_active_cores_y_with_full_x << ")" << endl;
    }
    CoreRangeSet all_active_cores(all_active_cores_set);
    std::set<CoreRange> noop_cores_set;
    if (total_noop_cores > 0) {
        assert(total_noop_cores == (num_cores_x - num_active_cores_x_last_y));
        noop_cores_set.insert((CoreRange) {.start = CoreCoord(num_active_cores_x_last_y, num_active_cores_y_with_full_x), .end = CoreCoord(num_cores_x - 1, num_active_cores_y_with_full_x)});
        // cout << "Noop core range - (" << num_active_cores_x_last_y << "," << num_active_cores_y_with_full_x << ") to (" << num_cores_x - 1 << "," << num_active_cores_y_with_full_x << ")" << endl;

    }
    CoreRangeSet noop_cores(noop_cores_set);

    // Mcast cores
    // If total_num_cores, there is no mcasting
    CoreCoord top_left_core = {(std::size_t) 0, (std::size_t) 0};
    CoreCoord top_left_core_plus_one = {(std::size_t) 1, (std::size_t) 1};
    CoreCoord bottom_right_core = {(std::size_t) num_cores_x - 1, (std::size_t) num_cores_y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto top_left_core_plus_one_physical = device->worker_core_from_logical_core(top_left_core_plus_one);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreRange mcast_sender_cores{.start=top_left_core, .end=top_left_core}; // If single core, this kernel doesn't do mcasting
    CoreRangeSet mcast_receiver_cores{{}};
    uint32_t weights_mcast_sender_semaphore;
    uint32_t weights_mcast_receiver_semaphore;
    // 2D mcast
    if (weight_width_sliced) {
        mcast_sender_cores = {.start=top_left_core, .end=CoreCoord(0, num_cores_y - 1)};
        mcast_receiver_cores = {{{.start=CoreCoord(1, 0), .end=bottom_right_core}}};
        weights_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        weights_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // 1D mcast
    } else {
        if (total_num_cores > 1) {
            std::set<CoreRange> mcast_receiver_set;
            if (num_cores_x > 1) {
                mcast_receiver_set.insert({.start=CoreCoord(1, 0), .end=CoreCoord(num_cores_x - 1, 0)});
            } if (num_cores_y > 1) {
                mcast_receiver_set.insert({.start=CoreCoord(0, 1), .end=bottom_right_core});
            }
            mcast_receiver_cores = mcast_receiver_set;
            weights_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
            weights_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        }
    }

    uint32_t num_weight_cb_tiles = weight_block_h_ntiles * weight_block_w_ntiles;
    if (per_core_weight_matrix_width_ntiles < 8) {
        num_weight_cb_tiles = num_weight_cb_tiles * 2;
    }
    if (rn50_first_conv) {
        num_weight_cb_tiles = weight_block_h_ntiles * weight_block_w_ntiles * num_blocks_weight_w * num_blocks_act_w;
    }
    uint32_t num_act_cb_tiles = act_block_h_ntiles * act_block_w_ntiles;
    if (conv_act_size_c < 256) {
        num_act_cb_tiles = num_act_cb_tiles * 2; // double buffered
    }
    uint32_t writer_output_block_num_tiles = out_block_h_ntiles * weight_block_w_ntiles;

    // if (!(conv_output_size_w == 14 || conv_output_size_w == 7)) {
    //     writer_output_block_num_tiles = writer_output_block_num_tiles * 2;
    // }

    std::optional<uint32_t> output_cb_address = std::nullopt;
    if (output.memory_config().is_sharded()) {
        output_cb_address = output.buffer()->address();
    }
    auto cb_output = create_CBs(
            program,
            a,
            all_cores,
            num_act_cb_tiles, // row major act cb
            num_weight_cb_tiles, // tiled weight cb
            act_block_h_ntiles * act_block_w_ntiles, // tiled act cb
            writer_output_block_num_tiles, // math output cb
            weight_block_w_ntiles, // reblock cb
            writer_output_block_num_tiles, // writer output cb, double bufferred
            num_bytes_of_df,
            untilize_out,
            bias_ntiles_per_core,
            has_bias,
            output_cb_address);

    string reader_kernel;
    string compute_kernel;
    string writer_mcast_sender_kernel;
    string writer_mcast_receiver_kernel;
    bool reader_with_indices = false;
    if (rn50_first_conv) {
        reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_fast_resnet50_first_conv.cpp";
        compute_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/bmm_tilize_untilize_all_weights_in_l1_single_output_block_width_dim.cpp";
        writer_mcast_sender_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_and_mcast_sender_weights_resnet50_first_conv_tiled_out.cpp";
        writer_mcast_receiver_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_and_mcast_receiver_weights_resnet50_first_conv_tiled_out.cpp";
    } else {
        if (weight_size_h == 1 && weight_size_w == 1) {
            // use custom 1x1 conv kernels
            reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv1x1_activations_fast_for_col_major_conv_out_blocks.cpp";
            assert(conv_act_size_c % act_block_w_datums == 0);
            assert(num_blocks_act_w == (conv_act_size_c / act_block_w_datums));
        }
        else {
            // non 1x1 conv
            if (act_block_w_equals_input_channels_x_filter_width) {
                reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_act_block_w_equals_channels_X_filter_width.cpp";
            }
            else {
                assert(act_block_w_datums == conv_act_size_c);
                assert(num_blocks_act_w == weight_size_w * weight_size_h);

                if (a.memory_config().is_sharded() && weight_size_h == 3 && weight_size_w == 3 && stride_h == 1 && not weight_width_sliced) {
                    reader_with_indices = true;
                    reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_padded_with_halo_3x3_weights.cpp";

                    CircularBufferConfig cb_for_reader_indices_config = CircularBufferConfig(act_block_h_datums * 4, {{cb_for_reader_indices, tt::DataFormat::Float16_b}})
		                .set_page_size(cb_for_reader_indices, 4);
                    auto cb_for_reader_indices_id = tt_metal::CreateCircularBuffer(program, all_cores, cb_for_reader_indices_config);
                } else {
                    reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_fast_for_col_major_conv_out_blocks.cpp";
                }
            }
        }
        compute_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/conv_bmm_tilize_col_major_out_blocks.cpp";
        writer_mcast_sender_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
        writer_mcast_receiver_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
    }
    TT_ASSERT(!(conv_act_size_c & (conv_act_size_c - 1))); // channel depth power of 2 is supported only
    TT_ASSERT(!(out_row_size_bytes & (out_row_size_bytes - 1))); // output channels power of 2 is supported only

    std::vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_rt_args;
    std::vector<uint32_t> writer_compile_time_args;

    reader_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
            (uint32_t) stride_h, (uint32_t) stride_w, (uint32_t) conv_act_size_w, (uint32_t) conv_output_size_w,
            (uint32_t) conv_act_size_c * num_bytes_of_df, (uint32_t) std::log2(conv_act_size_c * num_bytes_of_df), extra_padding_for_32B_alignment,
            (uint32_t) (conv_act_size_c/act_block_w_datums), act_block_w_datums * num_bytes_of_df};

    // define for bias
    std::map<string, string> writer_defines;
    std::map<string, string> writer_mcast_sender_defines;
    std::map<string, string> compute_defines;
    if (output.memory_config().is_sharded()) {
        writer_defines["SHARDED_OUT"] = "1";
        writer_mcast_sender_defines["SHARDED_OUT"] = "1";
    }
    if (total_num_cores == 1) {
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }
    if (has_bias) {
        writer_defines["FUSE_BIAS"] = "1";
        writer_mcast_sender_defines["FUSE_BIAS"] = "1";
        compute_defines["FUSE_BIAS"] = "1";
    }

    if (fuse_relu) {
        compute_defines["PACK_RELU"] = "1";
    }

    writer_compile_time_args = {
        (uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
        out0_cb,
        weight_cb,
        bias_cb,
        bias_log2_of_pagesize,
        bias_tile_nbytes,
        (uint32_t) (bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0))};

    vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h_per_core,
        num_blocks_act_w,
        num_blocks_weight_w_per_core,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        true,
        untilize_out,

        bias_ntiles_per_core
    };

    auto writer_mcast_noc = NOC::NOC_0;
    auto reader_noc = writer_mcast_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
    auto writer_mcast_sender_id = CreateDataMovementKernel(
    program,
    writer_mcast_sender_kernel,
    mcast_sender_cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = writer_mcast_noc,
        .compile_args = writer_compile_time_args,
        .defines = writer_mcast_sender_defines});

    KernelID writer_mcast_receiver_id;
    if (total_num_cores > 1) {
        writer_mcast_receiver_id = CreateDataMovementKernel(
        program,
        writer_mcast_receiver_kernel,
        mcast_receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = writer_mcast_noc,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});
    }

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a.dtype());
    auto reader_id = CreateDataMovementKernel(
        program,
        reader_kernel,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // Compile compute kernel for active cores only
    // Compile blank kernel for noop cores
    auto compute_id = CreateComputeKernel(
        program,
        compute_kernel,
        all_active_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    if (total_noop_cores > 0) {
        auto compute_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        noop_cores);
    }

    vector<KernelID> reader_ids;
    vector<KernelID> writer_ids;
    //tt_start_debug_print_server();
    for(uint32_t core_i = 0; core_i < total_num_cores; core_i++) {
        uint32_t core_x_i = core_i % num_cores_x;
        uint32_t core_y_i = core_i / num_cores_x;
        // cout << "core_x_i=" << core_x_i << ", core_y_i=" << core_y_i << endl;
        CoreRange core = {.start = CoreCoord(core_x_i, core_y_i), .end = CoreCoord(core_x_i, core_y_i)};
        bool noop_core = false;
        for (const auto & noop_core_range : noop_cores.ranges()) {
            if (noop_core_range.contains(core)) {
                // cout << "No op core" << endl;
                // cout << "core_x_i=" << core_x_i << ", core_y_i=" << core_y_i << endl;
                noop_core = true;
                break;
            }
        }
        // per core specific args
        uint32_t act_slice_i = core_i % (num_cores_y_per_weight_slice_width * num_cores_x);
        uint32_t weight_slice_i = core_i / (num_cores_y_per_weight_slice_width * num_cores_x);
        uint32_t total_h_start = act_slice_i * per_core_out_matrix_height_ntiles * TILE_HEIGHT;
        uint32_t n_start = total_h_start / (conv_output_size_h * conv_output_size_w);
        uint32_t matrix_h_start = total_h_start % (conv_output_size_h * conv_output_size_w);
        uint32_t out_h_start = matrix_h_start / conv_output_size_w;
        uint32_t out_w_start = matrix_h_start % conv_output_size_w;
        uint32_t in_h_start = (n_start * conv_act_size_h) + out_h_start * stride_h;
        uint32_t last_start_in_h_curr_image = 222 + (n_start * conv_act_size_h);
        uint32_t out_start_tile_id = (act_slice_i * per_core_out_matrix_height_ntiles * weight_matrix_width_ntiles) + (weight_slice_i * per_core_weight_matrix_width_ntiles);
        uint32_t out_start_tile_id_h = act_slice_i * per_core_out_matrix_height_ntiles;
        uint32_t out_start_tile_id_w = weight_slice_i * per_core_weight_matrix_width_ntiles;
        uint32_t bias_tile_offset = weight_slice_i * per_core_weight_matrix_width_ntiles;
        if (has_bias) {
            assert(bias_tile_offset < bias_ntiles);
        }
        // cout << "act_slice_i=" << act_slice_i << endl;
        // cout << "weight_slice_i=" << weight_slice_i << endl;
        // cout << "core_i=" << core_i << endl;
        // cout << "num_blocks_act_h_per_core=" << num_blocks_act_h_per_core << endl;
        // cout << "num_blocks_weight_w_per_core=" << num_blocks_weight_w_per_core << endl;
        // cout << "bias_tile_offset=" << bias_tile_offset << endl;
        // cout << "out_start_tile_id=" << out_start_tile_id << endl;
        // cout << "out_start_tile_id_w=" << out_start_tile_id_w << endl;
        // cout << "per_core_out_matrix_height_ntiles=" << per_core_out_matrix_height_ntiles << endl;
        // cout << "weight_matrix_width_ntiles=" << weight_matrix_width_ntiles <<  endl;
        // cout << "out_start_tile_id_h=" << out_start_tile_id_h << endl;
        // cout << endl;
        // cout << "total_h_start=" << total_h_start << endl;
        // cout << "in_h_start=" << in_h_start << endl;
        // cout << "out_h_start=" << out_h_start << endl;
        // cout << "out_w_start=" << out_w_start << endl;
        // cout << "matrix_h_start=" << matrix_h_start << endl;
        // cout << "n_start=" << n_start << endl;

        if (rn50_first_conv) {
            assert(pad_h == 0 && pad_w == 0);
            reader_rt_args = {
                act_dram_addr,
                conv_act_size_c,
                conv_output_size_w,
                weight_size_w,
                num_blocks_act_h_per_core,
                num_blocks_act_w,
                act_block_h_datums,
                act_block_num_tiles,
                in_h_start,
                out_w_start,
                last_start_in_h_curr_image,
                (uint32_t) noop_core
            };
        } else if (reader_with_indices) {
            /* Logic to compute:
             * NOTE: This logic is wrong if stride !=1
             * first_partial_right_aligned_row_width
             * skip_after_partial_right_aligned_row
             * first_partial_image_num_rows
             * skip_after_first_partial_image_row
             * num_full_images
             * skip_after_full_image
             * last_partial_image_num_rows
             * last_partial_left_aligned_row_width
             */
            uint32_t start_stick = core_i * act_block_h_datums;
            uint32_t end_stick = start_stick + act_block_h_datums;

            ShardingConfig sharding_config = get_specs_for_sharding_partition(start_stick, end_stick, conv_act_size_h, conv_act_size_w, weight_size_w, pad_h, pad_w);
            uint32_t first_partial_right_aligned_row_width = sharding_config.first_partial_right_aligned_row_width;
            uint32_t skip_after_partial_right_aligned_row = sharding_config.skip_after_partial_right_aligned_row;
            uint32_t first_partial_image_num_rows = sharding_config.first_partial_image_num_rows;
            uint32_t skip_after_first_partial_image_row = sharding_config.skip_after_first_partial_image_row;
            uint32_t num_full_images = sharding_config.num_full_images;
            uint32_t skip_after_full_image = sharding_config.skip_after_full_image;
            uint32_t last_partial_image_num_rows = sharding_config.last_partial_image_num_rows;
            uint32_t last_partial_left_aligned_row_width = sharding_config.last_partial_left_aligned_row_width;

            reader_rt_args = {
                // arguments for act
                act_dram_addr,
                act_noc_x,
                act_noc_y,

                conv_act_size_w,
                conv_act_size_h,
                conv_act_size_c,
                weight_size_h,
                weight_size_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                conv_output_size_h,
                conv_output_size_w,
                num_blocks_act_h_per_core, // per core
                num_blocks_act_w,
                num_blocks_weight_w_per_core,
                num_groups,

                act_matrix_height_unpadded,
                act_matrix_width_unpadded,
                act_matrix_height,
                act_matrix_width,
                act_matrix_height_ntiles,
                act_matrix_width_ntiles,
                act_block_h_datums,
                act_block_w_datums,
                act_block_h_ntiles,
                act_block_w_ntiles,
                act_block_num_tiles,

                src_dram_act_buffer_size_bytes,
                dst_l1_act_buffer_size_bytes,

                n_start,
                out_h_start,
                out_w_start,
                total_h_start,

                first_partial_right_aligned_row_width,
                skip_after_partial_right_aligned_row,
                first_partial_image_num_rows,
                skip_after_first_partial_image_row,
                num_full_images,
                skip_after_full_image,
                last_partial_image_num_rows,
                last_partial_left_aligned_row_width,

                (uint32_t) noop_core
            };
        } else {
            reader_rt_args = {
                // arguments for act
                act_dram_addr,
                act_noc_x,
                act_noc_y,

                conv_act_size_w,
                conv_act_size_h,
                conv_act_size_c,
                weight_size_h,
                weight_size_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                conv_output_size_h,
                conv_output_size_w,
                num_blocks_act_h_per_core, // per core
                num_blocks_act_w,
                num_blocks_weight_w_per_core,
                num_groups,

                act_matrix_height_unpadded,
                act_matrix_width_unpadded,
                act_matrix_height,
                act_matrix_width,
                act_matrix_height_ntiles,
                act_matrix_width_ntiles,
                act_block_h_datums,
                act_block_w_datums,
                act_block_h_ntiles,
                act_block_w_ntiles,
                act_block_num_tiles,

                src_dram_act_buffer_size_bytes,
                dst_l1_act_buffer_size_bytes,

                n_start,
                out_h_start,
                out_w_start,
                total_h_start,
                (uint32_t) noop_core
            };
        }

        writer_rt_args = {
            out_dram_addr,
            weight_dram_addr,
            bias_dram_addr,

            output_width_num_tiles, // out_next_tile_stride_h
            1, // out_next_tile_stride_w
            out_subblock_h_ntiles * output_width_num_tiles, // out_next_subblock_stride_h
            out_subblock_w_ntiles, // out_next_subblock_stride_w
            act_block_h_ntiles * output_width_num_tiles, // out_next_block_stride_h
            weight_block_w_ntiles, // out_next_block_stride_w
            out_subblock_h_ntiles,
            out_subblock_w_ntiles,
            out_subblock_num_tiles,
            act_block_h_ntiles / out_subblock_h_ntiles, // out_num_subblocks_h
            weight_block_w_ntiles / out_subblock_w_ntiles,   // out_num_subblocks_w
            num_blocks_act_h_per_core, // out_num_blocks_h
            num_blocks_weight_w_per_core, // out_num_blocks_w
            act_block_h_ntiles, // out_block_height_num_tiles
            output_height_num_tiles, // out_height_num_tiles without block shape padding
            output_width_num_tiles, // out_width_num_tiles withoug block shape padding
            out_start_tile_id,
            out_start_tile_id_h,
            out_start_tile_id_w,

            num_blocks_act_w, // = number of blocks of weight in height dim
            weight_block_num_tiles,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

            // bias
            bias_ntiles_per_core,
            bias_tile_offset,

            (uint32_t) noop_core
        };

        SetRuntimeArgs(
            program, reader_id, core,
            reader_rt_args
        );
        reader_ids.push_back(reader_id);

        // Mcast sender
        // 2D mcast
        if (weight_width_sliced) {
            CoreCoord right_core = {(std::size_t) num_cores_x - 1, (std::size_t) core_y_i};
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            // sender
            if (core_x_i == 0) {
                if (writer_mcast_noc == NOC::NOC_0) {
                    writer_rt_args.push_back(top_left_core_plus_one_physical.x); // weights_mcast_dest_noc_start_x
                    writer_rt_args.push_back(right_core_physical.y); // weights_mcast_dest_noc_start_y
                    writer_rt_args.push_back(bottom_right_core_physical.x); // weights_mcast_dest_noc_end_x
                    writer_rt_args.push_back(right_core_physical.y); // weights_mcast_dest_noc_end_y
                } else {
                    writer_rt_args.push_back(bottom_right_core_physical.x); // weights_mcast_dest_noc_start_x
                    writer_rt_args.push_back(right_core_physical.y); // weights_mcast_dest_noc_start_y
                    writer_rt_args.push_back(top_left_core_plus_one_physical.x); // weights_mcast_dest_noc_end_x
                    writer_rt_args.push_back(right_core_physical.y); // weights_mcast_dest_noc_end_y
                }

                writer_rt_args.push_back(num_cores_x - 1); // weights_mcast_num_dests
                writer_rt_args.push_back(num_cores_x - 1); // weights_mcast_num_cores
                writer_rt_args.push_back(weights_mcast_sender_semaphore);
                writer_rt_args.push_back(weights_mcast_receiver_semaphore);

                SetRuntimeArgs(
                    program, writer_mcast_sender_id, core,
                    writer_rt_args
                );
                writer_ids.push_back(writer_mcast_sender_id);
            // receiver
            } else {
                writer_rt_args.push_back(top_left_core_physical.x); // weights_mcast_sender_noc_x
                writer_rt_args.push_back(right_core_physical.y); // weights_mcast_sender_noc_y
                writer_rt_args.push_back(weights_mcast_sender_semaphore);
                writer_rt_args.push_back(weights_mcast_receiver_semaphore);

                SetRuntimeArgs(
                    program, writer_mcast_receiver_id, core,
                    writer_rt_args
                );
                writer_ids.push_back(writer_mcast_receiver_id);
            }
        // 1D mcast
        } else {
            // sender
            if (core_x_i == 0 and core_y_i == 0) {
                if (writer_mcast_noc == NOC::NOC_0) {
                    writer_rt_args.push_back(top_left_core_physical.x); // weights_mcast_dest_noc_start_x
                    writer_rt_args.push_back(top_left_core_physical.y); // weights_mcast_dest_noc_start_y
                    writer_rt_args.push_back(bottom_right_core_physical.x); // weights_mcast_dest_noc_end_x
                    writer_rt_args.push_back(bottom_right_core_physical.y); // weights_mcast_dest_noc_end_y
                } else {
                    writer_rt_args.push_back(bottom_right_core_physical.x); // weights_mcast_dest_noc_start_x
                    writer_rt_args.push_back(bottom_right_core_physical.y); // weights_mcast_dest_noc_start_y
                    writer_rt_args.push_back(top_left_core_physical.x); // weights_mcast_dest_noc_end_x
                    writer_rt_args.push_back(top_left_core_physical.y); // weights_mcast_dest_noc_end_y
                }
                writer_rt_args.push_back(total_active_num_cores - 1); // weights_mcast_num_dests
                writer_rt_args.push_back(total_num_cores - 1); // weights_mcast_num_cores
                writer_rt_args.push_back(weights_mcast_sender_semaphore);
                writer_rt_args.push_back(weights_mcast_receiver_semaphore);

                SetRuntimeArgs(
                    program, writer_mcast_sender_id, core,
                    writer_rt_args
                );
                writer_ids.push_back(writer_mcast_sender_id);
            // receiver
            } else {
                writer_rt_args.push_back(top_left_core_physical.x); // weights_mcast_sender_noc_x
                writer_rt_args.push_back(top_left_core_physical.y); // weights_mcast_sender_noc_y
                writer_rt_args.push_back(weights_mcast_sender_semaphore);
                writer_rt_args.push_back(weights_mcast_receiver_semaphore);

                SetRuntimeArgs(
                    program, writer_mcast_receiver_id, core,
                    writer_rt_args
                );
                writer_ids.push_back(writer_mcast_receiver_id);
            }
        }

    } // for num_cores

    auto override_runtime_arguments_callback = [
            reader_kernel_ids=reader_ids,
            writer_kernel_ids=writer_ids,
            cb_output=cb_output,
            total_num_cores=total_num_cores,
            num_cores_x=num_cores_x,
            num_cores_y=num_cores_y,
            has_bias=has_bias
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
        TT_ASSERT(output_tensors.size() == 1);

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        for(uint32_t core_i = 0; core_i < total_num_cores; core_i++) {
            uint32_t core_x_i = core_i % num_cores_x;
            uint32_t core_y_i = core_i / num_cores_x;
            CoreCoord core = {core_x_i, core_y_i};
            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_i], core);
                runtime_args[0] = src_buffer_a->address();
                SetRuntimeArgs(program, reader_kernel_ids[core_i], core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_i], core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[1] = src_buffer_b->address();
                if (has_bias) {
                    auto src_buffer_c = optional_input_tensors.at(0).value().buffer();
                    TT_ASSERT(src_buffer_c != nullptr);
                    runtime_args[2] = src_buffer_c->address();
                }
                SetRuntimeArgs(program, writer_kernel_ids[core_i], core, runtime_args);
            }
        }
        if (out_sharded) {
            auto& output_cb_config = GetCircularBufferConfig(program, cb_output);
            output_cb_config.set_globally_allocated_address(dst_buffer->address());
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

Tensor optimized_conv(const Tensor& a,
            const Tensor &b,
            std::optional<const Tensor> bias,
            const vector<int> conv_params,
            uint32_t output_channels,
            bool untilize_out,
            bool has_bias,
            bool fuse_relu,
            MathFidelity math_fidelity,
            const OptimizedConvParallelizationConfig& parallelization_config,
            const OptimizedConvBlockConfig& block_config,
            uint32_t extra_padding_for_32B_alignment,
            std::optional<MemoryConfig> output_mem_config,
            std::optional<std::array<std::uint32_t, 4>> input_tensor_shape
) {
    TT_ASSERT(!untilize_out, "Optimized conv only supports tiled out");
    TT_ASSERT(b.layout() == Layout::TILE); // Weights should already be formatted
    const auto& ashape = input_tensor_shape.has_value() ? Shape(input_tensor_shape.value()) : a.shape();
    auto padded_a_shape = Shape({ashape[0], ashape[1], ashape[2], round_up(ashape[3], 16)});
    FormatParams input_a_format_params = {.pad_shape=padded_a_shape, .pad_value=0.0, .target_layout=Layout::ROW_MAJOR};
    FormatParams input_b_format_params = {.pad_shape=b.shape(), .pad_value=0.0, .target_layout=Layout::TILE};
    FormatParams input_bias_format_params = {};
    if (has_bias) {
        input_bias_format_params = {.pad_shape=bias.value().shape(), .pad_value=0, .target_layout=Layout::TILE};
    }
    auto output_layout = untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (output_mem_config.has_value()) {
        TT_ASSERT((output_mem_config.value().is_sharded() || output_mem_config.value() == a.memory_config()));
    }
    return operation::run_without_autoformat(
        OptimizedConv(conv_params, output_channels, untilize_out, has_bias, fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, output_mem_config.value_or(a.memory_config()), ashape
        ),
        {a, b},
        {bias}).at(0);
}

void OptimizedConv::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
    TT_ASSERT(!input_tensor_b.memory_config().is_sharded());
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(!this->untilize_out);
        uint32_t out_block_h_ntiles = block_config.out_block_h_ntiles;
        auto [act_matrix_shape, act_matrix_shape_unpadded] = compute_opt_conv_activation_as_mm_shape(input_tensor_a.shape(), conv_params, out_block_h_ntiles, extra_padding_for_32B_alignment);
        uint32_t out_width_ntiles = this->compute_output_shapes(input_tensors).at(0)[-1] / TILE_WIDTH;
        if(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_ASSERT(this->parallelization_config.per_core_weight_matrix_width_ntiles == out_width_ntiles);
            TT_ASSERT(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1);
        } else if (this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_ASSERT(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1);
        }
    }
}

std::vector<Shape> OptimizedConv::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    // TODO: clean up here
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = compute_opt_conv_output_face_shape(conv_activation_h, conv_activation_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    if (untilize_out) {
        // RM output has unpadded output height and padded output width to 32.
        // pad the output channels to TILE_WIDTH as conv writer kernel does not remove padding for tile
        // TODO (nshanker): specify padding explicitly here with "Padding" object and add unit test
        assert(batch_size == 1); // batch size > 1 not tested with "untilize_out" (TODO)
        auto output_channels = round_up(this->output_channels, TILE_WIDTH);
        Shape output_tensor_shape = {batch_size, conv_output_h, conv_output_w, output_channels};
        return {output_tensor_shape};
    } else {
        // Tiled output shape is padded shape. Padded to tile shape.
        auto shape_w = batch_size * conv_output_h * conv_output_w;
        auto shape_c = output_channels;
        auto padded_shape_w = round_up(shape_w, TILE_HEIGHT);
        auto padded_shape_c = round_up(this->output_channels, TILE_WIDTH);
        auto output_padding = Padding({{0, 0}, {0, 0}, {0, (padded_shape_w - shape_w)}, {0, (padded_shape_c - shape_c)}}, Padding::PadValue::Any);
        auto output_tensor_shape = Shape({1, 1, padded_shape_w, padded_shape_c}, output_padding);
        return {output_tensor_shape};
    }
}

std::vector<Tensor> OptimizedConv::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->output_mem_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        if (this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t total_height_tiles = tt_metal::compute_volume(output_shape) / output_shape[-1] / TILE_HEIGHT;
            uint32_t num_cores = total_height_tiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            CoreRangeSet shard_grid = num_cores_to_corerange_set(num_cores, this->parallelization_config.grid_size, true);

            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, output_shape[-1]};
            auto shard_spec = ShardSpec{.shard_grid=shard_grid, .shard_shape=shard_shape, .shard_orientation=ShardOrientation::ROW_MAJOR};
            return {create_sharded_device_tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), this->output_mem_config, shard_spec)};
        } else {
            auto [act_matrix_shape, act_matrix_shape_unpadded] = compute_opt_conv_activation_as_mm_shape(input_tensor.shape(), conv_params, this->block_config.out_block_h_ntiles, extra_padding_for_32B_alignment);
            uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
            uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
            uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            uint32_t weight_matrix_width = weight_tensor.shape()[-1];
            uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;
            uint32_t num_weight_slices_width = weight_matrix_width_ntiles / this->parallelization_config.per_core_weight_matrix_width_ntiles ;
            uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
            CoreRangeSet shard_grid = num_cores_to_corerange_set(total_active_num_cores, this->parallelization_config.grid_size, true);
            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, this->parallelization_config.per_core_weight_matrix_width_ntiles * TILE_WIDTH};
            auto shard_spec = ShardSpec{.shard_grid=shard_grid, .shard_shape=shard_shape, .shard_orientation=ShardOrientation::COL_MAJOR};
            return {create_sharded_device_tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device(), this->output_mem_config, shard_spec)};
        }

    }
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), output_layout, this->output_mem_config);
}

operation::ProgramWithCallbacks OptimizedConv::create_program(const std::vector<Tensor>& input_tensors,
                                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                     std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {optimized_conv_(input_tensor_a, input_tensor_b, this->input_tensor_shape,
            input_tensor_bias, conv_params, output_channels,
            untilize_out, has_bias,
            fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, output_tensor)};
}

tt::stl::reflection::Attributes OptimizedConv::attributes() const {
    return {
        {"conv_params", this->conv_params},
        {"output_channels", this->output_channels},
        {"untilize_out", this->untilize_out},
        {"has_bias", this->has_bias},
        {"fuse_relu", this->fuse_relu},
        {"math_fidelity", this->math_fidelity},
    };
}

tt::stl::reflection::Attributes OptimizedConvParallelizationConfig::attributes() const {
    return {
        {"grid_size",  this->grid_size.str()},
        {"per_core_out_matrix_height_ntiles",  this->per_core_out_matrix_height_ntiles},
        {"per_core_weight_matrix_width_ntiles",  this->per_core_weight_matrix_width_ntiles},
    };
}

tt::stl::reflection::Attributes OptimizedConvBlockConfig::attributes() const {
    return {
        {"act_block_h_ntiles",  this->act_block_h_ntiles},
        {"act_block_w_ntiles",  this->act_block_w_ntiles},
        {"weight_block_w_ntiles",  this->weight_block_w_ntiles},
        {"out_block_h_ntiles",  this->out_block_h_ntiles},
        {"out_subblock_h_ntiles",  this->out_subblock_h_ntiles},
        {"out_subblock_w_ntiles",  this->out_subblock_w_ntiles},
    };
}

}  // namespace tt_metal

}  // namespace tt
