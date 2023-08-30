#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tt_stl/reflection.hpp"

#include "tt_dnn/op_library/auto_format.hpp"
using namespace tt::constants;
namespace tt {

namespace tt_metal {

const uint32_t act_cb                                 = CB::c_in0;
const uint32_t weight_cb                              = CB::c_in1;
const uint32_t bias_cb                                = CB::c_in2;
const uint32_t matmul_partials_cb                     = CB::c_intermed0;
const uint32_t tilize_mode_tilized_act_cb             = CB::c_intermed1;
const uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
const uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
const uint32_t out_for_bias_cb                        = CB::c_intermed4;
const uint32_t out0_cb                                = CB::c_out0;

pair<uint32_t, uint32_t> compute_opt_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t padding_for_32B_alignment=0) {
    uint32_t conv_output_h = ((conv_activation_h - filter_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_w = ((conv_activation_w - filter_w + (2 * pad_w) - padding_for_32B_alignment) / stride_w) + 1;
    return {conv_output_h, conv_output_w};
}
pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(Shape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t padding_for_32B_alignment=0) {
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
    uint32_t act_block_w_datums = act_block_w_ntiles * TILE_WIDTH;
    assert(act_block_w_datums >= conv_activation_shape[3] * filter_w);
    uint32_t num_cols_padded = act_block_w_datums * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}


void create_CBs(tt_metal::Program &program,
                                tt_metal::Device* device,
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
                                bool with_bias = false) {

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    // Invariants
    auto cb_act = CreateCircularBuffers(
        program,
        act_cb,
        core,
        num_cb0_tiles,
        num_cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    auto cb_weight = CreateCircularBuffers(
        program,
        weight_cb,
        core,
        num_cb1_tiles,
        num_cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    // Used for placing tilized activations
    auto cb_src0_tilized = CreateCircularBuffers(
        program,
        tilize_mode_tilized_act_cb,
        core,
        num_cb0_tilized_tiles,
        num_cb0_tilized_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    if (untilize_out) {
        auto cb_matmul_partials = CreateCircularBuffers(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = CreateCircularBuffers(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        auto cb_reblock = CreateCircularBuffers(
            program,
            untilize_mode_reblock_cb,
            core,
            num_reblock_cb_tiles,
            num_reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = CreateCircularBuffers(
            program,
            out0_cb,
            core,
            num_writer_output_tiles,
            num_writer_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    } else {
        CoreRangeSet cores(std::set<CoreRange>({core}));
        auto cb_matmul_partials = CreateCircularBuffers(
            program,
            {matmul_partials_cb, out0_cb},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }

    if (with_bias) {
        // bias input
        uint32_t bias_pagesize = single_tile_size;
        auto cb_bias = CreateCircularBuffers(
            program,
            bias_cb,
            core,
            bias_ntiles,
            bias_ntiles * bias_pagesize,
            DataFormat::Float16_b
        );
        // intermed mm output
        auto cb_out_for_bias = CreateCircularBuffers(
            program,
            out_for_bias_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
        log_debug("BIAS CBs: {} {} {}", bias_cb, bias_ntiles, bias_pagesize);
    }
}

operation::ProgramWithCallbacks optimized_conv_single_core(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, vector<int> conv_params,
                                       uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                                       uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity, uint32_t extra_padding_for_32B_alignment, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_ASSERT(output_channels <= b.shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] = compute_opt_conv_activation_as_mm_shape(a.shape(), conv_params, act_block_h_ntiles, act_block_w_ntiles, extra_padding_for_32B_alignment);
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

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

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
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core = {.start={0, 0}, .end={0, 0}};
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

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

    // more args for reader
    uint32_t conv_act_size_h = a.shape()[1];
    uint32_t conv_act_size_w = a.shape()[2];
    uint32_t conv_act_size_c = a.shape()[3];
    uint32_t weight_size_h = (uint32_t) conv_params[0];
    uint32_t weight_size_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];

    //uint32_t conv_output_size_h = ((conv_act_size_h - weight_size_h + (2 * pad_h)) / stride_h) + 1;
    //uint32_t conv_output_size_w = ((conv_act_size_w - weight_size_w + (2 * pad_w)) / stride_w) + 1;

    auto [conv_output_size_h, conv_output_size_w] = compute_opt_conv_output_face_shape(conv_act_size_h, conv_act_size_w, weight_size_h, weight_size_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    std::map<string, string> reader_defines;

    if(conv_act_size_c * weight_size_w != act_block_w_datums) {
        assert(act_block_w_datums > conv_act_size_c * weight_size_w);
        uint32_t conv_act_block_width_padding_bytes = (act_block_w_datums - (conv_act_size_c * weight_size_w)) * num_bytes_of_df;
        reader_defines["ACT_BLOCK_WIDTH_PADDING_BYTES"] = to_string(conv_act_block_width_padding_bytes);
    }
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
    uint32_t batch_size = 1;
    // output data format
    const auto out_df = datatype_to_dataformat_converter(a.dtype());
    // For debug
    {
        log_debug(tt::LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(tt::LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(tt::LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(tt::LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(tt::LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(tt::LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(tt::LogOp, "act_dram_addr: {}", act_dram_addr);
        log_debug(tt::LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(tt::LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(tt::LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(tt::LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(tt::LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(tt::LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(tt::LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(tt::LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
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

    bool rn50_first_conv = (conv_act_size_h == 230 && conv_act_size_w == (231 + extra_padding_for_32B_alignment) &&
                            conv_output_size_h == 112 && conv_output_size_w == 112 &&
                            weight_size_h == 7 && weight_size_w == 8 &&
                            stride_h == 2 && stride_w == 2 &&
                            num_blocks_weight_w == 1);

    uint32_t num_weight_tiles_in_cb = weight_block_h_ntiles * weight_block_w_ntiles;
    if (rn50_first_conv) {
        num_weight_tiles_in_cb = weight_block_h_ntiles * weight_block_w_ntiles * num_blocks_weight_w * num_blocks_act_w;
    }
    create_CBs(
        program,
        a.device(),
        core,
        act_block_h_ntiles * act_block_w_ntiles * 2, // row major act cb, double bufferred
        num_weight_tiles_in_cb, // tiled weight cb
        act_block_h_ntiles * act_block_w_ntiles, // tiled act cb
        act_block_h_ntiles * weight_block_w_ntiles, // math output cb
        weight_block_w_ntiles, // reblock cb
        act_block_h_ntiles * weight_block_w_ntiles * 2, // writer output cb, double bufferred
        num_bytes_of_df,
        untilize_out,
        bias_ntiles,
        has_bias);

    // define for bias
    std::map<string, string> all_defines;
    std::map<string, string> compute_defines;
    if (has_bias) {
        all_defines["FUSE_BIAS"] = "1";
        compute_defines["FUSE_BIAS"] = "1";
    }

    if (fuse_relu) {
        // auto relu_param = UnaryWithParam{.op_type = UnaryOpType.RELU};
        compute_defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::RELU, nullopt, "ACTIVATION", "i"));
        compute_defines["RELU_ACTIVATION"] = "1";
        if (has_bias) {
            compute_defines["FUSE_BIAS"] = "1";
        }
    }

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    string writer_kernel;
    string compute_kernel;
    TT_ASSERT(!(conv_act_size_c & (conv_act_size_c - 1))); // channel depth power of 2 is supported only
    TT_ASSERT(!(out_row_size_bytes & (out_row_size_bytes - 1))); // output channels power of 2 is supported only
    if (pad_h == 0 && pad_w == 0) {
        if(rn50_first_conv) {
            reader_kernel = "libs/tt_dnn/op_library/conv/kernels/reader_conv_activations_fast_resnet50_first_conv.cpp";
            compute_kernel = "libs/tt_dnn/op_library/conv/kernels/bmm_tilize_untilize_all_weights_in_l1_single_output_block_width_dim.cpp";
        } else {
            reader_kernel = "libs/tt_dnn/op_library/conv/kernels/reader_conv_activations_fast.cpp";
            compute_kernel = "tt_metal/kernels/compute/bmm_tilize_untilize.cpp";
        }
    } else {
        reader_kernel = "libs/tt_dnn/op_library/conv/kernels/reader_conv_activations_fast.cpp";
        compute_kernel = "tt_metal/kernels/compute/bmm_tilize_untilize.cpp";
    }
    reader_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
            (uint32_t) stride_h, (uint32_t) stride_w, (uint32_t) conv_act_size_w, (uint32_t) conv_output_size_w,
            (uint32_t) conv_act_size_c * num_bytes_of_df, (uint32_t) std::log2(conv_act_size_c * num_bytes_of_df), extra_padding_for_32B_alignment};

    if (rn50_first_conv) {
        assert(pad_h == 0 && pad_w == 0);
        reader_rt_args = {
            act_dram_addr,
            conv_act_size_c,
            conv_output_size_w,
            weight_size_w,
            num_blocks_act_h,
            num_blocks_act_w,
            act_block_h_datums,
            act_block_num_tiles
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
            num_blocks_act_h,
            num_blocks_act_w,
            num_blocks_weight_w,
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
        };
    }

    vector<uint32_t> writer_rt_args;
    std::vector<uint32_t> writer_compile_time_args;
    if (untilize_out) {
        if (rn50_first_conv) {
            writer_kernel = "libs/tt_dnn/op_library/conv/kernels/writer_and_reader_weights_resnet50_first_conv_untilize_out.cpp";
        } else {
            writer_kernel = "libs/tt_dnn/op_library/conv/kernels/writer_unary_stick_8bank_blocks_reader_weight_tile_with_pow2_addr_gen_fast.cpp";
        }
        writer_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0), out0_cb, weight_cb, (uint32_t) std::log2(out_row_size_bytes)};
        writer_rt_args = {
            out_dram_addr,
            weight_dram_addr,

            act_block_h_datums,
            out_block_row_size_bytes,
            1,
            num_blocks_act_h,
            num_blocks_weight_w,
            out_row_size_bytes,
            last_block_row_size_bytes,
            act_matrix_height_unpadded,

            num_blocks_act_w, // = number of blocks of weight in height dim
            weight_block_num_tiles,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

        };
    } else {
        if (rn50_first_conv) {
            writer_kernel = "libs/tt_dnn/op_library/conv/kernels/writer_and_reader_weights_resnet50_first_conv_tiled_out.cpp";
        } else {
            writer_kernel = "libs/tt_dnn/op_library/conv/kernels/writer_tiled_out_reader_conv_weights_tiled.cpp";
        }
        writer_compile_time_args = {
            (uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
            out0_cb,
            weight_cb,
            bias_cb,
            bias_log2_of_pagesize,
            bias_tile_nbytes,
            (uint32_t) (bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0))};
        writer_rt_args = {
            out_dram_addr,
            weight_dram_addr,

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
            num_blocks_act_h, // out_num_blocks_h
            num_blocks_weight_w, // out_num_blocks_w
            act_block_h_ntiles, // out_block_height_num_tiles
            output_height_num_tiles, // out_height_num_tiles without block shape padding
            output_width_num_tiles, // out_width_num_tiles withoug block shape padding

            num_blocks_act_w, // = number of blocks of weight in height dim
            weight_block_num_tiles,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

            // bias
            bias_dram_addr,
            bias_ntiles
        };
    }
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a.dtype());
    auto reader_id = CreateDataMovementKernel(
        program,
        reader_kernel,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    auto writer_id = CreateDataMovementKernel(
        program,
        writer_kernel,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = all_defines});

    vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h,
        num_blocks_act_w,
        num_blocks_weight_w,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        true,
        untilize_out,

        bias_ntiles
    };

    auto compute = CreateComputeKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    SetRuntimeArgs(
        program, reader_id, core,
        reader_rt_args
    );

    SetRuntimeArgs(
        program, writer_id, core,
        writer_rt_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_id,
        writer_kernel_id=writer_id,
        has_bias=has_bias
    ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        TT_ASSERT(input_buffers.size() == 3);
        TT_ASSERT(output_buffers.size() == 1);

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer_a->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = src_dram_buffer_b->address();
            if (has_bias) {
                auto src_dram_buffer_c = input_buffers.at(2);
                TT_ASSERT(src_dram_buffer_c != nullptr);
                runtime_args[25] = src_dram_buffer_c->address();
            }
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};

}

Tensor optimized_conv(const Tensor& a,
            const Tensor &b,
            std::optional<const Tensor> bias,
            const vector<int> conv_params,
            uint32_t act_block_h_ntiles,
            uint32_t act_block_w_ntiles,
            uint32_t weight_block_w_ntiles,
            uint32_t out_subblock_h_ntiles,
            uint32_t out_subblock_w_ntiles,
            uint32_t output_channels,
            bool untilize_out,
            bool has_bias,
            bool fuse_relu,
            MathFidelity math_fidelity,
            uint32_t extra_padding_for_32B_alignment) {
    TT_ASSERT(b.layout() == Layout::TILE); // Weights should already be formatted
    auto padded_a_shape = Shape({a.shape()[0], a.shape()[1], a.shape()[2], round_up(a.shape()[3], 16)});
    FormatParams input_a_format_params = {.pad_shape=padded_a_shape, .pad_value=0.0, .target_layout=Layout::ROW_MAJOR};
    FormatParams input_b_format_params = {.pad_shape=b.shape(), .pad_value=0.0, .target_layout=Layout::TILE};
    FormatParams input_bias_format_params = {};
    if (has_bias) {
        input_bias_format_params = {.pad_shape=bias.value().shape(), .pad_value=0, .target_layout=Layout::TILE};
    }
    auto output_layout = untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::run_without_autoformat(
        OptimizedConv(act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, conv_params, output_channels, untilize_out, has_bias, fuse_relu, math_fidelity, extra_padding_for_32B_alignment
        ),
        {a, b},
        {bias}).at(0);
}

void OptimizedConv::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
}

std::vector<Shape> OptimizedConv::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    uint32_t batch_size = input_tensor_a.shape()[0];
    uint32_t conv_activation_h = input_tensor_a.shape()[1];
    uint32_t conv_activation_w = input_tensor_a.shape()[2];
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
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), output_layout, input_tensor.memory_config());
}

operation::ProgramWithCallbacks OptimizedConv::create_program(const std::vector<Tensor>& input_tensors,
                                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                     std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {optimized_conv_single_core(input_tensor_a, input_tensor_b,
            input_tensor_bias, conv_params,
            act_block_h_ntiles, act_block_w_ntiles,
            weight_block_w_ntiles, out_subblock_h_ntiles,
            out_subblock_w_ntiles, output_channels,
            untilize_out, has_bias,
            fuse_relu, math_fidelity, extra_padding_for_32B_alignment, output_tensor)};
}

tt::stl::reflection::Attributes OptimizedConv::attributes() const {
    return {
        {"conv_params", this->conv_params},
        {"act_block_h_ntiles", this->act_block_h_ntiles},
        {"act_block_w_ntiles", this->act_block_w_ntiles},
        {"weight_block_w_ntiles", this->weight_block_w_ntiles},
        {"out_subblock_h_ntiles", this->out_subblock_h_ntiles},
        {"out_subblock_w_ntiles", this->out_subblock_w_ntiles},
        {"output_channels", this->output_channels},
        {"untilize_out", this->untilize_out},
        {"has_bias", this->has_bias},
        {"fuse_relu", this->fuse_relu},
        {"math_fidelity", this->math_fidelity},
    };
}

}  // namespace tt_metal

}  // namespace tt
