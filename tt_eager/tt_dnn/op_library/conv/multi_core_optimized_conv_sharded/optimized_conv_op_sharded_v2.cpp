// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/constants.hpp"

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
const uint32_t cb_for_reader_indices                  = CB::c_in4;
const uint32_t cb_for_reader_offsets                  = CB::c_in5;
const uint32_t act_cb_row_major_bfloat16              = CB::c_in6;
const uint32_t act_cb_second_reader                   = CB::c_in7;
const uint32_t matmul_partials_cb                     = CB::c_intermed0;
const uint32_t tilize_mode_tilized_act_cb             = CB::c_intermed1;
const uint32_t untilize_mode_reblock_cb               = CB::c_intermed2;
const uint32_t out0_cb                                = CB::c_out0;


// TODO: Add namespace for utilities?
tuple<CBHandle, CBHandle> create_CBs_for_sharded_input_v2(
    tt_metal::Program &program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    DataFormat act_df,
    DataFormat weight_df,
    DataFormat tilized_act_df,
    DataFormat out_df,
    DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc
) {

    tt::DataFormat interm0_df = packer_l1_acc ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);

    CBHandle cb_sharded_act = 0;
    if (input.memory_config().is_sharded()) {
        uint32_t num_bytes_for_df = datum_size(act_df);
        auto shard_shape = input.shard_spec().value().shape;
        // 2D-sys-conv already has uint16_t indicies, TODO: do the same for 1D-sys-conv
        TT_ASSERT(shard_shape[0] <= (1<<16), "Shard height must be less than 2^16, read pattern indicies are uint16_t");
        CircularBufferConfig cb_sharded_act_config = CircularBufferConfig(shard_shape[0] * shard_shape[1] * num_bytes_for_df, {{sharded_act_cb, act_df}})
		    .set_page_size(sharded_act_cb, shard_shape[1] * num_bytes_for_df);
        // incoming data is the input cb instead of raw l1/dram addr
        cb_sharded_act_config.set_globally_allocated_address(*input.buffer());
        cb_sharded_act = tt_metal::CreateCircularBuffer(program, core, cb_sharded_act_config);

        if (weight_width_sliced) {
            // For 2D convs, each core creates and tilizes full input matrix then mcasts round robin style
            // Each core receives input into act_cb, so won't need a separate cb to receive
            // However, we need a separate cb to push ROW_MAJOR BFLOAT16 data for tilizing and configure act cb to be output df

            // num_cb0_tiles is double buffered
            CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * tilized_act_tile_size, {{act_cb, tilized_act_df}})
            .set_page_size(act_cb, tilized_act_tile_size);
            auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);

            // num_cb0_tilized_tiles is single buffered
            CircularBufferConfig cb_act_row_major_bfloat16_config = CircularBufferConfig(num_cb0_tilized_tiles * act_tile_size, {{act_cb_row_major_bfloat16, act_df}})
            .set_page_size(act_cb_row_major_bfloat16, act_tile_size);
            auto cb_act_row_major_bfloat16 = tt_metal::CreateCircularBuffer(program, core, cb_act_row_major_bfloat16_config);
        } else {
            // For 1D convs, locally create act matrix in act_cb, which is always ROW_MAJOR BFLOAT16
            // Then, tilize input in compute

            // Extra cb for second reader if we split act reads across two RISCs
            // In this case, the regular reader only does first half of reads along output block h
            if (split_reader) {
                num_cb0_tiles /= 2;

                CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * act_tile_size, {{act_cb_second_reader, act_df}})
                .set_page_size(act_cb_second_reader, act_tile_size);
                auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);
            }

            CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * act_tile_size, {{act_cb, act_df}})
            .set_page_size(act_cb, act_tile_size);
            auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);
        }
    } else {
        TT_ASSERT(false, "Input must be sharded!");
    }

    CircularBufferConfig cb_weight_config = CircularBufferConfig(num_cb1_tiles * weight_tile_size, {{weight_cb, weight_df}})
		.set_page_size(weight_cb, weight_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, core, cb_weight_config);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config = CircularBufferConfig(num_cb0_tilized_tiles * tilized_act_tile_size, {{tilize_mode_tilized_act_cb, tilized_act_df}})
		.set_page_size(tilize_mode_tilized_act_cb, tilized_act_tile_size);
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

    CBHandle cb_output = 0;
    if (untilize_out) {
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * interm0_single_tile_size, {{matmul_partials_cb, interm0_df}})
		    .set_page_size(matmul_partials_cb, interm0_single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        CircularBufferConfig cb_reblock_config = CircularBufferConfig(num_reblock_cb_tiles * out_tile_size, {{untilize_mode_reblock_cb, out_df}})
		    .set_page_size(untilize_mode_reblock_cb, out_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        CircularBufferConfig cb_output_config = CircularBufferConfig(num_writer_output_tiles * out_tile_size, {{out0_cb, out_df}})
		    .set_page_size(out0_cb, out_tile_size);
        if (output.is_sharded()) {
            cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
        }
        cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    } else {
        //Share buffer if same data format
        if(interm0_df == out_df) {
            CoreRangeSet cores(std::set<CoreRange>({core}));
            std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
                {out0_cb, out_df},
                {matmul_partials_cb, out_df}
            };
            CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * out_tile_size, cb_output_data_format_spec)
                .set_page_size(out0_cb, out_tile_size)
                .set_page_size(matmul_partials_cb, out_tile_size);
            if (output.is_sharded()) {
                cb_matmul_partials_config = cb_matmul_partials_config.set_globally_allocated_address(*output.buffer());
            }
            cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_matmul_partials_config);
        } else {
            //Separate buffer if not same data format
            CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * interm0_single_tile_size, {{matmul_partials_cb, interm0_df}})
                .set_page_size(matmul_partials_cb, interm0_single_tile_size);
            auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

            CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * out_tile_size, {{out0_cb, out_df}})
                .set_page_size(out0_cb, out_tile_size);
            if (output.is_sharded()) {
                cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
            }
            cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
        }
    }

    if (with_bias) {
        uint32_t bias_tile_size = tt_metal::detail::TileSize(bias_df);
        // bias input
        uint32_t bias_pagesize = bias_tile_size;
        CircularBufferConfig cb_bias_config = CircularBufferConfig(bias_ntiles * bias_pagesize, {{bias_cb, bias_df}})
		    .set_page_size(bias_cb, bias_pagesize);
        auto cb_bias = tt_metal::CreateCircularBuffer(program, core, cb_bias_config);

        log_debug("BIAS CBs: {} {} {}", bias_cb, bias_ntiles, bias_pagesize);
    }

    return {cb_sharded_act, cb_output};
}

operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_(const Tensor& a, const Tensor &b, const Shape& ashape, std::optional<const Tensor> bias, const std::optional<const Tensor> conv_reader_indices, vector<int> conv_params, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const OptimizedConvParallelizationConfig& parallelization_config, const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment, bool use_shallow_conv_variant, Tensor &output, DeviceComputeKernelConfig compute_kernel_config) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.get_layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_ASSERT(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_ASSERT(output_channels <= b.get_legacy_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntiles;
    uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntiles;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    DataFormat weight_df = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    DataFormat bias_df = has_bias ? tt_metal::datatype_to_dataformat_converter(bias.value().get_dtype()) : DataFormat::Float16_b;
    DataFormat tilized_act_df = out_df;

    log_debug("act_df: {}", act_df);
    log_debug("weight_df: {}", weight_df);
    log_debug("out_df: {}", out_df);
    log_debug("bias_df: {}", bias_df);


    // compute kernel config
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
            packer_l1_acc = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            packer_l1_acc = compute_kernel_config.packer_l1_acc;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    if (fp32_dest_acc_en and (out_subblock_h_ntiles * out_subblock_w_ntiles > 4)) {
        if (out_subblock_w_ntiles >= 4) {
            out_subblock_h_ntiles = 1;
            out_subblock_w_ntiles = find_max_block_size(out_subblock_w_ntiles, 4);
        } else {
            while (out_subblock_h_ntiles * out_subblock_w_ntiles > 4) {
                uint32_t div = find_max_divisor(out_subblock_h_ntiles, out_subblock_h_ntiles-1);
                out_subblock_h_ntiles = find_max_block_size(out_subblock_h_ntiles, div);
            }
        }
    }
    //assert(out_block_h_ntiles == act_block_h_ntiles); // TODO: fix output block sizing
    TT_ASSERT(out_block_h_ntiles >= act_block_h_ntiles, "Output block height (in # of tiles) should be greater than or equal to activation block height (in # of tiles)");

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_ASSERT(b.get_layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_ASSERT(b.get_legacy_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_ASSERT(b.get_legacy_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.get_legacy_shape()[2];
    uint32_t weight_matrix_width = b.get_legacy_shape()[3];
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    // Partitions conv inner dim into blocks to support sharding along this dim
    // TODO: Only 2D convs with sharded input use this, but we can uplift to support generically
    // TODO: Only updated variables which is affected, but there may be more that needs to account for this
    // TODO: Loop naming in reader, writer, and compute kernels could also be cleaned up
    // TODO: Can conv_act_c_blocks be same as num_blocks_act_w?
    auto shard_shape = a.shard_spec().value().shape;

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = num_cores_x * num_cores_y;
    assert(num_cores_x < 13);
    assert(num_cores_y < 10);
    uint32_t per_core_out_matrix_height_ntiles = p_config.per_core_out_matrix_height_ntiles;
    uint32_t per_core_out_matrix_width_ntiles = p_config.per_core_out_matrix_width_ntiles;

    // weight_width_sliced determines is 1d-sysarr-conv or 2d-sysarr-conv
    bool weight_width_sliced = per_core_out_matrix_width_ntiles < weight_matrix_width_ntiles;
    uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t input_channels_padded = 0;
    if(weight_width_sliced) {
        TT_FATAL(conv_act_c_blocks == num_cores_y, "Expected conv_act_blocks to be equal to height of grid");
        input_channels_padded = shard_shape[1] * num_cores_y;
    } else {
        input_channels_padded = shard_shape[1];
    }
    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    // check is for 16-byte alignment
    TT_FATAL(input_channels_padded % 16 == 0, "Expected input channels to be padded for 16 byte alignment in L1"); // TODO: For bfp16, check if its divisible by 8 not 16.
    // Always use split reader for first conv in resnet which has input channels = 16
    // TODO: Expose option to split readers for 1D convs to python?
    bool split_reader = use_shallow_conv_variant;
    if (split_reader) {
        TT_FATAL(block_config.act_block_h_ntiles % block_config.out_subblock_h_ntiles == 0, "Out_block_h must be divisible by out_subblock_h!");
        TT_FATAL((block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles) % 2 == 0, "Number of out_subblock_h must be divisible by 2 for split reader!");
    }
    Shape ashape_with_channels_padded = {ashape[0], ashape[1], ashape[2], input_channels_padded};
    uint32_t conv_act_size_h = ashape_with_channels_padded[1];
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    uint32_t weight_size_h = (uint32_t) conv_params[0];
    uint32_t weight_size_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];

    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] = optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(ashape_with_channels_padded, conv_params, out_block_h_ntiles, extra_padding_for_32B_alignment);
    assert(act_matrix_shape.size() == 3);
    assert(act_matrix_shape[0] == 1);
    uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t) act_matrix_shape[2];
    uint32_t act_matrix_height_unpadded = (uint32_t) act_matrix_shape_unpadded[1];
    uint32_t act_matrix_width_unpadded = (uint32_t) act_matrix_shape_unpadded[2];

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_ASSERT(bias.has_value());
        TT_ASSERT(bias.value().buffer() != nullptr);
        auto bias_shape_without_padding = bias.value().get_legacy_shape().without_padding();
        TT_ASSERT(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
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

    assert(act_matrix_height_ntiles % act_block_h_ntiles == 0);
    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(weight_matrix_width_ntiles % weight_block_w_ntiles == 0);
    assert(act_matrix_height_ntiles % out_block_h_ntiles == 0);

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;
    TT_ASSERT((act_block_w_datums == round_up(conv_act_size_c * weight_size_w, TILE_WIDTH)) || ((act_block_w_datums <= conv_act_size_c) && (conv_act_size_c % act_block_w_datums == 0)));

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

    // sanity check
    assert(num_blocks_output_w == num_blocks_weight_w);

    uint32_t out_block_h_datums = out_block_h_ntiles * TILE_HEIGHT;

    tt_metal::Program program = tt_metal::CreateProgram();
    //CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    //CoreRange core = {{0, 0}, {0, 0}};

    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
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
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles = bias.value().get_legacy_shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
    }

    //uint32_t conv_output_size_h = ((conv_act_size_h - weight_size_h + (2 * pad_h)) / stride_h) + 1;
    //uint32_t conv_output_size_w = ((conv_act_size_w - weight_size_w + (2 * pad_w)) / stride_w) + 1;

    auto [conv_output_size_h, conv_output_size_w] = optimized_conv_op_utils::compute_opt_conv_output_face_shape(conv_act_size_h, conv_act_size_w, weight_size_h, weight_size_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    std::map<string, string> reader_defines;

    if (act_matrix_height_unpadded < act_matrix_height) {
        reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
    }

    if (conv_act_c_blocks > 1) {
        reader_defines["ACT_W_OUTER_BLOCKS"] = "1";
    }

    uint32_t output_height_padded_to_tile_height = round_up(act_matrix_height_unpadded, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    assert(output_height_num_tiles <= act_matrix_height_ntiles);

    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes = out_block_h_ntiles * act_block_w_ntiles * tt::tt_metal::detail::TileSize(act_df);
    uint32_t dst_l1_weight_buffer_size_bytes = weight_block_h_ntiles * weight_block_w_ntiles * tt::tt_metal::detail::TileSize(weight_df);


    // For debug
    {
        log_debug(tt::LogOp, "multi_core_optimized_conv_sharded_v2_");
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
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_groups: {}", num_groups);
        log_debug("math_fidelity: {}", math_fidelity);
        log_debug("math_approx_mode: {}", math_approx_mode);
        log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);
        log_debug("packer_l1_acc: {}", packer_l1_acc);
    }

    uint32_t window_outer;
    uint32_t window_inner;
    if (weight_width_sliced) {
        window_outer = 1; // window_outer = 1 becasue all of filter window is processed in the inner loop
        window_inner = 3; // window_inner = 9 / 3, ie. read 3 width coalesced
    } else {
        window_outer = num_blocks_act_w; // window_outer
        window_inner = weight_size_h * weight_size_w / num_blocks_act_w; // window_inner
    }
    reader_defines["WINDOW_INNER"] = std::to_string(window_inner);
    log_debug("window_outer: {}, window_inner: {}", window_outer, window_inner);

    assert(weight_matrix_width_ntiles % per_core_out_matrix_width_ntiles == 0);
    assert(per_core_out_matrix_width_ntiles % weight_block_w_ntiles == 0);
    uint32_t num_blocks_weight_w_per_core = per_core_out_matrix_width_ntiles / weight_block_w_ntiles;
    if (not weight_width_sliced) {
        assert(num_blocks_weight_w_per_core == num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
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

    assert(act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0);
    uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / per_core_out_matrix_height_ntiles;
    assert(total_active_num_cores_per_weight_slice <= total_num_cores_per_weight_slice);
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - total_active_num_cores_per_weight_slice;
    uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
    if (weight_width_sliced) {
        assert(total_noop_cores == 0);
        assert(total_active_num_cores == total_num_cores);
    }

    if (has_bias) {
    assert(bias_ntiles % num_weight_slices_width == 0);
    assert(bias_ntiles == weight_matrix_width_ntiles);
    }
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    CoreRange all_cores(CoreCoord(0, 0), CoreCoord(num_cores_x - 1, num_cores_y - 1));
    assert(total_active_num_cores >= num_cores_x);
    uint32_t num_active_cores_x = num_cores_x;
    uint32_t num_active_cores_y_with_full_x = total_active_num_cores / num_cores_x;
    uint32_t num_active_cores_x_last_y =  total_active_num_cores % num_cores_x;
    assert((num_active_cores_x * num_active_cores_y_with_full_x) + num_active_cores_x_last_y == total_active_num_cores);

    std::set<CoreRange> all_active_cores_set;
    all_active_cores_set.insert(CoreRange(CoreCoord(0, 0), CoreCoord(num_active_cores_x - 1, num_active_cores_y_with_full_x - 1)));
    if (num_active_cores_x_last_y > 0) {
        all_active_cores_set.insert(CoreRange(CoreCoord(0, num_active_cores_y_with_full_x), CoreCoord(num_active_cores_x_last_y - 1, num_active_cores_y_with_full_x)));
    }
    CoreRangeSet all_active_cores(all_active_cores_set);
    std::set<CoreRange> noop_cores_set;
    if (total_noop_cores > 0) {
        assert(total_noop_cores == (num_cores_x - num_active_cores_x_last_y));
        noop_cores_set.insert(CoreRange(CoreCoord(num_active_cores_x_last_y, num_active_cores_y_with_full_x), CoreCoord(num_cores_x - 1, num_active_cores_y_with_full_x)));

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

    CoreRange mcast_sender_cores(top_left_core, top_left_core); // If single core, this kernel doesn't do mcasting
    CoreRangeSet mcast_receiver_cores{{}};
    uint32_t weights_mcast_sender_semaphore;
    uint32_t weights_mcast_receiver_semaphore;
    uint32_t act_mcast_sender_semaphore = 0;
    uint32_t act_mcast_receiver_semaphore = 0;
    std::vector<uint32_t> act_mcast_noc_y;
    // 2D mcast
    if (weight_width_sliced) {
        mcast_sender_cores = CoreRange(top_left_core, CoreCoord(0, num_cores_y - 1));
        mcast_receiver_cores = {{CoreRange(CoreCoord(1, 0), bottom_right_core)}};
        weights_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        weights_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    // 1D mcast
    } else {
        if (total_num_cores > 1) {
            std::set<CoreRange> mcast_receiver_set;
            if (num_cores_x > 1) {
                mcast_receiver_set.insert(CoreRange(CoreCoord(1, 0), CoreCoord(num_cores_x - 1, 0)));
            } if (num_cores_y > 1) {
                mcast_receiver_set.insert(CoreRange(CoreCoord(0, 1), bottom_right_core));
            }
            mcast_receiver_cores = mcast_receiver_set;
            weights_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
            weights_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        }
    }

    bool read_3x3_window_in_inner_loop = false;
    uint32_t num_weight_cb_tiles = weight_block_h_ntiles * weight_block_w_ntiles / conv_act_c_blocks;
    bool fully_buffer_weights = false;
    uint32_t num_act_cb_tiles = act_block_h_ntiles * act_block_w_ntiles / conv_act_c_blocks;
    // TODO: This flag should be set in kernel logic but need this for create_CB
    if (a.memory_config().is_sharded() and weight_size_h == 3 and weight_size_w == 3 and (stride_h == 1 or stride_h == 2) and weight_width_sliced) {
        // If conv_act_c_blocks > 1 and we have 2D conv with sharded input, we always read entire 3x3 window before pushing in reader/writer
        // TODO: Generalize this to not make this assumption
        read_3x3_window_in_inner_loop = true;
        num_weight_cb_tiles *= weight_size_h * weight_size_w;
        num_act_cb_tiles *= weight_size_h * weight_size_w;
    } else if (num_blocks_act_h_per_core > 1) {
        fully_buffer_weights = true;
    }
    uint32_t num_cb0_tilized_tiles = num_act_cb_tiles;

    if (fully_buffer_weights) {
        num_weight_cb_tiles *= window_outer;
    } else if (per_core_out_matrix_width_ntiles < 5 && per_core_out_matrix_height_ntiles < 22) {
        num_weight_cb_tiles = num_weight_cb_tiles * 2;
    }

    if (conv_act_size_c / conv_act_c_blocks < 160 && per_core_out_matrix_height_ntiles < 22) {
        num_act_cb_tiles = num_act_cb_tiles * 2; // double buffered
    }

    uint32_t writer_output_block_num_tiles = out_block_h_ntiles * weight_block_w_ntiles;

    // TODO: Moving this function call to after kernel logic causes pcc fails
    // There are additional CBs and semaphores created in 2D conv in kernel logic,
    // so does order of create_cb calls matter?
    auto [cb_sharded_act, cb_output] = create_CBs_for_sharded_input_v2(
            program,
            a,
            all_cores,
            num_act_cb_tiles, // row major act cb
            num_weight_cb_tiles, // tiled weight cb
            num_cb0_tilized_tiles, // tiled act cb
            writer_output_block_num_tiles, // math output cb
            weight_block_w_ntiles, // reblock cb
            writer_output_block_num_tiles, // writer output cb, double bufferred
            untilize_out,
            act_df,
            weight_df,
            tilized_act_df,
            out_df,
            bias_df,
            weight_width_sliced,
            output,
            bias_ntiles_per_core,
            has_bias,
            split_reader,
            fp32_dest_acc_en,
            packer_l1_acc
    );

    string reader_kernel;
    string compute_kernel;
    string writer_mcast_sender_kernel;
    string writer_mcast_receiver_kernel;
    bool tilize_in0 = true;

    compute_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/conv_bmm_tilize_col_major_out_blocks.cpp";
    // Input should always be sharded in this conv; always use reader kernel for input shard with halo and padding
    if (weight_size_h == weight_size_w and weight_size_w > 1 and (stride_h == 1 or stride_h == 2)) {
        // 2D conv
        if (weight_width_sliced) {
            assert(read_3x3_window_in_inner_loop == true);
            reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp";
            writer_mcast_sender_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
            writer_mcast_receiver_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
            act_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
            act_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

            act_mcast_noc_y.reserve(num_cores_y);
            for(uint32_t core_idx_y = 0; core_idx_y < num_cores_y; ++core_idx_y) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }

            // For 2D convs, pre-tilize input and round robin self-mcast tilized act matrix to other cores
            tilize_in0 = false;
        }
        // 1D conv
        else {
            assert(act_block_w_datums == round_up(conv_act_size_c * weight_size_w, TILE_WIDTH));
            reader_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp";
            if (split_reader) {
                writer_mcast_sender_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
                writer_mcast_receiver_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
            } else {
                writer_mcast_sender_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
                writer_mcast_receiver_kernel = "tt_eager/tt_dnn/op_library/conv/kernels/writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
            }
        }

        // Local L1 to store array for reader indices
        // All convs use packed uint16 indices, so each entry can be 2B (not 4)
        CircularBufferConfig cb_for_reader_indices_config = CircularBufferConfig(out_block_h_datums * 2, {{cb_for_reader_indices, tt::DataFormat::Float16_b}})
		    .set_page_size(cb_for_reader_indices, out_block_h_datums * 2);
        cb_for_reader_indices_config.set_globally_allocated_address(*conv_reader_indices.value().buffer());
        auto cb_for_reader_indices_id = tt_metal::CreateCircularBuffer(program, all_cores, cb_for_reader_indices_config);

        // Local L1 to store array for reader offsets
        // TODO: this is not used in 2D-sys-conv, remove also from 1D-sys-conv
        CircularBufferConfig cb_for_reader_offsets_config = CircularBufferConfig(weight_size_h * weight_size_w * 4, {{cb_for_reader_offsets, tt::DataFormat::Float16_b}})
		    .set_page_size(cb_for_reader_offsets, 4);
        auto cb_for_reader_offsets_id = tt_metal::CreateCircularBuffer(program, all_cores, cb_for_reader_offsets_config);
    } else {
        TT_ASSERT(false, "Sharded input not supported for this conv yet!");
    }

    std::vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_rt_args;
    std::vector<uint32_t> writer_compile_time_args;

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    uint32_t act_block_w_extra_align_bytes = (round_up(conv_act_size_c * weight_size_w, TILE_WIDTH) - (conv_act_size_c * weight_size_w)) * a.element_size();
    reader_compile_time_args = {(uint32_t)
        (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
        (uint32_t) stride_h,
        (uint32_t) stride_w,
        (uint32_t) conv_act_size_w,
        (uint32_t) conv_output_size_w,
        (uint32_t) conv_act_c_read_bytes,

        (uint32_t) window_outer,
        (uint32_t) window_inner,
        (uint32_t) act_block_h_datums,
        (uint32_t) act_block_num_tiles / conv_act_c_blocks,
        (uint32_t) weight_size_w,
        (uint32_t) conv_act_size_w + (2 * pad_w),
        (uint32_t) act_block_w_extra_align_bytes, // only used for 1d systolic variant
        (uint32_t) weight_size_h,
        (uint32_t) num_blocks_act_h_per_core,
        };

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

    if (!tilize_in0) {
        compute_defines["PRE_TILIZE"] = "1";
    }

    if (split_reader) {
        reader_defines["SPLIT_READER"] = "1";
        compute_defines["SPLIT_READER"] = "1";
    }

    if (packer_l1_acc) {
        compute_defines["PACKER_L1_ACC"] = "1";
    }

    writer_compile_time_args = {
        (uint32_t) (dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
        out0_cb,
        weight_cb,
        bias_cb,
        (uint32_t) (bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0)),
    };
    if (split_reader) {
        std::vector<uint32_t> split_reader_args = {
            (uint32_t) act_block_h_datums,
            (uint32_t) act_block_num_tiles / conv_act_c_blocks,
            (uint32_t) conv_act_c_read_bytes,
            (uint32_t) weight_size_w * conv_act_c_read_bytes, // coalesced_read_bytes
            (uint32_t) (conv_act_size_w + 2 * pad_w) * conv_act_c_read_bytes, // window_outer_offset
            (uint32_t) act_block_w_extra_align_bytes, // only used for 1d systolic variant
        };
        writer_compile_time_args.insert(writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    }


    uint32_t in0_block_w = act_block_w_ntiles / conv_act_c_blocks;
    uint32_t in0_block_num_tiles = act_block_num_tiles / conv_act_c_blocks;
    uint32_t in0_subblock_num_tiles = act_subblock_num_tiles / conv_act_c_blocks;
    uint32_t in1_block_num_tiles = weight_block_num_tiles / conv_act_c_blocks;
    uint32_t in0_num_blocks_w = num_blocks_act_w * conv_act_c_blocks; // Fold outer c_block loop together with weight_block_num_tiles = 9
    if (read_3x3_window_in_inner_loop) {
        const uint32_t window_size = weight_size_h * weight_size_w;
        in0_block_w *= window_size;
        in0_block_num_tiles *= window_size;
        in0_subblock_num_tiles *= window_size;
        in1_block_num_tiles *= window_size;
        in0_num_blocks_w /= window_size;
    }

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        act_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        in1_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h_per_core,
        in0_num_blocks_w,
        num_blocks_weight_w_per_core,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        tilize_in0,
        untilize_out,

        bias_ntiles_per_core
    };

    auto writer_mcast_noc = NOC::NOC_0;
    auto reader_noc = writer_mcast_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
    auto writer_mcast_sender_id = CreateKernel(
    program,
    writer_mcast_sender_kernel,
    mcast_sender_cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = writer_mcast_noc,
        .compile_args = writer_compile_time_args,
        .defines = writer_mcast_sender_defines});

    KernelHandle writer_mcast_receiver_id;
    if (total_num_cores > 1) {
        writer_mcast_receiver_id = CreateKernel(
        program,
        writer_mcast_receiver_kernel,
        mcast_receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = writer_mcast_noc,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});
    }

    auto reader_id = CreateKernel(
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
    auto compute_id = CreateKernel(
        program,
        compute_kernel,
        all_active_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    if (total_noop_cores > 0) {
        auto compute_id = CreateKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        noop_cores, ComputeConfig{});
    }

    vector<KernelHandle> reader_ids;
    vector<KernelHandle> writer_ids;
    for(uint32_t core_i = 0; core_i < total_num_cores; core_i++) {
        uint32_t core_x_i = core_i % num_cores_x;
        uint32_t core_y_i = core_i / num_cores_x;
        CoreRange core(CoreCoord(core_x_i, core_y_i), CoreCoord(core_x_i, core_y_i));
        bool noop_core = false;
        for (const auto & noop_core_range : noop_cores.ranges()) {
            if (noop_core_range.contains(core)) {
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
        uint32_t out_start_tile_id = (act_slice_i * per_core_out_matrix_height_ntiles * weight_matrix_width_ntiles) + (weight_slice_i * per_core_out_matrix_width_ntiles);
        uint32_t out_start_tile_id_h = act_slice_i * per_core_out_matrix_height_ntiles;
        uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
        uint32_t bias_tile_offset = weight_slice_i * per_core_out_matrix_width_ntiles;
        if (has_bias) {
            assert(bias_tile_offset < bias_ntiles);
        }

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

        // If 2D, same image specs across a row
        uint32_t start_stick = weight_width_sliced ? core_x_i * out_block_h_datums : core_i * out_block_h_datums;
        uint32_t end_stick = start_stick + out_block_h_datums;

        ShardingConfig sharding_config = get_specs_for_sharding_partition(start_stick, end_stick, conv_act_size_h, conv_act_size_w, weight_size_w, pad_h, pad_w);
        uint32_t first_partial_right_aligned_row_width = sharding_config.first_partial_right_aligned_row_width;
        uint32_t skip_after_partial_right_aligned_row = sharding_config.skip_after_partial_right_aligned_row;
        uint32_t first_partial_image_num_rows = sharding_config.first_partial_image_num_rows;
        uint32_t skip_after_first_partial_image_row = sharding_config.skip_after_first_partial_image_row;
        uint32_t num_full_images = sharding_config.num_full_images;
        uint32_t skip_after_full_image = sharding_config.skip_after_full_image;
        uint32_t last_partial_image_num_rows = sharding_config.last_partial_image_num_rows;
        uint32_t last_partial_left_aligned_row_width = sharding_config.last_partial_left_aligned_row_width;

        if (weight_width_sliced) {
            auto shard_shape = a.shard_spec().value().shape;
            uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
            CoreCoord bottom_core = {(std::size_t) core_x_i, (std::size_t) num_cores_y - 1};
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

            bool reader_is_noc_0 = reader_noc == NOC::NOC_0;
            uint32_t act_mcast_dest_noc_start_x = bottom_core_physical.x;
            uint32_t act_mcast_dest_noc_start_y = reader_is_noc_0 ? top_left_core_physical.y : bottom_core_physical.y;
            uint32_t act_mcast_dest_noc_end_x = bottom_core_physical.x;
            uint32_t act_mcast_dest_noc_end_y = reader_is_noc_0 ? bottom_core_physical.y : top_left_core_physical.y;
            reader_rt_args = {
                conv_act_size_w,
                conv_act_size_h,
                weight_size_h,
                weight_size_w,
                num_blocks_act_h_per_core,
                act_block_h_datums,
                in0_block_num_tiles,
                conv_act_c_blocks,

                // Specs for reader indices
                first_partial_right_aligned_row_width,
                skip_after_partial_right_aligned_row,
                first_partial_image_num_rows,
                skip_after_first_partial_image_row,
                num_full_images,
                skip_after_full_image,
                last_partial_image_num_rows,
                last_partial_left_aligned_row_width,

                // Specs for reader offsets
                window_outer, // window_outer
                window_inner, // window_inner = 9 / 3, ie. read 3 width coalesced

                (uint32_t) noop_core,

                // mcast args
                act_mcast_dest_noc_start_x,
                act_mcast_dest_noc_start_y,
                act_mcast_dest_noc_end_x,
                act_mcast_dest_noc_end_y,
                num_cores_y - 1,
                num_cores_y - 1,
                act_mcast_sender_semaphore,
                act_mcast_receiver_semaphore,
                in0_block_num_tiles * tilized_act_tile_size, // act_mcast_sender_size_bytes
                core_y_i, // act_mcast_sender_id (goes down the column)
                (uint32_t) bottom_core_physical.x, // act_mcast_sender_noc_x
            };
            reader_rt_args.insert(reader_rt_args.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end()); // act_mcast_sender_noc_y
        } else {
            reader_rt_args = {
                conv_act_size_w,
                conv_act_size_h,
                weight_size_h,
                weight_size_w,
                num_blocks_act_h_per_core,
                // act_block_h_datums,
                // act_block_num_tiles / conv_act_c_blocks,

                // Specs for reader indices
                first_partial_right_aligned_row_width,
                skip_after_partial_right_aligned_row,
                first_partial_image_num_rows,
                skip_after_first_partial_image_row,
                num_full_images,
                skip_after_full_image,
                last_partial_image_num_rows,
                last_partial_left_aligned_row_width,

                // Specs for reader offsets
                window_outer, // window_outer
                window_inner, // window_inner

                (uint32_t) noop_core
            };
        }

        SetRuntimeArgs(
            program, reader_id, core,
            reader_rt_args
        );
        reader_ids.push_back(reader_id);

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
            in1_block_num_tiles,
            conv_act_c_blocks,
            weight_block_h_ntiles / conv_act_c_blocks,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

            // bias
            bias_ntiles_per_core,
            bias_tile_offset,

            (uint32_t) noop_core
        };

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
            cb_sharded_act=cb_sharded_act,
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

        // Reader config indices is an optional static sharded tensor, so no need to update address
        TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 4);
        TT_ASSERT(output_tensors.size() == 1);

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();
        auto src_a_is_sharded = input_tensors.at(0).memory_config().is_sharded();

        auto dst_buffer = output_tensors.at(0).buffer();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        for(uint32_t core_i = 0; core_i < total_num_cores; core_i++) {
            uint32_t core_x_i = core_i % num_cores_x;
            uint32_t core_y_i = core_i / num_cores_x;
            CoreCoord core = {core_x_i, core_y_i};

            if (!src_a_is_sharded) {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_i], core);
                runtime_args[0] = src_buffer_a->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_i], core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[1] = src_buffer_b->address();
                if (has_bias) {
                    auto src_buffer_c = optional_input_tensors.at(0).value().buffer();
                    TT_ASSERT(src_buffer_c != nullptr);
                    runtime_args[2] = src_buffer_c->address();
                }
            }
        }

        if (src_a_is_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_act, *src_buffer_a);
        }

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
