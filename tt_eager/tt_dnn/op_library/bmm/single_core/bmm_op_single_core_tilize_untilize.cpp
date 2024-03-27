// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {
namespace tt_metal {

Tensor bmm_tilize_untilize(const Tensor& a, const Tensor& b, const Tensor& bias, DataType out_dt,
                           uint32_t a_height_nblocks, uint32_t a_width_nblocks, uint32_t b_width_nblocks,
                           uint32_t a_block_height_ntiles, uint32_t a_block_width_ntiles, uint32_t b_block_width_ntiles,
                           uint32_t out_subblock_height_ntiles, uint32_t out_subblock_width_ntiles,
                           bool tilize_in0, bool untilize_out, bool has_bias,
                           std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::LoFi, true, false, false);

    // NOTE: Currently only single core implementation exists.
    return operation::run(BMMTilizeUntilize {
                            out_dt,
                            a_height_nblocks, a_width_nblocks, b_width_nblocks,
                            a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                            out_subblock_height_ntiles, out_subblock_width_ntiles,
                            tilize_in0, untilize_out,
                            has_bias, kernel_config_val},
                          {a, b, bias},
                          {}).at(0);
}

void create_cb_bmm_single_core_tilize_untilize(Program &program,
                                               CoreRange core,
                                               DataFormat in0_df,
                                               DataFormat in1_df,
                                               DataFormat bias_df,
                                               DataFormat out_df,
                                               uint32_t in0_block_w,
                                               uint32_t in0_block_h,
                                               uint32_t in1_block_w,
                                               uint32_t bias_width_ntiles,
                                               uint32_t in0_tile_nbytes,
                                               uint32_t in1_tile_nbytes,
                                               uint32_t bias_tile_nbytes,
                                               uint32_t out_tile_nbytes,
                                               bool tilize_in0 = true,
                                               bool untilize_out = true,
                                               bool has_bias = false) {
    // buffer indices
    uint32_t in0_cb                                 = CB::c_in0;
    uint32_t in1_cb                                 = CB::c_in1;
    uint32_t bias_cb                                = CB::c_in2;
    uint32_t matmul_partials_cb                     = CB::c_intermed0;
    uint32_t tilize_mode_tilized_in0_cb             = CB::c_intermed1;
    uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
    uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
    uint32_t out_for_bias_cb                        = CB::c_intermed4;
    uint32_t out_cb                                 = CB::c_out0;

    const uint32_t cb0_ntiles = in0_block_h * in0_block_w * 2;  // double buffer
    const uint32_t cb1_ntiles = in0_block_w * in1_block_w * 2;   // double buffer
    const uint32_t out_ntiles = in0_block_h * in1_block_w;

    // in0 (RM/TM)
    CircularBufferConfig cb_in0_config = CircularBufferConfig(cb0_ntiles * in0_tile_nbytes, {{in0_cb, in0_df}})
		.set_page_size(in0_cb, in0_tile_nbytes);
    auto cb_in0 = CreateCircularBuffer(program, core, cb_in0_config);

    // in1 (TM)
    CircularBufferConfig cb_in1_config = CircularBufferConfig(cb1_ntiles * in1_tile_nbytes, {{in1_cb, in1_df}})
		.set_page_size(in1_cb, in1_tile_nbytes);
    auto cb_in1 = CreateCircularBuffer(program, core, cb_in1_config);

    if (tilize_in0) {
        // in0 (RM -> TM)
        CircularBufferConfig cb_src0_tilized_config = CircularBufferConfig(cb0_ntiles * in0_tile_nbytes, {{tilize_mode_tilized_in0_cb, in0_df}})
		    .set_page_size(tilize_mode_tilized_in0_cb, in0_tile_nbytes);
        auto cb_src0_tilized = CreateCircularBuffer(program, core, cb_src0_tilized_config);
    }

    if (untilize_out) {
        // partial sums
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(out_ntiles * out_tile_nbytes, {{matmul_partials_cb, out_df}})
		    .set_page_size(matmul_partials_cb, out_tile_nbytes);
        auto cb_matmul_partials = CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // final partial sums
        CircularBufferConfig cb_final_matmul_partials_config = CircularBufferConfig(out_ntiles * out_tile_nbytes, {{untilize_mode_final_matmul_partials_cb, out_df}})
		    .set_page_size(untilize_mode_final_matmul_partials_cb, out_tile_nbytes);
        auto cb_final_matmul_partials = CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // to reorganize output blocks to fill the whole "per core output block width"
        // in1_block_w depicts single row of tiles
        CircularBufferConfig cb_reblock_config = CircularBufferConfig(in1_block_w * out_tile_nbytes, {{untilize_mode_reblock_cb, out_df}})
		    .set_page_size(untilize_mode_reblock_cb, out_tile_nbytes);
        auto cb_reblock = CreateCircularBuffer(program, core, cb_reblock_config);

        // output
        CircularBufferConfig cb_output_config = CircularBufferConfig(out_ntiles * out_tile_nbytes, {{out_cb, out_df}})
		    .set_page_size(out_cb, out_tile_nbytes);
        auto cb_output = CreateCircularBuffer(program, core, cb_output_config);
    } else {
        // partials and output share same memory
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec = {
            {out_cb, out_df},
            {matmul_partials_cb, out_df}
        };
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(out_ntiles * out_tile_nbytes, output_cb_data_format_spec)
		    .set_page_size(out_cb, out_tile_nbytes)
            .set_page_size(matmul_partials_cb, out_tile_nbytes);
        CoreRangeSet cores(std::set<CoreRange>{core});
        auto cb_matmul_partials = CreateCircularBuffer(program, cores, cb_matmul_partials_config);
    }

    if (has_bias) {
        tt_metal::CircularBufferConfig cb_in_bias_config = tt_metal::CircularBufferConfig(bias_width_ntiles * bias_tile_nbytes, {{bias_cb, bias_df}})
		    .set_page_size(bias_cb, bias_tile_nbytes);
        auto cb_in_bias = tt_metal::CreateCircularBuffer(program, core, cb_in_bias_config);

        tt_metal::CircularBufferConfig cb_out_for_bias_config = tt_metal::CircularBufferConfig(out_ntiles * out_tile_nbytes, {{out_for_bias_cb, out_df}})
		    .set_page_size(out_for_bias_cb, out_tile_nbytes);
        auto cb_out_for_bias = tt_metal::CreateCircularBuffer(program, core, cb_out_for_bias_config);
    }
}

operation::ProgramWithCallbacks bmm_single_core_tilize_untilize(
                                    const Tensor &in0,       // activations
                                    const Tensor &in1,       // weights
                                    const Tensor &bias,      // optional bias
                                    DataType out_dt,
                                    uint32_t in0_height_nblocks,
                                    uint32_t in0_width_nblocks,
                                    uint32_t in1_width_nblocks,
                                    uint32_t in0_block_height_ntiles,
                                    uint32_t in0_block_width_ntiles,
                                    uint32_t in1_block_width_ntiles,
                                    uint32_t out_subblock_height_ntiles,
                                    uint32_t out_subblock_width_ntiles,
                                    bool tilize_in0,
                                    bool untilize_out,
                                    bool has_bias,
                                    Tensor &out,
                                    DeviceComputeKernelConfig compute_kernel_config) {

    uint32_t in0_batch = in0.get_legacy_shape()[0];
    uint32_t in0_channel = in0.get_legacy_shape()[1];
    uint32_t in0_height = in0.get_legacy_shape()[2];
    uint32_t in0_width = in0.get_legacy_shape()[3];
    uint32_t in1_batch = in1.get_legacy_shape()[0];
    uint32_t in1_channel = in1.get_legacy_shape()[1];
    uint32_t in1_height = in1.get_legacy_shape()[2];
    uint32_t in1_width = in1.get_legacy_shape()[3];

    // input matrix shape checks
    TT_FATAL(in0_batch == 1, "Supports only batch = 1");
    TT_FATAL(in1_batch == in0_batch, "Batch dimension needs to match for two inputs");
    TT_FATAL(in0_channel == in1_channel, "Channel dimension needs to match for two inputs");
    TT_FATAL(in0_width == in1_height, "Input matrices should be compatible for multiplication");
    if (has_bias) {
        TT_FATAL(bias.get_legacy_shape()[3] == in1.get_legacy_shape()[3], "Bias shape mismatch");
    }

    // tile size checks
    TT_FATAL(in0_height % constants::TILE_HEIGHT == 0, "Input tensor in0 height needs to be divisible by TILE_HEIGHT");
    TT_FATAL(in1_height % constants::TILE_HEIGHT == 0, "Input tensor in1 height needs to be divisible by TILE_HEIGHT");
    TT_FATAL(in0_width % constants::TILE_WIDTH == 0, "Input tensor in0 width needs to be divisible by TILE_WIDTH");
    TT_FATAL(in1_width % constants::TILE_WIDTH == 0, "Input tensor in1 width needs to be divisible by TILE_WIDTH");
    if (has_bias) {
        TT_FATAL(bias.get_legacy_shape()[2] % constants::TILE_HEIGHT == 0);
        TT_FATAL(bias.get_legacy_shape()[3] % constants::TILE_WIDTH == 0);
    }

    // device compatibility checks
    TT_FATAL(in0.storage_type() == StorageType::DEVICE and in1.storage_type() == StorageType::DEVICE, "Operands need to be on the device!");
    TT_FATAL(in0.device() == in1.device(), "Operands need to be on the same device!");
    TT_FATAL(in0.buffer() != nullptr && in1.buffer() != nullptr, "Operands need to have buffers allocated on the device!");
    if (has_bias) {
        TT_FATAL(bias.storage_type() == StorageType::DEVICE);
        TT_FATAL(bias.device() == in0.device());
    }

    // input data type and formats
    const auto in0_dt = in0.get_dtype();
    const auto in1_dt = in1.get_dtype();
    const auto in0_df = datatype_to_dataformat_converter(in0_dt);
    const auto in1_df = datatype_to_dataformat_converter(in1_dt);

    // input data format checks
    TT_FATAL(in0_dt == DataType::BFLOAT16 || (in0_dt == DataType::BFLOAT8_B && !tilize_in0),
              "in0 only supports BFLOAT16 and BFLOAT8_B data types for now");
    TT_FATAL(in1_dt == DataType::BFLOAT16 || in1_dt == DataType::BFLOAT8_B, "in1 only supports BFLOAT16 and BFLOAT8_B formats for now!");
    if (has_bias) {
        TT_FATAL(bias.get_dtype() == DataType::BFLOAT16 || bias.get_dtype() == DataType::BFLOAT8_B);
    }

    // output data format
    const auto out_df = datatype_to_dataformat_converter(out_dt);

    // out dt checks
    TT_FATAL(!untilize_out || (untilize_out && out_dt == DataType::BFLOAT16));

    if (has_bias) {
        TT_FATAL(!untilize_out, "Untilize is not supported with bias");
    }

    // // TODO (AS): Certain mixed-prec cases do not currently work. Assert them out.
    // if (!(in0_dt == out_dt && in0_dt == in1_dt && in0_dt == DataType::BFLOAT16) && (tilize_in0 || untilize_out)) {
    //     TT_FATAL(false, "TODO: Cases to be debugged");
    // }

    const auto in0_tile_nbytes = tile_size(in0_df);
    const auto in1_tile_nbytes = tile_size(in1_df);
    const auto out_tile_nbytes = tile_size(out_df);

    Buffer *src0_dram_buffer = in0.buffer();
    Buffer *src1_dram_buffer = in1.buffer();

    TT_FATAL(src0_dram_buffer->size() % in0_tile_nbytes == 0, "Buffer size of tensor in0 must be multiple of tile size");
    TT_FATAL(src1_dram_buffer->size() % in1_tile_nbytes == 0, "Buffer size of tensor in1 must be multiple of tile size");

    Device *device = in0.device();
    CoreCoord core = {0, 0};

    Program program{};

    CoreRange core_range(core, core);

    Buffer *dst_dram_buffer = out.buffer();
    TT_FATAL(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // compute kernel config
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = in0_df == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    // TODO [AS]: support non-tile multiple shapes
    // Convert tensor dims to tile dims
    uint32_t in0_height_ntiles = in0_height / constants::TILE_HEIGHT;   // == in0_height_nblocks * in0_block_height_ntiles
    uint32_t in0_width_ntiles = in0_width / constants::TILE_WIDTH;      // == in0_width_nblocks * in0_block_width_ntiles
    uint32_t in1_width_ntiles = in1_width / constants::TILE_WIDTH;      // == in1_width_nblocks * in1_block_width_ntiles
    // Ensure the size arguments match the input tensors
    TT_FATAL(in0_height_ntiles == in0_height_nblocks * in0_block_height_ntiles, "Mismatch in tensor in0 height!");
    TT_FATAL(in0_width_ntiles == in0_width_nblocks * in0_block_width_ntiles, "Mismatch tensor in0 width!");
    TT_FATAL(in1_width_ntiles == in1_width_nblocks * in1_block_width_ntiles, "Mismatch in tensor in1 width! in1_width_ntiles = {}, in1_width_nblocks = {}, in1_block_width_ntiles = {}", in1_width_ntiles, in1_width_nblocks, in1_block_width_ntiles);

    // in0
    uint32_t in0_dram_addr = src0_dram_buffer->address();
    // in0 block info
    uint32_t in0_subblock_h = out_subblock_height_ntiles;
    uint32_t in0_num_blocks_w = in0_width_nblocks;
    uint32_t in0_num_blocks_h = in0_height_nblocks;
    uint32_t in0_block_w = in0_width_ntiles / in0_num_blocks_w;
    uint32_t in0_block_h = in0_height_ntiles / in0_num_blocks_h;
    uint32_t in0_block_num_tiles = in0_block_h * in0_block_w;
    uint32_t in0_num_subblocks = in0_block_h / in0_subblock_h;
    uint32_t in0_subblock_num_tiles = in0_subblock_h * in0_block_w;
    TT_FATAL(in0_block_h % out_subblock_height_ntiles == 0);

    // in1
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    // in1 block info
    uint32_t in1_subblock_w = out_subblock_width_ntiles;
    uint32_t in1_num_blocks_w = in1_width_nblocks;
    uint32_t in1_num_blocks_h = in0_width_nblocks;
    uint32_t in1_block_w = in1_block_width_ntiles;
    uint32_t in1_num_subblocks = in1_block_w / in1_subblock_w;
    uint32_t in1_block_h = in0_block_w;
    uint32_t in1_block_num_tiles = in1_block_w * in1_block_h;
    TT_FATAL(in1_block_w % out_subblock_width_ntiles == 0);

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_subblock_ntiles = out_subblock_height_ntiles * out_subblock_width_ntiles;
    TT_FATAL(out_subblock_ntiles <= 8, "Subblock can have at most 8 tiles to fit computed intermediates in dst[half]");

    // bias
    uint32_t bias_addr = 0;
    uint32_t bias_ntiles_w = 0;
    uint32_t bias_tile_nbytes = 0;
    uint32_t bias_log2_of_pagesize = 0;
    DataFormat bias_df = in0_df;
    if (has_bias) {
        bias_addr = bias.buffer()->address();
        bias_ntiles_w = bias.get_legacy_shape()[3] / constants::TILE_WIDTH;
        bias_df = datatype_to_dataformat_converter(bias.get_dtype());
        bias_tile_nbytes = tile_size(bias_df);
        bias_log2_of_pagesize = (uint32_t) std::log2((float) bias_tile_nbytes);
    }

    {   // debug
        // in0
        log_debug("in0_dram_addr: {}", in0_dram_addr);
        log_debug("in0_height_ntiles: {}", in0_height_ntiles);
        log_debug("in0_width_ntiles: {}", in0_width_ntiles);
        log_debug("in0_subblock_h: {}", in0_subblock_h);
        log_debug("in0_num_blocks_w: {}", in0_num_blocks_w);
        log_debug("in0_num_blocks_h: {}", in0_num_blocks_h);
        log_debug("in0_block_w: {}", in0_block_w);
        log_debug("in0_block_h: {}", in0_block_h);
        log_debug("in0_block_num_tiles: {}", in0_block_num_tiles);
        log_debug("in0_num_subblocks: {}", in0_num_subblocks);
        log_debug("in0_subblock_num_tiles: {}", in0_subblock_num_tiles);
        // in1
        log_debug("in1_dram_addr: {}", in1_dram_addr);
        log_debug("in1_width_ntiles: {}", in1_width_ntiles);
        log_debug("in1_subblock_w: {}", in1_subblock_w);
        log_debug("in1_num_subblocks: {}", in1_num_subblocks);
        log_debug("in1_block_num_tiles: {}", in1_block_num_tiles);
        log_debug("in1_block_w: {}", in1_block_w);
        log_debug("in1_block_h: {}", in1_block_h);
        log_debug("in1_num_blocks_w: {}", in1_num_blocks_w);
        log_debug("in1_num_blocks_h: {}", in1_num_blocks_h);
        // bias
        log_debug("has_bias: {}", has_bias);
        log_debug("bias_addr: {}", bias_addr);
        log_debug("bias_ntiles_w: {}", bias_ntiles_w);
        // out
        log_debug("out_dram_addr: {}", out_dram_addr);
        log_debug("out_subblock_height_ntiles: {}", out_subblock_height_ntiles);
        log_debug("out_subblock_width_ntiles: {}", out_subblock_width_ntiles);
        log_debug("out_subblock_ntiles: {}", out_subblock_ntiles);
        // extra
        log_debug("out size: {}", dst_dram_buffer->size());
        log_debug("out pagesize: {}", dst_dram_buffer->page_size());
        // data formats
        log_debug("in0_df: {}", in0_df);
        log_debug("in1_df: {}", in1_df);
        log_debug("bias_df: {}", bias_df);
        log_debug("out_df: {}", out_df);
    }

    create_cb_bmm_single_core_tilize_untilize(program,
                                              core_range,
                                              in0_df,
                                              in1_df,
                                              bias_df,
                                              out_df,
                                              in0_block_w,
                                              in0_block_h,
                                              in1_block_w,
                                              bias_ntiles_w,
                                              in0_tile_nbytes,
                                              in1_tile_nbytes,
                                              bias_tile_nbytes,
                                              out_tile_nbytes,
                                              tilize_in0,
                                              untilize_out,
                                              has_bias);

    // defines for Bias
    std::map<string, string> all_defines;
    if (has_bias) {
        all_defines["FUSE_BIAS"] = "1";
    }

    // Reader kernel
    std::string reader_kernel;
    std::vector<uint32_t> reader_rt_args;
    if (tilize_in0) {
        // in0 is row major, in1 is tiled
        // NOTE: this only makes sense for non-tile-shared datatypes for in0
        reader_kernel = "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_single_core_tilize_untilize.cpp";
        reader_rt_args = {
            // in0
            in0_dram_addr,
            in0_block_h,
            in0_num_blocks_h,
            in0_num_blocks_w,
            in0_block_num_tiles,
            in0_block_h * constants::TILE_HEIGHT,               // in0_block_nrows,
            in0.element_size(),                                         // UNUSED
            in0_width * in0.element_size(),                             // page size (size of an in0 row)
            in0_block_w * constants::TILE_WIDTH * in0.element_size(),   // size of partial row to fit within a block width
            // in1
            in1_dram_addr,
            in1_block_h,
            in1_block_w,
            in1_num_blocks_w,
            in1_block_num_tiles,
            in1_width_ntiles,
            in1_width_ntiles * in1_block_h,
            in1_block_w,
            0,                                              // unused
            0,                                              // unused
            0,                                              // unused
            0,                                              // unused
            0,                                              // unused
            bias_addr,
            bias_ntiles_w,
            bias_log2_of_pagesize,
            bias_tile_nbytes
        };
    } else {
        // in0 is tiled, in1 is tiled
        reader_kernel = "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_single_core.cpp";
        reader_rt_args = {
            // in0
            in0_dram_addr,
            in0_num_blocks_h,
            in0_num_blocks_w,
            1,                      // in0_stride_w
            in0_width_ntiles,       // in0_stride_h
            in0_block_w,            // in0_next_block_stride
            in0_block_w,            // in0_block_w
            in0_block_h,            // in0_block_h
            in0_block_num_tiles,    // in0_block_num_tiles
            // in1
            in1_dram_addr,          // in1_addr
            in1_num_blocks_w,
            0,                      // in1_start_tile_id
            1,                      // in1_stride_w
            in1_width_ntiles,       // in1_stride_h
            in0_block_w * in1_width_ntiles, // in1_next_block_stride UNUSED
            in1_block_w,                    // in1_block_w
            in1_block_h,                    // in1_block_h
            in1_block_num_tiles,            // in1_block_num_tiles
            in0_width_ntiles * in0_block_h, // in0_next_block_stride_h,
            in0_block_w,                    // in0_next_block_stride_w,
            in1_width_ntiles * in1_block_h, // in1_next_block_stride_h,
            in1_block_w,                    // in1_next_block_stride_w
            bias_addr,
            bias_ntiles_w,
            bias_log2_of_pagesize,
            bias_tile_nbytes
        };
    }
    auto reader_id = CreateKernel(
        program,                            // program
        reader_kernel,                      // file name
        core_range,                         // core
        tt_metal::ReaderDataMovementConfig({}, all_defines)
    );

    // number of data elements along height of an in0 block
    uint32_t in0_block_h_data = in0_height / in0_num_blocks_h;

    // Writer kernel
    std::string writer_kernel;
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        // out is row major
        writer_kernel = "tt_eager/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            in0_block_h_data,
            in1_block_w * constants::TILE_WIDTH * out.element_size(), // block_row_size
            1,                                                  // batch
            in0_num_blocks_h,
            in1_num_blocks_w,
            in1_width * out.element_size(),   // output_row_size
            in1_block_w * constants::TILE_WIDTH * out.element_size(), // last block_row_size (same as block row size)
            in0_height,
            0,
            0
        };
    } else {
        // out is tiled
        writer_kernel = "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/writer_bmm_single_core_tiled.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,                                              // UNUSED
            1,                                              // out_stride_w
            in1_width_ntiles,                               // out_stride_h
            out_subblock_width_ntiles,                      // out_next_subblock_stride_w
            out_subblock_height_ntiles * in1_width_ntiles,  // out_next_subblock_stride_h
            out_subblock_width_ntiles,                      // out_subblock_w
            out_subblock_height_ntiles,                     // out_subblock_h
            out_subblock_ntiles,                            // out_subblock_tile_count
            in1_width_ntiles / out_subblock_width_ntiles,   // out_num_subblocks_w
            in0_height_ntiles / out_subblock_height_ntiles // out_num_subblocks_h
        };
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
        (uint32_t) fp32_dest_acc_en
    };

    auto writer_id = CreateKernel(
        program,                        // program
        writer_kernel,                  // file name
        core_range,                     // core
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, all_defines)
    );

    // Compute kernel
    std::string compute_kernel = "tt_eager/tt_dnn/kernels/compute/bmm_tilize_untilize.cpp";
    std::vector<uint32_t> compute_comptime_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in0_subblock_h,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_block_w,
        in0_num_blocks_h,
        in0_num_blocks_w,
        in1_num_blocks_w,
        out_subblock_height_ntiles,
        out_subblock_width_ntiles,
        out_subblock_ntiles,
        tilize_in0,
        untilize_out,
        bias_ntiles_w
    };
    auto bmm_compute_id = CreateKernel(
        program,
        compute_kernel,
        core_range,
        tt_metal::ComputeConfig{
            .compile_args = compute_comptime_args,
            .defines = all_defines}
    );

    // Reader rt args
    SetRuntimeArgs(program, reader_id, core_range, reader_rt_args);
    // Writer rt args
    SetRuntimeArgs(program, writer_id, core_range, writer_rt_args);

    auto override_runtime_args_callback = [kernel_reader_id = reader_id, kernel_writer_id = writer_id](
                                            const Program &program,
                                            const std::vector<Buffer*>& input_buffers,
                                            const std::vector<Buffer*>& output_buffers) {
        auto in0_dram_buffer = input_buffers.at(0);
        auto in1_dram_buffer = input_buffers.at(1);
        auto bias_dram_buffer = input_buffers.at(2);
        auto out_dram_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = GetRuntimeArgs(program, kernel_reader_id, core);
            runtime_args[0] = in0_dram_buffer->address();
            runtime_args[9] = in1_dram_buffer->address();
            if (bias_dram_buffer != nullptr) {
                runtime_args[22] = bias_dram_buffer->address();
            }
        }
        {
            auto &runtime_args = GetRuntimeArgs(program, kernel_writer_id, core);
            runtime_args[0] = out_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
} // bmm_single_core_tilize_untilize()


void BMMTilizeUntilize::validate(const std::vector<Tensor>& inputs) const {
    const auto& in0 = inputs.at(0);
    const auto& in1 = inputs.at(1);
    const auto& bias = inputs.at(2);
    // TODO: Currently all validation is part of the primary function from create_program. Move them here.
}

std::vector<Shape> BMMTilizeUntilize::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    const auto& in0 = inputs.at(0);
    const auto& in1 = inputs.at(1);

    auto in0_batch = in0.get_legacy_shape()[0];
    auto in0_channel = in0.get_legacy_shape()[1];
    auto in0_height = in0.get_legacy_shape()[2];
    auto in0_width = in0.get_legacy_shape()[3];
    auto in1_batch = in1.get_legacy_shape()[0];
    auto in1_channel = in1.get_legacy_shape()[1];
    auto in1_height = in1.get_legacy_shape()[2];
    auto in1_width = in1.get_legacy_shape()[3];

    const Shape out_shape { in0_batch, in0_channel, in0_height, in1_width };
    return {out_shape};
}

std::vector<Tensor> BMMTilizeUntilize::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_layout = this->untilize_out_ ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::generic_create_output_tensors(*this, input_tensors, this->out_dt_, output_layout, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
}

operation::ProgramWithCallbacks BMMTilizeUntilize::create_program(const std::vector<Tensor>& inputs,
                                                                  std::vector<Tensor>& outputs) const {
    const auto& in0 = inputs.at(0);
    const auto& in1 = inputs.at(1);
    const auto& bias = inputs.at(2);
    auto& out = outputs.at(0);
    // NOTE: currently only single core version exists
    return bmm_single_core_tilize_untilize(in0, in1, bias, out_dt_,
                                           in0_nblocks_h_, in0_nblocks_w_, in1_nblocks_w_,
                                           in0_block_ntiles_h_, in0_block_ntiles_w_, in1_block_ntiles_w_,
                                           out_subblock_ntiles_h_, out_subblock_ntiles_w_,
                                           tilize_in0_, untilize_out_,
                                           has_bias_,
                                           out, this->compute_kernel_config);
}

}  // namespace tt_metal
}  // namespace tt
