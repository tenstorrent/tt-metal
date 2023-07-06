#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "tt_metal/llrt/tt_debug_print_server.hpp"
#include "tt_metal/hostdevcommon/debug_print_common.h"

namespace tt {
namespace tt_metal {

void create_cb_bmm_single_core_tilize_untilize(Program &program,
                                                Device* device,
                                                CoreCoord core,
                                                uint32_t in0_block_w,
                                                uint32_t in0_block_h,
                                                uint32_t in1_block_w,
                                                uint32_t dtype_nbytes,
                                                bool untilize_out = true) {
    // buffer indices
    uint32_t in0_cb                                 = CB::c_in0;
    uint32_t in1_cb                                 = CB::c_in1;
    uint32_t tilize_mode_tilized_in0_cb             = CB::c_intermed0;
    uint32_t matmul_partials_cb                     = CB::c_intermed1;
    uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
    uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
    uint32_t out_cb                                = CB::c_out0;

    const uint32_t tile_size_bytes = dtype_nbytes * constants::TILE_HW;
    const uint32_t cb0_ntiles = in0_block_h * in0_block_w * 2;  // double buffer
    const uint32_t cb1_ntiles = in0_block_w * in1_block_w * 2;   // double buffer
    const uint32_t out_ntiles = in0_block_h * in1_block_w;

    // inputs

    // in0 (RM)
    auto cb_in0 = CreateCircularBuffer(
        program,
        in0_cb,
        core,
        cb0_ntiles,
        cb0_ntiles * tile_size_bytes,
        DataFormat::Float16_b
    );
    // in1
    auto cb_in1 = CreateCircularBuffer(
        program,
        in1_cb,
        core,
        cb1_ntiles,
        cb1_ntiles * tile_size_bytes,
        DataFormat::Float16_b
    );

    // intermediates

    // in0 (TM)
    auto cb_src0_tilized = CreateCircularBuffer(
        program,
        tilize_mode_tilized_in0_cb,
        core,
        cb0_ntiles,
        cb0_ntiles * tile_size_bytes,
        DataFormat::Float16_b
    );
    CircularBuffer cb_matmul_partials;
    if (untilize_out) {
        cb_matmul_partials = CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            DataFormat::Float16_b
        );
    } else {
        CoreRangeSet cores(std::set<CoreRange>{CoreRange{.start=core, .end=core}});
        cb_matmul_partials = CreateCircularBuffers(
            program,
            {matmul_partials_cb, out_cb},
            cores,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            DataFormat::Float16_b
        );

    }

    if (untilize_out) {
        auto cb_final_matmul_partials = CreateCircularBuffer(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            DataFormat::Float16_b
        );
        // to reorganize output blocks to fill the whole "per core output block width"
        auto cb_reblock = CreateCircularBuffer(
            program,
            untilize_mode_reblock_cb,
            core,
            in1_block_w,                    // a single row of tiles
            in1_block_w * tile_size_bytes,
            DataFormat::Float16_b
        );
    }

    // output

    if (untilize_out) {
        auto cb_output = CreateCircularBuffer(
            program,
            out_cb,
            core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            DataFormat::Float16_b
        );
    }

    // std::cout << "NUMBER OF TILES IN OUT_CB = " << out_ntiles << " (" << out_ntiles * tile_size_bytes << " bytes)" << std::endl;
}

Tensor bmm_single_core_tilize_untilize(const Tensor &in0,
                                       const Tensor &in1,
                                       uint32_t in0_height_nblocks,
                                       uint32_t in0_width_nblocks,
                                       uint32_t in1_width_nblocks,
                                       uint32_t in0_block_height_ntiles,
                                       uint32_t in0_block_width_ntiles,
                                       uint32_t in1_block_width_ntiles,
                                       uint32_t out_subblock_height_ntiles,
                                       uint32_t out_subblock_width_ntiles,
                                       bool tilize_in0 = true,
                                       bool untilize_out = true) {
    const auto [in0_batch, in0_channel, in0_height, in0_width] = in0.shape();
    const auto [in1_batch, in1_channel, in1_height, in1_width] = in1.shape();

    constexpr uint32_t dtype_nbytes = 2;    // TODO (AS); Obtain from the datatype

    // input matrix shape checks
    TT_ASSERT(in0_batch == 1, "Supports only batch = 1");
    TT_ASSERT(in1_batch == in0_batch, "Batch dimension needs to match for two inputs");
    TT_ASSERT(in0_channel == in1_channel, "Channel dimension needs to match for two inputs");
    TT_ASSERT(in0_width == in1_height, "Input matrices should be compatible for multiplication");

    // tile size checks
    TT_ASSERT(in0_height % constants::TILE_HEIGHT == 0, "Input tensor in0 height needs to be divisible by TILE_HEIGHT");
    TT_ASSERT(in1_height % constants::TILE_HEIGHT == 0, "Input tensor in1 height needs to be divisible by TILE_HEIGHT");
    TT_ASSERT(in0_width % constants::TILE_WIDTH == 0, "Input tensor in0 width needs to be divisible by TILE_WIDTH");
    TT_ASSERT(in1_width % constants::TILE_WIDTH == 0, "Input tensor in1 width needs to be divisible by TILE_WIDTH");

    // device compatibility checks
    TT_ASSERT(in0.storage_type() == StorageType::DEVICE and in1.storage_type() == StorageType::DEVICE, "Operands need to be on the device!");
    TT_ASSERT(in0.device() == in1.device(), "Operands need to be on the same device!");
    TT_ASSERT(in0.buffer() != nullptr && in1.buffer() != nullptr, "Operands need to have buffers allocated on the device!");

    // Data format checks
    TT_ASSERT(in0.dtype() == in1.dtype() && in0.dtype() == DataType::BFLOAT16, "Datatypes of operands should match. Only BFLOAT16 supported for now");

    const uint32_t tile_size_bytes = dtype_nbytes * constants::TILE_HW;   // TODO: use datatype size
    Buffer *src0_dram_buffer = in0.buffer();
    Buffer *src1_dram_buffer = in1.buffer();

    TT_ASSERT(src0_dram_buffer->size() % tile_size_bytes == 0, "Buffer size of tensor in0 must be divisible by tile_size_bytes");
    TT_ASSERT(src1_dram_buffer->size() % tile_size_bytes == 0, "Buffer size of tensor in1 must be divisible by tile_size_bytes");

    CoreCoord core = {0, 0};
    CoreCoord debug_core = {1, 1};
    Program program = Program();
    Device *device = in0.device();

    // for kernel debug print
    // int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;
    tt_start_debug_print_server(device->cluster(), {0}, {debug_core});

    const Shape out_shape{in0_batch, in0_channel, in0_height, in1_width};
    Tensor output = create_device_tensor(out_shape,
                            in0.dtype(),
                            untilize_out ? Layout::ROW_MAJOR : Layout::TILE,
                            device);
    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // Convert tensor dims to tile dims
    uint32_t in0_height_ntiles = in0_height / constants::TILE_HEIGHT;   // == in0_height_nblocks * in0_block_height_ntiles
    uint32_t in0_width_ntiles = in0_width / constants::TILE_WIDTH;      // == in0_width_nblocks * in0_block_width_ntiles
    uint32_t in1_width_ntiles = in1_width / constants::TILE_WIDTH;      // == in1_width_nblocks * in1_block_width_ntiles
    // Ensure the size arguments match the input tensors
    TT_ASSERT(in0_height_ntiles == in0_height_nblocks * in0_block_height_ntiles, "Mismatch in tensor in0 height!");
    TT_ASSERT(in0_width_ntiles == in0_width_nblocks * in0_block_width_ntiles, "Mismatch tensor in0 width!");
    TT_ASSERT(in1_width_ntiles == in1_width_nblocks * in1_block_width_ntiles, "Mismatch tensor in1 width!");

    {   // debug
        log_debug("in0_height_ntiles = {}", in0_height_ntiles);
        log_debug("in0_width_ntiles = {}", in0_width_ntiles);
        log_debug("in1_width_ntiles = {}", in1_width_ntiles);
    }

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
    uint32_t in0_partial_row_size_bytes = (in0_block_w * constants::TILE_WIDTH) * dtype_nbytes; // TODO: use datatype
    TT_ASSERT(in0_block_h % out_subblock_height_ntiles == 0);

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
    TT_ASSERT(in1_block_w % out_subblock_width_ntiles == 0);

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();
    uint32_t out_dram_noc_x = out_dram_noc_xy.x;
    uint32_t out_dram_noc_y = out_dram_noc_xy.y;
    uint32_t out_row_size_bytes = in1_width * dtype_nbytes;  // TODO: use datatype info
    uint32_t out_subblock_ntiles = out_subblock_height_ntiles * out_subblock_width_ntiles;
    TT_ASSERT(out_subblock_ntiles <= 8, "Subblock can have at most 8 tiles to fit computed intermediates in dst[half]");

    {   // debug
        // in0
        log_debug("in0_dram_addr: {}", in0_dram_addr);
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
        log_debug("in1_subblock_w: {}", in1_subblock_w);
        log_debug("in1_num_subblocks: {}", in1_num_subblocks);
        log_debug("in1_block_num_tiles: {}", in1_block_num_tiles);
        log_debug("in1_block_w: {}", in1_block_w);
        log_debug("in1_block_h: {}", in1_block_h);
        log_debug("in1_num_blocks_w: {}", in1_num_blocks_w);
        log_debug("in1_num_blocks_h: {}", in1_num_blocks_h);
        // out
        log_debug("out_dram_addr: {}", out_dram_addr);
        log_debug("out_subblock_height_ntiles: {}", out_subblock_height_ntiles);
        log_debug("out_subblock_width_ntiles: {}", out_subblock_width_ntiles);
        log_debug("out_subblock_ntiles: {}", out_subblock_ntiles);
    }

    create_cb_bmm_single_core_tilize_untilize(
        program,
        in0.device(),
        core,
        in0_block_w,
        in0_block_h,
        in1_block_w,
        dtype_nbytes,
        untilize_out);

    // Reader kernel
    std::string reader_kernel;
    std::vector<uint32_t> reader_rt_args;
    if (tilize_in0) {
        // in0 is row major, in1 is tiled
        reader_kernel = "tt_metal/kernels/dataflow/reader_bmm_single_core_tilize_untilize.cpp";
        reader_rt_args = {
            // in0
            in0_dram_addr,
            in0_block_h,
            in0_num_blocks_h,
            in0_num_blocks_w,
            in0_block_num_tiles,
            in0_block_h * constants::TILE_HEIGHT,               // in0_block_nrows,
            0,                                                  // start row id
            in0_width * dtype_nbytes,                           // size of an in0 row
            in0_block_w * constants::TILE_WIDTH * dtype_nbytes, // size of partial row to fit within a block width
            // in1
            in1_dram_addr,
            in1_block_h,
            in1_block_w,
            in1_num_blocks_w,
            in1_block_num_tiles,
            in1_width_ntiles,
            in1_width_ntiles * in1_block_h,
            in1_block_w
        };
    } else {
        // in0 is tiled, in1 is tiled
        reader_kernel = "tt_metal/kernels/dataflow/reader_bmm_single_core.cpp";
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
        };
    }
    auto reader = CreateDataMovementKernel(
        program,                            // program
        reader_kernel,                      // file name
        core,                               // core
        DataMovementProcessor::RISCV_1,     // processor type
        NOC::RISCV_1_default                // noc
    );

    // number of data elements along height of an in0 block
    uint32_t in0_block_h_data = in0_height / in0_num_blocks_h;

    // Writer kernel
    std::string writer_kernel;
    vector<uint32_t> writer_rt_args;

    if (untilize_out) {
        // out is row major
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            in0_block_h_data,
            in1_block_w * constants::TILE_WIDTH * dtype_nbytes, // block_row_size
            1,                                                  // batch
            in0_num_blocks_h,
            in1_num_blocks_w,
            in1_width * dtype_nbytes                            // output_row_size
        };
    } else {
        // out is tiled
        writer_kernel = "tt_metal/kernels/dataflow/writer_bmm_single_core_tiled.cpp";
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
            in0_height_ntiles / out_subblock_height_ntiles, // out_num_subblocks_h
        };
    }
    for (auto param_val : writer_rt_args) {
        log_debug("==> {}", param_val);
    }
    auto writer = CreateDataMovementKernel(
        program,                        // program
        writer_kernel,                  // file name
        core,                           // core
        DataMovementProcessor::RISCV_0, // processor type
        NOC::RISCV_0_default);          // noc

    // Compute kernel
    std::string compute_kernel = "tt_metal/kernels/compute/bmm_tilize_untilize.cpp";
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
        untilize_out
    };
    auto bmm_compute = CreateComputeKernel(
        program,
        compute_kernel,
        core,
        compute_comptime_args,
        MathFidelity::HiFi4,
        false,  // fp32_dest_acc_en
        false   // math_approx_mode
    );

    // Reader rt args
    SetRuntimeArgs(reader, core, reader_rt_args);
    // Writer rt args
    SetRuntimeArgs(writer, core, writer_rt_args);

    // Compile and launch
    bool pass = CompileProgram(device, program, false);
    pass &= ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, program);
    pass &= LaunchKernels(device, program);

    TT_ASSERT(pass);

    return output;
} // bmm_single_core_tilize_untilize()

}  // namespace tt_metal
}  // namespace tt
