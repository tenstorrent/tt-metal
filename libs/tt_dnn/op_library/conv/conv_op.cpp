#include "tt_dnn/op_library/conv/conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "libs/dtx/dtx.hpp"
#include "libs/dtx/dtx_passes.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt::constants;
namespace tt {

namespace tt_metal {
void create_CBs_for_fused_matmul_new_alloc(tt_metal::Program* program,
                                tt_metal::Device* device,
                                tt_xy_pair core,
                                uint32_t act_block_size,
                                uint32_t weight_block_size,
                                uint32_t output_block_size,
                                uint32_t reblock_size,
                                uint32_t num_bytes_for_df,
                                bool untilize_out) {
    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = output_block_size;

    // Invariants
    uint32_t cb0_tiles = act_block_size * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t cb1_tiles = weight_block_size * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    // Used for placing tilized activations
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
        program,
        device,
        tilize_mode_tilized_in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    if(untilize_out) {
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = reblock_size; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
    else {

        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_matmul_partials_addr = cb_matmul_partials->address();

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            cb_matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

    }
}

Tensor create_output_dram_buffer_(Device * device, DataType data_type, std::array<uint32_t, 4> cshape, bool untilize_out) {
    tt::tt_metal::Layout out_layout;
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        data_type,
        untilize_out ? tt::tt_metal::Layout::ROW_MAJOR : tt::tt_metal::Layout::TILE,
        device);

    return output;
}

std::tuple<uint32_t, uint32_t, uint32_t, string> compute_conv_op_block_info(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t single_tile_size_bytes = 2 * 1024;
    std::string report_string = "";

    // Constraint 1: in0 and in1 should fit in L1. If not, slice into blocks
    // in0 slice should contain whole column and in1 slice should contain whole row
    // Can only do in0 width slicing and in1 height slicing
    // Harcoded max sizes that can fit all CBs into 1MB L1 space - reserved space
    // TODO: remove hardcoded values and calculate them.
    uint32_t max_in0_block_bytes = 50 * 1024;
    uint32_t max_in1_block_bytes = 50 * 1024;
    uint32_t max_in0_block_tiles = max_in0_block_bytes / single_tile_size_bytes;
    uint32_t max_in1_block_tiles = max_in1_block_bytes / single_tile_size_bytes;
    uint32_t num_blocks = 1;
    uint32_t in_block_w = K;
    bool input_and_weight_can_fit_after_slicing = M <= max_in0_block_tiles && N <= max_in1_block_tiles;
    if(M > max_in0_block_tiles) {
        report_string += "Cannot run conv on TT device because activation matrix height (in tiles) =" + to_string(M) + " > " + to_string(max_in0_block_tiles) + "\n";
        report_string += "Connot fit conv in L1.\n";
    }
    if(N > max_in1_block_tiles) {
        report_string += "Cannot run conv on TT device because weight matrix width (in tiles) =" + to_string(N) + " > " + to_string(max_in1_block_tiles) + "\n";
        report_string += "Connot fit conv in L1.\n";
    }
    if(!input_and_weight_can_fit_after_slicing) {
        return std::make_tuple(0, 0, 0, report_string);
    }
    assert(input_and_weight_can_fit_after_slicing);
    uint32_t max_in_block_w = std::min((max_in0_block_tiles/M), (max_in1_block_tiles/N));
    while (in_block_w > max_in_block_w || K % num_blocks != 0) {
        num_blocks += 1;
        assert(num_blocks <= K);
        in_block_w = K / num_blocks;
    }

    // Constraint 2: whole output should fit in L1
    uint32_t max_out_block_bytes = 120 * 1024;
    uint32_t max_out_block_tiles = max_out_block_bytes / single_tile_size_bytes;
    uint32_t max_n_reblock_bytes = 50 * 1024;
    uint32_t max_n_reblock_tiles = max_n_reblock_bytes / single_tile_size_bytes;
    log_debug(tt::LogOp, "max_out_block_tiles: {}", max_out_block_tiles);
    bool output_fits = M*N <= max_out_block_tiles;
    bool one_row_output_fits_in_reblock_cb = N <= max_n_reblock_tiles;
    if(M*N > max_out_block_tiles) {
        report_string += "Cannot run conv on TT device because output matrix volume (in tiles) =" + to_string(M*N) + " > " + to_string(max_out_block_tiles) + "\n";
        report_string += "Connot fit conv in L1.\n";
    }
    if(N > max_n_reblock_tiles) {
        report_string += "Cannot run conv on TT device because output matrix width (in tiles) =" + to_string(N) + " > " + to_string(max_n_reblock_tiles) + "\n";
        report_string += "Connot fit conv in L1.\n";
    }
    if(!output_fits || !one_row_output_fits_in_reblock_cb) {
        return std::make_tuple(0, 0, 0, report_string);
    }
    assert (output_fits && one_row_output_fits_in_reblock_cb);
    log_debug(tt::LogOp, "num_blocks: {}", num_blocks);
    log_debug(tt::LogOp, "in_block_w: {}", in_block_w);
    // Constraint 3: output should should fit in half DST (8 tiles). If not, divide into output sublocks
    uint32_t out_subblock_h = M;
    uint32_t out_subblock_w = N;
    uint32_t num_out_subblocks_h = 1;
    uint32_t num_out_subblocks_w = 1;
    bool divide_h_next = true;
    while (out_subblock_h*out_subblock_w > 8) {
        if (divide_h_next) {
            if(num_out_subblocks_h < M) {
                num_out_subblocks_h += 1;
                while(M % num_out_subblocks_h != 0) {
                    num_out_subblocks_h += 1;
                }
            }
            out_subblock_h = M / num_out_subblocks_h;
            divide_h_next = false;
        }
        else {
            if(num_out_subblocks_w < N) {
                num_out_subblocks_w += 1;
                while(N % num_out_subblocks_w != 0) {
                    num_out_subblocks_w += 1;
                }
            }
            out_subblock_w = N / num_out_subblocks_w;
            divide_h_next = true;
        }
    }
    return std::make_tuple(num_blocks, out_subblock_h, out_subblock_w, "pass");
}

vector<uint32_t> compute_conv_as_mm_shape(vector<int> shape, vector<int> conv_params) {
    int conv_input_x = shape[2];
    int conv_input_y = shape[1];
    int conv_input_z = shape[0];
    int R = conv_params[0];
    int S = conv_params[1];
    int U = conv_params[2];
    int V = conv_params[3];
    int Pad_H = conv_params[4];
    int Pad_W = conv_params[5];
    int conv_output_h = ((conv_input_x - R + (2 * Pad_H)) / U) + 1;
    int conv_output_w = ((conv_input_y - S + (2 * Pad_W)) / V) + 1;
    {
        log_debug(tt::LogOp, "conv_input_x: {}", conv_input_x);
        log_debug(tt::LogOp, "conv_input_y: {}", conv_input_y);
        log_debug(tt::LogOp, "conv_input_z: {}", conv_input_z);
        log_debug(tt::LogOp, "kernel_size_y: {}", R);
        log_debug(tt::LogOp, "kernel_size_x: {}", S);
        log_debug(tt::LogOp, "stride_y: {}", U);
        log_debug(tt::LogOp, "stride_x: {}", V);
        log_debug(tt::LogOp, "pad_y: {}", Pad_H);
        log_debug(tt::LogOp, "pad_x: {}", Pad_W);
        log_debug(tt::LogOp, "conv_output_x: {}", conv_output_w);
        log_debug(tt::LogOp, "conv_output_y: {}", conv_output_h);
    }
    // pad height
    uint32_t num_rows = (uint32_t) conv_output_h*conv_output_w;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) TILE_HEIGHT ) * TILE_HEIGHT);
    uint32_t num_cols = conv_input_z*R*S;
    uint32_t num_cols_padded = (uint32_t) (ceil((double) num_cols / (double) TILE_WIDTH ) * TILE_HEIGHT);
    return {1,num_rows_padded, num_cols_padded};
}
// TODO(whoever gets a chance!): Refactor this so it's a part of matmul_single_core_... keeping it
// independent for now as it's easier for me to progress
Tensor conv_as_large_bmm_single_core_(const Tensor& a, const Tensor &b, vector<int> conv_params, bool untilize_out=true) {
    bool pass = true;
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");
    TT_ASSERT(a.shape()[0] == 1, "Only batch size 1 supported.");
    uint32_t activation_C = a.shape()[1];
    //TT_ASSERT(activation_C % TILE_WIDTH == 0, "Channel depth must be divisible by tile width(32).");
    // Compute the 2d matrix shape
    vector<int> shape = {(int)a.shape()[1], (int)a.shape()[2], (int)a.shape()[3]};
    auto matrix_shape = compute_conv_as_mm_shape(shape , conv_params);
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];

    // More Checks
    uint32_t Ba = 1;
    uint32_t Ca = 1;
    auto Ha = num_rows;
    auto Wa = num_cols;
    const auto [Bb, Cb, Hb, Wb] = b.shape();
    // Normal matrix shape checks
    TT_ASSERT(Ba == 1, "So far, large matmul op has only been tested for batch one.");
    TT_ASSERT(Ba == Bb, "Batch dimension needs to match");
    TT_ASSERT(Ca == Cb, "Channel dimension needs to match");
    TT_ASSERT(Wa == Hb, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");
    // Convert tensor dims to tile dims
    uint32_t B   = Ba;
    uint32_t Hat = Ha / TILE_HEIGHT;
    uint32_t Wat = Wa / TILE_WIDTH;
    uint32_t Wbt = Wb / TILE_WIDTH;
    log_debug(tt::LogOp, "Hat(MM Activation H in tiles): {}", Hat);
    log_debug(tt::LogOp, "Wat(MM Activation W (MM Weight H) in tiles): {}", Wat);
    log_debug(tt::LogOp, "Wbt(MM Weight W in tiles): {}", Wbt);
    // compute block info
    auto [num_blocks, out_subblock_h, out_subblock_w, report_string] = compute_conv_op_block_info(Hat, Wat, Wbt);
    assert(report_string == "pass");
    // uint32_t num_blocks = 2;
    // uint32_t out_subblock_h = 1;
    // uint32_t out_subblock_w = 1;
    // in0 block info
    uint32_t in0_block_w = Wat / num_blocks; // Two blocks in the W dimension
    uint32_t in0_block_w_datums = Wa / num_blocks;
    std::pair<vector<int>,vector<int>> block_info;
    block_info.first = {0,1,2};
    block_info.second = {(int)num_rows, (int)in0_block_w_datums};

    DataTransformations * dtx = conv_transform(shape, conv_params, block_info);

    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;
    uint32_t t_bytes = 0;

    // Generate address map for reader kernel
    TT_ASSERT(dtx->transformations.size() == 2, "DTX transformations not collapsed.");
    TT_ASSERT(dtx->transformations.back()->groups[0]->transfers.size() > 0, "Addresses not generated in DTX.");
    uint32_t block_size_bytes = num_rows * in0_block_w_datums * 2;
    uint32_t b_bytes = 0;
    uint32_t n_blocks = 0;
    for(auto transfer : dtx->transformations.back()->groups[0]->transfers){
        assert(transfer->size*2 % 32 == 0);
        assert(transfer->src_address*2 % 32 == 0);
        assert(transfer->dst_address*2 % 32 == 0);
        address_map.push_back(transfer->src_address*2);
        address_map.push_back(transfer->dst_address*2);
        address_map.push_back(transfer->size*2);
        address_map.push_back(transfer->pad);

        t_bytes += transfer->size*2;
        b_bytes += transfer->size*2;
        if(b_bytes == block_size_bytes) {
            b_bytes = 0;
            n_blocks++;
        }
    }
    uint32_t total_bytes = num_rows * num_cols * 2; // 2 for bfloat16
    assert(b_bytes == 0);
    assert(n_blocks == num_blocks);
    assert(total_bytes == t_bytes);
    assert(total_bytes % num_blocks == 0);
    uint32_t in0_block_size_bytes = total_bytes / num_blocks;
    assert(in0_block_size_bytes == block_size_bytes);
    //delete dtx;
    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});


    uint32_t single_tile_size = 2 * 1024; // TODO(agrebenisan): Refactor on df
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};

    Tensor output = create_output_dram_buffer_(a.device(), a.dtype(), cshape, untilize_out);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, address_map.size() * sizeof(uint32_t));
    uint32_t address_map_l1_addr = l1_b0->address();

    // Keep for now, but need to fix when you get to multibank
    uint32_t out_dram_addr = dst_dram_buffer->address();
    auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t out_dram_noc_x = out_dram_noc_xy.x;
    uint32_t out_dram_noc_y = out_dram_noc_xy.y;

    // out
    uint32_t out_row_size = Wb * 2;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // in0
    uint32_t in0_dram_addr = src0_dram_buffer->address();
    auto in0_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t in0_noc_x = in0_dram_noc_xy.x;
    uint32_t in0_noc_y = in0_dram_noc_xy.y;

    assert(Wat % in0_block_w == 0);
    uint32_t in0_num_blocks_w = Wat / in0_block_w;
    assert(Hat % out_subblock_h == 0);
    uint32_t in0_num_subblocks = (Hat / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    assert(in0_block_size_bytes == single_tile_size * in0_block_num_tiles);

    // in1
    uint32_t in1_dram_addr = src1_dram_buffer->address();

    // in1 block info
    assert(Wbt % out_subblock_w == 0);
    uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
    uint32_t in1_block_w = out_subblock_w * in1_num_subblocks;
    uint32_t in1_block_h = in0_block_w;

    // For debug
    {
        log_debug(tt::LogOp, "num_blocks: {}", num_blocks);
        log_debug(tt::LogOp, "in0_dram_addr: {}", in0_dram_addr);
        log_debug(tt::LogOp, "in0_block_w: {}", in0_block_w);
        log_debug(tt::LogOp, "in0_num_blocks_w: {}", in0_num_blocks_w);
        log_debug(tt::LogOp, "in0_num_subblocks: {}", in0_num_subblocks);
        log_debug(tt::LogOp, "in0_block_num_tiles: {}", in0_block_num_tiles);
        log_debug(tt::LogOp, "in0_subblock_h: {}", in0_subblock_h);
        log_debug(tt::LogOp, "in0_subblock_num_tiles: {}", in0_subblock_num_tiles);
        log_debug(tt::LogOp, "in1_dram_addr: {}", in1_dram_addr);
        log_debug(tt::LogOp, "in1_num_subblocks: {}", in1_num_subblocks);
        log_debug(tt::LogOp, "in1_block_num_tiles: {}", in1_block_num_tiles);
        log_debug(tt::LogOp, "in1_block_w: {}", in1_block_w);
        log_debug(tt::LogOp, "in1_block_h: {}", in1_block_h);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h: {}", out_subblock_h);
        log_debug(tt::LogOp, "out_subblock_w: {}", out_subblock_w);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
    }

    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        Hat * in0_block_w,
        Wbt * in0_block_w,
        Hat * Wbt,
        Wbt,
        2,
        untilize_out);

    uint32_t in1_tensor_start_tile_id = 0;
    uint32_t in1_tensor_stride_w = 1;
    uint32_t in1_tensor_stride_h = Wbt;
    uint32_t in1_tensor_next_block_stride = in0_block_w * Wbt;

    string reader_kernel;
    vector<uint32_t> reader_rt_args;

    reader_kernel = "tt_metal/kernels/dataflow/reader_binary_dtx.cpp";
    reader_rt_args = {
        num_blocks,
        // arguments for in0
        in0_dram_addr,
        in0_noc_x,
        in0_noc_y,
        in0_block_num_tiles,
        address_map_l1_addr,
        in0_block_size_bytes,
        // arguments for in1
        in1_dram_addr,
        in1_block_w,
        in1_block_h,
        in1_block_num_tiles,
        in1_tensor_start_tile_id,
        in1_tensor_stride_w,
        in1_tensor_stride_h,
        in1_tensor_next_block_stride
    };

    string writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp";
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp";
        writer_rt_args = {
            out_dram_addr,
            Ha,
            out_row_size
        };
    } else {
        writer_kernel = "tt_metal/kernels/dataflow/writer_matmul_tile_layout.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,
            1,
            Wbt,
            out_subblock_w,
            out_subblock_h * Wbt,

            out_subblock_w,
            out_subblock_h,
            out_subblock_w * out_subblock_h,
            Wbt / out_subblock_w,
            Hat / out_subblock_h
        };
    }
    auto reader = tt_metal::CreateDataMovementKernel(
        program,
        reader_kernel,
        core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    auto writer = tt_metal::CreateDataMovementKernel(
        program,
        writer_kernel,
        core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in0_subblock_h,

        in1_num_subblocks,
        in1_block_num_tiles,
        in1_block_w,

        num_blocks,

        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,

        true,
        untilize_out
    };

    tt_metal::ComputeKernelArgs *bmm_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/matmul_large_block.cpp",
        core,
        bmm_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device, reader, core,
        reader_rt_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device, writer, core,
        writer_rt_args
    );

    pass &= tt_metal::CompileProgram(device, program, false);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteToDeviceL1(device, core, address_map, address_map_l1_addr);

        pass &= tt_metal::LaunchKernels(device, program);


    TT_ASSERT(pass);
    delete program;
    return output;
}
std::pair<bool, string> conv_fits_on_single_core(vector<uint32_t> conv_act_shape, vector<uint32_t> conv_weight_shape, vector<int> conv_params) {
    assert(conv_act_shape[0] == 1);
    // Compute the 2d matrix shape
    vector<int> shape = {(int)conv_act_shape[1], (int)conv_act_shape[2], (int)conv_act_shape[3]};
    auto matrix_shape = compute_conv_as_mm_shape(shape , conv_params);
    auto Ha = matrix_shape[1];
    auto Wa = matrix_shape[2];
    auto Wb = (uint32_t) (ceil((double) conv_weight_shape[0] / (double) TILE_WIDTH ) * TILE_HEIGHT);
    auto Hat = Ha / TILE_HEIGHT;
    auto Wat = Wa / TILE_WIDTH;
    auto Wbt = Wb / TILE_WIDTH;
    auto [a,b,c,report_string] = compute_conv_op_block_info(Hat, Wat, Wbt);
    return std::pair<bool, string>(report_string == "pass", report_string);
}
Tensor conv(const Tensor& a, const Tensor &b, vector<int> conv_params, bool untilize_out) {

    Tensor output = conv_as_large_bmm_single_core_(a, b, conv_params, untilize_out);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
