#include "tt_dnn/op_library/conv/conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "llrt/tt_debug_print_server.hpp"

using namespace tt::constants;
namespace tt {

namespace tt_metal {
vector<uint32_t> compute_conv_activation_as_mm_shape(vector<int> shape, const std::vector<int>& conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles) {
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
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_input_z*R*S;
    uint32_t act_block_w_datums = act_block_w_ntiles * TILE_WIDTH;
    uint32_t num_cols_padded = (uint32_t) (ceil((double) num_cols / (double) act_block_w_datums ) * act_block_w_datums);
    return {1,num_rows_padded, num_cols_padded};
}


void create_CBs_for_fused_matmul_new_alloc(tt_metal::Program &program,
                                tt_metal::Device* device,
                                CoreRange core,
                                uint32_t act_block_size,
                                uint32_t weight_block_size,
                                uint32_t output_block_size,
                                uint32_t reblock_size,
                                uint32_t num_bytes_for_df,
                                bool untilize_out) {
    uint32_t act_cb                                   = 0;
    uint32_t weight_cb                                   = 1;
    uint32_t tilize_mode_tilized_act_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = output_block_size;

    // Invariants
    uint32_t cb0_tiles = act_block_size;
    auto cb_act = tt_metal::CreateCircularBuffers(
        program,
        act_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t cb1_tiles = weight_block_size;
    auto cb_weight = tt_metal::CreateCircularBuffers(
        program,
        weight_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    // Used for placing tilized activations
    auto cb_src0_tilized = tt_metal::CreateCircularBuffers(
        program,
        tilize_mode_tilized_act_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    if(untilize_out) {
        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = reblock_size; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffers(
            program,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
    else {

        CoreRangeSet cores(std::set<CoreRange>({core}));
        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            {matmul_partials_cb, out0_cb},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
}

Tensor create_output_dram_buffer_(Device * device, DataType data_type, std::array<uint32_t, 4> cshape, bool untilize_out) {
    tt::tt_metal::Layout out_layout;
    tt_metal::Tensor output = tt_metal::create_device_tensor(
        cshape,
        data_type,
        untilize_out ? tt::tt_metal::Layout::ROW_MAJOR : tt::tt_metal::Layout::TILE,
        device);

    return output;
}

Program conv_as_large_bmm_single_core_(const Tensor& a, const Tensor &b, vector<int> conv_params,
                                       uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                                       uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, bool untilize_out, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");
    uint32_t act_batch_size = a.shape()[0];
    TT_ASSERT(act_batch_size == 1, "Only batch size 1 supported.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    // Compute the 2d matrix shape
    vector<int> activation_shape = {(int)a.shape()[1], (int)a.shape()[2], (int)a.shape()[3]};    // TODO: Update types to use just one kind
    // Shape activation_shape_shape = {a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]};
    auto act_matrix_shape = compute_conv_activation_as_mm_shape(activation_shape, conv_params, act_block_h_ntiles, act_block_w_ntiles);
    assert(act_matrix_shape.size() == 3);
    assert(act_matrix_shape[0] == 1);
    uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t) act_matrix_shape[2];

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_ASSERT(b.layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_ASSERT(b.shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_ASSERT(b.shape()[1] == 1, "Conv weight matrix shape is invalid");
    const auto [notused1, notused2, weight_matrix_height, weight_matrix_width] = b.shape();
    // Normal matrix shape check
    TT_ASSERT(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(act_matrix_height % TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_ASSERT(act_matrix_width % TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_height % TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_width % TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(a.storage_type() == StorageType::DEVICE and b.storage_type() == StorageType::DEVICE, "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
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

    tt_metal::Program program = tt_metal::Program();
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core = {.start={0, 0}, .end={0, 0}};
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    std::array<uint32_t, 4> cshape{1, 1, act_matrix_height, weight_matrix_width};

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

    // more args for reader
    uint32_t conv_act_size_w = a.shape()[3];
    uint32_t conv_act_size_h = a.shape()[2];
    uint32_t conv_act_size_c = a.shape()[1];
    uint32_t weight_size_h = (uint32_t) conv_params[0];
    uint32_t weight_size_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    uint32_t conv_output_size_h = ((conv_act_size_h - weight_size_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_size_w = ((conv_act_size_w - weight_size_w + (2 * pad_w)) / stride_w) + 1;

    uint32_t act_matrix_height_unpadded = conv_output_size_h * conv_output_size_w;
    uint32_t act_matrix_width_unpadded = conv_act_size_c * weight_size_h * weight_size_w;
    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes = act_block_h_ntiles * act_block_w_ntiles * single_tile_size;
    uint32_t dst_l1_weight_buffer_size_bytes = weight_block_h_ntiles * weight_block_w_ntiles * single_tile_size;


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
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_groups: {}", num_groups);
    }

    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        act_block_h_ntiles * act_block_w_ntiles,
        weight_block_h_ntiles * weight_block_w_ntiles,
        act_block_h_ntiles * weight_block_w_ntiles,
        weight_block_w_ntiles,
        num_bytes_of_df,
        untilize_out);

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    reader_kernel = "tt_metal/kernels/dataflow/reader_conv_activations_and_weights.cpp";
    reader_rt_args = {
        // arguments for act
        act_dram_addr,
        act_noc_x,
        act_noc_y,

        weight_dram_addr,
        weight_noc_x,
        weight_noc_y,
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

        weight_matrix_height,
        weight_matrix_width,
        weight_matrix_height_ntiles,
        weight_matrix_width_ntiles,
        weight_block_w_datums,
        weight_block_h_ntiles,
        weight_block_w_ntiles,
        weight_block_num_tiles,

        src_dram_act_buffer_size_bytes,
        dst_l1_act_buffer_size_bytes,
        src_dram_weight_buffer_size_bytes,
        dst_l1_weight_buffer_size_bytes,
    };

    string writer_kernel;
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            act_block_h_datums,
            weight_block_w_ntiles*TILE_WIDTH*num_bytes_of_df,
            1,
            num_blocks_act_h,
            num_blocks_weight_w,
            weight_matrix_width*num_bytes_of_df
        };
    } else {
        assert(false && "Tiled output unsupported");
        writer_kernel = "tt_metal/kernels/dataflow/writer_matmul_tile_layout.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,
            1,
            weight_matrix_width_ntiles,
            out_subblock_w_ntiles,
            out_subblock_h_ntiles * weight_matrix_width_ntiles,

            out_subblock_w_ntiles,
            out_subblock_h_ntiles,
            out_subblock_w_ntiles * out_subblock_h_ntiles,
            weight_matrix_width_ntiles / out_subblock_w_ntiles,
            act_matrix_height_ntiles / out_subblock_h_ntiles
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
        untilize_out
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_tilize_untilize.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        reader, core,
        reader_rt_args
    );

    tt_metal::SetRuntimeArgs(
        writer, core,
        writer_rt_args
    );
    return program;
}

Tensor conv(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles) {
    return operation::run(Conv(act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, conv_params, true), {a, b}).at(0);
}

Program conv_single_core(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, bool untilize_out, Tensor &output) {
    return conv_as_large_bmm_single_core_(a, b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, untilize_out, output);
}

void Conv::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
}

std::vector<Shape> Conv::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    vector<int> input_tensor_a_shape = { (int) input_tensor_a.shape()[1], (int) input_tensor_a.shape()[2], (int) input_tensor_a.shape()[3]};
    auto mm_shape = compute_conv_activation_as_mm_shape(input_tensor_a_shape, conv_params, act_block_h_ntiles, act_block_w_ntiles);
    // TODO: Update batch size below
    Shape output_tensor_shape = {1, 1, mm_shape[1], input_tensor_b.shape()[3] };
    return {output_tensor_shape};
}

std::vector<Tensor> Conv::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0);
    Tensor output = create_output_dram_buffer_(input_tensor.device(), input_tensor.dtype(), output_shape, untilize_out);
    std::vector<Tensor> output_tensors;
    // TODO: check if anything else needs to be done here.
    output_tensors.emplace_back(output);
    return output_tensors;
}

operation::ProgramWithCallbacks Conv::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    return {conv_single_core(input_tensor_a, input_tensor_b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, untilize_out, output_tensor)};
}

// generates address map for reader kernel which reads from dram buffer (tiled layout) into l1 buffer
std::pair<vector<uint32_t>, vector<uint32_t>> generate_conv_weight_address_map(
                            const std::array<uint32_t, 4>& weight_shape,
                            uint32_t weight_block_h_datums,
                            uint32_t weight_block_w_datums,
                            uint32_t num_blocks_act_h,
                            uint32_t num_blocks_weight_h,
                            uint32_t num_blocks_weight_w,
                            uint32_t num_bytes_df) {
    vector<uint32_t> address_map;
    vector<uint32_t> address_map_metadata;
    assert(weight_shape[0] == 1 && weight_shape[1] == 1);
    uint32_t matrix_height = weight_shape[2];
    uint32_t matrix_width = weight_shape[3];
    assert(matrix_height % weight_block_h_datums == 0);
    assert(matrix_width % weight_block_w_datums == 0);
    uint32_t src_dram_buffer_size_bytes = matrix_height * matrix_width * num_bytes_df;
    uint32_t dst_l1_buffer_size_bytes = weight_block_h_datums * weight_block_w_datums * num_bytes_df;
    uint32_t num_groups = num_blocks_act_h * num_blocks_weight_h * num_blocks_weight_w;
    assert(matrix_height % TILE_HEIGHT == 0);
    uint32_t matrix_height_ntiles = matrix_height / TILE_HEIGHT;
    assert(matrix_width % TILE_WIDTH == 0);
    uint32_t matrix_width_ntiles = matrix_width / TILE_WIDTH;
    assert(matrix_height_ntiles % num_blocks_weight_h == 0);
    uint32_t block_height_ntiles = matrix_height_ntiles / num_blocks_weight_h;
    assert(matrix_width_ntiles % num_blocks_weight_w == 0);
    uint32_t block_width_ntiles = matrix_width_ntiles / num_blocks_weight_w;
    uint32_t matrix_size_ntiles = matrix_height_ntiles * matrix_width_ntiles;
    assert(weight_block_h_datums % TILE_HEIGHT == 0);
    assert(weight_block_w_datums % TILE_WIDTH == 0);
    assert(block_height_ntiles == weight_block_h_datums / TILE_HEIGHT);
    assert(block_width_ntiles == weight_block_w_datums / TILE_WIDTH);
    address_map_metadata.push_back(num_groups);
    uint32_t address_map_current_group_dram_address_offset = 0;
    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {
        // Weight blocks are col major
        uint32_t block_idx_h = (uint32_t) (group_idx % num_blocks_weight_h);
        uint32_t block_idx_w = (uint32_t) (group_idx / num_blocks_weight_h) % (num_blocks_weight_w);
        uint32_t block_idx = (block_idx_w * num_blocks_weight_h) + block_idx_h;
        uint32_t start_block_tile_h_index = block_idx_h * block_height_ntiles;
        uint32_t start_block_tile_w_index = block_idx_w * block_width_ntiles;
        uint32_t single_tile_size_bytes = TILE_HEIGHT * TILE_WIDTH * num_bytes_df;
        uint32_t address_map_current_group_size = 0;
        // Weight tiles are in row major order within block
        for(uint32_t tile_h_index_in_block = 0; tile_h_index_in_block < block_height_ntiles; tile_h_index_in_block++) {
            for(uint32_t tile_w_index_in_block = 0; tile_w_index_in_block < block_width_ntiles; tile_w_index_in_block++) {
                uint32_t tile_index_h_in_matrix = tile_h_index_in_block + start_block_tile_h_index;
                uint32_t tile_index_w_in_matrix = tile_w_index_in_block + start_block_tile_w_index;
                // Weight tiles are in row major order in weight matrix in dram
                uint32_t tile_index_in_matrix = (tile_index_h_in_matrix * block_width_ntiles * num_blocks_weight_w) + tile_index_w_in_matrix;
                assert(tile_index_in_matrix < matrix_size_ntiles);
                // Weight tiles are in row major order in weight block in l1
                uint32_t tile_index_in_block = tile_h_index_in_block * block_width_ntiles + tile_w_index_in_block;
                uint32_t src_address_offset_dram = tile_index_in_matrix * single_tile_size_bytes;
                uint32_t read_size_bytes = single_tile_size_bytes;
                uint32_t dst_address_offset_l1 = tile_index_in_block * single_tile_size_bytes;
                uint32_t pad = 0;
                assert(read_size_bytes > 0);
                assert(pad == 0 || pad == 1);
                assert(src_address_offset_dram < src_dram_buffer_size_bytes);
                assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                address_map.push_back(src_address_offset_dram);
                address_map.push_back(dst_address_offset_l1);
                address_map.push_back(read_size_bytes);
                address_map.push_back(pad);
                address_map_current_group_size += 4;
            }
        }
        // DRAM reads should be 32B aligned
        assert(address_map_current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(address_map_current_group_dram_address_offset);
        address_map_metadata.push_back(address_map_current_group_size);
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t address_map_current_group_size_padded = (uint32_t) (ceil((double) address_map_current_group_size / (double) 8) * 8);
        if(address_map_current_group_size_padded != address_map_current_group_size) {
            assert(address_map_current_group_size_padded > address_map_current_group_size);
            address_map.insert(address_map.end(), address_map_current_group_size_padded - address_map_current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        address_map_current_group_dram_address_offset += (address_map_current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

std::pair<vector<uint32_t>, vector<uint32_t>> generate_conv_activation_address_map(
                            const std::array<uint32_t, 4>& activation_shape,
                            const vector<int>& conv_params,
                            uint32_t act_block_h_datums,
                            uint32_t act_block_w_datums,
                            uint32_t weight_block_w_datums,
                            uint32_t num_blocks_act_h,
                            uint32_t num_blocks_act_w,
                            uint32_t num_blocks_weight_w,
                            uint32_t num_bytes_df) {
    vector<uint32_t> address_map;
    vector<uint32_t> address_map_metadata;
    uint32_t conv_input_x = activation_shape[3];
    uint32_t conv_input_y = activation_shape[2];
    uint32_t conv_input_z = activation_shape[1];
    uint32_t R = conv_params[0];
    uint32_t S = conv_params[1];
    uint32_t U = conv_params[2];
    uint32_t V = conv_params[3];
    uint32_t Pad_H = conv_params[4];
    uint32_t Pad_W = conv_params[5];
    uint32_t src_dram_buffer_size_bytes = conv_input_x * conv_input_y * conv_input_z * num_bytes_df;
    uint32_t dst_l1_buffer_size_bytes = act_block_h_datums * act_block_w_datums * num_bytes_df;
    int conv_output_h = ((conv_input_x - R + (2 * Pad_H)) / U) + 1;
    int conv_output_w = ((conv_input_y - S + (2 * Pad_W)) / V) + 1;
    uint32_t matrix_height_unpadded = conv_output_h * conv_output_w;
    uint32_t matrix_width_unpadded = conv_input_z * R * S;
    uint32_t matrix_height = (uint32_t) (ceil((double) matrix_height_unpadded / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t matrix_width = (uint32_t) (ceil((double) matrix_width_unpadded / (double) act_block_w_datums ) * act_block_w_datums);

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    uint32_t channel_stick_size = conv_input_z;
    uint32_t address_map_current_group_dram_address_offset = 0;
    address_map_metadata.push_back(num_groups);
    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {
        uint32_t block_idx_h = (uint32_t) (group_idx / num_blocks_act_w) / (num_blocks_weight_w);
        uint32_t block_idx_w = (uint32_t) (group_idx % num_blocks_act_w);
        uint32_t block_idx = (block_idx_h * num_blocks_act_w) + block_idx_w;
        uint32_t start_block_2d_index_h = block_idx_h * act_block_h_datums;
        uint32_t start_block_2d_index_w = block_idx_w * act_block_w_datums;
        uint32_t start_block_2d_index = (start_block_2d_index_h * act_block_w_datums * num_blocks_act_w) + start_block_2d_index_w;
        assert(start_block_2d_index_w < matrix_width_unpadded);
        uint32_t address_map_current_group_size = 0;
        for(uint32_t h_b = 0; h_b < act_block_h_datums; h_b++) {
            uint32_t h = start_block_2d_index_h + h_b;
            uint32_t dst_address_offset_l1 = h_b * act_block_w_datums * num_bytes_df;
            if (h >= matrix_height_unpadded) {
                // pad (block shape padding for height dim)
                uint32_t pad_size_bytes = act_block_w_datums * num_bytes_df;
                assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                address_map.push_back(0); // src address not used
                address_map.push_back(dst_address_offset_l1);
                address_map.push_back(pad_size_bytes);
                address_map.push_back(1); // pad = 1
                address_map_current_group_size += 4;
            }
            else {
                uint32_t w = start_block_2d_index_w;
                uint32_t end_block_2d_index_w = start_block_2d_index_w + act_block_w_datums - 1;
                assert(end_block_2d_index_w < matrix_width);
                while (w <= end_block_2d_index_w) {
                    uint32_t src_address_offset_dram = 0;
                    uint32_t read_size_bytes = 0;
                    uint32_t pad = 0;
                    if (w >= matrix_width_unpadded) {
                        // pad (block shape padding for width dim)
                        assert(end_block_2d_index_w == matrix_width-1);
                        read_size_bytes = (end_block_2d_index_w - w + 1) * num_bytes_df;
                        pad = 1;
                    }
                    else {
                        uint32_t channel_stick_offset = w % channel_stick_size;
                        uint32_t channel_stick_col_id = w / channel_stick_size;
                        uint32_t channel_stick_row_id = h;
                        assert(channel_stick_offset % (32/num_bytes_df) == 0); // DRAM read address must be aligned to 32 bytes
                        uint32_t channel_stick_row_id_x = channel_stick_row_id % conv_output_w;
                        uint32_t channel_stick_row_id_y = channel_stick_row_id / conv_output_w;
                        uint32_t act_tensor_start_x = channel_stick_row_id_x * V;
                        uint32_t act_tensor_start_y = channel_stick_row_id_y * U;
                        uint32_t act_tensor_padded_x = act_tensor_start_x + (channel_stick_col_id % S);
                        uint32_t act_tensor_padded_y = act_tensor_start_y + (channel_stick_col_id / S);
                        assert(w <= end_block_2d_index_w);
                        uint32_t read_size = min(channel_stick_size - channel_stick_offset, (end_block_2d_index_w+1)-w);
                        read_size_bytes = read_size * num_bytes_df;
                        if(act_tensor_padded_x < Pad_W || act_tensor_padded_x >= (Pad_W + conv_input_x) || act_tensor_padded_y < Pad_H || act_tensor_padded_y >= (Pad_H + conv_input_y)) {
                            // pad (conv padding)
                            pad = 1;
                        }
                        else {
                            uint32_t act_tensor_x = act_tensor_padded_x - Pad_W;
                            uint32_t act_tensor_y = act_tensor_padded_y - Pad_H;
                            assert(act_tensor_x < conv_input_x && act_tensor_x >= 0 && act_tensor_y < conv_input_y && act_tensor_y >= 0);
                            uint32_t act_tensor_channel_id = act_tensor_y * conv_input_x + act_tensor_x;
                            src_address_offset_dram = ((act_tensor_channel_id * channel_stick_size) + channel_stick_offset) * num_bytes_df;
                            assert(src_address_offset_dram % 32 == 0); // DRAM read address must be aligned to 32 bytes
                        }
                    }
                    assert(read_size_bytes > 0);
                    assert(pad == 0 || pad == 1);
                    assert(src_address_offset_dram < src_dram_buffer_size_bytes);
                    assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                    address_map.push_back(src_address_offset_dram);
                    address_map.push_back(dst_address_offset_l1);
                    address_map.push_back(read_size_bytes);
                    address_map.push_back(pad);
                    address_map_current_group_size += 4;
                    dst_address_offset_l1 += read_size_bytes;
                    w += (read_size_bytes/num_bytes_df);
                    assert(w <= end_block_2d_index_w+1);
                }
            }
        }
        // DRAM reads should be 32B aligned
        assert(address_map_current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(address_map_current_group_dram_address_offset);
        address_map_metadata.push_back(address_map_current_group_size);
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t address_map_current_group_size_padded = (uint32_t) (ceil((double) address_map_current_group_size / (double) 8) * 8);
        if(address_map_current_group_size_padded != address_map_current_group_size) {
            assert(address_map_current_group_size_padded > address_map_current_group_size);
            address_map.insert(address_map.end(), address_map_current_group_size_padded - address_map_current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        address_map_current_group_dram_address_offset += (address_map_current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

std::pair<vector<uint32_t>, vector<uint32_t>> populate_address_map_vectors_for_reader_kernel(vector<uint32_t> address_map_raw) {
    // This function is called twice i.e., for activation and weight address maps
    // "address_map_raw" is the DTX address map vector returned from DTX "conv_transform" function.
    // "address_map_raw" contains metadata along with the address map data for all groups
    // To keep the reader kernel simple, the metadata is separated into a different buffer
    // So two buffers are created -
    // First buffer is in DRAM containing the address map for all groups
    //      This DRAM buffer is big and is streamed into L1 scratchpad
    // Second buffer contains the metadata and is copied to L1 from host
    // It contains number of groups in its first index, followed by group info for each group -
    //      1. dram read address offset of address map group in dram buffer (in bytes)
    //      2. size of address map group in dram buffer (in datums, not bytes)
    // TODO (nshanker), support for streaming the second buffer from dram if it does not fit in L1
    vector<uint32_t> address_map; // will be in dram
    vector<uint32_t> address_map_metadata; // will be in l1

    uint32_t num_address_map_fields_per_transfer = 4; // TODO (nshanker): remove hardcoded 4 and get this value from output of DTX
    uint32_t num_dtx_groups = address_map_raw[0];
    address_map_metadata.push_back(address_map_raw[0]);
    uint32_t address_map_raw_index = 1;
    uint32_t current_group_dram_address_offset = 0;
    for(uint32_t g = 0; g < num_dtx_groups; g++) {
        // insert group's dram read address (in bytes) in metadata buffer
        // Separate reads are issued for each "address map group"
        // DRAM reads should be 32B aligned
        assert(current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(current_group_dram_address_offset);
        // insert group size (datums, not in bytes) into metadata buffer
        uint32_t current_group_size = address_map_raw[address_map_raw_index];
        address_map_metadata.push_back(current_group_size);
        address_map_raw_index += 1;
        // insert address map for this group into the address map buffer
        auto address_map_raw_current_group_start = address_map_raw.begin() + address_map_raw_index;
        address_map.insert(address_map.end(),
                                address_map_raw_current_group_start,
                                address_map_raw_current_group_start + current_group_size);
        address_map_raw_index += current_group_size;
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t current_group_size_padded = (uint32_t) (ceil((double) current_group_size / (double) 8) * 8);
        if(current_group_size_padded != current_group_size) {
            assert(current_group_size_padded > current_group_size);
            address_map.insert(address_map.end(), current_group_size_padded - current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        current_group_dram_address_offset += (current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

Program conv_as_large_bmm_with_address_map_single_core_(const Tensor& a, const Tensor &b, vector<int> conv_params,
                                       uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                                       uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, bool untilize_out, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");
    TT_ASSERT(a.shape()[0] == 1, "Only batch size 1 supported.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    uint32_t activation_C = a.shape()[1];
    //TT_ASSERT(activation_C % TILE_WIDTH == 0, "Channel depth must be divisible by tile width(32).");
    // Compute the 2d matrix shape
    vector<int> activation_shape = {(int)a.shape()[1], (int)a.shape()[2], (int)a.shape()[3]};    // TODO: Update types to use just one kind
    // Shape activation_shape_shape = {a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]};
    auto matrix_shape = compute_conv_activation_as_mm_shape(activation_shape, conv_params, act_block_h_ntiles, act_block_w_ntiles);
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
    TT_ASSERT(a.storage_type() == StorageType::DEVICE and b.storage_type() == StorageType::DEVICE, "Operands to large matmul need to be on device!");
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

    assert(Hat % act_block_h_ntiles == 0);
    assert(Wat % act_block_w_ntiles == 0);
    assert(Wbt % weight_block_w_ntiles == 0);

    uint32_t num_blocks_act_h = Hat / act_block_h_ntiles;
    uint32_t num_blocks_act_w = Wat / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = Wbt / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = Wa / num_blocks_act_w;
    uint32_t act_block_h_datums = Ha / num_blocks_act_h;

    // weight block info
    uint32_t weight_block_w_datums = Wb / num_blocks_weight_w;
    assert(weight_block_w_ntiles % out_subblock_w_ntiles == 0);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    // DTX conv activation transform data access pattern
    auto [act_address_map, act_address_map_metadata] = generate_conv_activation_address_map(a.shape(), conv_params, act_block_h_datums, act_block_w_datums, weight_block_w_datums,
                                                            num_blocks_act_h, num_blocks_act_w, num_blocks_weight_w, num_bytes_of_df);

    auto [weight_address_map, weight_address_map_metadata] = generate_conv_weight_address_map(b.shape(), act_block_w_datums, weight_block_w_datums,
                                                                num_blocks_act_h, num_blocks_act_w, num_blocks_weight_w, num_bytes_of_df);

    // sanity check
    uint32_t num_dtx_groups = act_address_map_metadata[0];
    assert(weight_address_map_metadata[0] == num_dtx_groups);

    // debug prints
    int detailed_debug = 1;
    if(detailed_debug > 0) {
        log_debug(tt::LogOp, "Printing activation and weight address maps.");
        log_debug(tt::LogOp, "DTX groups: {}", num_dtx_groups);
        uint32_t act_metadata_index = 1;
        uint32_t weight_metadata_index = 1;
        uint32_t act_addr_map_index = 0;
        uint32_t weight_addr_map_index = 0;
        for(uint32_t g = 0; g < num_dtx_groups; g++) {
            log_debug(tt::LogOp, "  DTX group: {}", g);
            uint32_t act_current_group_address = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            uint32_t act_current_group_size = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            log_debug(tt::LogOp, "      act_current_group_address: {}", act_current_group_address);
            log_debug(tt::LogOp, "      act_current_group_size: {}", act_current_group_size);
            if(detailed_debug > 1) {
                uint32_t act_current_group_index = act_current_group_address/sizeof(uint32_t);
                for(uint32_t i = act_current_group_index; i < act_current_group_index + act_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          act_addr_map[0]: {}", act_address_map[i]);
                    log_debug(tt::LogOp, "          act_addr_map[1]: {}", act_address_map[i+1]);
                    log_debug(tt::LogOp, "          act_addr_map[2]: {}", act_address_map[i+2]);
                    log_debug(tt::LogOp, "          act_addr_map[3]: {}", act_address_map[i+3]);
                }
            }
            uint32_t weight_current_group_address = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            uint32_t weight_current_group_size = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            log_debug(tt::LogOp, "      weight_current_group_address: {}", weight_current_group_address);
            log_debug(tt::LogOp, "      weight_current_group_size: {}", weight_current_group_size);
            if(detailed_debug > 1) {
                uint32_t weight_current_group_index = weight_current_group_address/sizeof(uint32_t);
                for(uint32_t i = weight_current_group_index; i < weight_current_group_index + weight_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          weight_addr_map[0]: {}", weight_address_map[i]);
                    log_debug(tt::LogOp, "          weight_addr_map[1]: {}", weight_address_map[i+1]);
                    log_debug(tt::LogOp, "          weight_addr_map[2]: {}", weight_address_map[i+2]);
                    log_debug(tt::LogOp, "          weight_addr_map[3]: {}", weight_address_map[i+3]);
                }
            }
        }
    }

    uint32_t dram_bank_id = 0;
    auto act_address_map_buffer_size_in_dram = act_address_map.size() * sizeof(uint32_t);
    auto act_address_map_dram_buffer = tt_metal::Buffer(device, act_address_map_buffer_size_in_dram, act_address_map_buffer_size_in_dram, tt_metal::BufferType::DRAM);
    auto weight_address_map_buffer_size_in_dram = weight_address_map.size() * sizeof(uint32_t);
    auto weight_address_map_dram_buffer = tt_metal::Buffer(device, weight_address_map_buffer_size_in_dram, weight_address_map_buffer_size_in_dram, tt_metal::BufferType::DRAM);
    uint32_t act_address_map_dram_addr = act_address_map_dram_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(act_address_map_dram_addr%32 == 0);
    auto act_address_map_dram_noc_xy = act_address_map_dram_buffer.noc_coordinates();
    uint32_t act_address_map_dram_noc_x = act_address_map_dram_noc_xy.x;
    uint32_t act_address_map_dram_noc_y = act_address_map_dram_noc_xy.y;
    uint32_t weight_address_map_dram_addr = weight_address_map_dram_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(weight_address_map_dram_addr%32 == 0);
    auto weight_address_map_dram_noc_xy = weight_address_map_dram_buffer.noc_coordinates();
    uint32_t weight_address_map_dram_noc_x = weight_address_map_dram_noc_xy.x;
    uint32_t weight_address_map_dram_noc_y = weight_address_map_dram_noc_xy.y;

    // Write address maps to DRAM
    WriteToDeviceDRAMChannel(device, dram_bank_id, act_address_map_dram_addr, act_address_map);
    WriteToDeviceDRAMChannel(device, dram_bank_id, weight_address_map_dram_addr, weight_address_map);

    tt_metal::Program program = tt_metal::Program();
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core = {.start={0, 0}, .end={0, 0}};
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // L1 buffers
    // Create scratchpad buffer in L1 to stream in dtx address map from dram
    // One scratchpad buffer is used for both activation and weight address maps
    uint32_t num_address_map_fields_per_transfer = 4; // TODO: (nshanker): remove hardcoded 4 and get this value from output of DTX
    // Scratchpad buffer size must be a multiple of 32B to ensure DRAM->L1 addresses align 32B
    auto scratch_pad_for_address_map_in_l1_b0_size_bytes = 32;
    // Scratchpad buffer size must also be a multiple of address map fields per transfer. We need all address map fields for a transfer in scratchpad.
    assert(scratch_pad_for_address_map_in_l1_b0_size_bytes % (num_address_map_fields_per_transfer*sizeof(uint32_t)) == 0);
    auto scratch_pad_for_address_map_l1_buffer = tt_metal::Buffer(device, scratch_pad_for_address_map_in_l1_b0_size_bytes, scratch_pad_for_address_map_in_l1_b0_size_bytes, tt_metal::BufferType::L1);
    uint32_t scratch_pad_for_address_map_l1_address = scratch_pad_for_address_map_l1_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(scratch_pad_for_address_map_l1_address%32 == 0);
    // Create address map metadata buffers in L1
    // Metadata vectors are copied to L1 buffers from host before calling LaunchKernels
    auto act_address_map_metadata_l1_b0_size = act_address_map_metadata.size() * sizeof(uint32_t);
    auto act_address_map_metadata_l1_buffer = tt_metal::Buffer(device, act_address_map_metadata_l1_b0_size, act_address_map_metadata_l1_b0_size, tt_metal::BufferType::L1);
    uint32_t act_address_map_metadata_l1_address = act_address_map_metadata_l1_buffer.address();
    auto weight_address_map_metadata_l1_b0_size = weight_address_map_metadata.size() * sizeof(uint32_t);
    auto weight_address_map_metadata_l1_buffer = tt_metal::Buffer(device, weight_address_map_metadata_l1_b0_size, weight_address_map_metadata_l1_b0_size, tt_metal::BufferType::L1);
    uint32_t weight_address_map_metadata_l1_address = weight_address_map_metadata_l1_buffer.address();

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_row_size = Wb * num_bytes_of_df;
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;

    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    assert(Wat % act_block_w_ntiles == 0);
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

    // For debug
    {
        log_debug(tt::LogOp, "Hat (activation height in tiles): {}", Hat);
        log_debug(tt::LogOp, "Wat (activation width in tiles): {}", Wat);
        log_debug(tt::LogOp, "Wbt (weight width in tiles): {}", Wbt);
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
        log_debug(tt::LogOp, "act_address_map_dram_addr: {}", act_address_map_dram_addr);
        log_debug(tt::LogOp, "act_address_map_metadata_l1_address: {}", act_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(tt::LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(tt::LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(tt::LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(tt::LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(tt::LogOp, "weight_address_map_dram_addr: {}", weight_address_map_dram_addr);
        log_debug(tt::LogOp, "weight_address_map_metadata_l1_address: {}", weight_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(tt::LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_dtx_groups: {}", num_dtx_groups);
        log_debug(tt::LogOp, "scratch_pad_for_address_map_l1_address: {}", scratch_pad_for_address_map_l1_address);
    }

    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        act_block_h_ntiles * act_block_w_ntiles,
        weight_block_h_ntiles * weight_block_w_ntiles,
        act_block_h_ntiles * weight_block_w_ntiles,
        weight_block_w_ntiles,
        num_bytes_of_df,
        untilize_out);

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    reader_kernel = "tt_metal/kernels/dataflow/reader_binary_dtx.cpp";
    reader_rt_args = {
        // arguments for act
        act_dram_addr,
        act_noc_x,
        act_noc_y,
        act_address_map_dram_addr,
        act_address_map_dram_noc_x,
        act_address_map_dram_noc_y,
        act_address_map_metadata_l1_address,
        act_block_num_tiles,

        // arguments for weight
        weight_dram_addr,
        weight_noc_x,
        weight_noc_y,
        weight_address_map_dram_addr,
        weight_address_map_dram_noc_x,
        weight_address_map_dram_noc_y,
        weight_address_map_metadata_l1_address,
        weight_block_num_tiles,

        scratch_pad_for_address_map_l1_address,
    };

    string writer_kernel;
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            act_block_h_datums,
            weight_block_w_ntiles*TILE_WIDTH*num_bytes_of_df,
            1,
            num_blocks_act_h,
            num_blocks_weight_w,
            Wb*num_bytes_of_df
        };
    } else {
        assert(false && "Tiled output unsupported");
        writer_kernel = "tt_metal/kernels/dataflow/writer_matmul_tile_layout.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,
            1,
            Wbt,
            out_subblock_w_ntiles,
            out_subblock_h_ntiles * Wbt,

            out_subblock_w_ntiles,
            out_subblock_h_ntiles,
            out_subblock_w_ntiles * out_subblock_h_ntiles,
            Wbt / out_subblock_w_ntiles,
            Hat / out_subblock_h_ntiles
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
        untilize_out
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_tilize_untilize.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        reader, core,
        reader_rt_args
    );

    tt_metal::SetRuntimeArgs(
        writer, core,
        writer_rt_args
    );

    tt_metal::WriteToDeviceL1(device, core_coord, act_address_map_metadata_l1_address, act_address_map_metadata);
    tt_metal::WriteToDeviceL1(device, core_coord, weight_address_map_metadata_l1_address, weight_address_map_metadata);

    return program;
}

Tensor conv_with_address_map(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles) {
    return operation::run(ConvWithAddressMap(act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, conv_params, true), {a, b}).at(0);
}

Program conv_with_address_map_single_core(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, bool untilize_out, Tensor &output) {
    return conv_as_large_bmm_with_address_map_single_core_(a, b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, untilize_out, output);
}

void ConvWithAddressMap::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
}

std::vector<Shape> ConvWithAddressMap::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    vector<int> input_tensor_a_shape = { (int) input_tensor_a.shape()[1], (int) input_tensor_a.shape()[2], (int) input_tensor_a.shape()[3]};
    auto mm_shape = compute_conv_activation_as_mm_shape(input_tensor_a_shape, conv_params, act_block_h_ntiles, act_block_w_ntiles);
    // TODO: Update batch size below
    Shape output_tensor_shape = {1, 1, mm_shape[1], input_tensor_b.shape()[3] };
    return {output_tensor_shape};
}

std::vector<Tensor> ConvWithAddressMap::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0);
    Tensor output = create_output_dram_buffer_(input_tensor.device(), input_tensor.dtype(), output_shape, untilize_out);
    std::vector<Tensor> output_tensors;
    // TODO: check if anything else needs to be done here.
    output_tensors.emplace_back(output);
    return output_tensors;
}

operation::ProgramWithCallbacks ConvWithAddressMap::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    return {conv_with_address_map_single_core(input_tensor_a, input_tensor_b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, untilize_out, output_tensor)};
}

}  // namespace tt_metal

}  // namespace tt
