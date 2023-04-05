#include "tt_dnn/op_library/conv/conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "libs/dtx/dtx.hpp"
#include "libs/dtx/dtx_passes.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

void create_CBs_for_fused_matmul_c(tt_metal::Program* program,
                                tt_metal::Device* device,
                                tt_xy_pair core,
                                uint32_t M,
                                uint32_t N,
                                uint32_t in0_block_w,
                                uint32_t out_subblock_h,
                                uint32_t num_bytes_for_df) {
    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t src0_cb_addr = 120 * 1024;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_addr = 220 * 1024;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b
    );

    // Used for placing tilized activations
    uint32_t tilized_cb_addr = 320 * 1024;
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
        program,
        device,
        tilize_mode_tilized_in0_cb,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tilized_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t cb_matmul_partials_addr = 440 * 1024;
    auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
        program,
        device,
        matmul_partials_cb,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_matmul_partials_addr,
        tt::DataFormat::Float16_b
    );

    // Shares same address space as matmul partials
    uint32_t temp_addr = 560 * 1024;
    auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
        program,
        device,
        untilize_mode_final_matmul_partials_cb,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        temp_addr,
        tt::DataFormat::Float16_b
    );

    // Supposed to be a small CB only responsible for reorganizing
    // the output blocks to fill the whole "per core output block width"
    uint32_t reblock_cb_addr = 680 * 1024;
    uint32_t reblock_cb_tiles = N; // Only space for one row
    auto cb_reblock = tt_metal::CreateCircularBuffer(
        program,
        device,
        untilize_mode_reblock_cb,
        core,
        reblock_cb_tiles,
        reblock_cb_tiles * single_tile_size,
        reblock_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t output_cb_addr = 730 * 1024;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        out0_cb,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b
    );
}

Tensor create_output_dram_buffer_(Device * device, DataType data_type, std::array<uint32_t, 4> cshape) {
    tt::tt_metal::Layout out_layout;
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        data_type,
        tt::tt_metal::Layout::ROW_MAJOR,
        device);

    return output;
}

std::tuple<uint32_t, uint32_t, uint32_t> compute_block_info_(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t single_tile_size_bytes = 2 * 1024;

    // Constraint 1: in0 and in1 should fit in L1. If not, divide into blocks
    // Max sizes based on hard coded CB addressing
    uint32_t max_in0_bytes = 50 * 1024;
    uint32_t max_in1_bytes = 50 * 1024;
    uint32_t max_in0_tiles = max_in0_bytes / single_tile_size_bytes;
    uint32_t max_in1_tiles = max_in1_bytes / single_tile_size_bytes;
    std::cout << "max_in0_tiles=" << max_in0_tiles << std::endl;
    std::cout << "max_in1_tiles=" << max_in1_tiles << std::endl;
    uint32_t num_blocks = 1;
    uint32_t in_block_w = K;
    assert(M <= max_in0_tiles && N <= max_in1_tiles);
    uint32_t max_in_block_w = std::min((max_in0_tiles/M), (max_in1_tiles/N));
    while (in_block_w > max_in_block_w || K % num_blocks != 0) {
        num_blocks += 1;
        assert(num_blocks <= K);
        in_block_w = K / num_blocks;
    }
    std::cout << "Num blocks=" << num_blocks << std::endl;
    std::cout << "in0_block_w=" << in_block_w << std::endl;

    // Constraint 2: output should fit in L1
    uint32_t max_out_bytes = 120 * 1024;
    uint32_t max_out_tiles = max_out_bytes / single_tile_size_bytes;
    std::cout << "max_out_tiles=" << max_out_tiles << std::endl;
    assert (M*N <= max_out_tiles);

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
    std::cout << "out_subblock_h=" << out_subblock_h << std::endl;
    std::cout << "out_subblock_w=" << out_subblock_w << std::endl;
    return std::make_tuple(num_blocks, out_subblock_h, out_subblock_w);
}

// TODO(whoever gets a chance!): Refactor this so it's a part of matmul_single_core_... keeping it
// independent for now as it's easier for me to progress
Tensor conv_as_large_bmm_single_core_(const Tensor& a, const Tensor &b) {

    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");

    //vector<int> shape = {5, 4,4};
    vector<int> shape = {(int) a.shape()[1], (int) a.shape()[2], (int) a.shape()[3]};
    auto activation_C = a.shape()[1];
    TT_ASSERT(activation_C % TILE_WIDTH == 0, "Channel depth of tensor needs to be divisible by 32");
    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1(dtx_right);
    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    pass &= row_major_memory_store(dtx_right);

    //cout << "\n\nDTX_RIGHT" << endl;
    //dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    //cout << "\n\nDTX_LEFT" << endl;
    //dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();

    pass &= optimize_away_transpose(combined);
    //cout << "\n\nDTX_OPTIMIZED" << endl;
    //combined->print();

    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    //combined->print();
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;
    uint32_t t_bytes = 0;

    // Generate address map for reader kernel and
    // verify for the assumptions taken by the reader kernel about the DTX address map
    uint32_t transfer_size = combined->transformations.back()->groups[0]->transfers[0]->size*2; // 2 for bfloat16
    // Channels last layout -> so each transfer size should = activation_C
    assert(transfer_size == activation_C*2); // 2 for bfloat16
    for(auto transfer : combined->transformations.back()->groups[0]->transfers){
        address_map.push_back(transfer->src_address*2); // 2 for bfloat16
        // TODO: remove dst address. It is not used by the reader kernel because it writes to L1 destination contiguously
        address_map.push_back(transfer->dst_address*2);
        // transfer size should be the same for each transfer
        assert(transfer->size*2 == transfer_size);
        address_map.push_back(transfer->size*2);
        // Using multi bank reader
        // add stick id to address map which will be used to determine stick bank address in the reader kernel
        // data is in channels last "stick" layout in 8 banks where each stick size = activation_C
        assert(transfer->src_address % activation_C == 0); // src address points to the start of a stick
        auto stick_id = transfer->src_address / activation_C;
        address_map.push_back(stick_id);
        t_bytes += transfer->size*2;
    }

    uint32_t total_bytes = num_rows * num_cols * 2; // 2 for bfloat16
    assert(total_bytes == t_bytes);

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
    //TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    uint32_t single_tile_size = 2 * 1024; // TODO(agrebenisan): Refactor on df
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    // same condition as above, different message
    //TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor a must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};
    // pad height
    cshape[2] = (uint32_t) (ceil((double) cshape[2] / (double) TILE_HEIGHT ) * TILE_HEIGHT);

    Tensor output = create_output_dram_buffer_(a.device(), a.dtype(), cshape);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    uint32_t address_map_l1_addr = 980 * 1024;
    assert(address_map.size() * sizeof(uint32_t) <= 19 * 1024);
    auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, address_map.size() * sizeof(uint32_t), address_map_l1_addr);
    // Keep for now, but need to fix when you get to multibank
    uint32_t out_dram_addr = dst_dram_buffer->address();
    auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t out_dram_noc_x = out_dram_noc_xy.x;
    uint32_t out_dram_noc_y = out_dram_noc_xy.y;
    {
        // Convert tensor dims to tile dims
        uint32_t B   = Ba;
        uint32_t Hat = (uint32_t) ceil((double) Ha / (double) TILE_HEIGHT );
        uint32_t Wat = Wa / TILE_WIDTH;
        uint32_t Wbt = Wb / TILE_WIDTH;
        std::cout << "Hat=" << Hat << std::endl;
        std::cout << "Wat=" << Wat << std::endl;
        std::cout << "Wbt=" << Wbt << std::endl;
        // out
        uint32_t out_dram_addr = dst_dram_buffer->address();
        uint32_t out_row_size = Wb * 2;

        // out block info
        auto [num_blocks, out_subblock_h, out_subblock_w] = compute_block_info_(Hat, Wat, Wbt);
        //uint32_t out_subblock_h = 4;
        //uint32_t out_subblock_w = 2;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

        // in0
        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in0_row_size = Wa * 2; // some row major data needed in case we want to tilize A

        // Important, dictates in0 block width, in1 block height
        //uint32_t num_blocks = 2;

        // in0 block info
        uint32_t in0_block_w = Wat / num_blocks; // Two blocks in the W dimension
        uint32_t in0_channel_stick_size = Wat * 32 * 2;
        uint32_t in0_partial_channel_stick_size = (in0_block_w * 32) * 2;
        std::cout << "in0_channel_stick_size=" << in0_channel_stick_size << std::endl;
        std::cout << "in0_partial_channel_stick_size=" << in0_partial_channel_stick_size << std::endl;
        uint32_t in0_num_blocks_w = Wat / in0_block_w;
        uint32_t in0_num_rows = num_rows;
        uint32_t in0_num_subblocks = (Hat / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        // For height padding in reader kernel
        uint32_t total_zeroes_bytes_per_block = (cshape[2] - Ha) * in0_block_w * 32 * 2; // 2 for bfloat16
        std::cout << "num rows to pad = " << (cshape[2] - Ha) << std::endl;
        std::cout << "row size per block = " << in0_block_w * 32 << std::endl;
        uint32_t zero_buffer_size = l1_mem::address_map::ZEROS_SIZE;
        uint32_t num_bytes_of_zeroes_per_read = 0;
        uint32_t num_reads_of_zeroes = 0;
        uint32_t num_bytes_of_zeroes_remainder = 0;

        if(total_zeroes_bytes_per_block > zero_buffer_size) {
            num_bytes_of_zeroes_per_read = zero_buffer_size;
            num_reads_of_zeroes = total_zeroes_bytes_per_block / zero_buffer_size;
            num_bytes_of_zeroes_remainder = total_zeroes_bytes_per_block % zero_buffer_size;
        }
        else if(total_zeroes_bytes_per_block > 0) {
            num_bytes_of_zeroes_per_read = total_zeroes_bytes_per_block;
            num_reads_of_zeroes = 1;
        }

        // in1
        uint32_t in1_dram_addr = src1_dram_buffer->address();

        // in1 block info
        uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
        uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
        uint32_t in1_block_w = out_subblock_w * in1_num_subblocks;
        uint32_t in1_block_h = in0_block_w;

        // For debug, uncomment this
        /*
        std::cout << "in0 information" << std::endl;
        std::cout << "\t in0_dram_addr: " << in0_dram_addr << std::endl;
        std::cout << "\t in0_row_size: " << in0_row_size << std::endl;
        std::cout << "\t in0_block_w: " << in0_block_w << std::endl;
        std::cout << "\t in0_partial_row_size: " << in0_partial_row_size << std::endl;
        std::cout << "\t in0_num_blocks_w: " << in0_num_blocks_w << std::endl;
        std::cout << "\t in0_block_h: " << in0_block_h << std::endl;
        std::cout << "\t in0_num_subblocks: " << in0_num_subblocks << std::endl;
        std::cout << "\t in0_block_num_tiles: " << in0_block_num_tiles << std::endl;
        std::cout << "\t in0_subblock_h: " << in0_subblock_h << std::endl;
        std::cout << "\t in0_subblock_num_tiles: " << in0_subblock_num_tiles << std::endl;

        std::cout << "in1 information" << std::endl;
        std::cout << "\t in1_dram_addr: " << in1_dram_addr << std::endl;
        std::cout << "\t in1_num_subblocks: " << in1_num_subblocks << std::endl;
        std::cout << "\t in1_block_num_tiles: " << in1_block_num_tiles << std::endl;
        std::cout << "\t in1_block_w: " << in1_block_w << std::endl;
        std::cout << "\t in1_block_h: " << in1_block_h << std::endl;

        std::cout << "out information" << std::endl;
        std::cout << "\t out_dram_addr: " << out_dram_addr << std::endl;
        std::cout << "\t out_row_size: " << out_row_size << std::endl;
        std::cout << "\t out_subblock_h: " << out_subblock_h << std::endl;
        std::cout << "\t out_subblock_w: " << out_subblock_w << std::endl;
        std::cout << "\t out_subblock_num_tiles: " << out_subblock_num_tiles << std::endl;
        */

        {
            create_CBs_for_fused_matmul_c(
                program,
                a.device(),
                core,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2); // TODO(agrebenisan): fix df num bytes

            std::cout << "Reader kernel args - " << std::endl;
            std::cout << "Num reads of zeroes - " << num_reads_of_zeroes << std::endl;
            std::cout << "Num bytes of zeroes per read - " << num_bytes_of_zeroes_per_read << std::endl;
            std::cout << "Num bytes of zeroes remainder - " << num_bytes_of_zeroes_remainder << std::endl;

            uint32_t in1_tensor_start_tile_id = 0;
            uint32_t in1_tensor_stride_w = 1;
            uint32_t in1_tensor_stride_h = Wbt;
            uint32_t in1_tensor_next_block_stride = in0_block_w * Wbt;
            uint32_t in0_num_channel_sticks_per_row = 1; // For 1x1 conv
            string reader_kernel;
            vector<uint32_t> reader_rt_args;
            reader_kernel = "tt_metal/kernels/dataflow/reader_multi_bank_matmul_blocked_cl_acts_tl_weights_dtx.cpp";
            reader_rt_args = {
                num_blocks,
                // arguments for in1
                in1_dram_addr,
                in1_block_w,
                in1_block_h,
                in1_block_num_tiles,
                in1_tensor_start_tile_id,
                in1_tensor_stride_w,
                in1_tensor_stride_h,
                in1_tensor_next_block_stride,
                // arguments for in0
                in0_dram_addr,
                in0_block_num_tiles,
                in0_num_rows,
                in0_num_channel_sticks_per_row,
                in0_channel_stick_size,
                in0_partial_channel_stick_size,
                num_bytes_of_zeroes_per_read,
                num_reads_of_zeroes,
                num_bytes_of_zeroes_remainder,
                address_map_l1_addr,
                (uint32_t)address_map.size()
            };

            string writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp";
            vector<uint32_t> writer_rt_args = {
                out_dram_addr,
                cshape[2], // padded height
                out_row_size
            };
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
                true
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
        }


        pass &= tt_metal::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    return output;
}

Tensor conv_as_large_bmm_single_core(const Tensor& a, const Tensor &b) {

    Tensor output = conv_as_large_bmm_single_core_(a, b);
    return output;
}

}  // namespace tt_metal

}  // namespace tt
