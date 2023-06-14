#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "hostdevcommon/common_values.hpp"

using namespace tt::constants;
using namespace tt;


namespace mcast_reuse_generalized_helpers {
using namespace tt::constants;
using namespace tt;

tt_metal::Program create_program_mcast_in0_in1(
    tt_metal::Device *device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord start_core,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    uint32_t in0_dram_addr, uint32_t in1_dram_addr, uint32_t out_dram_addr
) {

    tt_metal::Program program = tt_metal::Program();

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange left_column{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange all_except_left_column{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange in0_sender_in1_sender{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y}};

    CoreRange in0_sender_in1_receiver{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange in0_receiver_in1_sender{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        .end{(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y}};

    CoreRange in0_receiver_in1_receiver{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    std::vector<uint32_t> reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = {1, (std::uint32_t)log2(single_tile_size)};
    } else {
        reader_writer_compile_time_args = {0, 0};
    }

    auto mm_reader_kernel_in0_sender_in1_sender = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        in0_sender_in1_sender,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);

    auto mm_reader_kernel_in0_sender_in1_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        in0_sender_in1_receiver,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);

    auto mm_reader_kernel_in0_receiver_in1_sender = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        in0_receiver_in1_sender,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto mm_reader_kernel_in0_receiver_in1_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        in0_receiver_in1_receiver,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel_noc0 = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_except_left_column,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    auto unary_writer_kernel_noc1 = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_tile_layout.cpp",
        left_column,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        compute_kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode
    );

    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);
    auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);

    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        all_cores,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
        all_cores,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_data_format
    );

    uint32_t interm0_cb_index = 24;
    auto cb_interm0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        interm0_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_output->address(),
        cb_data_format
    );

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            CoreCoord left_core    = {(std::size_t) start_core_x, (std::size_t) core.y};
            CoreCoord left_core_plus_one    = {(std::size_t) start_core_x + 1, (std::size_t) core.y};
            CoreCoord right_core   = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) core.y};
            CoreCoord top_core     = {(std::size_t) core.x, (std::size_t) start_core_y};
            CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
            CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

            auto left_core_physical = device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = device->worker_core_from_logical_core(right_core);
            auto top_core_physical = device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)  in0_dram_addr, // in0_tensor_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)  1, // in0_tensor_stride_w
                (std::uint32_t)  K, // in0_tensor_stride_h
                (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

                (std::uint32_t)  in1_dram_addr, // in1_tensor_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)  1, // in1_tensor_stride_w
                (std::uint32_t)  N, // in1_tensor_stride_h
                (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  K / in0_block_w, // num_blocks

                (std::uint32_t)  right_core_physical.x, // in0_mcast_dest_noc_start_x
                (std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  left_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
                (std::uint32_t)  left_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
                (std::uint32_t)  (num_cores_c - 1), // in0_mcast_num_dests
                (std::uint32_t)  left_core_physical.x, // in0_mcast_sender_noc_x
                (std::uint32_t)  left_core_physical.y, // in0_mcast_sender_noc_y
                (std::uint32_t)  in0_mcast_sender_semaphore->address(),
                (std::uint32_t)  in0_mcast_receiver_semaphore->address(),

                (std::uint32_t)  bottom_core_physical.x, // in0_mcast_dest_noc_start_x
                (std::uint32_t)  bottom_core_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  top_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
                (std::uint32_t)  top_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
                (std::uint32_t)  (num_cores_r - 1), // in0_mcast_num_dests
                (std::uint32_t)  top_core_physical.x, // in0_mcast_sender_noc_x
                (std::uint32_t)  top_core_physical.y, // in0_mcast_sender_noc_y
                (std::uint32_t)  in1_mcast_sender_semaphore->address(),
                (std::uint32_t)  in1_mcast_receiver_semaphore->address(),

                (std::uint32_t)  M * K, // MtKt
                (std::uint32_t)  K * N, // KtNt
                (std::uint32_t)  B, // batch
                (std::uint32_t)  bcast_batch // bcast_B
            };
            std::vector<uint32_t> writer_args = {
                (std::uint32_t) out_dram_addr, // out_tensor_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id
                (std::uint32_t) 1, // out_tensor_stride_w
                (std::uint32_t) N,  // out_tensor_stride_h
                (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

                (std::uint32_t) M * N, // MtNt
                (std::uint32_t) B // batch
            };

            if(core_idx_x == 0 and core_idx_y == 0) {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_in0_sender_in1_sender, core, mm_reader_args); // RISCV_0_default
                tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel_noc1, core, writer_args); // RISCV_1_default
            } else if (core_idx_x == 0 and core_idx_y != 0) {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_in0_sender_in1_receiver, core, mm_reader_args); // RISCV_0_default
                tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel_noc1, core, writer_args); // RISCV_1_default
            } else if (core_idx_x != 0 and core_idx_y == 0) {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_in0_receiver_in1_sender, core, mm_reader_args); // RISCV_1_default
                tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel_noc0, core, writer_args); // RISCV_0_default
            } else {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_in0_receiver_in1_receiver, core, mm_reader_args); // RISCV_1_default
                tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel_noc0, core, writer_args); // RISCV_0_default
            }

        }
    }

    return std::move(program);
}

tt_metal::Program create_program_mcast_in0(
    tt_metal::Device *device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord start_core,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    uint32_t in0_dram_addr, uint32_t in1_dram_addr, uint32_t out_dram_addr
) {

    tt_metal::Program program = tt_metal::Program();

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange mcast_senders{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x, (std::size_t) start_core_y + num_cores_r - 1}};
    CoreRange mcast_receivers{
        .start={(std::size_t) start_core_x + 1, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    std::vector<uint32_t> reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = {1, (std::uint32_t)log2(single_tile_size)};
    } else {
        reader_writer_compile_time_args = {0, 0};
    }

    auto mm_reader_kernel_sender = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_mcast_sender.cpp",
        mcast_senders,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto mm_reader_kernel_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0_mcast_receiver.cpp",
        mcast_receivers,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_cores,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B // batch
    };

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);

    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        all_cores,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
        all_cores,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_data_format
    );

    uint32_t interm0_cb_index = 24;
    auto cb_interm0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        interm0_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_output->address(),
        cb_data_format
    );

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            CoreCoord mcast_sender = {(std::size_t) start_core_x, core.y};
            CoreCoord core_start = {(std::size_t) start_core_x + 1, core.y};
            CoreCoord core_end = {(std::size_t) start_core_x + (num_cores_c - 1), core.y};
            auto mcast_sender_phyiscal = device->worker_core_from_logical_core(mcast_sender);
            auto core_start_physical = device->worker_core_from_logical_core(core_start);
            auto core_end_physical = device->worker_core_from_logical_core(core_end);

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)  in0_dram_addr, // in0_tensor_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)  1, // in0_tensor_stride_w
                (std::uint32_t)  K, // in0_tensor_stride_h
                (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

                (std::uint32_t)  in1_dram_addr, // in1_tensor_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)  1, // in1_tensor_stride_w
                (std::uint32_t)  N, // in1_tensor_stride_h
                (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  K / in0_block_w, // num_blocks

                (std::uint32_t)  core_end_physical.x, // in0_mcast_dest_noc_start_x
                (std::uint32_t)  core_end_physical.y, // in0_mcast_dest_noc_start_y
                (std::uint32_t)  core_start_physical.x, // in0_mcast_dest_noc_end_x
                (std::uint32_t)  core_start_physical.y, // in0_mcast_dest_noc_end_y
                (std::uint32_t)  num_cores_c - 1, // in0_mcast_num_dests
                (std::uint32_t)  mcast_sender_phyiscal.x, //in0_mcast_sender_noc_x
                (std::uint32_t)  mcast_sender_phyiscal.y, //in0_mcast_sender_noc_y
                (std::uint32_t) in0_mcast_sender_semaphore->address(),
                (std::uint32_t) in0_mcast_receiver_semaphore->address(),

                (std::uint32_t)  M * K, // MtKt
                (std::uint32_t)  K * N, // KtNt
                (std::uint32_t)  B, // batch
                (std::uint32_t)  bcast_batch // bcast_B
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) out_dram_addr, // out_tensor_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id
                (std::uint32_t) 1, // out_tensor_stride_w
                (std::uint32_t) N,  // out_tensor_stride_h
                (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

                (std::uint32_t) M * N, // MtNt
                (std::uint32_t) B // batch
            };

            if(core_idx_x == 0) {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_sender, core, mm_reader_args);
            } else {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_receiver, core, mm_reader_args);
            }
            tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel, core, writer_args);

        }
    }

    return std::move(program);
}

tt_metal::Program create_program_mcast_in1(
    tt_metal::Device *device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord start_core,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    uint32_t in0_dram_addr, uint32_t in1_dram_addr, uint32_t out_dram_addr
) {

    tt_metal::Program program = tt_metal::Program();

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange mcast_senders{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y}};
    CoreRange mcast_receivers{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y + 1},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    std::vector<uint32_t> reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = {1, (std::uint32_t)log2(single_tile_size)};
    } else {
        reader_writer_compile_time_args = {0, 0};
    }

    auto mm_reader_kernel_sender = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_mcast_sender.cpp",
        mcast_senders,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto mm_reader_kernel_receiver = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in1_mcast_receiver.cpp",
        mcast_receivers,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_tile_layout.cpp",
        all_cores,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles,
        B // batch
    };

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm.cpp",
        all_cores,
        compute_kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode
    );

    auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, device, all_cores, INVALID);

    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        all_cores,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
        all_cores,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_data_format
    );

    uint32_t interm0_cb_index = 24;
    auto cb_interm0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        interm0_cb_index,
        all_cores,
        out_CB_tiles,
        out_CB_size,
        cb_output->address(),
        cb_data_format
    );

    for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            CoreCoord mcast_sender = {core.x, (std::size_t) start_core_y};
            CoreCoord core_start = {core.x, (std::size_t) start_core_y + 1};
            CoreCoord core_end = {core.x, (std::size_t) start_core_y + (num_cores_r - 1)};
            auto mcast_sender_physical = device->worker_core_from_logical_core(mcast_sender);
            auto core_start_physical = device->worker_core_from_logical_core(core_start);
            auto core_end_physical = device->worker_core_from_logical_core(core_end);

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)  in0_dram_addr, // in0_tensor_addr
                (std::uint32_t)  K * per_core_M * core_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)  1, // in0_tensor_stride_w
                (std::uint32_t)  K, // in0_tensor_stride_h
                (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

                (std::uint32_t)  in0_block_w, // in0_block_w
                (std::uint32_t)  per_core_M, // in0_block_h
                (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

                (std::uint32_t)  in1_dram_addr, // in1_tensor_addr
                (std::uint32_t)  per_core_N * core_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)  1, // in1_tensor_stride_w
                (std::uint32_t)  N, // in1_tensor_stride_h
                (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

                (std::uint32_t)  per_core_N, // in1_block_w
                (std::uint32_t)  in0_block_w, //in1_block_h
                (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

                (std::uint32_t)  K / in0_block_w, // num_blocks

                (std::uint32_t)  core_end_physical.x, // in1_mcast_dest_noc_start_x
                (std::uint32_t)  core_end_physical.y, // in1_mcast_dest_noc_start_y
                (std::uint32_t)  core_start_physical.x, // in1_mcast_dest_noc_end_x
                (std::uint32_t)  core_start_physical.y, // in1_mcast_dest_noc_end_y
                (std::uint32_t)  num_cores_r - 1, // in1_mcast_num_dests
                (std::uint32_t)  mcast_sender_physical.x, //in1_mcast_sender_noc_x
                (std::uint32_t)  mcast_sender_physical.y, //in1_mcast_sender_noc_y
                (std::uint32_t)  in1_mcast_sender_semaphore->address(),
                (std::uint32_t)  in1_mcast_receiver_semaphore->address(),

                (std::uint32_t)  M * K, // MtKt
                (std::uint32_t)  K * N, // KtNt
                (std::uint32_t)  B, // batch
                (std::uint32_t)  bcast_batch // bcast_B
            };
            std::vector<uint32_t> writer_args = {
                (std::uint32_t) out_dram_addr, // out_tensor_addr
                (std::uint32_t) core_idx_x * per_core_N + core_idx_y * per_core_M * N, // out_tensor_start_tile_id
                (std::uint32_t) 1, // out_tensor_stride_w
                (std::uint32_t) N,  // out_tensor_stride_h
                (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
                (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

                (std::uint32_t) out_subblock_w, // out_subblock_w
                (std::uint32_t) out_subblock_h, // out_subblock_h
                (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
                (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
                (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

                (std::uint32_t) M * N, // MtNt
                (std::uint32_t) B // batch
            };

            if(core_idx_y == 0) {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_sender, core, mm_reader_args);
            } else {
                tt_metal::WriteRuntimeArgsToDevice(device, mm_reader_kernel_receiver, core, mm_reader_args);
            }
            tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel, core, writer_args);

        }
    }

    return std::move(program);
}

}

namespace tt {

namespace tt_metal {


Program matmul_multi_core_reuse_mcast_generalized_(const Tensor &a, const Tensor &b, Tensor& output, bool bcast_batch) {

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    TT_ASSERT(a.dtype() == b.dtype());
    TT_ASSERT(a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0);
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0);

    TT_ASSERT(ashape[3] == bshape[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
    TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(bshape[3] % TILE_WIDTH == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t in0_block_w = 2;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;

    // Get large matmul params
    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_blocks_total = (Mt / per_core_M) * (Nt / per_core_N);
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreCoord start_core = {0, 0};
    CoreCoord core_range = bmm_op_utils::get_core_range((Mt / per_core_M), (Nt / per_core_N), num_cores_y, num_cores_x);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    std::array<uint32_t, 4> cshape = output.shape(); // C=A*B, N1MK*11KN->N1MN
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t in0_dram_addr = src0_dram_buffer->address();
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    uint32_t out_dram_addr = dst_dram_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Program program;

    if (core_range.x > 1 && core_range.y > 1) {
        program = mcast_reuse_generalized_helpers::create_program_mcast_in0_in1(
            device,
            cb_data_format,
            math_fidelity,
            single_tile_size,
            start_core,
            core_range,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_dram_addr, in1_dram_addr, out_dram_addr
        );
    } else if (core_range.x > 1) {
        program = mcast_reuse_generalized_helpers::create_program_mcast_in0(
            device,
            cb_data_format,
            math_fidelity,
            single_tile_size,
            start_core,
            core_range,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_dram_addr, in1_dram_addr, out_dram_addr
        );
    } else {
        program = mcast_reuse_generalized_helpers::create_program_mcast_in1(
            device,
            cb_data_format,
            math_fidelity,
            single_tile_size,
            start_core,
            core_range,
            B, Mt, Nt, Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h, out_subblock_w,
            per_core_M, per_core_N,
            in0_dram_addr, in1_dram_addr, out_dram_addr
        );
    }

    // output does not hold any data, contains pointer to buffer on device with the data
    return program;
}

Program matmul_multi_core_reuse_mcast_generalized(const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor) {
    return matmul_multi_core_reuse_mcast_generalized_(input_tensor_a, input_tensor_b, output_tensor, true);
}

Program bmm_multi_core_reuse_mcast_generalized(const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor) {
    return matmul_multi_core_reuse_mcast_generalized_(input_tensor_a, input_tensor_b, output_tensor, false);
}

}  // namespace tt_metal

}  // namespace tt
