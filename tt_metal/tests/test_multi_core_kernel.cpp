#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

tt_metal::Device *initialize_device() {
    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int pci_express_slot = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    tt_metal::InitializeDevice(device);;
    return device;
}

tt_metal::Program *create_program(
    tt_metal::Device *device,
    uint32_t single_tile_size,
    const tt_metal::CoreRange &all_cores,
    tt_metal::ComputeKernelArgs *eltwise_unary_args) {
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair start_core = all_cores.first;
    tt_xy_pair end_core = all_cores.second;
    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = tt_xy_pair(x, y);
            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 8;
            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src0_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                src0_cb_addr,
                tt::DataFormat::Float16_b
            );

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t output_cb_addr = 300 * 1024;
            uint32_t num_output_tiles = 1;
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                output_cb_addr,
                tt::DataFormat::Float16_b
            );
        }
    }

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_push_4.cpp",
        all_cores,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary.cpp",
        all_cores,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_copy.cpp",
        all_cores,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    return program;
}

void compile_and_configure_program(
    tt_metal::Device *device,
    tt_metal::Program *program,
    std::vector<uint32_t> &src_vec,
    tt_metal::DramBuffer *src_dram_buffer) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    tt_metal::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::WriteToDeviceDRAM(src_dram_buffer, src_vec);

    tt_metal::ConfigureDeviceWithProgram(device, program);
}

void write_same_runtime_args_to_device(
    tt_metal::Device *device, tt_metal::Program *program, const tt_metal::CoreRange &core_range, int32_t num_tiles, tt_metal::DramBuffer *src_dram_buffer, tt_metal::DramBuffer *dst_dram_buffer) {
    auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    std::vector<uint32_t> unary_reader_args{
    (std::uint32_t)src_dram_buffer->address(),
    (std::uint32_t)dram_src_noc_xy.x,
    (std::uint32_t)dram_src_noc_xy.y,
    (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args{
    (std::uint32_t)dst_dram_buffer->address(),
    (std::uint32_t)dram_dst_noc_xy.x,
    (std::uint32_t)dram_dst_noc_xy.y,
    (std::uint32_t)num_tiles};

    for (auto dm_kernel : program->data_movement_kernels()) {
        if (dm_kernel->name() == "reader_unary_push_4") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_range, unary_reader_args);
        } else if (dm_kernel->name() == "writer_unary") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_range, unary_writer_args);
        }
    }
}

void write_unique_writer_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program *program,
    const tt_metal::CoreRange &core_range,
    const tt_metal::CoreBlocks &core_blocks,
    int32_t num_tiles,
    tt_metal::DramBuffer *src_dram_buffer,
    tt_metal::DramBuffer *dst_dram_buffer_1,
    tt_metal::DramBuffer *dst_dram_buffer_2,
    tt_metal::DramBuffer *dst_dram_buffer_3
) {
    auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    // All dst buffers use the same DRAM channel
    auto dram_dst_noc_xy = dst_dram_buffer_1->noc_coordinates();

    // Same readers args because all kernels read from same src
    std::vector<uint32_t> unary_reader_args{
        (std::uint32_t)src_dram_buffer->address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_1{
        dst_dram_buffer_1->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_2{
        dst_dram_buffer_2->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_3{
        dst_dram_buffer_3->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    for (auto dm_kernel : program->data_movement_kernels()) {
        if (dm_kernel->name() == "reader_unary_push_4") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_range, unary_reader_args);
        } else if (dm_kernel->name() == "writer_unary") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_blocks, {unary_writer_args_1, unary_writer_args_2, unary_writer_args_3});
        }
    }
}

void write_unique_reader_writer_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program *program,
    const tt_metal::CoreBlocks &core_blocks,
    int32_t num_tiles_1,
    int32_t num_tiles_2,
    int32_t num_tiles_3,
    tt_metal::DramBuffer *src_dram_buffer,
    tt_metal::DramBuffer *dst_dram_buffer_1,
    tt_metal::DramBuffer *dst_dram_buffer_2,
    tt_metal::DramBuffer *dst_dram_buffer_3
) {
    auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    // All dst buffers use the same DRAM channel
    auto dram_dst_noc_xy = dst_dram_buffer_1->noc_coordinates();

    // Data movement kernels across core groups read and write different number of tiles
    std::vector<uint32_t> unary_reader_args_1{
        src_dram_buffer->address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles_1};

    std::vector<uint32_t> unary_reader_args_2{
        src_dram_buffer->address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles_2};

    std::vector<uint32_t> unary_reader_args_3{
        src_dram_buffer->address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles_3};

    std::vector<uint32_t> unary_writer_args_1{
        dst_dram_buffer_1->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles_1};

    std::vector<uint32_t> unary_writer_args_2{
        dst_dram_buffer_2->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles_2};

    std::vector<uint32_t> unary_writer_args_3{
        dst_dram_buffer_3->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles_3};

    for (auto dm_kernel : program->data_movement_kernels()) {
        if (dm_kernel->name() == "reader_unary_push_4") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_blocks, {unary_reader_args_1, unary_reader_args_2, unary_reader_args_3});
        } else if (dm_kernel->name() == "writer_unary") {
            tt_metal::WriteRuntimeArgsToDevice(device, dm_kernel, core_blocks, {unary_writer_args_1, unary_writer_args_2, unary_writer_args_3});
        }
    }
}

bool test_multi_core_kernel_same_runtime_same_compile_time_args(tt_metal::Device *device) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {2, 2};

    tt_metal::CoreRange all_cores(start_core, end_core);

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src_addr = 0;
    int dram_src_channel_id = 0;

    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_dst_channel_id = 0;

    auto src_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
    auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    // Same compile time args for all cores
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(all_cores, compute_kernel_args);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program *program = create_program(device, single_tile_size, all_cores, eltwise_unary_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, src_dram_buffer);

    write_same_runtime_args_to_device(device, program, all_cores, num_tiles, src_dram_buffer, dst_dram_buffer);

    tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec);

    return pass;
}

bool test_multi_core_kernel_unique_runtime_same_compile_time_args(tt_metal::Device *device) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {1, 1};
    tt_metal::CoreRange core_group({0, 1}, {1, 1});
    tt_xy_pair single_core = {1, 0};
    tt_metal::CoreRange all_cores(start_core, end_core);
    tt_metal::CoreBlocks core_blocks = {start_core, single_core, core_group};

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src_addr = 0;
    int dram_src_channel_id = 0;

    uint32_t dram_buffer_dst_addr_1 = 512 * 1024 * 1024; // 512 MB (upper half)
    uint32_t dram_buffer_dst_addr_2 = dram_buffer_dst_addr_1 + dram_buffer_size;
    uint32_t dram_buffer_dst_addr_3 = dram_buffer_dst_addr_2 + dram_buffer_size;
    int dram_dst_channel_id = 0;

    auto src_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
    auto dst_dram_buffer_1 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr_1);
    auto dst_dram_buffer_2 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr_2);
    auto dst_dram_buffer_3 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr_3);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(all_cores, compute_kernel_args);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program *program = create_program(device, single_tile_size, all_cores, eltwise_unary_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, src_dram_buffer);

    write_unique_writer_runtime_args_to_device(
        device, program, all_cores, core_blocks, num_tiles, src_dram_buffer, dst_dram_buffer_1, dst_dram_buffer_2, dst_dram_buffer_3);

    tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> result_vec_1;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_1, result_vec_1);

    std::vector<uint32_t> result_vec_2;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_2, result_vec_2);

    std::vector<uint32_t> result_vec_3;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_3, result_vec_3);


    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec_1);
    pass &= (src_vec == result_vec_2);
    pass &= (src_vec == result_vec_3);

    return pass;
}

bool test_multi_core_kernel_unique_runtime_unique_compile_time_args(tt_metal::Device *device) {
    bool pass = true;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {1, 1};
    tt_metal::CoreRange core_group({0, 1}, {1, 1});
    tt_xy_pair single_core = {1, 0};
    tt_metal::CoreRange all_cores(start_core, end_core);
    tt_metal::CoreBlocks core_blocks = {start_core, single_core, core_group};

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles_1 = 2048;
    int32_t num_tiles_2 = num_tiles_1 / 2;
    int32_t num_tiles_3 = num_tiles_2 / 2;
    uint32_t dram_buffer_size_1 = single_tile_size * num_tiles_1; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_2 = single_tile_size * num_tiles_2; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_3 = single_tile_size * num_tiles_3; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src_addr = 0;
    int dram_src_channel_id = 0;

    uint32_t dram_buffer_dst_addr_1 = 512 * 1024 * 1024; // 512 MB (upper half)
    uint32_t dram_buffer_dst_addr_2 = dram_buffer_dst_addr_1 + dram_buffer_size_1;
    uint32_t dram_buffer_dst_addr_3 = dram_buffer_dst_addr_2 + dram_buffer_size_2;
    int dram_dst_channel_id = 0;

    auto src_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size_1, dram_buffer_src_addr);
    auto dst_dram_buffer_1 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size_1, dram_buffer_dst_addr_1);
    auto dst_dram_buffer_2 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size_2, dram_buffer_dst_addr_2);
    auto dst_dram_buffer_3 = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size_3, dram_buffer_dst_addr_3);


    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    vector<vector<uint32_t>> compute_kernel_args_for_all_cores = {
        {uint(num_tiles_1)},
        {uint(num_tiles_2)},
        {uint(num_tiles_3)}
    };

    // Difference in number of tiles read/written specified by different compile time args on compute kernel
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(
        core_blocks,
        compute_kernel_args_for_all_cores
    );

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program *program = create_program(device, single_tile_size, all_cores, eltwise_unary_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, src_dram_buffer);

    write_unique_reader_writer_runtime_args_to_device(
        device, program, core_blocks, num_tiles_1, num_tiles_2, num_tiles_3, src_dram_buffer, dst_dram_buffer_1, dst_dram_buffer_2, dst_dram_buffer_3
    );

    tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> result_vec_1;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_1, result_vec_1);

    std::vector<uint32_t> result_vec_2;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_2, result_vec_2);

    std::vector<uint32_t> result_vec_3;
    tt_metal::ReadFromDeviceDRAM(dst_dram_buffer_3, result_vec_3);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> src_vec_2(src_vec.begin(), src_vec.begin() + (src_vec.size() / 2));
    std::vector<uint32_t> src_vec_3(src_vec.begin(), src_vec.begin() + (src_vec.size() / 4));

    pass &= (src_vec == result_vec_1);
    pass &= (src_vec_2 == result_vec_2);
    pass &= (src_vec_3 == result_vec_3);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {

        tt_metal::Device *device = initialize_device();

        pass &= test_multi_core_kernel_same_runtime_same_compile_time_args(device);

        pass &= test_multi_core_kernel_unique_runtime_same_compile_time_args(device);

        pass &= test_multi_core_kernel_unique_runtime_unique_compile_time_args(device);

        ////////////////////////////////////////////////////////////////////////////
        //                          Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
