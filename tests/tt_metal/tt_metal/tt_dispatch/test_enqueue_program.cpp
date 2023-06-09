#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;

u32 NUM_TILES = 2048;

void zero_out_sysmem(Device *device) {
    // Prior to running anything, need to clear out system memory
    // to prevent anything being stale. Potentially make it a static
    // method on command queue
    vector<u32> zeros(1024 * 1024 * 1024 / sizeof(u32), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);
}

tt_metal::Program generate_eltwise_unary_program(Device *device) {
    // TODO(agrebenisan): This is directly copy and pasted from test_eltwise_binary.
    // We need to think of a better way to generate test data, so this section needs to be heavily refactored.

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = NUM_TILES;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src0_addr = 0;
    int dram_src0_channel_id = 0;

    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024;  // 512 MB (upper half)
    int dram_dst_channel_id = 0;

    uint32_t page_size = single_tile_size;
    auto src0_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_src0_addr, dram_src0_channel_id, page_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, page_size, tt_metal::BufferType::DRAM);

    auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    unary_writer_kernel->add_define("DEVICE_DISPATCH_MODE", "1");

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    vector<uint32_t> compute_kernel_args = {
        NUM_TILES,  // per_core_block_cnt
        1,          // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);

    tt_metal::CompileProgram(device, program);
    return program;
}

void test_enqueue_program(std::function<tt_metal::Program(tt_metal::Device *device)> create_program) {


    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    tt_metal::InitializeDevice(device);

    tt_metal::Program program = create_program(device);

    // if (1) {
    //     auto brisc_binary_path = program.kernels().at(0)->binary_path() + "/brisc/brisc.hex";
    //     auto ncrisc_binary_path = program.kernels().at(1)->binary_path() + "/ncrisc/ncrisc.hex";
    //     auto trisc0_binary_path = program.kernels().at(2)->binary_path() + "/tensix_thread0/tensix_thread0.hex";
    //     auto trisc1_binary_path = program.kernels().at(2)->binary_path() + "/tensix_thread1/tensix_thread1.hex";
    //     auto trisc2_binary_path = program.kernels().at(2)->binary_path() + "/tensix_thread2/tensix_thread2.hex";

    //     tt::llrt::test_load_write_read_risc_binary(device->cluster(), brisc_binary_path, 0, {1, 1}, 0);
    //     tt::llrt::test_load_write_read_risc_binary(device->cluster(), ncrisc_binary_path, 0, {1, 1}, 1);
    //     tt::llrt::test_load_write_read_risc_binary(device->cluster(), trisc0_binary_path, 0, {1, 1}, 2);
    //     tt::llrt::test_load_write_read_risc_binary(device->cluster(), trisc1_binary_path, 0, {1, 1}, 3);
    //     tt::llrt::test_load_write_read_risc_binary(device->cluster(), trisc2_binary_path, 0, {1, 1}, 4);

    //     auto vec = tt::llrt::read_hex_vec_from_core(device->cluster(), 0, {1, 1}, 0, 1024 * 1024);
    //     for (u32 el: vec) {
    //         std::cout << el << std::endl;
    //     }
    //     return;
    // }


    CoreCoord worker_core(0, 0);
    zero_out_sysmem(device);
    vector<u32> inp = create_random_vector_of_bfloat16(NUM_TILES * 2048, 100, 0);

    vector<u32> out_vec;
    {
        CommandQueue cq(device);

        // Enqueue program inputs
        Buffer buf(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);
        Buffer out(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);

        RuntimeArgs rt_args;
        map<Riscv, vector<u32>> worker_core_rt_args = {{Riscv::B, {out.address(), 0, 0, NUM_TILES}}, {Riscv::N, {buf.address(), 0, 0, NUM_TILES}}};
        rt_args[worker_core] = worker_core_rt_args;

        EnqueueWriteBuffer(cq, buf, inp, false);
        EnqueueProgram(cq, program, rt_args, false);

        EnqueueReadBuffer(cq, out, out_vec, true);
    }

    TT_ASSERT(out_vec == inp);

}

int main() {
    // test_program_to_device_map();
    test_enqueue_program(generate_eltwise_unary_program);
    // test_enqueue_program(generate_simple_brisc_program);

    // test_relay_program_to_dram(generate_simple_brisc_program);
    // test_compare_program_binaries_with_bins_on_disk();
}
