#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;
u32 NUM_TILES = 2048;

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

    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024;  // 512 MB (upper half)

    uint32_t page_size = single_tile_size;
    auto src0_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_src0_addr, page_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer = tt_metal::Buffer(
        device, dram_buffer_size, dram_buffer_dst_addr, page_size, tt_metal::BufferType::DRAM);

    auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        src0_cb_addr);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        output_cb_addr);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    vector<uint32_t> compute_kernel_args = {
        NUM_TILES,  // per_core_block_cnt
        1,          // per_core_block_size
    };

    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    tt_metal::CompileProgram(device, program);
    return program;
}

void test_enqueue_program(std::function<tt_metal::Program(tt_metal::Device *device)> create_program) {


    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    tt_metal::InitializeDevice(device);

    tt_metal::Program program = create_program(device);

    CoreCoord worker_core(0, 0);
    vector<u32> inp = create_random_vector_of_bfloat16(NUM_TILES * 2048, 100, 0);

    vector<u32> out_vec;
    {
        CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;

        // Enqueue program inputs
        Buffer buf(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);
        Buffer out(device, NUM_TILES * 2048, 0, 2048, BufferType::DRAM);

        // Absolutely disgusting way to query for the kernel I want to set runtime args for... needs to be cleaned up
        SetRuntimeArgs(program, program.kernels_on_core(worker_core).riscv0_id.value(), worker_core, {out.address(), 0, 0, NUM_TILES});
        SetRuntimeArgs(program, program.kernels_on_core(worker_core).riscv1_id.value(), worker_core, {buf.address(), 0, 0, NUM_TILES});

        EnqueueWriteBuffer(cq, buf, inp, false);
        EnqueueProgram(cq, program, false);

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
