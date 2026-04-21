#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

#endif
int main() {
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(std::int8_t) * n_elements_per_tile;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig distributed_buffer_config{.size = single_tile_size};
    auto src0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());

    constexpr uint32_t num_tiles = 1;
    auto make_cb_config = [&](CBIndex cb_index) {
        return CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, DataFormat::Int8}})
            .set_page_size(cb_index, single_tile_size);
    };

    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));

    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/reader_elwsub.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/writer_elwsub.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute_elwsub.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});

    std::vector<std::uint8_t> src0_vec(n_elements_per_tile, 0);
    std::vector<std::uint8_t> src1_vec(n_elements_per_tile, 0);
    std::mt19937 rng(std::random_device{}());
    src0_vec[0] = 0;  // 0
    src1_vec[0] = 0;  // 0

    src0_vec[1] = 0b01111000;  // 120
    src1_vec[1] = 0b0;         // 0

    src0_vec[2] = 0;           // 0
    src1_vec[2] = 0b00110100;  // 52

    src0_vec[3] = 0b00011000;  // 24
    src1_vec[3] = 0b00011000;  // 24

    src0_vec[4] = 0b00000001;  // 1
    src1_vec[4] = 0b00000010;  // 2

    src0_vec[5] = 0b00000100;  // 4
    src1_vec[5] = 0b00000011;  // 3

    src0_vec[6] = 0b00110100;  // 52
    src1_vec[6] = 0b10011110;  // -30

    src0_vec[7] = 0b11100010;  // -98
    src1_vec[7] = 0b10001111;  // -15

    src0_vec[8] = 0b10100001;  // -33
    src1_vec[8] = 0b10000101;  // -5

    src0_vec[9] = 0b01111111;  // 127
    src1_vec[9] = 0b01111111;  // 127

    src0_vec[10] = 0b11111111;  // -127
    src1_vec[10] = 0b11111111;  // -127

    src0_vec[11] = 0b01111111;  // 127
    src1_vec[11] = 0b11111111;  // -127

    src0_vec[12] = 0b11111111;  // -127
    src1_vec[12] = 0b01111111;  // 127

    src0_vec[13] = 0b01111111;  // 127
    src1_vec[13] = 0b00000001;  // 1

    src0_vec[14] = 0b11111111;  // -127
    src1_vec[14] = 0b00000001;  // 1

    src0_vec[15] = 0b11111111;  // -127
    src1_vec[15] = 0b10000001;  // -1

    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {(uint32_t)src0_dram_buffer->address(), (uint32_t)src1_dram_buffer->address()});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {(uint32_t)dst_dram_buffer->address()});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint8_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    bool success = true;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        printf("%x\n", result_vec[i]);
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
    mesh_device->close();
}
