#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

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
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});

    std::vector<std::uint8_t> src0_vec(n_elements_per_tile, 0);
    std::vector<std::uint8_t> src1_vec(n_elements_per_tile, 0);
    std::vector<std::uint8_t> golden(n_elements_per_tile, 0);
    std::mt19937 rng(std::random_device{}());
    src0_vec[0] = 0;  // 0
    src1_vec[0] = 0;  // 0
    golden[0] = 0;

    src0_vec[1] = 0b00000001;  // 1
    src1_vec[1] = 0b00000010;  // 2
    golden[1] = 0b00000011;

    src0_vec[2] = 0b10000001;  // -1
    src1_vec[2] = 0b10000010;  // -2
    golden[2] = 0b10000011;

    src0_vec[3] = 0b00000101;  // 5
    src1_vec[3] = 0b10000011;  // -3
    golden[3] = 0b00000010;

    src0_vec[4] = 0b10000101;  // -5
    src1_vec[4] = 0b00000011;  // 3
    golden[4] = 0b10000010;

    src0_vec[5] = 0b01111111;  // 127
    src1_vec[5] = 0b00000000;  // 0
    golden[5] = 0b01111111;

    src0_vec[6] = 0b11111111;  // -127
    src1_vec[6] = 0b00000000;  // 0
    golden[6] = 0b11111111;

    src0_vec[7] = 0b01111111;  // 127
    src1_vec[7] = 0b00000001;  // 1
    golden[7] = 0b01111111;

    src0_vec[8] = 0b11111111;  // -127
    src1_vec[8] = 0b00000001;  // 1
    golden[8] = 0b11111110;

    src0_vec[9] = 0b01111111;  // 127
    src1_vec[9] = 0b11111111;  // -127
    golden[9] = 0;

    src0_vec[10] = 0b11111111;  // -127
    src1_vec[10] = 0b01111111;  // 127
    golden[10] = 0;

    src0_vec[11] = 0b11111111;  // -127
    src1_vec[11] = 0b10000001;  // -1
    golden[11] = 0b11111111;

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
        if (result_vec[i] != golden[i]) {
            printf("Mismatch at index %zu: ", i);
            printf("golden = ");
            for (int b = 7; b >= 0; --b) {
                printf("%d", (golden[i] >> b) & 1);
            }
            printf(", result = ");
            for (int b = 7; b >= 0; --b) {
                printf("%d", (result_vec[i] >> b) & 1);
            }

            printf("\n");
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
    mesh_device->close();
}
