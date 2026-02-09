#include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/buffer.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    constexpr tt::tt_metal::CoreCoord core = tt::tt_metal::CoreCoord(0, 0);

    distributed::DeviceLocalBufferConfig dram_config{.page_size = sizeof(uint32_t), .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_config{.page_size = sizeof(uint32_t), .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{
        .size = sizeof(uint32_t),
    };

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src0_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    auto src1_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    auto dst_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::vector<uint32_t> src0_vec = {55};
    std::vector<uint32_t> src1_vec = {17};

    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/false);

    KernelHandle kernel_id = CreateKernel(
        program,
        "/tt-metal/evans_projects/addition_riscv/kernels/compute.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {static_cast<uint32_t>(src0_dram_buffer->address()),
         static_cast<uint32_t>(src1_dram_buffer->address()),
         static_cast<uint32_t>(dst_dram_buffer->address()),
         static_cast<uint32_t>(src0_l1_buffer->address()),
         static_cast<uint32_t>(src1_l1_buffer->address()),
         static_cast<uint32_t>(dst_l1_buffer->address())});
    workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    Finish(cq);

    std::vector<uint32_t> result_vec;
    EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);
    if (result_vec.size() != 1) {
        std::cout << "Error: Expected result vector size of 1, got " << result_vec.size() << std::endl;
        mesh_device->close();
        return -1;
    }
    if (result_vec[0] != 72) {
        std::cout << "Error: Expected result of 72, got " << result_vec[0] << std::endl;
        mesh_device->close();
        return -1;
    }
    std::cout << "Success: Result is " << result_vec[0] << std::endl;
    mesh_device->close();
    return 0;
}
