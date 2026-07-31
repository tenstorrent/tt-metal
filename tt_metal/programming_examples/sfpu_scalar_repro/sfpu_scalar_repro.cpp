// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal repro for a customer report: add_binary_tile() then a scalar op (div/mul/add/sub/rsub
// _unary_tile) inside one tile_regs_acquire()/commit() block. Takes op_mode + the raw param1 bits
// to hand the scalar op straight from argv, so every configuration in the investigation table can
// be run without rebuilding.
//
//   argv[1] op_mode:     0=div_unary_tile 1=mul_unary_tile 2=add_unary_tile 3=sub_unary_tile
//                        4=rsub_unary_tile
//   argv[2] scalar_bits: uint32_t, the exact bits handed to the SFPU op as param1

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    TT_FATAL(argc == 3, "Usage: {} <op_mode 0-4> <scalar_bits uint32_t>", argv[0]);
    uint32_t op_mode = std::strtoul(argv[1], nullptr, 0);
    uint32_t scalar_bits = std::strtoul(argv[2], nullptr, 0);

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = single_tile_size};
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    constexpr uint32_t cb_in0 = CBIndex::c_0;
    tt_metal::CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(single_tile_size, {{cb_in0, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in0, single_tile_size));

    constexpr uint32_t cb_in1 = CBIndex::c_1;
    tt_metal::CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(single_tile_size, {{cb_in1, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in1, single_tile_size));

    constexpr uint32_t cb_out = CBIndex::c_16;
    tt_metal::CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(single_tile_size, {{cb_out, tt::DataFormat::Float16_b}})
            .set_page_size(cb_out, single_tile_size));

    std::vector<uint32_t> reader_compile_time_args = {cb_in0, cb_in1};
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_scalar_repro/kernels/dataflow/reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {cb_out};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_scalar_repro/kernels/dataflow/writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {cb_in0, cb_in1, cb_out};
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_scalar_repro/kernels/compute/compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});
    SetRuntimeArgs(program, compute_kernel_id, core, {op_mode, scalar_bits});
    (void)reader_kernel_id;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    fmt::print(
        "op_mode={} scalar_bits={} (0x{:08x}) -> DST[0][0]={}\n",
        op_mode,
        scalar_bits,
        scalar_bits,
        static_cast<float>(result_vec[0]));

    mesh_device->close();
    return 0;
}
