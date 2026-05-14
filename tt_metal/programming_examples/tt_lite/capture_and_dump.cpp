// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-lite PoC: Capture an eltwise-add trace and dump it to a .ttb file.
// This program:
//  1. Opens a MeshDevice (auto-detects system shape)
//  2. Allocates replicated input/output DRAM buffers
//  3. Creates an eltwise-add program on a single Tensix core
//  4. Writes input data + runs the program once (to warm JIT caches)
//  5. Captures a trace of the program execution
//  6. Dumps the trace command stream + buffer metadata to a .ttb file
//  7. Verifies the output is correct

#include <iostream>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_metal/distributed/mesh_trace.hpp"

#include "trace_binary.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static Program CreateEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    uint32_t tile_size_bytes,
    uint32_t num_tiles) {
    auto program = CreateProgram();
    auto core = CoreRange(CoreCoord{0, 0});

    constexpr uint32_t src0_cb = tt::CBIndex::c_0;
    constexpr uint32_t src1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_size_bytes, {{src0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb, tile_size_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_size_bytes, {{src1_cb, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb, tile_size_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_size_bytes, {{out_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb, tile_size_bytes));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*a->get_reference_buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*b->get_reference_buffer()).append_to(reader_ct_args);
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/interleaved_tile_read.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*c->get_reference_buffer()).append_to(writer_ct_args);
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {},
            .defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}}});

    SetRuntimeArgs(program, reader, core, {a->address(), b->address(), num_tiles});
    SetRuntimeArgs(program, writer, core, {c->address(), num_tiles});
    SetRuntimeArgs(program, compute, core, {num_tiles});

    return program;
}

int main(int argc, char** argv) {
    std::string output_path = "model.ttb";
    if (argc > 1) {
        output_path = argv[1];
    }

    // 1. Open device with single-chip mesh (sufficient for PoC)
    auto mesh_shape = MeshShape{1, 1};
    std::cout << "Using mesh shape: " << mesh_shape << std::endl;

    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(mesh_shape),
        0,        // l1_small_size
        16 << 20  // trace_region_size = 16 MB
    );

    auto& cq = mesh_device->mesh_command_queue();

    // 2. Allocate buffers (replicated across mesh)
    constexpr uint32_t num_tiles = 4;
    uint32_t tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t buf_size = num_tiles * tile_size;

    ReplicatedBufferConfig global_cfg{.size = buf_size};
    DeviceLocalBufferConfig local_cfg{.page_size = tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    auto buf_a = MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    auto buf_b = MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    auto buf_c = MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());

    // 3. Populate input data
    constexpr float val_a = 2.0f;
    constexpr float val_b = 3.0f;
    auto data_a = create_constant_vector_of_bfloat16(buf_size, val_a);
    auto data_b = create_constant_vector_of_bfloat16(buf_size, val_b);

    EnqueueWriteMeshBuffer(cq, buf_a, data_a, /*blocking=*/false);
    EnqueueWriteMeshBuffer(cq, buf_b, data_b, /*blocking=*/false);

    // 4. Create program and run once (JIT compile + warm caches)
    auto program = CreateEltwiseAddProgram(buf_a, buf_b, buf_c, tile_size, num_tiles);
    auto device_range = MeshCoordinateRange(mesh_device->shape());
    auto workload = MeshWorkload();
    workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // Verify initial run
    std::vector<uint32_t> result(data_a.size(), 0);
    EnqueueReadMeshBuffer(cq, result, buf_c, /*blocking=*/true);

    bfloat16* result_bf16 = reinterpret_cast<bfloat16*>(result.data());
    float expected = val_a + val_b;
    bool pass = true;
    for (size_t i = 0; i < result.size() * 2; i++) {
        if (!is_close(static_cast<float>(result_bf16[i]), expected, 0.01f)) {
            pass = false;
            break;
        }
    }
    std::cout << "Initial run: " << (pass ? "PASS" : "FAIL") << std::endl;
    if (!pass) return 1;

    // 5. Capture trace
    std::cout << "Capturing trace..." << std::endl;
    auto trace_id = BeginTraceCapture(mesh_device.get(), 0);
    EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device->end_mesh_trace(0, trace_id);
    std::cout << "Trace captured." << std::endl;

    // 6. Verify trace replay works
    // Write fresh input
    constexpr float val_a2 = 10.0f;
    constexpr float val_b2 = 7.0f;
    auto data_a2 = create_constant_vector_of_bfloat16(buf_size, val_a2);
    auto data_b2 = create_constant_vector_of_bfloat16(buf_size, val_b2);
    EnqueueWriteMeshBuffer(cq, buf_a, data_a2, /*blocking=*/true);
    EnqueueWriteMeshBuffer(cq, buf_b, data_b2, /*blocking=*/true);

    mesh_device->replay_mesh_trace(0, trace_id, /*blocking=*/true);

    std::vector<uint32_t> result2(data_a.size(), 0);
    EnqueueReadMeshBuffer(cq, result2, buf_c, /*blocking=*/true);
    bfloat16* result2_bf16 = reinterpret_cast<bfloat16*>(result2.data());
    float expected2 = val_a2 + val_b2;
    bool pass2 = true;
    for (size_t i = 0; i < result2.size() * 2; i++) {
        if (!is_close(static_cast<float>(result2_bf16[i]), expected2, 0.01f)) {
            pass2 = false;
            break;
        }
    }
    std::cout << "Trace replay verify: " << (pass2 ? "PASS" : "FAIL") << std::endl;
    if (!pass2) return 1;

    // 7. Dump trace to .ttb file
    auto trace_buffer = mesh_device->get_mesh_trace(trace_id);
    auto& desc = *trace_buffer->desc;

    tt::lite::TraceBinary ttb{};
    ttb.header.magic = tt::lite::TTB_MAGIC;
    ttb.header.version = tt::lite::TTB_VERSION;
    ttb.header.num_worker_descs = desc.descriptors.size();
    ttb.header.num_trace_streams = desc.ordered_trace_data.size();
    ttb.header.num_persistent_buffers = 0;
    ttb.header.num_io_buffers = 3;

    for (auto& [sub_device_id, wd] : desc.descriptors) {
        ttb.worker_descs.push_back(tt::lite::TraceWorkerDesc{
            .sub_device_id = static_cast<uint8_t>(*sub_device_id),
            .num_completion_worker_cores = wd.num_completion_worker_cores,
            .num_mcast_programs = wd.num_traced_programs_needing_go_signal_multicast,
            .num_unicast_programs = wd.num_traced_programs_needing_go_signal_unicast,
        });
    }

    for (auto& td : desc.ordered_trace_data) {
        ttb.trace_streams.push_back(td.data);
    }

    auto add_io_buf = [&](const std::shared_ptr<MeshBuffer>& buf, const std::string& name) {
        ttb.io_buffers.push_back(tt::lite::BufferPlacement{
            .address = buf->address(),
            .size = buf->size(),
            .page_size = buf->page_size(),
        });
        ttb.io_buffer_names.push_back(name);
    };
    add_io_buf(buf_a, "input_a");
    add_io_buf(buf_b, "input_b");
    add_io_buf(buf_c, "output_c");

    ttb.trace_buf_address = trace_buffer->mesh_buffer->address();
    ttb.trace_buf_page_size = trace_buffer->mesh_buffer->page_size();
    ttb.trace_buf_num_pages = trace_buffer->mesh_buffer->num_pages();

    if (tt::lite::write_trace_binary(ttb, output_path)) {
        std::cout << "Trace binary written to: " << output_path << std::endl;
        std::cout << "  Trace streams: " << ttb.trace_streams.size() << std::endl;
        for (size_t i = 0; i < ttb.trace_streams.size(); i++) {
            std::cout << "    Stream " << i << ": " << ttb.trace_streams[i].size() * 4 << " bytes" << std::endl;
        }
        std::cout << "  Trace buffer addr: 0x" << std::hex << ttb.trace_buf_address << std::dec << std::endl;
        std::cout << "  Trace buffer page_size: " << ttb.trace_buf_page_size << std::endl;
        std::cout << "  Trace buffer num_pages: " << ttb.trace_buf_num_pages << std::endl;
        std::cout << "  IO buffers:" << std::endl;
        for (size_t i = 0; i < ttb.io_buffers.size(); i++) {
            std::cout << "    " << ttb.io_buffer_names[i] << " addr=0x" << std::hex << ttb.io_buffers[i].address
                      << " size=" << std::dec << ttb.io_buffers[i].size
                      << " page_size=" << ttb.io_buffers[i].page_size << std::endl;
        }
    } else {
        std::cerr << "Failed to write trace binary!" << std::endl;
        return 1;
    }

    mesh_device->release_mesh_trace(trace_id);
    mesh_device->close();
    return 0;
}
