// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared single-core, classic-circular-buffer program runners for the id-free (2.0) compute-API golden tests.
// Each runner builds a one-core program that streams tiles through a compute kernel (selected by path) and
// returns the raw device output; the caller validates it against a host-computed golden. These are the proven
// harnesses the 2.0 kernels were authored against (c_0[/c_1] in via reader_unary/reader_binary, c_16 out via
// writer_unary), factored out of test_fp8_typecast.cpp so every llk home test file can reuse them.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::unit_tests::llk::single_core {

using std::vector;

// Run a single-input datacopy-style kernel (c_0 -> c_16). The unpacker reads input_fmt, the packer writes
// output_fmt, so a format mismatch performs an implicit typecast. fp32_dest_acc_en selects Dest 32-bit mode.
// cb_depth_tiles must be >= any block size the kernel keeps resident (a 1-tile CB deadlocks wait_front(BLOCK)).
inline vector<std::uint32_t> run_unary(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<std::uint32_t>& src_vec,
    std::uint32_t num_tiles,
    bool fp32_dest_acc_en,
    const std::string& compute_kernel,
    std::uint32_t cb_depth_tiles = 1) {
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    std::uint32_t input_tile_size = tt::tile_size(input_fmt);
    std::uint32_t output_tile_size = tt::tile_size(output_fmt);

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = num_tiles * input_tile_size,
        .page_size = num_tiles * input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = num_tiles * output_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

    CircularBufferConfig cb_src_config =
        CircularBufferConfig(cb_depth_tiles * input_tile_size, {{tt::CBIndex::c_0, input_fmt}})
            .set_page_size(tt::CBIndex::c_0, input_tile_size);
    CreateCircularBuffer(program, core, cb_src_config);

    CircularBufferConfig cb_dst_config =
        CircularBufferConfig(cb_depth_tiles * output_tile_size, {{tt::CBIndex::c_16, output_fmt}})
            .set_page_size(tt::CBIndex::c_16, output_tile_size);
    CreateCircularBuffer(program, core, cb_dst_config);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles}});

    detail::WriteToBuffer(src_buffer, src_vec);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Run a two-input kernel: c_0, c_1 in (reader_binary) -> c_16 out (writer_unary). All tiles Float16_b.
// out_tiles defaults to num_tiles (elementwise N->N); reducing ops that collapse the block to fewer outputs
// must pass out_tiles explicitly or writer_unary over-reads c_16 and deadlocks. cb_depth_tiles must be >= any
// resident block size. compute_defines lets a shipping kernel (e.g. eltwise_binary.cpp) be steered via defines.
// extra_compile_args are appended after num_tiles as get_compile_time_arg_val(1..N) -- fused kernels read a mode
// enum (e.g. BroadcastType, or PoolType/ReduceDim) from them so one kernel covers a whole family.
inline vector<std::uint32_t> run_binary(
    distributed::MeshDevice& mesh_device,
    const vector<std::uint32_t>& src0_vec,
    const vector<std::uint32_t>& src1_vec,
    std::uint32_t num_tiles,
    const std::string& compute_kernel,
    const std::map<std::string, std::string>& compute_defines = {},
    std::uint32_t cb_depth_tiles = 1,
    std::uint32_t out_tiles = 0,
    const vector<std::uint32_t>& extra_compile_args = {}) {
    if (out_tiles == 0) {
        out_tiles = num_tiles;
    }
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const tt::DataFormat fmt = tt::DataFormat::Float16_b;
    std::uint32_t tile_bytes = tt::tile_size(fmt);

    auto make_dram = [&](std::uint32_t ntiles) {
        InterleavedBufferConfig cfg{
            .device = dev,
            .size = ntiles * tile_bytes,
            .page_size = ntiles * tile_bytes,
            .buffer_type = BufferType::DRAM};
        return CreateBuffer(cfg);
    };
    auto src0_buffer = make_dram(num_tiles);
    auto src1_buffer = make_dram(num_tiles);
    auto dst_buffer = make_dram(out_tiles);

    auto make_cb = [&](tt::CBIndex idx) {
        CircularBufferConfig cb_cfg =
            CircularBufferConfig(cb_depth_tiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cb_cfg);
    };
    make_cb(tt::CBIndex::c_0);
    make_cb(tt::CBIndex::c_1);
    make_cb(tt::CBIndex::c_16);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<std::uint32_t> compile_args = {num_tiles};
    compile_args.insert(compile_args.end(), extra_compile_args.begin(), extra_compile_args.end());
    auto compute = CreateKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = false, .compile_args = compile_args, .defines = compute_defines});

    detail::WriteToBuffer(src0_buffer, src0_vec);
    detail::WriteToBuffer(src1_buffer, src1_vec);
    SetRuntimeArgs(program, reader, core, {src0_buffer->address(), 0, src1_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, out_tiles});
    // Shipping eltwise_binary.cpp reads runtime args {per_core_block_cnt, per_core_block_size, acc_to_dst};
    // the id-free kernels read only compile-time num_tiles and ignore these (harmless). One tile per block.
    SetRuntimeArgs(program, compute, core, {num_tiles, 1, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Run a single-tile matmul: c_0 -> SrcB, c_1 -> SrcA -> c_16. All tiles Float16_b. 7 matmul compile args all 1.
inline vector<std::uint32_t> run_matmul_single(
    distributed::MeshDevice& mesh_device,
    const vector<std::uint32_t>& src0_vec,
    const vector<std::uint32_t>& src1_vec,
    const std::string& compute_kernel) {
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const tt::DataFormat fmt = tt::DataFormat::Float16_b;
    std::uint32_t tile_bytes = tt::tile_size(fmt);

    auto make_dram = [&]() {
        InterleavedBufferConfig cfg{
            .device = dev, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(cfg);
    };
    auto src0_buffer = make_dram();
    auto src1_buffer = make_dram();
    auto dst_buffer = make_dram();

    auto make_cb = [&](tt::CBIndex idx) {
        CircularBufferConfig cb_cfg = CircularBufferConfig(tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cb_cfg);
    };
    make_cb(tt::CBIndex::c_0);
    make_cb(tt::CBIndex::c_1);
    make_cb(tt::CBIndex::c_16);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // 7 matmul compile args, all 1: block_tile_dim, dst_tile_rows, dst_tile_cols, block_cnt,
    // in0_block_tile_cnt, in1_block_tile_cnt, out_block_tile_cnt.
    CreateKernel(
        program, compute_kernel, core, ComputeConfig{.fp32_dest_acc_en = false, .compile_args = {1, 1, 1, 1, 1, 1, 1}});

    detail::WriteToBuffer(src0_buffer, src0_vec);
    detail::WriteToBuffer(src1_buffer, src1_vec);
    SetRuntimeArgs(program, reader, core, {src0_buffer->address(), 0, src1_buffer->address(), 0, 1});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, 1});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Run a block matmul: A (rt_dim x kt_dim) -> c_0 -> SrcB, B (kt_dim x ct_dim) -> c_1 -> SrcA, C (rt_dim x ct_dim)
// -> c_16. reader_binary reads the same tile count into c_0 and c_1, so callers must satisfy rt_dim == ct_dim.
// CBs hold the whole block resident. Compile args = {ct_dim, rt_dim, kt_dim}.
inline vector<std::uint32_t> run_matmul_block(
    distributed::MeshDevice& mesh_device,
    const vector<std::uint32_t>& src0_vec,  // in0 (A) block: rt_dim*kt_dim tiles
    const vector<std::uint32_t>& src1_vec,  // in1 (B) block: kt_dim*ct_dim tiles
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    const std::string& compute_kernel) {
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const tt::DataFormat fmt = tt::DataFormat::Float16_b;
    std::uint32_t tile_bytes = tt::tile_size(fmt);

    const std::uint32_t in0_tiles = rt_dim * kt_dim;  // A block
    const std::uint32_t in1_tiles = kt_dim * ct_dim;  // B block
    const std::uint32_t out_tiles = rt_dim * ct_dim;  // C block
    // reader_binary reads the same count into c_0 and c_1 -> this runner requires rt_dim == ct_dim.
    TT_FATAL(in0_tiles == in1_tiles, "run_matmul_block: reader_binary needs in0_tiles == in1_tiles (rt==ct)");

    auto make_dram = [&](std::uint32_t n) {
        InterleavedBufferConfig cfg{
            .device = dev, .size = n * tile_bytes, .page_size = n * tile_bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(cfg);
    };
    auto src0_buffer = make_dram(in0_tiles);
    auto src1_buffer = make_dram(in1_tiles);
    auto dst_buffer = make_dram(out_tiles);

    auto make_cb = [&](tt::CBIndex idx, std::uint32_t depth) {
        CircularBufferConfig cfg =
            CircularBufferConfig(depth * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cfg);
    };
    make_cb(tt::CBIndex::c_0, in0_tiles);
    make_cb(tt::CBIndex::c_1, in1_tiles);
    make_cb(tt::CBIndex::c_16, out_tiles);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = false, .compile_args = {ct_dim, rt_dim, kt_dim}});

    detail::WriteToBuffer(src0_buffer, src0_vec);
    detail::WriteToBuffer(src1_buffer, src1_vec);
    SetRuntimeArgs(program, reader, core, {src0_buffer->address(), 0, src1_buffer->address(), 0, in0_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, out_tiles});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

}  // namespace tt::tt_metal::unit_tests::llk::single_core
