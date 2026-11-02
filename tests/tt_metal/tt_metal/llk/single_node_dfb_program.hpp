// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared single-node Metal 2.0 test program: each input streams DRAM -> direct_reader_unary_2_0 -> DFB into the
// compute kernel, whose output DFB drains via direct_writer_unary_2_0 -> DRAM. Runs on node {0,0}.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::unit_tests::llk::single_node_dfb {

// One input: its host data, tile format and count, the compute kernel's accessor, and the DFB depth.
struct Input {
    std::vector<std::uint32_t> data;
    tt::DataFormat format = tt::DataFormat::Invalid;
    std::uint32_t num_tiles = 0;
    std::string accessor;
    std::uint32_t num_entries = 0;
};

struct Output {
    tt::DataFormat format = tt::DataFormat::Invalid;
    std::uint32_t num_tiles = 0;
    std::string accessor;
    std::uint32_t num_entries = 0;
};

struct Config {
    std::string name;
    std::vector<Input> inputs;
    Output output;
    std::string compute_kernel;
    experimental::KernelSpec::CompileTimeArgs compile_time_args;
    experimental::ComputeHardwareConfig hw_config;
};

// Runs the program and returns the output DRAM buffer.
inline std::vector<std::uint32_t> run(distributed::MeshDevice& mesh_device, const Config& p) {
    const experimental::NodeCoord node{0, 0};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // The direct reader/writer kernels address a single DRAM bank (bank_id 0). Use page_size = whole buffer so
    // the allocator places each buffer in one bank, and advance the DRAM pointer by the native tile size.
    auto make_dram = [&](std::uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            {.page_size = size, .buffer_type = BufferType::DRAM},
            &mesh_device);
    };

    experimental::ProgramSpec spec{.name = p.name};
    experimental::WorkUnitSpec wu{.name = "main", .target_nodes = node};
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = p.compute_kernel,
        .num_threads = 1,
        .compile_time_args = p.compile_time_args,
        .hw_config = p.hw_config,
    };
    std::vector<experimental::KernelSpecName> readers;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> src_buffers;
    for (std::size_t i = 0; i < p.inputs.size(); i++) {
        const Input& in = p.inputs[i];
        const experimental::DFBSpecName dfb{"in" + std::to_string(i) + "_dfb"};
        const experimental::KernelSpecName reader{"reader" + std::to_string(i)};
        spec.dataflow_buffers.push_back({
            .unique_id = dfb,
            .entry_size = tt::tile_size(in.format),
            .num_entries = in.num_entries,
            .data_format_metadata = in.format,
        });
        spec.kernels.push_back({
            .unique_id = reader,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ProducerOf(dfb, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        });
        compute_spec.dfb_bindings.push_back(experimental::ConsumerOf(dfb, in.accessor));
        wu.kernels.push_back(reader);
        readers.push_back(reader);
        src_buffers.push_back(make_dram(in.num_tiles * tt::tile_size(in.format)));
    }
    const std::uint32_t out_tile_size = tt::tile_size(p.output.format);
    auto dst_buffer = make_dram(p.output.num_tiles * out_tile_size);
    spec.dataflow_buffers.push_back({
        .unique_id = OUT_DFB,
        .entry_size = out_tile_size,
        .num_entries = p.output.num_entries,
        .data_format_metadata = p.output.format,
    });
    spec.kernels.push_back({
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    });
    compute_spec.dfb_bindings.push_back(experimental::ProducerOf(OUT_DFB, p.output.accessor));
    spec.kernels.push_back(std::move(compute_spec));
    wu.kernels.push_back(WRITER);
    wu.kernels.push_back(COMPUTE);
    spec.work_units = {wu};

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    auto& cq = mesh_device.mesh_command_queue();
    experimental::ProgramRunArgs params;
    for (std::size_t i = 0; i < p.inputs.size(); i++) {
        const Input& in = p.inputs[i];
        distributed::EnqueueWriteMeshBuffer(cq, src_buffers[i], in.data, /*blocking=*/true);
        params.kernel_run_args.push_back({
            .kernel = readers[i],
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_buffers[i]->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", in.num_tiles},
                 {"dram_page_stride", tt::tile_size(in.format)}}),
        });
    }
    params.kernel_run_args.push_back({
        .kernel = WRITER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"dst_addr", dst_buffer->address()},
             {"dst_bank_id", 0u},
             {"num_tiles", p.output.num_tiles},
             {"dram_page_stride", out_tile_size}}),
    });
    params.kernel_run_args.push_back({.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_buffer, /*blocking=*/true);
    return result_vec;
}

}  // namespace tt::tt_metal::unit_tests::llk::single_node_dfb
