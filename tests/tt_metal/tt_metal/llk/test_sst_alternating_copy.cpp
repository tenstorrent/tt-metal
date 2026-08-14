// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// M1 benchmark for static state tracking: alternating-format copy.
//
// Two streams of different data formats (values: Float16_b, indices: UInt16)
// alternate through the same SrcA -> DST -> pack path — the inner-loop shape
// of the sort / SDPA kernels. Both kernels are identity copies per stream, so
// validation is a bit-exact compare against the inputs. The LLK 1.0 baseline
// re-inits UNPACK/MATH and reconfigures PACK on every format swap; the SST
// kernel re-emits only the tracked descriptor deltas.

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using std::vector;
using namespace tt;

namespace unit_tests::compute::sst_alternating_copy {

struct AltCopyConfig {
    // Number of A/B alternation iterations; each moves 2 tiles per stream.
    std::uint32_t num_iters = 1;
    // Compute kernel path (LLK 1.0 baseline or SST implementation).
    std::string compute_kernel;
    // When > 0, re-enqueue the validated workload this many times and report
    // wall time per iteration (pipelined behind one Finish, after warmup).
    std::uint32_t perf_iterations = 0;
};

void run_alternating_copy_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const AltCopyConfig& config,
    double* perf_us_per_iter_out = nullptr) {
    auto& cq = mesh_device->mesh_command_queue();
    auto* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    constexpr std::uint32_t tile_size_bytes = 2 * 1024;  // 32x32, 2-byte datums (both formats)
    const std::uint32_t num_tiles = 2 * config.num_iters;  // per stream
    const std::uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = dev,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config);  // Float16_b values
    auto src1_dram_buffer = CreateBuffer(dram_config);  // UInt16 indices
    auto dst0_dram_buffer = CreateBuffer(dram_config);
    auto dst1_dram_buffer = CreateBuffer(dram_config);

    const experimental::DFBSpecName IN0{"in0_dfb"};
    const experimental::DFBSpecName IN1{"in1_dfb"};
    const experimental::DFBSpecName OUT0{"out0_dfb"};
    const experimental::DFBSpecName OUT1{"out1_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name, tt::DataFormat fmt) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size_bytes,
            .num_entries = num_tiles,
            .data_format_metadata = fmt,
        };
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN0, "in0"), experimental::ProducerOf(IN1, "in1")},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
                .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT0, "in0"), experimental::ConsumerOf(OUT1, "in1")},
        .runtime_arg_schema =
            {.runtime_arg_names = {"dst0_addr", "dst0_bank_id", "dst1_addr", "dst1_bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
                .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = config.compute_kernel,
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT0,
                 .accessor_name = "out0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT1,
                 .accessor_name = "out1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_iters", config.num_iters}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fp32_dest_acc_en = false,
                .dst_full_sync_en = false,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "sst_alternating_copy",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {make_dfb(IN0, tt::DataFormat::Float16_b),
             make_dfb(IN1, tt::DataFormat::UInt16),
             make_dfb(OUT0, tt::DataFormat::Float16_b),
             make_dfb(OUT1, tt::DataFormat::UInt16)},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    // Values stream: bfloat16 ramp. Index stream: uint16 ramp (two per word).
    std::vector<std::uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);
    std::vector<std::uint32_t> src1_vec(dram_buffer_size / sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < src1_vec.size(); ++i) {
        const std::uint32_t lo = (2 * i) & 0xFFFF;
        const std::uint32_t hi = (2 * i + 1) & 0xFFFF;
        src1_vec[i] = (hi << 16) | lo;
    }
    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);
    tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = READER,
        .runtime_arg_values =
            {{node,
              {{"src0_addr", static_cast<std::uint32_t>(src0_dram_buffer->address())},
               {"src0_bank_id", 0u},
               {"src1_addr", static_cast<std::uint32_t>(src1_dram_buffer->address())},
               {"src1_bank_id", 0u},
               {"num_tiles", num_tiles}}}},
    });
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = WRITER,
        .runtime_arg_values =
            {{node,
              {{"dst0_addr", static_cast<std::uint32_t>(dst0_dram_buffer->address())},
               {"dst0_bank_id", 0u},
               {"dst1_addr", static_cast<std::uint32_t>(dst1_dram_buffer->address())},
               {"dst1_bank_id", 0u},
               {"num_tiles", num_tiles}}}},
    });
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program_, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result0_vec;
    std::vector<std::uint32_t> result1_vec;
    tt_metal::detail::ReadFromBuffer(dst0_dram_buffer, result0_vec);
    tt_metal::detail::ReadFromBuffer(dst1_dram_buffer, result1_vec);

    // Identity copy per stream: bit-exact.
    ASSERT_EQ(src0_vec, result0_vec) << "values stream (Float16_b) mismatch, kernel = " << config.compute_kernel;
    ASSERT_EQ(src1_vec, result1_vec) << "index stream (UInt16) mismatch, kernel = " << config.compute_kernel;

    if (config.perf_iterations > 0) {
        constexpr std::uint32_t warmup_iterations = 20;
        for (std::uint32_t i = 0; i < warmup_iterations; ++i) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        distributed::Finish(cq);

        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < config.perf_iterations; ++i) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        distributed::Finish(cq);
        const auto t1 = std::chrono::steady_clock::now();

        const double us_per_iter =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / config.perf_iterations;
        log_info(
            tt::LogTest,
            "perf: kernel = {}, num_iters = {} ({} format swaps), iters = {}, {:.2f} us/iter, {:.1f} ns/swap",
            config.compute_kernel,
            config.num_iters,
            2 * config.num_iters,
            config.perf_iterations,
            us_per_iter,
            1000.0 * us_per_iter / (2.0 * config.num_iters));
        if (perf_us_per_iter_out != nullptr) {
            *perf_us_per_iter_out = us_per_iter;
        }
    }
}

}  // namespace unit_tests::compute::sst_alternating_copy

TEST_F(LLKMeshDeviceFixture, TensixAlternatingFormatCopy) {
    for (const auto& kernel :
         {std::string("tests/tt_metal/tt_metal/test_kernels/compute/alternating_copy.cpp"),
          std::string("experiments/static-state-tracking/kernels/alternating_copy_sst.cpp")}) {
        for (std::uint32_t num_iters : {1u, 4u, 16u}) {
            unit_tests::compute::sst_alternating_copy::AltCopyConfig config{
                .num_iters = num_iters, .compute_kernel = kernel};
            unit_tests::compute::sst_alternating_copy::run_alternating_copy_program(this->devices_.at(0), config);
        }
    }
}

// Perf A/B on the alternating-format shape: LLK 1.0 pays a full
// reconfig+init on UNPACK/MATH and a pack reconfig per swap; SST re-emits
// only the tracked descriptor deltas. Swap count scales with num_iters.
TEST_F(LLKMeshDeviceFixture, TensixAlternatingFormatCopyPerfSSTvsBaseline) {
    vector<std::uint32_t> iter_counts = {8, 32, 64};
    std::uint32_t kPerfIters = 400;
    if (std::getenv("TT_SST_PERF_PROFILE") != nullptr) {
        iter_counts = {32};
        if (const char* shape = std::getenv("TT_SST_PERF_SHAPE")) {
            std::uint32_t n = 0;
            if (std::sscanf(shape, "%u", &n) == 1 && n > 0) {
                iter_counts = {n};
            }
        }
        kPerfIters = 30;
    }
    for (std::uint32_t num_iters : iter_counts) {
        double baseline_us = 0.0;
        double sst_us = 0.0;

        unit_tests::compute::sst_alternating_copy::AltCopyConfig base_config{
            .num_iters = num_iters,
            .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/alternating_copy.cpp",
            .perf_iterations = kPerfIters};
        unit_tests::compute::sst_alternating_copy::run_alternating_copy_program(
            this->devices_.at(0), base_config, &baseline_us);

        auto sst_config = base_config;
        sst_config.compute_kernel = "experiments/static-state-tracking/kernels/alternating_copy_sst.cpp";
        unit_tests::compute::sst_alternating_copy::run_alternating_copy_program(
            this->devices_.at(0), sst_config, &sst_us);

        log_info(
            tt::LogTest,
            "perf summary (alternating copy): num_iters = {} ({} swaps): baseline = {:.2f} us, SST = {:.2f} us, "
            "SST/baseline = {:.3f}",
            num_iters,
            2 * num_iters,
            baseline_us,
            sst_us,
            sst_us / baseline_us);
    }
}

}  // namespace tt::tt_metal
