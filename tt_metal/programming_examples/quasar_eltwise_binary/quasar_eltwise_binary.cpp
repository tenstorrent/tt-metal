// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone Metal 2.0 demo: a single-tile bf16 eltwise ADD driven through the
// ProgramSpec / DataflowBuffer (DFB) path that Quasar uses. Its host structure is a
// faithful copy of the gtest helper `single_core_binary()` in
//   tests/tt_metal/tt_metal/llk/test_single_core_binary_compute.cpp
// and it reuses the exact same three kernels:
//   - reader  : tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp
//   - writer  : tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp
//   - compute : tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0.cpp
//
// The gtest `LLKQuasarMeshDeviceSingleCardFixture.QuasarEltwiseBinaryAdd` is the
// AUTHORITATIVE version of this proof (golden comparison with tolerances, runs under CI).
// This example is a human-readable demo / craq-sim smoke test that slots into the
// simulator bring-up flow, mirroring how `metal_example_add_2_integers_in_riscv` is used.
//
// STATUS (2026-06-18): the gtest above has been run on the craq-sim Quasar functional
// simulator — it boots and runs end-to-end but currently FAILS the PCC check (output
// reads back as zero; functional bug in the single-tile Quasar add path, not the harness).
// This example itself has been compiled/linked but NOT yet run.
//
// Run on the craq-sim Quasar simulator (stage a QSR libttsim.so + quasar_32_arch.yaml as
// soc_descriptor.yaml in <sim_dir>; see craq-sim/scripts/run_quasar_metal_llk_gtests.py):
//   TT_METAL_SIMULATOR=<sim_dir>/libttsim.so TT_SIMULATOR_LOCALHOST=1 \
//     ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
//     LD_LIBRARY_PATH=$PWD/build/lib \
//     ./build/programming_examples/metal_example_quasar_eltwise_binary
// (craq-sim requires MODE=0 mtvec → needs tt-metal #46916 reverted, or a craq-sim fix.)

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include <fmt/base.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace tt::tt_metal;

int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    try {
        // ---------------------------------------------------------------------
        // Problem definition: a single bf16 tile (32x32), ADD.
        // ---------------------------------------------------------------------
        constexpr uint32_t num_tiles = 1;
        constexpr uint32_t tile_byte_size = 2 * 32 * 32;  // bf16 (2 bytes) * 32 * 32
        constexpr size_t byte_size = num_tiles * tile_byte_size;
        constexpr size_t num_elems = byte_size / sizeof(bfloat16);
        const CoreCoord core{0, 0};
        const tt::DataFormat data_format = tt::DataFormat::Float16_b;
        const Tile tile = Tile({32, 32});

        // Open a unit (1x1) mesh on device 0; the same APIs scale to larger meshes.
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();
        const auto zero_coord = distributed::MeshCoordinate(0, 0);
        const experimental::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

        // ---------------------------------------------------------------------
        // Host stimulus + golden (a + b), bf16.
        // ---------------------------------------------------------------------
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<bfloat16> a(num_elems);
        std::vector<bfloat16> b(num_elems);
        std::vector<bfloat16> golden(num_elems);
        for (size_t i = 0; i < num_elems; ++i) {
            a[i] = bfloat16(dist(rng));
            b[i] = bfloat16(dist(rng));
            golden[i] = bfloat16(static_cast<float>(a[i]) + static_cast<float>(b[i]));
        }

        // ---------------------------------------------------------------------
        // Four DRAM buffers (3 inputs + 1 output), mirroring the gtest helper.
        // The compute path for plain ADD only consumes in0/in1; in2 is created and
        // written for parity with the shared reader kernel / template (it is read only
        // under dest-reuse / accumulation defines, which we do not set here).
        // ---------------------------------------------------------------------
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = byte_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
        distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

        auto input0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto input1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto input2_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, a, false);
        distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, b, false);
        distributed::EnqueueWriteMeshBuffer(cq, input2_dram_buffer, b, false);

        // ---------------------------------------------------------------------
        // Compute defines for the "add" case (subset of the gtest's
        // build_binary_defines(): ELTWISE_OP=add_tiles, ELWADD, full init).
        // ---------------------------------------------------------------------
        experimental::KernelSpec::CompilerOptions::Defines defines;
        defines.emplace("ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD");
        defines.emplace("ELTWISE_OP", "add_tiles");
        defines.emplace("ELTWISE_OP_INIT", "add_tiles_init");
        defines.emplace("FULL_INIT", "1");

        // ---------------------------------------------------------------------
        // Dataflow buffers (DFB): 3 inputs + 1 output, one tile each.
        // ---------------------------------------------------------------------
        const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
        const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
        const experimental::DFBSpecName INP2_DFB{"inp2_dfb"};
        const experimental::DFBSpecName OUT_DFB{"out_dfb"};
        const experimental::KernelSpecName READER{"reader"};
        const experimental::KernelSpecName WRITER{"writer"};
        const experimental::KernelSpecName COMPUTE{"compute"};

        auto make_input_dfb = [&](const experimental::DFBSpecName& name) {
            return experimental::DataflowBufferSpec{
                .unique_id = name,
                .entry_size = tile_byte_size,
                .num_entries = num_tiles,
                .data_format_metadata = data_format,
                .tile_format_metadata = tile,
            };
        };

        experimental::DataflowBufferSpec inp0_dfb_spec = make_input_dfb(INP0_DFB);
        experimental::DataflowBufferSpec inp1_dfb_spec = make_input_dfb(INP1_DFB);
        experimental::DataflowBufferSpec inp2_dfb_spec = make_input_dfb(INP2_DFB);
        experimental::DataflowBufferSpec out_dfb_spec{
            .unique_id = OUT_DFB,
            .entry_size = tile_byte_size,
            .num_entries = num_tiles,
            .data_format_metadata = data_format,
            .tile_format_metadata = tile,
        };

        // ---------------------------------------------------------------------
        // Reader kernel: DRAM -> DFB (producer of in0/in1/in2).
        // ---------------------------------------------------------------------
        experimental::KernelSpec reader_spec{
            .unique_id = READER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = defines},
            .dfb_bindings =
                {{
                     .dfb_spec_name = INP0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = INP1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = INP2_DFB,
                     .accessor_name = "in2",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 }},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src0_addr",
                      "src0_bank_id",
                      "src1_addr",
                      "src1_bank_id",
                      "num_tiles",
                      "src2_addr",
                      "src2_bank_id"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen1_config =
                        experimental::DataMovementHardwareConfig::Gen1Config{
                            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
                    .gen2_config =
                        experimental::DataMovementHardwareConfig::Gen2Config{
                            .disable_implicit_sync_for = {INP0_DFB, INP1_DFB, INP2_DFB}}},
        };

        // ---------------------------------------------------------------------
        // Writer kernel: DFB -> DRAM (consumer of out).
        // ---------------------------------------------------------------------
        experimental::KernelSpec writer_spec{
            .unique_id = WRITER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
            .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen1_config =
                        experimental::DataMovementHardwareConfig::Gen1Config{
                            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
                    .gen2_config =
                        experimental::DataMovementHardwareConfig::Gen2Config{.disable_implicit_sync_for = {OUT_DFB}}},
        };

        // ---------------------------------------------------------------------
        // Compute kernel: consumes in0/in1/in2, produces out (eltwise ADD).
        // ---------------------------------------------------------------------
        experimental::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = defines},
            .dfb_bindings =
                {{
                     .dfb_spec_name = INP0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = INP1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = INP2_DFB,
                     .accessor_name = "in2",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 }},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt", "per_core_block_size", "acc_to_dst"}},
            .hw_config =
                experimental::ComputeHardwareConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                },
        };

        experimental::WorkUnitSpec wu{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = node,
        };

        experimental::ProgramSpec spec{
            .name = "quasar_eltwise_binary_add",
            .kernels = {reader_spec, writer_spec, compute_spec},
            .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, inp2_dfb_spec, out_dfb_spec},
            .work_units = {wu},
        };

        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

        // ---------------------------------------------------------------------
        // Runtime args (addresses + tile counts), matching the schema above.
        // ---------------------------------------------------------------------
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = READER,
                .runtime_arg_values =
                    {{node,
                      {{"src0_addr", input0_dram_buffer->address()},
                       {"src0_bank_id", 0u},
                       {"src1_addr", input1_dram_buffer->address()},
                       {"src1_bank_id", 0u},
                       {"num_tiles", num_tiles},
                       {"src2_addr", input2_dram_buffer->address()},
                       {"src2_bank_id", 0u}}}},
            },
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values =
                    {{node, {{"dst_addr", output_dram_buffer->address()}, {"bank_id", 0u}, {"num_tiles", num_tiles}}}},
            },
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = COMPUTE,
                .runtime_arg_values =
                    {{node, {{"per_core_block_cnt", num_tiles}, {"per_core_block_size", 1u}, {"acc_to_dst", 0u}}}},
            },
        };
        experimental::SetProgramRunArgs(program, params);

        // Slow-dispatch launch (same path the gtest helper uses for the ProgramSpec flow).
        auto* dev = mesh_device->get_devices()[0];
        detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

        // ---------------------------------------------------------------------
        // Read back and compare against the host golden (a + b).
        // ---------------------------------------------------------------------
        std::vector<bfloat16> result;
        distributed::ReadShard(cq, result, output_dram_buffer, zero_coord, true);

        constexpr float eps = 0.0155f;  // loose bf16 tolerance, matching the gtest helper
        if (result.size() != golden.size()) {
            pass = false;
            fmt::print(stderr, "Result size mismatch: expected {}, got {}\n", golden.size(), result.size());
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                const float expected = static_cast<float>(golden[i]);
                const float actual = static_cast<float>(result[i]);
                if (std::abs(expected - actual) > eps) {
                    pass = false;
                    fmt::print(stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, actual);
                    break;
                }
            }
        }

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
