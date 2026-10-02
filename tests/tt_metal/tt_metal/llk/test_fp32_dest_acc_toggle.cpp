// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mid-kernel FP32 dest-acc toggling on Quasar (set_fp32_dest_acc / restore_fp32_dest_acc, which wrap
// enable_fp32_dest_acc / disable_fp32_dest_acc).
//
// The compute kernel (fp32_dest_acc_toggle_quasar.cpp) is built for 16-bit or 32-bit dest and runs kNumPhases
// phases alternating between the compiled width W and !W (W, !W, W, !W). Each phase accumulates
// kTilesPerPhase (in0 + in1) tiles into dest tile 0 with add_tiles(acc_to_dest) and packs it to its own
// output DFB.
//
// Per phase, tile 0 of in0 holds x = 1 + k * 2^-7 (k random) and the other 8 tiles hold 2^-10; in1 is zero.
// Every addend is below half a bf16 ULP of x (2^-8):
//   - 16-bit dest drops each addend, under round-to-nearest or truncation: result == x (packed to Float16_b);
//   - 32-bit dest keeps them: result == x + 8 * 2^-10 == x + 2^-7 (packed to Float32, exact).
// Every phase checks its own width, so a toggle that fails to take effect, leaks into the next phase, or leaves
// dest unclear at the previous width shows up as a mismatch in a specific phase.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"
#include "tt_metal/test_utils/packing.hpp"

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::compute::fp32_dest_acc_toggle {

constexpr std::uint32_t kTilesPerPhase = 9;  // must match TILES_PER_PHASE in the compute kernel
constexpr std::uint32_t kNumPhases = 4;      // must match the phase sequence in the compute kernel
constexpr float kAddend = 0.0009765625f;     // 2^-10
constexpr float kOneUlpAtOne = 0.0078125f;   // 2^-7, bf16 ULP in [1, 2)

// Phases alternate starting from the compiled width.
constexpr bool phase_is_fp32(bool compiled_fp32, std::uint32_t phase) { return compiled_fp32 != (phase % 2 == 1); }

bool run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const bool compiled_fp32) {
    constexpr std::uint32_t elems_per_tile = tt::constants::TILE_HW;
    const std::uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t fp32_tile_size = tt::tile_size(tt::DataFormat::Float32);
    const std::uint32_t num_in_tiles = kNumPhases * kTilesPerPhase;
    const std::uint32_t in_bytes = num_in_tiles * bf16_tile_size;

    const CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::NodeCoord node{static_cast<std::uint32_t>(core.x), static_cast<std::uint32_t>(core.y)};

    // Single-page buffers: the reader/writers walk one DRAM bank with a running address.
    auto make_dram = [&](std::uint32_t bytes) {
        distributed::DeviceLocalBufferConfig local{
            .page_size = bytes, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
        distributed::ReplicatedBufferConfig replicated{.size = bytes};
        return distributed::MeshBuffer::create(replicated, local, mesh_device.get());
    };
    auto in0_dram = make_dram(in_bytes);
    auto in1_dram = make_dram(in_bytes);

    const experimental::DFBSpecName IN0_DFB{"in0"};
    const experimental::DFBSpecName IN1_DFB{"in1"};
    const std::array<experimental::DFBSpecName, kNumPhases> OUT_DFBS{
        experimental::DFBSpecName{"out0"},
        experimental::DFBSpecName{"out1"},
        experimental::DFBSpecName{"out2"},
        experimental::DFBSpecName{"out3"}};
    const experimental::KernelSpecName READER{"reader"};
    const std::array<experimental::KernelSpecName, kNumPhases> WRITERS{
        experimental::KernelSpecName{"writer0"},
        experimental::KernelSpecName{"writer1"},
        experimental::KernelSpecName{"writer2"},
        experimental::KernelSpecName{"writer3"}};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [](const experimental::DFBSpecName& name,
                       std::uint32_t entry_size,
                       std::uint32_t num_entries,
                       tt::DataFormat format) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = format,
        };
    };

    std::vector<experimental::DataflowBufferSpec> dfb_specs{
        make_dfb(IN0_DFB, bf16_tile_size, 2, tt::DataFormat::Float16_b),
        make_dfb(IN1_DFB, bf16_tile_size, 2, tt::DataFormat::Float16_b)};
    std::vector<std::shared_ptr<distributed::MeshBuffer>> out_dram;
    for (std::uint32_t phase = 0; phase < kNumPhases; ++phase) {
        const bool fp32 = phase_is_fp32(compiled_fp32, phase);
        const std::uint32_t tile_size = fp32 ? fp32_tile_size : bf16_tile_size;
        dfb_specs.push_back(
            make_dfb(OUT_DFBS[phase], tile_size, 1, fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b));
        out_dram.push_back(make_dram(tile_size));
    }

    using DFBEndpoint = experimental::DFBEndpointType;
    using DFBAccess = experimental::DFBAccessPattern;
    auto dfb_binding = [](const experimental::DFBSpecName& name, DFBEndpoint endpoint) {
        return experimental::DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = name.get(),
            .endpoint_type = endpoint,
            .access_pattern = DFBAccess::STRIDED,
        };
    };
    const experimental::DataMovementHardwareConfig dm_hw_config{
        .config_2xx =
            experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };

    std::vector<experimental::KernelSpec> kernel_specs;
    kernel_specs.push_back(experimental::KernelSpec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {dfb_binding(IN0_DFB, DFBEndpoint::PRODUCER), dfb_binding(IN1_DFB, DFBEndpoint::PRODUCER)},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config = dm_hw_config,
    });
    for (std::uint32_t phase = 0; phase < kNumPhases; ++phase) {
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = WRITERS[phase],
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ConsumerOf(OUT_DFBS[phase], "in")},
            .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
            .hw_config = dm_hw_config,
        });
    }

    std::vector<experimental::DFBBinding> compute_bindings{
        dfb_binding(IN0_DFB, DFBEndpoint::CONSUMER), dfb_binding(IN1_DFB, DFBEndpoint::CONSUMER)};
    for (const auto& out : OUT_DFBS) {
        compute_bindings.push_back(dfb_binding(out, DFBEndpoint::PRODUCER));
    }
    kernel_specs.push_back(experimental::KernelSpec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/fp32_dest_acc_toggle_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings = compute_bindings,
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = MathFidelity::LoFi,
                .enable_32_bit_dest = compiled_fp32,
                .double_buffer_dest = false,  // SyncFull: the toggle does not re-seat SyncHalf bank offsets
            },
    });

    std::vector<experimental::KernelSpecName> wu_kernels{READER, COMPUTE};
    for (const auto& writer : WRITERS) {
        wu_kernels.push_back(writer);
    }
    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = wu_kernels,
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "fp32_dest_acc_toggle_quasar",
        .kernels = kernel_specs,
        .dataflow_buffers = dfb_specs,
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Stimulus. x = 1 + k * 2^-7 with k in [0, 126], so x and x + 2^-7 are both exact bf16 values in [1, 2].
    std::mt19937 rng(0x5eed);
    std::uniform_int_distribution<int> k_dist(0, 126);
    std::vector<bfloat16> in0_values;
    in0_values.reserve(num_in_tiles * elems_per_tile);
    std::vector<std::vector<float>> x_per_phase(kNumPhases, std::vector<float>(elems_per_tile));
    for (std::uint32_t phase = 0; phase < kNumPhases; ++phase) {
        for (std::uint32_t e = 0; e < elems_per_tile; ++e) {
            const float x = 1.0f + static_cast<float>(k_dist(rng)) * kOneUlpAtOne;
            x_per_phase[phase][e] = x;
            in0_values.emplace_back(x);
        }
        for (std::uint32_t e = 0; e < (kTilesPerPhase - 1) * elems_per_tile; ++e) {
            in0_values.emplace_back(kAddend);
        }
    }
    const std::vector<bfloat16> in1_values(num_in_tiles * elems_per_tile, bfloat16(0.0f));

    auto in0_packed = pack_vector<std::uint32_t, bfloat16>(in0_values);
    auto in1_packed = pack_vector<std::uint32_t, bfloat16>(in1_values);
    distributed::WriteShard(cq, in0_dram, in0_packed, zero_coord, false);
    distributed::WriteShard(cq, in1_dram, in1_packed, zero_coord, false);

    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = READER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"src0_addr", static_cast<std::uint32_t>(in0_dram->address())},
             {"src0_bank_id", 0u},
             {"src1_addr", static_cast<std::uint32_t>(in1_dram->address())},
             {"src1_bank_id", 0u},
             {"num_tiles", num_in_tiles}}),
    });
    for (std::uint32_t phase = 0; phase < kNumPhases; ++phase) {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITERS[phase],
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(out_dram[phase]->address())},
                 {"bank_id", 0u},
                 {"num_tiles", 1u}}),
        });
    }
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    bool pass = true;
    for (std::uint32_t phase = 0; phase < kNumPhases; ++phase) {
        const bool fp32 = phase_is_fp32(compiled_fp32, phase);
        std::vector<std::uint32_t> raw;
        distributed::ReadShard(cq, raw, out_dram[phase], zero_coord, false);

        std::vector<float> got(elems_per_tile);
        if (fp32) {
            for (std::uint32_t e = 0; e < elems_per_tile; ++e) {
                got[e] = std::bit_cast<float>(raw[e]);
            }
        } else {
            const auto bf16 = unpack_vector<bfloat16, std::uint32_t>(raw);
            for (std::uint32_t e = 0; e < elems_per_tile; ++e) {
                got[e] = static_cast<float>(bf16[e]);
            }
        }

        std::uint32_t mismatches = 0;
        int first = -1;
        for (std::uint32_t e = 0; e < elems_per_tile; ++e) {
            // 16-bit dest drops every addend; 32-bit dest keeps all of them (exactly one bf16 ULP).
            const float expected = x_per_phase[phase][e] + (fp32 ? kOneUlpAtOne : 0.0f);
            if (got[e] != expected) {
                if (first < 0) {
                    first = static_cast<int>(e);
                }
                ++mismatches;
            }
        }

        const std::string label = "phase " + std::to_string(phase) + " (" + (fp32 ? "32" : "16") + "-bit dest)";
        if (mismatches == 0) {
            log_info(tt::LogTest, "{}: PASS", label);
        } else {
            pass = false;
            log_error(
                tt::LogTest,
                "{}: FAIL ({}/{} mismatches; first idx {}: got {}, expected {})",
                label,
                mismatches,
                elems_per_tile,
                first,
                got[first],
                x_per_phase[phase][first] + (fp32 ? kOneUlpAtOne : 0.0f));
        }
    }

    return pass;
}

}  // namespace unit_tests::compute::fp32_dest_acc_toggle

// Compiled 16-bit: 16 -> 32 -> 16 -> 32 (set/restore_fp32_dest_acc<true> run enable/disable).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixFp32DestAccToggleQuasarFrom16Bit) {
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::fp32_dest_acc_toggle::run(device, false /*compiled_fp32*/));
    }
}

// Compiled 32-bit: 32 -> 16 -> 32 -> 16 (set/restore_fp32_dest_acc<false> run disable/enable). The 16-bit
// phases also exercise the packer IN_DATA_FORMAT derivation for a width the host table was not built for.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixFp32DestAccToggleQuasarFrom32Bit) {
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::fp32_dest_acc_toggle::run(device, true /*compiled_fp32*/));
    }
}

}  // namespace tt::tt_metal
