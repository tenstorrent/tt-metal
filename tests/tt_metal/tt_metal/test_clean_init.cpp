// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/discovery/cluster.hpp>
#include <array>
#include <exception>
#include <map>
#include <memory>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/memory/buffer/buffer_types.hpp>
#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/execution/distributed.hpp>

/*
 * Similar to loopback programming example, except run on al devices and skip device teardown to check if we can
 * recover from a "bad" state.
 */

using std::vector;
using namespace tt::tt_metal;

int main(int argc, char** /*argv*/) {
    // Any arg means that we shouldn't do teardown.
    bool skip_teardown = (argc > 1);
    if (skip_teardown) {
        log_info(tt::LogTest, "Running loopback test with no teardown, to see if we can recover next run.");
    } else {
        log_info(tt::LogTest, "Running loopback test with proper teardown");
    }

    bool pass = true;
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    vector<tt::ChipId> ids;
    ids.reserve(num_devices);
    for (unsigned int id = 0; id < num_devices; id++) {
        ids.push_back(id);
    }

    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().resolve_dispatch_core_config();
    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

    for (int device_id = 0; device_id < num_devices; device_id++) {
        try {
            /*
             * Silicon accelerator setup
             */
            auto device = devices[device_id];

            /*
             * Setup program and command queue to execute along with its buffers and kernels to use
             */
            auto& cq = device->mesh_command_queue();

            constexpr uint32_t single_tile_size = 2 * (32 * 32);
            constexpr uint32_t num_tiles = 50;
            constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

            distributed::DeviceLocalBufferConfig local_l1_config{
                .page_size = dram_buffer_size, .buffer_type = tt::tt_metal::BufferType::L1};

            distributed::DeviceLocalBufferConfig local_dram_config{
                .page_size = dram_buffer_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

            distributed::ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};

            auto l1_buffer = distributed::MeshBuffer::create(global_buffer_config, local_l1_config, device.get());
            auto input_dram_buffer =
                distributed::MeshBuffer::create(global_buffer_config, local_dram_config, device.get());

            auto output_dram_buffer =
                distributed::MeshBuffer::create(global_buffer_config, local_dram_config, device.get());

            auto mesh_workload = distributed::MeshWorkload();

            const NodeCoord core{0, 0};

            // Metal 2.0 program: one data-movement kernel that copies the input DRAM buffer to L1 and from there
            // to the output DRAM buffer.
            const KernelSpecName DRAM_COPY{"dram_copy"};
            KernelSpec dram_copy_spec{
                .unique_id = DRAM_COPY,
                .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_named_args.cpp",
                .num_threads = 1,
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"l1_buffer_addr",
                          "dram_buffer_src_addr",
                          "dram_src_bank_id",
                          "dram_buffer_dst_addr",
                          "dram_dst_bank_id",
                          "dram_buffer_size"}},
                .hw_config =
                    DataMovementHardwareConfig{
                        .config_1xx =
                            DataMovementHardwareConfig::DataMovement1XXConfig{
                                .processor = DataMovementProcessor::RISCV_0,
                                .noc = NOC::RISCV_0_default,
                            },
                    },
            };
            ProgramSpec spec{
                .name = "clean_init_dram_loopback",
                .kernels = {dram_copy_spec},
                .work_units = {{.name = "main", .kernels = {DRAM_COPY}, .target_nodes = core}},
            };
            Program program(*device, spec);

            /*
             * Create input data and runtime arguments, then execute
             */
            std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
            distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

            // Each DRAM buffer is a single page, which lives in DRAM bank 0.
            ProgramRunArgs run_args;
            run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
                .kernel = DRAM_COPY,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    core,
                    {{"l1_buffer_addr", static_cast<uint32_t>(l1_buffer->address())},
                     {"dram_buffer_src_addr", static_cast<uint32_t>(input_dram_buffer->address())},
                     {"dram_src_bank_id", 0u},
                     {"dram_buffer_dst_addr", static_cast<uint32_t>(output_dram_buffer->address())},
                     {"dram_dst_bank_id", 0u},
                     {"dram_buffer_size", dram_buffer_size}}),
            }};
            program.set_program_run_args(run_args);
            mesh_workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
            distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
            log_info(tt::LogTest, "Started program");
            distributed::Finish(cq);
            log_info(tt::LogTest, "Finished program");

            /*
             * Validation & Teardown
             */
            std::vector<uint32_t> result_vec;
            distributed::ReadShard(cq, result_vec, output_dram_buffer, distributed::MeshCoordinate(0, 0));

            pass &= input_vec == result_vec;

        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Test failed with exception!");
            log_error(tt::LogTest, "{}", e.what());

            throw;
        }
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    // Skip teardown by throwing.
    if (skip_teardown) {
        TT_THROW("Skip teardown by throwing");
    } else {
        for (auto& [device_id, device] : devices) {
            device->close();
        }
    }

    // Error out with non-zero return code if we don't detect a pass
    TT_FATAL(pass, "Error");

    return 0;
}
