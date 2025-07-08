// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/distributed/utils.hpp"

#include <fmt/base.h>
#include <array>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/semaphore.hpp>
#include <tt_stl/span.hpp>
#include "tests/tt_metal/tt_metal/dispatch/dispatch_test_utils.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

namespace tt::tt_metal::distributed::test::utils {

std::vector<std::shared_ptr<Program>> create_eltwise_bin_programs(
    std::shared_ptr<MeshDevice>& mesh_device,
    std::vector<std::shared_ptr<MeshBuffer>>& src0_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& src1_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& output_bufs) {
    const std::vector<std::string> op_id_to_op_define = {"add_tiles", "mul_tiles", "sub_tiles"};
    const std::vector<std::string> op_id_to_op_type_define = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWMUL", "EltwiseBinaryType::ELWSUB"};

    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();

    std::vector<std::shared_ptr<Program>> programs = {
        std::make_shared<Program>(), std::make_shared<Program>(), std::make_shared<Program>()};
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

    for (std::size_t eltwise_op = 0; eltwise_op < op_id_to_op_define.size(); eltwise_op++) {
        auto& program = *programs[eltwise_op];
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t page_size = single_tile_size;

        ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};
        DeviceLocalBufferConfig per_device_buffer_config{
            .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = true};

        bool allocate_bufs = src0_bufs.empty();
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                if (allocate_bufs) {
                    auto src0_dram_buffer =
                        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                    src0_bufs.push_back(src0_dram_buffer);
                    auto src1_dram_buffer =
                        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                    src1_bufs.push_back(src1_dram_buffer);
                }
                auto dst_dram_buffer =
                    MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                output_bufs.push_back(dst_dram_buffer);
            }
        }

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src0_config);

        uint32_t src1_cb_index = tt::CBIndex::c_1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, full_grid, cb_output_config);

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {};

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        std::map<std::string, std::string> binary_defines = {
            {"ELTWISE_OP", op_id_to_op_define[eltwise_op]}, {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};
        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            full_grid,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

        SetRuntimeArgs(program, eltwise_binary_kernel, full_grid, {2048, 1});

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                CoreCoord curr_core = {col_idx, row_idx};
                const std::array<uint32_t, 7> reader_args = {
                    src0_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    src1_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    0};

                const std::array<uint32_t, 3> writer_args = {
                    output_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(), 0, num_tiles};

                SetRuntimeArgs(program, unary_writer_kernel, curr_core, writer_args);
                SetRuntimeArgs(program, binary_reader_kernel, curr_core, reader_args);
            }
        }
    }
    return programs;
}

std::vector<std::shared_ptr<Program>> create_random_programs(
    uint32_t num_programs,
    CoreCoord worker_grid_size,
    uint32_t seed,
    const std::unordered_set<CoreCoord>& active_eth_cores) {
    uint32_t MAX_LOOP = 100;
    uint32_t page_size = 1024;
    uint32_t max_eth_cores = 3;

    uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
    uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
    uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
    uint32_t ERISC_OUTER_LOOP, ERISC_MIDDLE_LOOP, ERISC_INNER_LOOP;
    bool USE_MAX_RT_ARGS;

    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    std::vector<std::shared_ptr<Program>> programs;

    std::map<std::string, std::string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
    std::map<std::string, std::string> compute_defines = {{"COMPUTE", "1"}};
    std::map<std::string, std::string> erisc_defines = {{"ERISC", "1"}};

    for (uint32_t i = 0; i < num_programs; i++) {
        Program& program = *programs.emplace_back(std::make_shared<Program>());
        // ========== Set configs for BRISC ==========
        if (i == 0) {
            // Ensures that we get at least one compilation with the max amount to
            // ensure it compiles and runs
            BRISC_OUTER_LOOP = MAX_LOOP;
            BRISC_MIDDLE_LOOP = MAX_LOOP;
            BRISC_INNER_LOOP = MAX_LOOP;
            NUM_CBS = NUM_CIRCULAR_BUFFERS;
            NUM_SEMS = NUM_SEMAPHORES;
            USE_MAX_RT_ARGS = true;
        } else {
            BRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
            NUM_CBS = rand() % (NUM_CIRCULAR_BUFFERS) + 1;
            NUM_SEMS = rand() % (NUM_SEMAPHORES) + 1;
            USE_MAX_RT_ARGS = false;
        }
        // Create CBs
        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}})
                                                 .set_page_size(j, page_size * (j + 1));
            auto cb = CreateCircularBuffer(program, cr_set, cb_config);
        }

        // Create Semaphores
        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program, cr_set, j + 1);
            uint32_t curr_idx = 0;
            if (active_eth_cores.size()) {
                auto active_eth_core = active_eth_cores.begin();
                for (int k = 0; k < max_eth_cores && active_eth_core != active_eth_cores.end();
                     ++i, ++active_eth_core) {
                    CreateSemaphore(program, *active_eth_core, j + 1, CoreType::ETH);
                }
            }
        }

        // Create RTAs
        auto [brisc_unique_rtargs, brisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_brisc_unique_rtargs = brisc_unique_rtargs.size();
        uint32_t num_brisc_common_rtargs = brisc_common_rtargs.size();
        std::vector<uint32_t> brisc_compile_args = {
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_brisc_unique_rtargs,
            num_brisc_common_rtargs,
            page_size};

        // ========== Set configs for NCRISC ==========
        if (i == 0) {
            NCRISC_OUTER_LOOP = MAX_LOOP;
            NCRISC_MIDDLE_LOOP = MAX_LOOP;
            NCRISC_INNER_LOOP = MAX_LOOP;
        } else {
            NCRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [ncrisc_unique_rtargs, ncrisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_ncrisc_unique_rtargs = ncrisc_unique_rtargs.size();
        uint32_t num_ncrisc_common_rtargs = ncrisc_common_rtargs.size();
        std::vector<uint32_t> ncrisc_compile_args = {
            NCRISC_OUTER_LOOP,
            NCRISC_MIDDLE_LOOP,
            NCRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_ncrisc_unique_rtargs,
            num_ncrisc_common_rtargs,
            page_size};

        // ========== Set configs for TRISC ==========
        if (i == 0) {
            TRISC_OUTER_LOOP = MAX_LOOP;
            TRISC_MIDDLE_LOOP = MAX_LOOP;
            TRISC_INNER_LOOP = MAX_LOOP;
        } else {
            TRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [trisc_unique_rtargs, trisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_trisc_unique_rtargs = trisc_unique_rtargs.size();
        uint32_t num_trisc_common_rtargs = trisc_common_rtargs.size();
        std::vector<uint32_t> trisc_compile_args = {
            TRISC_OUTER_LOOP,
            TRISC_MIDDLE_LOOP,
            TRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_trisc_unique_rtargs,
            num_trisc_common_rtargs,
            page_size};

        if (i == 0) {
            ERISC_OUTER_LOOP = MAX_LOOP;
            ERISC_MIDDLE_LOOP = MAX_LOOP;
            ERISC_INNER_LOOP = MAX_LOOP;
        } else {
            ERISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            ERISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            ERISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }
        // Only setup RTAs on ERISC. No Common RTAs.
        uint32_t max_erisc_rtas = 64;
        uint32_t num_erisc_rtas = rand() % (max_erisc_rtas + 1);
        auto [erisc_unique_rtargs, erisc_common_rtargs] = create_runtime_args(num_erisc_rtas, 0, 0, 0);
        uint32_t num_erisc_unique_rtargs = erisc_unique_rtargs.size();
        uint32_t num_erisc_common_rt_args = erisc_common_rtargs.size();

        std::vector<uint32_t> erisc_compile_time_args = {
            ERISC_OUTER_LOOP,
            ERISC_MIDDLE_LOOP,
            ERISC_INNER_LOOP,
            0, /* CBs are not supported on ERISC cores */
            NUM_SEMS,
            num_erisc_unique_rtargs,
            num_erisc_common_rt_args,
            page_size};

        // Create Kernels
        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = brisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = ncrisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                ComputeConfig{
                    .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
            SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = rand() % 3 + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = brisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = ncrisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    ComputeConfig{
                        .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
                SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            } else {
                TT_THROW("Invalid");
            }
        }
        if (active_eth_cores.size()) {
            auto active_eth_core = active_eth_cores.begin();
            for (int k = 0; k < max_eth_cores && active_eth_core != active_eth_cores.end(); ++i, ++active_eth_core) {
                auto dummy_erisc_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    *active_eth_core,
                    EthernetConfig{
                        .noc = NOC::NOC_0, .compile_args = erisc_compile_time_args, .defines = erisc_defines});
                SetRuntimeArgs(program, dummy_erisc_kernel, *active_eth_core, erisc_unique_rtargs);
            }
        }
    }
    return programs;
}

ScopedEnvVar::ScopedEnvVar(const char* name, const char* value) : name_(name) {
    // Save original value
    const char* original = std::getenv(name);
    if (original) {
        original_value_ = original;
        had_original_ = true;
    }

    // Set new value
    if (value) {
        setenv(name, value, /*overwrite=*/1);
    } else {
        unsetenv(name);
    }
}

ScopedEnvVar::~ScopedEnvVar() {
    // Restore original value
    if (had_original_) {
        setenv(name_, original_value_.c_str(), /*overwrite=*/1);
    } else {
        unsetenv(name_);
    }
}

}  // namespace tt::tt_metal::distributed::test::utils
