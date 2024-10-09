// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/distributed/mesh_program.hpp"

using namespace tt::tt_metal;

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    uint32_t num_cbs;
    uint32_t num_sems;
};

struct DummyProgramMultiCBConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
    uint32_t num_sems;
};

void initialize_dummy_kernels(MeshProgram& mesh_program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateKernel(
        mesh_program, "tt_metal/kernels/dataflow/blank.cpp", cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_writer_kernel = CreateKernel(
        mesh_program, "tt_metal/kernels/dataflow/blank.cpp", cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_compute_kernel = CreateKernel(mesh_program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

std::vector<CBHandle> initialize_dummy_circular_buffers(MeshProgram& mesh_program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs)
{
    std::vector<CBHandle> cb_handles;
    for (uint32_t i = 0; i < cb_configs.size(); i++) {
        const CBConfig& cb_config = cb_configs[i];
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config = CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(mesh_program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

void initialize_dummy_semaphores(MeshProgram& mesh_program, const std::variant<CoreRange, CoreRangeSet>& core_ranges, const vector<uint32_t>& init_values)
{
    for (uint32_t i = 0; i < init_values.size(); i++)
    {
        CreateSemaphore(mesh_program, core_ranges, init_values[i]);
    }
}

bool cb_config_successful(std::shared_ptr<MeshDevice> mesh_device, MeshProgram &mesh_program, const DummyProgramMultiCBConfig & program_config){
    bool pass = true;

    // Need to use old APIs to read since we cannot allocate a buffer in the reserved space we're trying
    // to read from
    vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        for (const CoreCoord& core_coord : core_range) {
            auto sem_base_addrs_across_mesh = mesh_program.get_sem_base_addr(mesh_device, core_coord, CoreType::WORKER);
            uint32_t dev_idx = 0;
            for (auto device : mesh_device->get_devices()) {
                tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord,
                    sem_base_addrs_across_mesh.at(dev_idx),
                    cb_config_buffer_size, cb_config_vector);

                uint32_t cb_addr = device->get_base_allocator_addr(HalMemType::L1);
                for (uint32_t i = 0; i < program_config.cb_config_vector.size(); i++) {
                    const uint32_t index = program_config.cb_config_vector[i].cb_id * sizeof(uint32_t);
                    const uint32_t cb_num_pages = program_config.cb_config_vector[i].num_pages;
                    const uint32_t cb_size = cb_num_pages * program_config.cb_config_vector[i].page_size;
                    const bool addr_match = cb_config_vector.at(index) == ((cb_addr) >> 4);
                    const bool size_match = cb_config_vector.at(index + 1) == (cb_size >> 4);
                    const bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                    pass &= (addr_match and size_match and num_pages_match);

                    cb_addr += cb_size;
                }
                dev_idx++;
            }
        }
    }
    return pass;
}

bool test_dummy_EnqueueProgram_with_cbs(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, DummyProgramMultiCBConfig& program_config) {
    MeshProgram mesh_program(mesh_device->get_devices().size());

    initialize_dummy_circular_buffers(mesh_program, program_config.cr_set, program_config.cb_config_vector);
    initialize_dummy_kernels(mesh_program, program_config.cr_set);
    const bool is_blocking_op = false;
    EnqueueMeshProgram(cq_id, mesh_program, mesh_device, is_blocking_op);
    Finish(mesh_device, cq_id);
    // return true;
    return cb_config_successful(mesh_device, mesh_program, program_config);
}

bool test_dummy_EnqueueProgram_with_sems(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, MeshProgram& mesh_program, const DummyProgramConfig& program_config, const vector<vector<uint32_t>>& expected_semaphore_vals) {
    TT_ASSERT(program_config.cr_set.size() == expected_semaphore_vals.size());

    bool are_all_semaphore_values_correct = true;

    const bool is_blocking_op = false;
    EnqueueMeshProgram(cq_id, mesh_program, mesh_device, is_blocking_op);
    Finish(mesh_device, cq_id);

    uint32_t expected_semaphore_vals_idx = 0;
    for (const CoreRange& core_range : program_config.cr_set.ranges())
    {
        const vector<uint32_t>& expected_semaphore_vals_for_core = expected_semaphore_vals[expected_semaphore_vals_idx];
        TT_ASSERT(expected_semaphore_vals_for_core.size() == program_config.num_sems);
        expected_semaphore_vals_idx++;
        for (const CoreCoord& core_coord : core_range)
        {
            auto sem_base_addrs_across_mesh = mesh_program.get_sem_base_addr(mesh_device, core_coord, CoreType::WORKER);
            uint32_t dev_idx = 0;
            for (auto device : mesh_device->get_devices()) {
                vector<uint32_t> semaphore_vals;
                uint32_t expected_semaphore_vals_for_core_idx = 0;
                const uint32_t semaphore_buffer_size = program_config.num_sems * hal.get_alignment(HalMemType::L1);
                tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, sem_base_addrs_across_mesh.at(dev_idx), semaphore_buffer_size, semaphore_vals);
                for (uint32_t i = 0; i < semaphore_vals.size(); i += (hal.get_alignment(HalMemType::L1) / sizeof(uint32_t)))
                {
                    const bool is_semaphore_value_correct = semaphore_vals[i] == expected_semaphore_vals_for_core[expected_semaphore_vals_for_core_idx];
                    expected_semaphore_vals_for_core_idx++;
                    if (!is_semaphore_value_correct)
                    {
                        are_all_semaphore_values_correct = false;
                    }
                }
                dev_idx++;
            }
        }
    }

    return are_all_semaphore_values_correct;
}

bool test_dummy_EnqueueProgram_with_sems(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, const DummyProgramConfig& program_config) {
    MeshProgram mesh_program(mesh_device->get_devices().size());

    vector<uint32_t> expected_semaphore_values;

    for (uint32_t initial_sem_value = 0; initial_sem_value < program_config.num_sems; initial_sem_value++) {
        expected_semaphore_values.push_back(initial_sem_value);
    }

    initialize_dummy_semaphores(mesh_program, program_config.cr_set, expected_semaphore_values);
    return test_dummy_EnqueueProgram_with_sems(mesh_device, cq_id, mesh_program, program_config, {expected_semaphore_values});
}

TEST(MeshProgram, TestMeshProgramCB) {
    std::shared_ptr<MeshDevice> mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape{2, 4}, MeshType::RowMajor));
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id=0, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config} };
    EXPECT_EQ(true, test_dummy_EnqueueProgram_with_cbs(mesh_device, 0, config));
    mesh_device->close_devices();
}

TEST(MeshProgram, TestMeshProgramSem) {
    std::shared_ptr<MeshDevice> mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape{2, 4}, MeshType::RowMajor));
    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    EXPECT_TRUE(test_dummy_EnqueueProgram_with_sems(mesh_device, 0, config));
    mesh_device->close_devices();
}
