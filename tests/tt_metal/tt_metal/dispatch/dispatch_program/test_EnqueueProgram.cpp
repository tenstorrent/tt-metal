// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>

#include "buffer_types.hpp"
#include "command_queue_fixture.hpp"
#include "dispatch_test_utils.hpp"
#include "env_lib.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"
#include "multi_command_queue_fixture.hpp"
#include "random_program_fixture.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>

// Access to internal API: ProgramImpl::get_cb_base_addr, get_kernel
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {
class CommandQueue;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;

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

struct IncrementKernelsSet {
    // Kernels that were created
    std::vector<KernelHandle> kernel_handles;
    // L1 address for unique args
    uint32_t unique_args_addr;
    // L1 address for common args
    uint32_t common_args_addr;
};

namespace local_test_functions {

// Helper function to create a kernel
KernelHandle create_kernel(
    HalProcessorIdentifier processor,
    Program& program,
    const CoreRangeSet& cr_set,
    const std::vector<uint32_t>& compile_args,
    const std::string& kernel_path) {
    auto [core_type, processor_class, processor_id] = processor;

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (processor_class) {
                case HalProcessorClassType::DM:
                    return CreateKernel(
                        program,
                        kernel_path,
                        cr_set,
                        DataMovementConfig{
                            .processor = static_cast<DataMovementProcessor>(processor_id),
                            .noc = static_cast<NOC>(processor_id),
                            .compile_args = compile_args,
                        });
                    break;
                case HalProcessorClassType::COMPUTE:
                    return CreateKernel(
                        program,
                        kernel_path,
                        cr_set,
                        tt::tt_metal::ComputeConfig{
                            .compile_args = compile_args,
                        });
                    break;
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH:
            return CreateKernel(
                program,
                kernel_path,
                cr_set,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = core_type == HalProgrammableCoreType::IDLE_ETH ? Eth::IDLE : Eth::RECEIVER,
                    .noc = static_cast<NOC>(processor_id),
                    .processor = static_cast<DataMovementProcessor>(processor_id),
                    .compile_args = compile_args,
                });
        case HalProgrammableCoreType::COUNT: TT_THROW("bad core type"); break;
    }
    TT_THROW("Unreachable");
}

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

void initialize_dummy_semaphores(
    Program& program, const std::variant<CoreRange, CoreRangeSet>& core_ranges, const vector<uint32_t>& init_values) {
    for (unsigned int init_value : init_values) {
        CreateSemaphore(program, core_ranges, init_value);
    }
}

std::vector<CBHandle> initialize_dummy_circular_buffers(
    Program& program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs) {
    std::vector<CBHandle> cb_handles;
    for (const auto& cb_config : cb_configs) {
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config =
            CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

bool cb_config_successful(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    distributed::MeshWorkload& workload,
    const DummyProgramMultiCBConfig& program_config) {
    bool pass = true;

    // Need to use old APIs to read since we cannot allocate a buffer in the reserved space we're trying
    // to read from
    vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size =
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    auto* device = mesh_device->get_devices()[0];
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        for (const CoreCoord& core_coord : core_range) {
            tt::tt_metal::detail::ReadFromDeviceL1(
                device,
                core_coord,
                workload.get_cb_base_addr(mesh_device, core_coord, CoreType::WORKER),
                cb_config_buffer_size,
                cb_config_vector);

            uint32_t cb_addr = l1_unreserved_base;
            for (const auto& config : program_config.cb_config_vector) {
                const uint32_t index = config.cb_id * sizeof(uint32_t);
                const uint32_t cb_num_pages = config.num_pages;
                const uint32_t cb_size = cb_num_pages * config.page_size;
                const bool addr_match = cb_config_vector.at(index) == cb_addr;
                const bool size_match = cb_config_vector.at(index + 1) == cb_size;
                const bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                pass &= (addr_match and size_match and num_pages_match);

                cb_addr += cb_size;
            }
        }
    }

    return pass;
}

void test_dummy_EnqueueProgram_with_runtime_args(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CoreCoord& eth_core_coord, DataMovementProcessor erisc_processor = DataMovementProcessor::RISCV_0) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program;
    auto* device = mesh_device->get_devices()[0];
    auto eth_noc_xy = mesh_device->ethernet_core_from_logical_core(eth_core_coord);

    constexpr uint32_t num_runtime_args0 = 9;
    uint32_t rta_base0 = MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    std::map<std::string, std::string> dummy_defines0 = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args0)},
        {"RESULTS_ADDR", std::to_string(rta_base0)}};
    auto dummy_kernel0 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        eth_core_coord,
        tt::tt_metal::EthernetConfig{
            .noc = static_cast<tt_metal::NOC>(erisc_processor),
            .processor = erisc_processor,
            .defines = dummy_defines0});

    constexpr int k_NumDummyArgs = 9;
    vector<uint32_t> dummy_kernel0_args(k_NumDummyArgs);
    // Generate 9 random numbers
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);
        for (uint32_t i = 0; i < k_NumDummyArgs; i++) {
            dummy_kernel0_args[i] = dis(gen);
        }
    }
    tt::tt_metal::SetRuntimeArgs(program, dummy_kernel0, eth_core_coord, dummy_kernel0_args);

    auto& cq = mesh_device->mesh_command_queue();
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    vector<uint32_t> dummy_kernel0_args_readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(),
        eth_noc_xy,
        MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED),
        dummy_kernel0_args.size() * sizeof(uint32_t));

    ASSERT_EQ(dummy_kernel0_args, dummy_kernel0_args_readback);
}

bool test_dummy_EnqueueProgram_with_cbs(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const DummyProgramMultiCBConfig& program_config) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program;

    initialize_dummy_circular_buffers(program, program_config.cr_set, program_config.cb_config_vector);
    initialize_dummy_kernels(program, program_config.cr_set);
    const bool is_blocking_op = false;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, is_blocking_op);
    Finish(cq);

    return cb_config_successful(mesh_device, workload, program_config);
}

bool test_dummy_EnqueueProgram_with_cbs_update_size(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const DummyProgramMultiCBConfig& program_config) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program;

    const std::vector<CBHandle>& cb_handles =
        initialize_dummy_circular_buffers(program, program_config.cr_set, program_config.cb_config_vector);
    initialize_dummy_kernels(program, program_config.cr_set);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    const bool is_cb_config_before_update_successful = cb_config_successful(mesh_device, workload, program_config);

    DummyProgramMultiCBConfig program_config_2 = program_config;
    for (uint32_t cb_id = 0; cb_id < program_config.cb_config_vector.size(); cb_id++) {
        CBConfig& cb_config = program_config_2.cb_config_vector[cb_id];
        cb_config.num_pages *= 2;
        const uint32_t cb_size = cb_config.num_pages * cb_config.page_size;
        UpdateCircularBufferTotalSize(workload.get_programs().at(device_range), cb_handles[cb_id], cb_size);
    }
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    const bool is_cb_config_after_update_successful = cb_config_successful(mesh_device, workload, program_config_2);
    return is_cb_config_before_update_successful && is_cb_config_after_update_successful;
}

bool test_dummy_EnqueueProgram_with_sems(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    distributed::MeshCommandQueue& cq,
    distributed::MeshWorkload& workload,
    const DummyProgramConfig& program_config,
    const vector<vector<uint32_t>>& expected_semaphore_vals) {
    TT_FATAL(
        program_config.cr_set.size() == expected_semaphore_vals.size(),
        "cr_set size {} must match expected_semaphore_vals size {}",
        program_config.cr_set.size(),
        expected_semaphore_vals.size());

    bool are_all_semaphore_values_correct = true;

    const bool is_blocking_op = false;
    distributed::EnqueueMeshWorkload(cq, workload, is_blocking_op);
    Finish(cq);

    auto* device = mesh_device->get_devices()[0];
    uint32_t expected_semaphore_vals_idx = 0;
    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        const vector<uint32_t>& expected_semaphore_vals_for_core = expected_semaphore_vals[expected_semaphore_vals_idx];
        TT_FATAL(
            expected_semaphore_vals_for_core.size() == program_config.num_sems,
            "expected_semaphore_vals_for_core size {} must match num_sems {}",
            expected_semaphore_vals_for_core.size(),
            program_config.num_sems);
        expected_semaphore_vals_idx++;
        for (const CoreCoord& core_coord : core_range) {
            vector<uint32_t> semaphore_vals;
            uint32_t expected_semaphore_vals_for_core_idx = 0;
            const uint32_t semaphore_buffer_size =
                program_config.num_sems * MetalContext::instance().hal().get_alignment(HalMemType::L1);
            uint32_t semaphore_base = workload.get_sem_base_addr(mesh_device, core_coord, CoreType::WORKER);
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, semaphore_base, semaphore_buffer_size, semaphore_vals);
            for (uint32_t i = 0; i < semaphore_vals.size();
                 i += (MetalContext::instance().hal().get_alignment(HalMemType::L1) / sizeof(uint32_t))) {
                const bool is_semaphore_value_correct =
                    semaphore_vals[i] == expected_semaphore_vals_for_core[expected_semaphore_vals_for_core_idx];
                expected_semaphore_vals_for_core_idx++;
                if (!is_semaphore_value_correct) {
                    are_all_semaphore_values_correct = false;
                }
            }
        }
    }

    return are_all_semaphore_values_correct;
}

bool test_dummy_EnqueueProgram_with_sems(
    const std::shared_ptr<distributed::MeshDevice>& device,
    distributed::MeshCommandQueue& cq,
    const DummyProgramConfig& program_config) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program;

    vector<uint32_t> expected_semaphore_values;
    expected_semaphore_values.reserve(program_config.num_sems);
    for (uint32_t initial_sem_value = 0; initial_sem_value < program_config.num_sems; initial_sem_value++) {
        expected_semaphore_values.push_back(initial_sem_value);
    }

    initialize_dummy_semaphores(program, program_config.cr_set, expected_semaphore_values);
    workload.add_program(device_range, std::move(program));

    return test_dummy_EnqueueProgram_with_sems(device, cq, workload, program_config, {expected_semaphore_values});
}

bool test_dummy_EnqueueProgram_with_runtime_args(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const DummyProgramConfig& program_config,
    uint32_t num_runtime_args_dm0,
    uint32_t num_runtime_args_dm1,
    uint32_t num_runtime_args_compute,
    uint32_t num_iterations,
    bool do_checks = true) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto* device = mesh_device->get_devices()[0];
    Program program;
    bool pass = true;

    CoreRangeSet cr_set = program_config.cr_set;

    uint32_t rta_base_dm0 = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t rta_base_dm1 = rta_base_dm0 + (num_runtime_args_dm0 * sizeof(uint32_t));
    uint32_t rta_base_compute = rta_base_dm1 + (num_runtime_args_dm1 * sizeof(uint32_t));
    std::map<std::string, std::string> dm_defines0 = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args_dm0)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm0)}};
    std::map<std::string, std::string> dm_defines1 = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args_dm1)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm1)}};
    std::map<std::string, std::string> compute_defines = {
        {"COMPUTE", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args_compute)},
        {"RESULTS_ADDR", std::to_string(rta_base_compute)}};

    auto dm_kernel0 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = dm_defines0});

    auto dm_kernel1 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = dm_defines1});

    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        ComputeConfig{.defines = compute_defines});

    vector<uint32_t> dm_kernel0_args;
    vector<uint32_t> dm_kernel1_args;
    vector<uint32_t> compute_kernel_args;

    uint32_t idx;
    for (idx = 0; idx < num_runtime_args_dm0; idx++) {
        dm_kernel0_args.push_back(idx);
    }
    for (; idx < num_runtime_args_dm0 + num_runtime_args_dm1; idx++) {
        dm_kernel1_args.push_back(idx);
    }
    for (; idx < num_runtime_args_dm0 + num_runtime_args_dm1 + num_runtime_args_compute; idx++) {
        compute_kernel_args.push_back(idx);
    }

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        for (const CoreCoord& core_coord : core_range) {
            SetRuntimeArgs(program, dm_kernel0, core_coord, dm_kernel0_args);
            SetRuntimeArgs(program, dm_kernel1, core_coord, dm_kernel1_args);
            SetRuntimeArgs(program, compute_kernel, core_coord, compute_kernel_args);
        }
    }

    workload.add_program(device_range, std::move(program));
    for (uint32_t i = 0; i < num_iterations; i++) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);

    // Early return to skip slow dispatch path
    if (!do_checks) {
        return pass;
    }

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        for (const CoreCoord& core_coord : core_range) {
            vector<uint32_t> dm_kernel0_args_readback;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, rta_base_dm0, dm_kernel0_args.size() * sizeof(uint32_t), dm_kernel0_args_readback);
            pass &= (dm_kernel0_args == dm_kernel0_args_readback);

            vector<uint32_t> dm_kernel1_args_readback;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, rta_base_dm1, dm_kernel1_args.size() * sizeof(uint32_t), dm_kernel1_args_readback);
            pass &= (dm_kernel1_args == dm_kernel1_args_readback);

            vector<uint32_t> compute_kernel_args_readback;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_compute,
                compute_kernel_args.size() * sizeof(uint32_t),
                compute_kernel_args_readback);
            pass &= (compute_kernel_args == compute_kernel_args_readback);
        }
    }

    return pass;
}

bool test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const DummyProgramConfig& program_config,
    uint32_t num_runtime_args_for_cr0,
    uint32_t num_runtime_args_for_cr1,
    uint32_t num_iterations) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];
    Program program;
    bool pass = true;

    // TODO: this test would be better if it varied args across core ranges and kernel type

    CoreRangeSet cr_set = program_config.cr_set;
    constexpr uint32_t kCommonRTASeparation = 1024 * sizeof(uint32_t);

    uint32_t rta_base_dm0 = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t rta_base_dm1 = rta_base_dm0 + (2048 * sizeof(uint32_t));
    uint32_t rta_base_compute = rta_base_dm1 + (4096 * sizeof(uint32_t));
    // Copy max # runtime args in the kernel for simplicity
    std::map<std::string, std::string> dm_defines0 = {
        {"COMMON_RUNTIME_ARGS", "1"},
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm0)}};
    std::map<std::string, std::string> dm_defines1 = {
        {"COMMON_RUNTIME_ARGS", "1"},
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm1)}};
    std::map<std::string, std::string> compute_defines = {
        {"COMMON_RUNTIME_ARGS", "1"},
        {"COMPUTE", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_compute)}};

    auto dummy_kernel0 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = dm_defines0});

    auto dummy_kernel1 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = dm_defines1});

    auto dummy_compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        cr_set,
        ComputeConfig{.defines = compute_defines});

    vector<uint32_t> dummy_cr0_args;
    vector<uint32_t> dummy_cr1_args;
    vector<uint32_t> dummy_common_args;

    auto it = program_config.cr_set.ranges().begin();
    CoreRange core_range_0 = *it;
    std::advance(it, 1);
    CoreRange core_range_1 = *it;

    uint32_t idx = 0;
    constexpr uint32_t num_common_runtime_args = 13;
    workload.add_program(device_range, std::move(program));

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        auto& program_ = workload.get_programs().at(device_range);
        SCOPED_TRACE(iter);
        dummy_cr0_args.clear();
        dummy_cr1_args.clear();
        dummy_common_args.clear();

        for (uint32_t i = 0; i < num_runtime_args_for_cr0; i++) {
            dummy_cr0_args.push_back(idx++);
        }

        for (uint32_t i = 0; i < num_runtime_args_for_cr1; i++) {
            dummy_cr1_args.push_back(idx++);
        }

        for (uint32_t i = 0; i < num_common_runtime_args; i++) {
            dummy_common_args.push_back(idx++);
        }

        bool first = true;
        for (const CoreCoord& core_coord : core_range_0) {
            // Don't set RTAs on all cores
            if (first) {
                first = false;
                continue;
            }

            SetRuntimeArgs(program_, dummy_kernel0, core_coord, dummy_cr0_args);
            SetRuntimeArgs(program_, dummy_kernel1, core_coord, dummy_cr0_args);
            SetRuntimeArgs(program_, dummy_compute_kernel, core_coord, dummy_cr0_args);
        }

        first = true;
        for (const CoreCoord& core_coord : core_range_1) {
            // Don't set RTAs on all cores
            if (first) {
                first = false;
                continue;
            }

            SetRuntimeArgs(program_, dummy_kernel0, core_coord, dummy_cr1_args);
            SetRuntimeArgs(program_, dummy_kernel1, core_coord, dummy_cr1_args);
            SetRuntimeArgs(program_, dummy_compute_kernel, core_coord, dummy_cr1_args);
        }

        if (iter == 0) {
            SetCommonRuntimeArgs(program_, dummy_kernel0, dummy_common_args);
            SetCommonRuntimeArgs(program_, dummy_kernel1, dummy_common_args);
            SetCommonRuntimeArgs(program_, dummy_compute_kernel, dummy_common_args);
        } else {
            memcpy(
                GetCommonRuntimeArgs(program_, dummy_kernel0).rt_args_data,
                dummy_common_args.data(),
                dummy_common_args.size() * sizeof(uint32_t));
            memcpy(
                GetCommonRuntimeArgs(program_, dummy_kernel1).rt_args_data,
                dummy_common_args.data(),
                dummy_common_args.size() * sizeof(uint32_t));
            memcpy(
                GetCommonRuntimeArgs(program_, dummy_compute_kernel).rt_args_data,
                dummy_common_args.data(),
                dummy_common_args.size() * sizeof(uint32_t));
        }

        distributed::EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);

        first = true;
        for (const CoreCoord& core_coord : core_range_0) {
            // Don't test RTAs on first cores
            if (first) {
                first = false;
                continue;
            }
            {
                vector<uint32_t> dummy_kernel0_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm0,
                    dummy_cr0_args.size() * sizeof(uint32_t),
                    dummy_kernel0_args_readback);
                pass &= (dummy_cr0_args == dummy_kernel0_args_readback);

                vector<uint32_t> dummy_kernel1_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm1,
                    dummy_cr0_args.size() * sizeof(uint32_t),
                    dummy_kernel1_args_readback);
                pass &= (dummy_cr0_args == dummy_kernel1_args_readback);

                vector<uint32_t> dummy_compute_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_compute,
                    dummy_cr0_args.size() * sizeof(uint32_t),
                    dummy_compute_args_readback);
                pass &= (dummy_cr0_args == dummy_compute_args_readback);
            }
            {
                vector<uint32_t> dummy_kernel0_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm0 + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_kernel0_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_kernel0_args_readback);
                pass &= (dummy_common_args == dummy_kernel0_args_readback);

                vector<uint32_t> dummy_kernel1_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm1 + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_kernel1_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_kernel1_args_readback);
                pass &= (dummy_common_args == dummy_kernel1_args_readback);

                vector<uint32_t> dummy_compute_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_compute + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_compute_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_compute_args_readback);
                pass &= (dummy_common_args == dummy_compute_args_readback);
            }
        }

        first = true;
        for (const CoreCoord& core_coord : core_range_1) {
            // Don't test RTAs on first cores
            if (first) {
                first = false;
                continue;
            }
            {
                vector<uint32_t> dummy_kernel0_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm0,
                    dummy_cr1_args.size() * sizeof(uint32_t),
                    dummy_kernel0_args_readback);
                pass &= (dummy_cr1_args == dummy_kernel0_args_readback);

                vector<uint32_t> dummy_kernel1_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm1,
                    dummy_cr1_args.size() * sizeof(uint32_t),
                    dummy_kernel1_args_readback);
                pass &= (dummy_cr1_args == dummy_kernel1_args_readback);

                vector<uint32_t> dummy_compute_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_compute,
                    dummy_cr1_args.size() * sizeof(uint32_t),
                    dummy_compute_args_readback);
                pass &= (dummy_cr1_args == dummy_compute_args_readback);
            }
            {
                vector<uint32_t> dummy_kernel0_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm0 + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_kernel0_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_kernel0_args_readback);
                pass &= (dummy_common_args == dummy_kernel0_args_readback);

                vector<uint32_t> dummy_kernel1_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_dm1 + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_kernel1_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_kernel1_args_readback);
                pass &= (dummy_common_args == dummy_kernel1_args_readback);

                vector<uint32_t> dummy_compute_args_readback;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    rta_base_compute + kCommonRTASeparation,
                    dummy_common_args.size() * sizeof(uint32_t),
                    dummy_compute_args_readback);
                EXPECT_EQ(dummy_common_args, dummy_compute_args_readback);
                pass &= (dummy_common_args == dummy_compute_args_readback);
            }
        }
    }

    return pass;
}

// Verify RT args for a core at a given address by comparing to expected values.
bool verify_rt_args(
    bool unique,
    IDevice* device,
    CoreCoord logical_core,
    HalProgrammableCoreType core_type,
    uint32_t addr,
    std::vector<uint32_t> expected_rt_args,
    uint32_t incr_val) {
    bool pass = true;
    std::string label = unique ? "Unique" : "Common";
    // Same idea as ReadFromDeviceL1() but with ETH support.
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    auto noc_xy = (core_type == HalProgrammableCoreType::ACTIVE_ETH || core_type == HalProgrammableCoreType::IDLE_ETH)
                      ? device->ethernet_core_from_logical_core(logical_core)
                      : device->worker_core_from_logical_core(logical_core);
    std::vector<uint32_t> args_readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(), noc_xy, addr, expected_rt_args.size() * sizeof(uint32_t));
    log_debug(
        tt::LogTest,
        "Verifying {} {} RT args for {} (Logical: {}) at addr: 0x{:x} w/ incr_val: {}",
        expected_rt_args.size(),
        label,
        noc_xy,
        logical_core.str(),
        addr,
        incr_val);

    for (int i = 0; i < expected_rt_args.size(); i++) {
        uint32_t expected_val = expected_rt_args[i] + incr_val;
        log_debug(
            tt::LogTest,
            "Checking {} RT Arg. i: {} expected: {} observed: {}",
            label,
            i,
            expected_val,
            args_readback[i]);
        EXPECT_EQ(args_readback[i], expected_val);
        pass &= (args_readback[i] == expected_val);
    }
    return pass;
}

// Returns L1 address for {unique RTA, common RTA}
std::pair<uint32_t, uint32_t> get_args_addr(const IDevice* device, HalProcessorIdentifier processor) {
    uint32_t unique_args_addr;
    uint32_t common_args_addr;
    auto [core_type, processor_class, processor_id] = processor;
    switch (core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (processor_class) {
                case HalProcessorClassType::DM:
                    TT_FATAL(
                        0 <= processor_id && processor_id < 2, "processor_id {} must be 0 or 1 for DM", processor_id);
                    unique_args_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1) +
                                       processor_id * 256 * sizeof(uint32_t);
                    common_args_addr = unique_args_addr + (3 + processor_id) * 256 * sizeof(uint32_t);
                    break;
                case HalProcessorClassType::COMPUTE:
                    unique_args_addr =
                        device->allocator()->get_base_allocator_addr(HalMemType::L1) + 2 * 256 * sizeof(uint32_t);
                    common_args_addr = unique_args_addr + 5 * 256 * sizeof(uint32_t);
                    break;
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH:
            unique_args_addr = MetalContext::instance().hal().get_dev_addr(core_type, HalL1MemAddrType::UNRESERVED);
            common_args_addr = unique_args_addr + 1 * 256 * sizeof(uint32_t);
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("bad core type");
    }
    return {unique_args_addr, common_args_addr};
}

// Call CreateKernel for the program configs
// Returns a struct with the kernel IDs, and L1 addresses to check CRTA/RTAs.
IncrementKernelsSet create_increment_kernels(
    const IDevice* device,
    Program& program,
    const std::vector<DummyProgramConfig>& program_configs,
    HalProcessorIdentifier processor,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args) {
    // Tell kernel how many unique and common RT args to expect. Will increment each.
    std::vector<KernelHandle> kernels;
    const auto [unique_args_addr, common_args_addr] = get_args_addr(device, processor);
    std::vector<uint32_t> compile_args{num_unique_rt_args, num_common_rt_args, unique_args_addr, common_args_addr};

    const std::string increment_kernel_path =
        processor.processor_class == HalProcessorClassType::COMPUTE
            ? "tests/tt_metal/tt_metal/test_kernels/compute/increment_runtime_arg.cpp"
            : "tests/tt_metal/tt_metal/test_kernels/misc/increment_runtime_arg.cpp";

    // CreateKernel on each core range set
    for (const auto& program_config : program_configs) {
        const auto& cr_set = program_config.cr_set;
        KernelHandle kernel_id = create_kernel(processor, program, cr_set, compile_args, increment_kernel_path);

        kernels.push_back(kernel_id);
    }

    return IncrementKernelsSet{
        .kernel_handles = kernels, .unique_args_addr = unique_args_addr, .common_args_addr = common_args_addr};
}

// Write unique and common RT args, increment in kernel, and verify correctness via readback.
// Multiple program_configs may be provided to create multiple kernels on the same program.
bool test_increment_runtime_args_sanity(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<DummyProgramConfig>& program_configs,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args,
    HalProcessorIdentifier processor) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];
    Program program;
    bool pass = true;

    auto configured_kernels =
        create_increment_kernels(device, program, program_configs, processor, num_unique_rt_args, num_common_rt_args);

    // Args will be at this addr in L1
    uint32_t unique_args_addr = configured_kernels.unique_args_addr;
    uint32_t common_args_addr = configured_kernels.common_args_addr;

    // Generate Runtime Args.
    std::vector<uint32_t> unique_runtime_args;
    unique_runtime_args.reserve(num_unique_rt_args);
    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        unique_runtime_args.push_back(i * 0x10101010);
    }

    // Generate Common Runtime Args.
    std::vector<uint32_t> common_runtime_args;
    common_runtime_args.reserve(num_common_rt_args);
    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        common_runtime_args.push_back(1000 + 0x10101010);
    }

    // Call SetRuntimeArgs. Set for core ranges that are running the kernel
    // zip the kernel_id and cr set from program_config
    for (int i = 0; i < program_configs.size(); ++i) {
        const auto& cr_set = program_configs[i].cr_set;
        const auto& kernel_id = configured_kernels.kernel_handles[i];

        SetRuntimeArgs(program, kernel_id, cr_set, unique_runtime_args);
    }

    // Call SetCommonRuntimeArgs for kernels. Does not take into account core range as it's common.
    for (const auto& kernel_id : configured_kernels.kernel_handles) {
        SetCommonRuntimeArgs(program, kernel_id, common_runtime_args);
    }

    // Compile and Launch the Program now.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    Finish(mesh_device->mesh_command_queue());

    // Read all cores for all kernels
    constexpr uint32_t unique_arg_incr_val = 10;
    constexpr uint32_t common_arg_incr_val = 100;
    for (const auto& kernel_id : configured_kernels.kernel_handles) {
        const auto& kernel = workload.get_programs()[device_range].impl().get_kernel(kernel_id);

        for (auto& core_range : kernel->logical_coreranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord core_coord(x, y);
                    pass &= verify_rt_args(
                        true,
                        device,
                        core_coord,
                        processor.core_type,
                        unique_args_addr,
                        unique_runtime_args,
                        unique_arg_incr_val);
                    pass &= verify_rt_args(
                        false,
                        device,
                        core_coord,
                        processor.core_type,
                        common_args_addr,
                        common_runtime_args,
                        common_arg_incr_val);
                }
            }
        }
    }

    return pass;
}

bool test_increment_runtime_args_sanity(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const DummyProgramConfig& program_config,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args,
    HalProcessorIdentifier processor) {
    return test_increment_runtime_args_sanity(
        mesh_device,
        std::vector<DummyProgramConfig>{program_config},
        num_unique_rt_args,
        num_common_rt_args,
        processor);
}

void test_my_coordinates(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, HalProcessorIdentifier processor, size_t cq_id = 0) {
    const std::string k_kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/read_my_coordinates.cpp";
    // All logical cores
    CoreRangeSet cr{CoreRange{{2, 2}, {6, 6}}};

    uint32_t cb_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    std::vector<uint32_t> compile_args{
        cb_addr,
    };

    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    create_kernel(processor, program, CoreRangeSet{cr}, compile_args, k_kernel_path);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(cq_id), workload, false);
    Finish(mesh_device->mesh_command_queue(cq_id));

    tt::tt_metal::verify_kernel_coordinates(
        processor.core_type, cr, mesh_device.get(), tt::tt_metal::SubDeviceId{0}, cb_addr);
}

void test_basic_dispatch_functions(const std::shared_ptr<distributed::MeshDevice>& mesh_device, int cq_id) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    constexpr uint32_t k_DataSize = 64 * 1024;
    constexpr uint32_t k_PageSize = 4 * 1024;
    constexpr uint32_t k_Iterations = 10;
    constexpr uint32_t k_LoopPerDev = 100;

    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    auto* device = mesh_device->get_devices()[0];
    log_info(tt::LogTest, "Running On Device {} CQ{}", mesh_device->id(), cq_id);

    log_info(tt::LogTest, "Running On Device {} CQ{}", device->id(), cq_id);

    // Alternate write patterns
    std::vector<uint32_t> src_data_1(k_DataSize / sizeof(uint32_t));
    std::vector<uint32_t> src_data_2(k_DataSize / sizeof(uint32_t));
    for (int i = 0; i < k_DataSize / sizeof(uint32_t); ++i) {
        src_data_1[i] = (device->id() + rand()) * 0xdeadbeef;
        src_data_2[i] = (device->id() + rand()) * 0xabcd1234;
    }
    distributed::DeviceLocalBufferConfig l1_buffer_config{.page_size = k_PageSize, .buffer_type = BufferType::L1};
    distributed::DeviceLocalBufferConfig dram_buffer_config{.page_size = k_PageSize, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig replicated_buffer_config{.size = k_DataSize};

    auto buffer = distributed::MeshBuffer::create(replicated_buffer_config, l1_buffer_config, mesh_device.get());
    auto dram_buffer = distributed::MeshBuffer::create(replicated_buffer_config, dram_buffer_config, mesh_device.get());

    auto& cq = mesh_device->mesh_command_queue(cq_id);

    for (int iteration = 0; iteration < k_Iterations; ++iteration) {
        for (int i = 0; i < k_LoopPerDev; ++i) {
            EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
                mesh_device, cq, dummy_program_config, 24, 12, 15, k_LoopPerDev, false));

            std::vector<uint32_t> dst_data;
            if (i & 1) {
                distributed::EnqueueWriteMeshBuffer(cq, buffer, src_data_1, false);
                distributed::ReadShard(cq, dst_data, buffer, distributed::MeshCoordinate{0, 0}, true);
                EXPECT_EQ(src_data_1, dst_data);
            } else {
                distributed::EnqueueWriteMeshBuffer(cq, dram_buffer, src_data_2, false);
                distributed::ReadShard(cq, dst_data, dram_buffer, distributed::MeshCoordinate{0, 0}, true);
                EXPECT_EQ(src_data_2, dst_data);
            }
        }
    }

    // non blocking fast data movement APIs
    for (int iteration = 0; iteration < k_Iterations; ++iteration) {
        for (int i = 0; i < k_LoopPerDev; ++i) {
            distributed::EnqueueWriteMeshBuffer(cq, buffer, src_data_1, false);
        }
    }

    std::vector<uint32_t> dst_data;
    for (int iteration = 0; iteration < k_Iterations; ++iteration) {
        for (int i = 0; i < k_LoopPerDev; ++i) {
            distributed::ReadShard(cq, dst_data, buffer, distributed::MeshCoordinate{0, 0}, true);
        }
    }

    distributed::Finish(cq);
}

}  // namespace local_test_functions

namespace basic_tests {

namespace compiler_workaround_hardware_bug_tests {

TEST_F(UnitMeshCQFixture, TensixTestArbiterDoesNotHang) {
    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});
        // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
        // added separately
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
            cr_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
        Finish(device->mesh_command_queue());
    }
}
}  // namespace compiler_workaround_hardware_bug_tests
namespace single_core_tests {

TEST_F(UnitMeshCQFixture, TensixTestSingleCbConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id = 0, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config}};

    for (const auto& device : devices_) {
        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestMultiCbSeqConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {
        .cr_set = cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};

    for (const auto& device : devices_) {
        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestMultiCbRandomConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 1, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 0, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 24, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 16, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {
        .cr_set = cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};

    for (const auto& device : devices_) {
        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestMultiCBSharedAddressSpaceSentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    uint32_t intermediate_cb = 24;
    uint32_t out_cb = 16;
    std::map<uint8_t, tt::DataFormat> intermediate_and_out_data_format_spec = {
        {intermediate_cb, tt::DataFormat::Float16_b}, {out_cb, tt::DataFormat::Float16_b}};
    uint32_t num_bytes_for_df = 2;
    uint32_t single_tile_size = num_bytes_for_df * 1024;
    uint32_t num_tiles = 2;
    uint32_t cb_size = num_tiles * single_tile_size;

    uint32_t cb_config_buffer_size =
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    CoreCoord core_coord(0, 0);

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);

        CircularBufferConfig cb_config = CircularBufferConfig(cb_size, intermediate_and_out_data_format_spec)
                                             .set_page_size(intermediate_cb, single_tile_size)
                                             .set_page_size(out_cb, single_tile_size);
        CreateCircularBuffer(program_, cr_set, cb_config);

        local_test_functions::initialize_dummy_kernels(program_, cr_set);

        distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
        Finish(device->mesh_command_queue());

        vector<uint32_t> cb_config_vector;

        auto address = program_.impl().get_cb_base_addr(device->get_devices()[0], core_coord, CoreType::WORKER);
        tt::tt_metal::detail::ReadFromDeviceL1(
            device->get_devices()[0], core_coord, address, cb_config_buffer_size, cb_config_vector);
        uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t intermediate_index = intermediate_cb * sizeof(uint32_t);

        bool addr_match_intermediate = cb_config_vector.at(intermediate_index) == (cb_addr);
        bool size_match_intermediate = cb_config_vector.at(intermediate_index + 1) == (cb_size);
        bool num_pages_match_intermediate = cb_config_vector.at(intermediate_index + 2) == num_tiles;
        bool pass_intermediate = (addr_match_intermediate and size_match_intermediate and num_pages_match_intermediate);
        EXPECT_TRUE(pass_intermediate);

        uint32_t out_index = out_cb * sizeof(uint32_t);
        bool addr_match_out = cb_config_vector.at(out_index) == cb_addr;
        bool size_match_out = cb_config_vector.at(out_index + 1) == cb_size;
        bool num_pages_match_out = cb_config_vector.at(out_index + 2) == num_tiles;
        bool pass_out = (addr_match_out and size_match_out and num_pages_match_out);
        EXPECT_TRUE(pass_out);
    }
}

TEST_F(UnitMeshCQFixture, TensixTestSingleCbConfigCorrectlyUpdateSizeSentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id = 0, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config}};

    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(
            device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestSingleSemaphoreConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    for (const auto& device : devices_) {
        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_sems(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAutoInsertedBlankBriscKernelInDeviceDispatchMode) {
    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);
        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});
        // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
        // added separately
        CreateKernel(
            program_,
            "tt_metal/kernels/dataflow/blank.cpp",
            cr_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
        Finish(device->mesh_command_queue());
    }
}

// Sanity test for setting and verifying common and unique runtime args to a single core, the simplest case.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanitySingleCoreCompute) {
    CoreRange cr0({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr0});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device, dummy_program_config, 8, 8, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    }
}

// Test setting common runtime args across multiple kernel in the same program
// This test will ensure a multicast (or unicast for eth) gets created for each time the
// user calls SetCommonRuntimeArgs.
TEST_F(UnitMeshCQFixture, TensixSetCommonRuntimeArgsMultipleCreateKernel) {
    const CoreRange core_range_0(CoreCoord(1, 1), CoreCoord(2, 2));
    const CoreRange core_range_1(CoreCoord(3, 3), CoreCoord(4, 4));

    const CoreRangeSet core_range_set_0(std::vector{core_range_0});
    const CoreRangeSet core_range_set_1(std::vector{core_range_1});

    std::vector<DummyProgramConfig> configs{
        {.cr_set = core_range_set_0},
        {.cr_set = core_range_set_1},
    };

    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device, configs, 8, 8, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    }
}

TEST_F(UnitMeshCQFixture, ActiveEthEnqueueDummyProgram) {
    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    if (erisc_count != 2) {
        GTEST_SKIP() << "Skipping test as this test requires 2 active ethernet cores";
    }
    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_devices()[0]->get_active_ethernet_cores(true)) {
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
                log_info(
                    tt::LogTest,
                    "Test active ethernet enqueue dummy program with runtime args for eth_core: {} DM{}",
                    eth_core.str(),
                    erisc_idx);
                local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
                    device, eth_core, static_cast<DataMovementProcessor>(erisc_idx));
            }
        }
    }
}

// Test to see we can launch a kernel at the same time on both active ethernet cores
// If they can't handshake it means only 1 was able to launch
TEST_F(UnitMeshCQFixture, ActiveEthTwoRiscsHandshake) {
    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    if (erisc_count < 2) {
        GTEST_SKIP() << "Skipping test as this test requires 2 ethernet cores";
    }
    for (const auto& mesh_device : devices_) {
        auto& cq = mesh_device->mesh_command_queue();
        distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        for (const auto& eth_core : mesh_device->get_devices()[0]->get_active_ethernet_cores(true)) {
            auto program = tt::tt_metal::CreateProgram();
            auto primary = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/local_handshake_2.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig{.noc = tt::tt_metal::NOC::NOC_0, .processor = DataMovementProcessor::RISCV_0}
            );
            auto secondary = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/local_handshake_2.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig{.noc = tt::tt_metal::NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1}
            );

            uint32_t unreserved_l1 = hal::get_erisc_l1_unreserved_base();
            uint32_t init_value = rand();
            log_info(tt::LogTest,
                "Test active ethernet handshake for eth_core: {} DM0 and DM1, init value: {} unreserved_l1: 0x{:x}",
                eth_core.str(),
                init_value, unreserved_l1);

            std::vector<uint32_t> primary_kernel_args = {1, unreserved_l1, init_value};
            std::vector<uint32_t> secondary_kernel_args = {0, unreserved_l1, init_value};

            tt::tt_metal::SetRuntimeArgs(program, primary, eth_core, primary_kernel_args);
            tt::tt_metal::SetRuntimeArgs(program, secondary, eth_core, secondary_kernel_args);

            distributed::MeshWorkload workload;
            workload.add_program(device_range, std::move(program));
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }

        distributed::Finish(cq);
    }
}

// Sanity test for setting and verifying common and unique runtime args to single cores via ERISC. Some arch may return
// 0 active eth cores, that's okay.
TEST_F(UnitMeshCQFixture, ActiveEthIncrementRuntimeArgsSanitySingleCoreDataMovementErisc) {
    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_devices()[0]->get_active_ethernet_cores(true)) {
            CoreRange cr0(eth_core);
            CoreRangeSet cr_set({cr0});
            DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
            log_info(tt::LogTest, "Issuing test for eth_core: {} using cr_set: {}", eth_core.str(), cr_set.str());
            const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                HalProgrammableCoreType::ACTIVE_ETH);
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
                log_info(
                    tt::LogTest,
                    "Test active ethernet runtime args for eth_core: {} DM{} using cr_set: {}",
                    eth_core.str(),
                    erisc_idx,
                    cr_set.str());
                EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
                    device,
                    dummy_program_config,
                    16,
                    16,
                    {HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, erisc_idx}));
            }
        }
    }
}

// Sanity test for setting and verifying common and unique runtime args to single cores via ERISC(IDLE). Some arch may
// return 0 active eth cores, that's okay.
// FIXME - Re-enable when FD-on-idle-eth is supported
TEST_F(UnitMeshCQFixture, DISABLED_ActiveEthIncrementRuntimeArgsSanitySingleCoreDataMovementEriscIdle) {
    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_devices()[0]->get_active_ethernet_cores(true)) {
            CoreRange cr0(eth_core);
            CoreRangeSet cr_set({cr0});
            DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
            log_info(tt::LogTest, "Issuing test for idle eth_core: {} using cr_set: {}", eth_core.str(), cr_set.str());
            EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
                device,
                dummy_program_config,
                16,
                16,
                {HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 0}));
        }
    }
}

// Sanity test for setting and verifying common and unique runtime args to single cores via inactive ERISC cores. Some
// arch may return 0 active eth cores, that's okay.
// FIXME - Re-enable when FD-on-idle-eth is supported
TEST_F(UnitMeshCQFixture, DISABLED_IdleEthIncrementRuntimeArgsSanitySingleCoreDataMovementEriscInactive) {
    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_devices()[0]->get_inactive_ethernet_cores()) {
            CoreRange cr0(eth_core);
            CoreRangeSet cr_set({cr0});
            DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
            log_info(
                tt::LogTest, "Issuing test for inactive eth_core: {} using cr_set: {}", eth_core.str(), cr_set.str());
            EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
                device,
                dummy_program_config,
                16,
                16,
                {HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 0}));
        }
    }
}

TEST_F(UnitMeshCQFixture, TensixTestRuntimeArgsCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (auto& device : devices_) {
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
            device, device->mesh_command_queue(), dummy_program_config, 9, 12, 15, 1);
    }
}

}  // end namespace single_core_tests

namespace multicore_tests {
TEST_F(UnitMeshCQFixture, TensixTestAllCbConfigsCorrectlySentMultiCore) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        cb_config_vector[i].cb_id = i;
    }

    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllCbConfigsCorrectlySentUpdateSizeMultiCore) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        cb_config_vector[i].cb_id = i;
    }

    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(
            device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestMultiCbConfigsCorrectlySentUpdateSizeMultiCore) {
    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(
            device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllCbConfigsCorrectlySentMultipleCoreRanges) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        cb_config_vector[i].cb_id = i;
    }

    for (const auto& device : devices_) {
        CoreRange cr0({0, 0}, {1, 1});

        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        CoreRange cr1(
            {worker_grid_size.x - 2, worker_grid_size.y - 2}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

        CoreRangeSet core_ranges(std::vector{cr0, cr1});

        DummyProgramMultiCBConfig config = {.cr_set = core_ranges, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllCbConfigsCorrectlySentUpdateSizeMultipleCoreRanges) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        cb_config_vector[i].cb_id = i;
    }

    for (const auto& device : devices_) {
        CoreRange cr0({0, 0}, {1, 1});

        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        CoreRange cr1(
            {worker_grid_size.x - 2, worker_grid_size.y - 2}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

        CoreRangeSet core_ranges(std::vector{cr0, cr1});

        DummyProgramMultiCBConfig config = {.cr_set = core_ranges, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(
            device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestMultiCbConfigsCorrectlySentUpdateSizeMultipleCoreRanges) {
    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    for (const auto& device : devices_) {
        CoreRange cr0({0, 0}, {1, 1});

        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        CoreRange cr1(
            {worker_grid_size.x - 2, worker_grid_size.y - 2}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

        CoreRangeSet core_ranges(std::vector{cr0, cr1});

        DummyProgramMultiCBConfig config = {.cr_set = core_ranges, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(
            device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllSemConfigsCorrectlySentMultiCore) {
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(cr);

        DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};
        EXPECT_TRUE(
            local_test_functions::test_dummy_EnqueueProgram_with_sems(device, device->mesh_command_queue(), config));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllSemaphoreConfigsCorrectlySentMultipleCoreRanges) {
    for (const auto& device : devices_) {
        CoreRange first_cr({0, 0}, {1, 1});

        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        CoreRange second_cr(
            {worker_grid_size.x - 2, worker_grid_size.y - 2}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

        CoreRangeSet cr_set(std::vector{first_cr, second_cr});

        Program program;
        DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

        vector<vector<uint32_t>> expected_semaphore_vals;

        uint32_t semaphore_val = 0;
        vector<uint32_t> initial_semaphore_vals;
        for (uint32_t i = 0; i < config.num_sems; i++) {
            initial_semaphore_vals.push_back(semaphore_val);
            semaphore_val++;
        }

        local_test_functions::initialize_dummy_semaphores(program, first_cr, initial_semaphore_vals);
        expected_semaphore_vals.push_back(initial_semaphore_vals);

        initial_semaphore_vals.clear();
        for (uint32_t i = 0; i < config.num_sems; i++) {
            initial_semaphore_vals.push_back(semaphore_val);
            semaphore_val++;
        }

        local_test_functions::initialize_dummy_semaphores(program, second_cr, initial_semaphore_vals);
        expected_semaphore_vals.push_back(initial_semaphore_vals);
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_sems(
            device, device->mesh_command_queue(), workload, config, expected_semaphore_vals));
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllRuntimeArgsCorrectlySentMultiCore) {
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(cr);

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
            device, device->mesh_command_queue(), dummy_program_config, 13, 17, 19, 1);
    }
}

TEST_F(UnitMeshCQFixture, TensixTestAllRuntimeArgsCorrectlySentMultiCore_MaxRuntimeArgs_PerKernel) {
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(cr);

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        auto n_rt = tt::tt_metal::max_runtime_args;
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
            device, device->mesh_command_queue(), dummy_program_config, n_rt, n_rt, n_rt, 1);
    }
}

TEST_F(UnitMeshCQFixture, TensixTestSendRuntimeArgsMultiCoreRange) {
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(std::vector{cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->mesh_command_queue(), dummy_program_config, 12, 9, 2);
    }
}

TEST_F(UnitMeshCQFixture, TensixTestSendRuntimeArgsMultiNonOverlappingCoreRange) {
    // Core ranges get merged in kernel groups, this one does not
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 5}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(std::vector{cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->mesh_command_queue(), dummy_program_config, 9, 12, 2);
    }
}

TEST_F(UnitMeshCQFixture, TensixTestUpdateRuntimeArgsMultiCoreRange) {
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 5}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(std::vector{cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->mesh_command_queue(), dummy_program_config, 9, 31, 10);
    }
}

// Sanity test for setting and verifying common and unique runtime args to multiple cores.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanityMultiCoreCompute) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set(std::vector{cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device,
            dummy_program_config,
            16,
            16,
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    }
}

// Max number of max_runtime_args unique RT args.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanityMultiCoreCompute_MaxRuntimeArgs_UniqueArgs) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set(std::vector{cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device,
            dummy_program_config,
            tt::tt_metal::max_runtime_args,
            0,
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    }
}

// Max number of max_runtime_args common RT args.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanityMultiCoreCompute_MaxRuntimeArgs_CommonArgs) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set(std::vector{cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device,
            dummy_program_config,
            0,
            tt::tt_metal::max_runtime_args,
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    }
}

// Sanity test for setting and verifying common and unique runtime args to multiple cores via BRISC.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanityMultiCoreDataMovementBrisc) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set(std::vector{cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device, dummy_program_config, 16, 16, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0}));
    }
}

// Sanity test for setting and verifying common and unique runtime args to multiple cores via NCRISC.
TEST_F(UnitMeshCQFixture, TensixIncrementRuntimeArgsSanityMultiCoreDataMovementNcrisc) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set(std::vector{cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (const auto& device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(
            device, dummy_program_config, 16, 16, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1}));
    }
}

// Ensure the data movement core can access their own logical coordinate. Same binary enqueued to multiple cores.
TEST_F(UnitMeshCQFixture, TestLogicalCoordinatesDataMovement) {
    for (const auto& device : devices_) {
        local_test_functions::test_my_coordinates(
            device, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0});
        local_test_functions::test_my_coordinates(
            device, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1});
    }
}

// Ensure the compute core can access their own logical coordinate. Same binary enqueued to multiple cores.
TEST_F(UnitMeshCQFixture, TestLogicalCoordinatesCompute) {
    for (const auto& device : devices_) {
        local_test_functions::test_my_coordinates(
            device, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0});
    }
}

// Ensure the eth core can access their own logical coordinate. Same binary enqueued to multiple cores.
TEST_F(UnitMeshCQFixture, TestLogicalCoordinatesEth) {
    GTEST_SKIP() << "Mesh device does not support logical / relative coordinates on Eth";
    for (const auto& device : devices_) {
        if (!does_device_have_active_eth_cores(device->get_devices()[0])) {
            GTEST_SKIP() << "Skipping test because device " << device->id()
                         << " does not have any active ethernet cores";
        }
        const auto erisc_count =
            tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test logical coordinates active ethernet DM{}", erisc_idx);
            local_test_functions::test_my_coordinates(
                device, {HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, erisc_idx});
        }
    }
}

// Ensure the data movement core can access their own logical coordinate. Same binary enqueued to multiple cores.
TEST_F(UnitMeshMultiCQSingleDeviceProgramFixture, TestLogicalCoordinatesDataMovement) {
    local_test_functions::test_my_coordinates(device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0});
    local_test_functions::test_my_coordinates(
        device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0}, 1);
    local_test_functions::test_my_coordinates(device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1});
    local_test_functions::test_my_coordinates(
        device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1}, 1);
}

// Ensure the compute core can access their own logical coordinate. Same binary enqueued to multiple cores.
TEST_F(UnitMeshMultiCQSingleDeviceProgramFixture, TestLogicalCoordinatesCompute) {
    local_test_functions::test_my_coordinates(
        device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0});
    local_test_functions::test_my_coordinates(
        device_, {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}, 1);
}

}  // end namespace multicore_tests
}  // namespace basic_tests

namespace stress_tests {

TEST_F(UnitMeshMultiCQSingleDeviceProgramFixture, TensixTestRandomizedProgram) {
    uint32_t NUM_WORKLOADS = 100;
    uint32_t MAX_LOOP = 100;
    uint32_t page_size = 1024;

    if (this->arch_ == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP();  // Running on second CQ is hanging on CI
    }

    // Make random
    auto random_seed = 0;  // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    CoreCoord worker_grid_size = device_->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_WORKLOADS);

    vector<distributed::MeshWorkload> workloads;
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        workloads.push_back(distributed::MeshWorkload());
        distributed::MeshWorkload& workload = workloads.back();

        Program program;
        workload.add_program(this->device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(this->device_range_);

        std::map<std::string, std::string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
        std::map<std::string, std::string> compute_defines = {{"COMPUTE", "1"}};

        // brisc
        uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
        bool USE_MAX_RT_ARGS;

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

        log_debug(
            tt::LogTest,
            "Compiling program {}/{} w/ BRISC_OUTER_LOOP: {} BRISC_MIDDLE_LOOP: {} BRISC_INNER_LOOP: {} NUM_CBS: {} "
            "NUM_SEMS: {} USE_MAX_RT_ARGS: {}",
            i + 1,
            NUM_WORKLOADS,
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            USE_MAX_RT_ARGS);

        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}})
                                                 .set_page_size(j, page_size * (j + 1));
            CreateCircularBuffer(program_, cr_set, cb_config);
        }

        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program_, cr_set, j + 1);
        }

        auto [brisc_unique_rtargs, brisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_brisc_unique_rtargs = brisc_unique_rtargs.size();
        uint32_t num_brisc_common_rtargs = brisc_common_rtargs.size();
        vector<uint32_t> brisc_compile_args = {
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_brisc_unique_rtargs,
            num_brisc_common_rtargs,
            page_size};

        // ncrisc
        uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
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
        vector<uint32_t> ncrisc_compile_args = {
            NCRISC_OUTER_LOOP,
            NCRISC_MIDDLE_LOOP,
            NCRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_ncrisc_unique_rtargs,
            num_ncrisc_common_rtargs,
            page_size};

        // trisc
        uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
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
        vector<uint32_t> trisc_compile_args = {
            TRISC_OUTER_LOOP,
            TRISC_MIDDLE_LOOP,
            TRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_trisc_unique_rtargs,
            num_trisc_common_rtargs,
            page_size};

        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = brisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program_, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_brisc_kernel, brisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = ncrisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program_, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                ComputeConfig{
                    .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
            SetRuntimeArgs(program_, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_trisc_kernel, trisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = (rand() % 3) + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = brisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program_, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_brisc_kernel, brisc_common_rtargs);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = ncrisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program_, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    ComputeConfig{
                        .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
                SetRuntimeArgs(program_, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_trisc_kernel, trisc_common_rtargs);
            } else {
                TT_THROW("Invalid");
            }
        }
    }

    for (uint8_t cq_id = 0; cq_id < device_->num_hw_cqs(); ++cq_id) {
        log_info(tt::LogTest, "Running {} MeshWorkloads on cq {} for cache warmup.", workloads.size(), (uint32_t)cq_id);
        // This loop caches program and runs
        for (distributed::MeshWorkload& wl : workloads) {
            distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), wl, false);
        }

        // This loops assumes already cached
        uint32_t NUM_ITERATIONS = 500;  // TODO(agrebenisan): Bump this to 5000, saw hangs for very large number of
                                        // iterations, need to come back to that

        log_info(
            tt::LogTest,
            "Running {} programs on cq {} for {} iterations now.",
            workloads.size(),
            (uint32_t)cq_id,
            NUM_ITERATIONS);
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            auto rng = std::default_random_engine{};
            std::shuffle(std::begin(workloads), std::end(workloads), rng);
            if (i % 10 == 0) {
                log_debug(
                    tt::LogTest,
                    "Enqueuing {} programs on cq {} for iter: {}/{} now.",
                    workloads.size(),
                    (uint32_t)cq_id,
                    i + 1,
                    NUM_ITERATIONS);
            }
            for (distributed::MeshWorkload& wl : workloads) {
                EnqueueMeshWorkload(device_->mesh_command_queue(), wl, false);
            }
        }

        log_info(tt::LogTest, "Calling Finish.");
        Finish(device_->mesh_command_queue(cq_id));
    }
}

TEST_F(UnitMeshCQFixture, DISABLED_TensixTestFillDispatchCoreBuffer) {
    uint32_t NUM_ITER = 100000;
    for (const auto& device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set(cr);

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};

        local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
            device, device->mesh_command_queue(), dummy_program_config, 256, 256, 256, NUM_ITER);
    }
}

TEST_F(UnitMeshCQProgramFixture, TensixTestRandomizedProgram) {
    uint32_t NUM_WORKLOADS = 100;
    uint32_t MAX_LOOP = 100;
    uint32_t page_size = 1024;

    // Make random
    auto random_seed = 0;  // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    auto device = this->devices_.at(0);

    CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_WORKLOADS);

    vector<distributed::MeshWorkload> workloads;
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        workloads.push_back(distributed::MeshWorkload());
        distributed::MeshWorkload& workload = workloads.back();

        Program program;
        workload.add_program(this->device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(this->device_range_);

        std::map<std::string, std::string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
        std::map<std::string, std::string> compute_defines = {{"COMPUTE", "1"}};

        // brisc
        uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
        bool USE_MAX_RT_ARGS;

        if (i % 10 == 0) {
            log_info(tt::LogTest, "Compiling program {} of {}", i + 1, NUM_WORKLOADS);
        }

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

        log_debug(
            tt::LogTest,
            "Compiling program {}/{} w/ BRISC_OUTER_LOOP: {} BRISC_MIDDLE_LOOP: {} BRISC_INNER_LOOP: {} NUM_CBS: {} "
            "NUM_SEMS: {} USE_MAX_RT_ARGS: {}",
            i + 1,
            NUM_WORKLOADS,
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            USE_MAX_RT_ARGS);

        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}})
                                                 .set_page_size(j, page_size * (j + 1));
            CreateCircularBuffer(program_, cr_set, cb_config);
        }

        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program_, cr_set, j + 1);
        }

        auto [brisc_unique_rtargs, brisc_common_rtargs] = create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_brisc_unique_rtargs = brisc_unique_rtargs.size();
        uint32_t num_brisc_common_rtargs = brisc_common_rtargs.size();
        vector<uint32_t> brisc_compile_args = {
            BRISC_OUTER_LOOP,
            BRISC_MIDDLE_LOOP,
            BRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_brisc_unique_rtargs,
            num_brisc_common_rtargs,
            page_size};
        // ncrisc
        uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
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
        vector<uint32_t> ncrisc_compile_args = {
            NCRISC_OUTER_LOOP,
            NCRISC_MIDDLE_LOOP,
            NCRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_ncrisc_unique_rtargs,
            num_ncrisc_common_rtargs,
            page_size};

        // trisc
        uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
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
        vector<uint32_t> trisc_compile_args = {
            TRISC_OUTER_LOOP,
            TRISC_MIDDLE_LOOP,
            TRISC_INNER_LOOP,
            NUM_CBS,
            NUM_SEMS,
            num_trisc_unique_rtargs,
            num_trisc_common_rtargs,
            page_size};

        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = brisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program_, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_brisc_kernel, brisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = ncrisc_compile_args,
                    .defines = data_movement_defines});
            SetRuntimeArgs(program_, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                cr_set,
                ComputeConfig{
                    .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
            SetRuntimeArgs(program_, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
            SetCommonRuntimeArgs(program_, dummy_trisc_kernel, trisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = (rand() % 3) + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = brisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program_, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_brisc_kernel, brisc_common_rtargs);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = ncrisc_compile_args,
                        .defines = data_movement_defines});
                SetRuntimeArgs(program_, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp",
                    cr_set,
                    ComputeConfig{
                        .math_approx_mode = false, .compile_args = trisc_compile_args, .defines = compute_defines});
                SetRuntimeArgs(program_, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
                SetCommonRuntimeArgs(program_, dummy_trisc_kernel, trisc_common_rtargs);
            } else {
                TT_THROW("Invalid");
            }
        }
    }

    log_info(tt::LogTest, "Running {} programs for cache warmup.", workloads.size());
    // This loop caches program and runs
    for (distributed::MeshWorkload& wl : workloads) {
        distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
    }

    // This loops assumes already cached
    uint32_t NUM_ITERATIONS = 500;  // TODO(agrebenisan): Bump this to 5000, saw hangs for very large number of
                                    // iterations, need to come back to that

    log_info(tt::LogTest, "Running {} programs for {} iterations now.", workloads.size(), NUM_ITERATIONS);
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(workloads), std::end(workloads), rng);
        if (i % 50 == 0) {
            log_info(
                tt::LogTest, "Enqueuing {} programs for iter: {}/{} now.", workloads.size(), i + 1, NUM_ITERATIONS);
        }
        for (distributed::MeshWorkload& wl : workloads) {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
        }
    }

    log_info(tt::LogTest, "Calling Finish.");
    Finish(device->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, TensixTestSimplePrograms) {
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);
        this->create_kernel(program_, CoreType::WORKER, true);
        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, TensixActiveEthTestSimplePrograms) {
    for (const auto& device : device_->get_devices()) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id()
                         << " does not have any active ethernet cores";
        }
    }

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            this->create_kernel(program_, CoreType::ETH, true);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            this->create_kernel(program_, CoreType::WORKER, true);
        }

        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, ActiveEthTestPrograms) {
    for (const auto& device : device_->get_devices()) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id()
                         << " does not have any active ethernet cores";
        }
    }

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program_, CoreType::ETH, true);
        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, TensixActiveEthTestPrograms) {
    for (const auto& device : device_->get_devices()) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id()
                         << " does not have any active ethernet cores";
        }
    }

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
            // and the max kernel size to ensure that the kernel can fit in the ring buffer
            KernelProperties kernel_properties;
            kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
            kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program_, CoreType::ETH, false, kernel_properties);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            KernelProperties kernel_properties;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program_, CoreType::WORKER, false, kernel_properties);
        }

        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, TensixTestAlternatingLargeAndSmallPrograms) {
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);

        KernelProperties kernel_properties;
        if (i % 2 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program_, CoreType::WORKER, false, kernel_properties);
        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, NIGHTLY_TensixTestLargeProgramFollowedBySmallPrograms) {
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        ;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);

        KernelProperties kernel_properties;
        if (i == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program_, CoreType::WORKER, false, kernel_properties);
        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

TEST_F(UnitMeshRandomProgramFixture, TensixTestLargeProgramInBetweenFiveSmallPrograms) {
    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range_, std::move(program));
        auto& program_ = workload.get_programs().at(device_range_);
        KernelProperties kernel_properties;
        if (i % 6 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program_, CoreType::WORKER, false, kernel_properties);
        distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, false);
    }

    Finish(device_->mesh_command_queue());
}

}  // namespace stress_tests

}  // namespace tt::tt_metal
