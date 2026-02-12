// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <map>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/base.h>
#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/experimental/host_api.hpp>

#include "device_fixture.hpp"
#include <umd/device/types/xy_pair.hpp>

// Access to internal API: ProgramImpl::num_kernel, get_kernel
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace unit_tests::runtime_args {

enum class KernelType {
    DATA_MOVEMENT = 0,
    COMPUTE = 1,
};

uint32_t get_runtime_arg_addr(
    uint32_t l1_unreserved_base, tt_metal::HalProcessorClassType processor_class, int processor_id, bool is_common) {
    uint32_t result_base = 0;

    // Spread results out a bit, overly generous
    constexpr uint32_t runtime_args_space = 1024 * sizeof(uint32_t);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t num_dm = hal.get_processor_types_count(
        HalProgrammableCoreType::TENSIX, ttsl::as_underlying_type(tt::tt_metal::HalProcessorClassType::DM));
    const uint32_t uncached_l1_offset = (hal.get_arch() == tt::ARCH::QUASAR)
                                            ? hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE)
                                            : 0;

    switch (processor_class) {
        case tt::tt_metal::HalProcessorClassType::DM:
            TT_FATAL(
                0 <= processor_id && processor_id < num_dm,
                "processor_id {} must be 0 to {} for DM",
                processor_id,
                num_dm - 1);
            result_base = l1_unreserved_base + processor_id * runtime_args_space;
            break;
        case tt::tt_metal::HalProcessorClassType::COMPUTE:
            result_base = l1_unreserved_base + num_dm * runtime_args_space;
            break;
        default: TT_THROW("Unknown processor");
    }

    // Common args go after all unique arg slots
    uint32_t total_processors = hal.get_num_risc_processors(HalProgrammableCoreType::TENSIX);
    uint32_t offset = is_common ? total_processors * runtime_args_space : 0;
    return (result_base + offset + uncached_l1_offset);
};

distributed::MeshWorkload initialize_program_data_movement(
    const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/, const CoreRangeSet& core_range_set) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    workload.add_program(device_range, std::move(program));
    return workload;
}

distributed::MeshWorkload initialize_program_data_movement_rta(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreRangeSet& core_range_set,
    uint32_t num_unique_rt_args,
    bool common_rtas = false) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t rta_base_dm = get_runtime_arg_addr(
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
        tt::tt_metal::HalProcessorClassType::DM,
        0,
        common_rtas);
    std::map<std::string, std::string> dm_defines = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_unique_rt_args)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm)}};
    if (common_rtas) {
        dm_defines["COMMON_RUNTIME_ARGS"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = dm_defines});

    workload.add_program(device_range, std::move(program));
    return workload;
}

// Quasar-specific helper - handles all Quasar DM kernel patterns
std::pair<distributed::MeshWorkload, std::vector<KernelHandle>> initialize_program_data_movement_rta_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreRangeSet& core_range_set,
    uint32_t num_runtime_args,
    bool common_rtas,
    uint32_t num_kernels,
    uint32_t dm_processors_per_kernel) {
    TT_ASSERT(MetalContext::instance().hal().get_arch() == tt::ARCH::QUASAR, "This helper is only for Quasar");

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = tt_metal::CreateProgram();

    std::vector<KernelHandle> kernel_handles(num_kernels);
    uint32_t rta_base = get_runtime_arg_addr(
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
        tt::tt_metal::HalProcessorClassType::DM,
        0,
        common_rtas);

    std::map<std::string, std::string> dm_defines = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args)},
        {"RESULTS_ADDR", std::to_string(rta_base)}};
    uint32_t max_dms = MetalContext::instance().hal().get_processor_types_count(
        HalProgrammableCoreType::TENSIX, ttsl::as_underlying_type(HalProcessorClassType::DM));
    dm_defines["MAX_DMS"] = std::to_string(max_dms);
    if (common_rtas) {
        dm_defines["COMMON_RUNTIME_ARGS"] = "1";
    }

    for (uint32_t k = 0; k < num_kernels; k++) {
        kernel_handles[k] = tt::tt_metal::experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
            core_range_set,
            tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = dm_processors_per_kernel, .defines = dm_defines});
    }

    workload.add_program(device_range, std::move(program));
    return {std::move(workload), kernel_handles};
}

tt::tt_metal::KernelHandle initialize_program_compute(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_range_set,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args) {
    // Tell kernel how many unique and common RT args to expect. Will increment each.
    uint32_t rta_base_compute = get_runtime_arg_addr(
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
        tt::tt_metal::HalProcessorClassType::COMPUTE,
        0,
        false);
    uint32_t common_rta_base_compute = get_runtime_arg_addr(
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
        tt::tt_metal::HalProcessorClassType::COMPUTE,
        0,
        true);
    std::vector<uint32_t> compile_args = {
        num_unique_rt_args, num_common_rt_args, rta_base_compute, common_rta_base_compute};
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/increment_runtime_arg.cpp",
        core_range_set,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compile_args});

    return compute_kernel_id;
}

std::pair<distributed::MeshWorkload, tt::tt_metal::KernelHandle> initialize_program_compute(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreRangeSet& core_range_set,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = tt_metal::CreateProgram();

    auto kernel_id =
        initialize_program_compute(mesh_device, program, core_range_set, num_unique_rt_args, num_common_rt_args);
    workload.add_program(device_range, std::move(program));
    return {std::move(workload), kernel_id};
}

std::pair<distributed::MeshWorkload, std::vector<tt::tt_metal::KernelHandle>>
initialize_program_compute_multi_range_sets(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<CoreRangeSet>& core_range_sets,
    uint32_t num_unique_rt_args,
    uint32_t num_common_rt_args) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = tt_metal::CreateProgram();
    std::vector<tt::tt_metal::KernelHandle> kernel_ids;

    kernel_ids.reserve(core_range_sets.size());
    for (const auto& core_range_set : core_range_sets) {
        kernel_ids.push_back(
            initialize_program_compute(mesh_device, program, core_range_set, num_unique_rt_args, num_common_rt_args));
    }
    workload.add_program(device_range, std::move(program));
    return {std::move(workload), kernel_ids};
}

// Verify the runtime args for a single core (apply optional non-zero increment amounts to values written to match
// compute kernel)
void verify_core_rt_args(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool is_common,
    CoreCoord core,
    uint32_t base_addr,
    const std::vector<uint32_t>& written_args,
    const uint32_t incr_val) {
    std::vector<uint32_t> observed_args;
    auto* device = mesh_device->get_devices()[0];
    tt_metal::detail::ReadFromDeviceL1(device, core, base_addr, written_args.size() * sizeof(uint32_t), observed_args);

    for (size_t i = 0; i < written_args.size(); i++) {
        uint32_t expected_result = written_args.at(i) + incr_val;
        log_debug(
            tt::LogTest,
            "Validating {} Args. Core: {} at addr: 0x{:x} idx: {} - Expected: {:#x} Observed: {:#x}",
            is_common ? "Common" : "Unique",
            core.str(),
            base_addr,
            i,
            expected_result,
            observed_args[i]);
        EXPECT_EQ(observed_args.at(i), expected_result) << (is_common ? "(common rta)" : "(unique rta)");
    }
}

// Iterate over all cores unique and common runtime args, and verify they match expected values.
void verify_results(
    bool are_args_incremented,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const distributed::MeshWorkload& workload,
    const std::map<CoreCoord, std::vector<uint32_t>>& core_to_rt_args,
    const std::vector<uint32_t>& common_rt_args = {}) {
    // These increment amounts model what is done by compute kernel in this test.
    uint32_t unique_arg_incr_val = are_args_incremented ? 10 : 0;
    uint32_t common_arg_incr_val = are_args_incremented ? 100 : 0;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    const auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        const auto kernel = program.impl().get_kernel(kernel_id);
        auto rt_args_base_addr = get_runtime_arg_addr(
            device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
            kernel->get_kernel_processor_class(),
            kernel->get_kernel_processor_type(0),
            false);

        // Verify Unique RT Args (per core)
        for (const auto& logical_core : kernel->cores_with_runtime_args()) {
            const auto& expected_rt_args = core_to_rt_args.at(logical_core);
            auto rt_args = kernel->runtime_args(logical_core);
            EXPECT_EQ(rt_args, expected_rt_args) << "(unique rta)";

            verify_core_rt_args(
                mesh_device, false, logical_core, rt_args_base_addr, expected_rt_args, unique_arg_incr_val);
        }

        // Verify common RT Args (same for all cores) if they exist.
        if (!common_rt_args.empty()) {
            auto common_rt_args_base_addr = get_runtime_arg_addr(
                device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1),
                kernel->get_kernel_processor_class(),
                kernel->get_kernel_processor_type(0),
                true);

            for (auto& core_range : kernel->logical_coreranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core({x, y});
                        auto rt_args = kernel->common_runtime_args();
                        EXPECT_EQ(rt_args, common_rt_args) << "(common rta)";
                        verify_core_rt_args(
                            mesh_device,
                            true,
                            logical_core,
                            common_rt_args_base_addr,
                            common_rt_args,
                            common_arg_incr_val);
                    }
                }
            }
        }
    }
}

void verify_quasar_crtas(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const std::vector<std::vector<uint32_t>>& per_kernel_crtas,
    bool expect_shared_address) {
    auto* device = mesh_device->get_devices()[0];
    uint32_t l1_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint32_t results_base = get_runtime_arg_addr(l1_base, tt::tt_metal::HalProcessorClassType::DM, 0, true);
    uint32_t max_dms = MetalContext::instance().hal().get_processor_types_count(
        HalProgrammableCoreType::TENSIX, ttsl::as_underlying_type(HalProcessorClassType::DM));

    // Kernel writes to memory using hartid as index, so verification must cover all DMs
    TT_ASSERT(
        per_kernel_crtas.size() == max_dms,
        "per_kernel_crtas.size() ({}) must equal max_dms ({})",
        per_kernel_crtas.size(),
        max_dms);

    constexpr uint32_t kCommonRTASeparation = 1024;
    // For this test, all CRTAs in all kernels are the same size
    const uint32_t num_crtas = per_kernel_crtas[0].size();
    std::vector<uint32_t> crta_addrs;

    for (uint32_t dm_id = 0; dm_id < max_dms; dm_id++) {
        const auto& expected_crtas = per_kernel_crtas[dm_id];
        uint32_t dm_crta_addr = results_base + ((kCommonRTASeparation + dm_id * num_crtas) * sizeof(uint32_t));

        std::vector<uint32_t> observed;
        tt_metal::detail::ReadFromDeviceL1(device, core, dm_crta_addr, num_crtas * sizeof(uint32_t), observed);

        for (size_t i = 0; i < num_crtas; i++) {
            EXPECT_EQ(observed[i], expected_crtas[i]) << "DM" << dm_id << " CRTA[" << i << "]";
        }
    }

    // Address slot starts right after CRTA values
    uint32_t addr_base = results_base + ((kCommonRTASeparation + max_dms * num_crtas) * sizeof(uint32_t));
    for (uint32_t dm_id = 0; dm_id < max_dms; dm_id++) {
        uint32_t addr_offset = addr_base + (dm_id * sizeof(uint32_t));
        std::vector<uint32_t> addr;
        tt_metal::detail::ReadFromDeviceL1(device, core, addr_offset, sizeof(uint32_t), addr);
        crta_addrs.push_back(addr[0]);
    }

    if (expect_shared_address) {
        // All DMs should have same CRTA address
        for (size_t i = 1; i < crta_addrs.size(); i++) {
            EXPECT_EQ(crta_addrs[0], crta_addrs[i]) << "All DMs should share same CRTA L1 address";
        }
    } else {
        // All DMs should have different CRTA addresses
        for (size_t i = 0; i < crta_addrs.size(); i++) {
            for (size_t j = i + 1; j < crta_addrs.size(); j++) {
                EXPECT_NE(crta_addrs[i], crta_addrs[j])
                    << "DM" << i << " and DM" << j << " have same CRTA address: 0x" << std::hex << crta_addrs[i];
            }
        }
    }
}

}  // namespace unit_tests::runtime_args

namespace tt::tt_metal {

// Write unique and common runtime args to device and readback to verify written correctly.
TEST_F(MeshDeviceFixture, TensixLegallyModifyRTArgsDataMovement) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto* device = this->devices_.at(id)->get_devices()[0];
        auto workload =
            unit_tests::runtime_args::initialize_program_data_movement_rta(this->devices_.at(id), core_range_set, 2);
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        auto& program = workload.get_programs().at(device_range);
        ASSERT_TRUE(program.impl().num_kernels() == 1);
        std::vector<uint32_t> initial_runtime_args = {0xfeadbeef, 0xabababab};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        detail::WriteRuntimeArgsToDevice(device, program);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(false, mesh_device, workload, core_to_rt_args);

        std::vector<uint32_t> second_runtime_args = {0x12341234, 0xcafecafe};
        SetRuntimeArgs(program, 0, first_core_range, second_runtime_args);
        detail::WriteRuntimeArgsToDevice(device, program);
        for (auto x = first_core_range.start_coord.x; x <= first_core_range.end_coord.x; x++) {
            for (auto y = first_core_range.start_coord.y; y <= first_core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = second_runtime_args;
            }
        }
        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(false, mesh_device, workload, core_to_rt_args);

        auto workload2 =
            unit_tests::runtime_args::initialize_program_data_movement_rta(mesh_device, core_range_set, 4, true);
        auto& program2 = workload2.get_programs().at(device_range);
        // Set common runtime args, automatically sent to all cores used by kernel.
        std::vector<uint32_t> common_runtime_args = {0x30303030, 0x60606060, 0x90909090, 1234};
        SetCommonRuntimeArgs(program2, 0, common_runtime_args);
        detail::WriteRuntimeArgsToDevice(device, program2);
        distributed::EnqueueMeshWorkload(cq, workload2, false);
        unit_tests::runtime_args::verify_results(false, mesh_device, workload2, core_to_rt_args, common_runtime_args);
    }
}

TEST_F(MeshDeviceFixture, TensixLegallyModifyRTArgsCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> initial_runtime_args = {0xfeadbeee, 0xabababab};
        std::vector<uint32_t> common_runtime_args = {0x11001100, 0x22002200, 0x33003300, 0x44004400};
        auto [workload, kernel] = unit_tests::runtime_args::initialize_program_compute(
            mesh_device, core_range_set, initial_runtime_args.size(), common_runtime_args.size());

        auto& program = workload.get_programs().at(device_range);
        SetRuntimeArgs(program, kernel, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, kernel, common_runtime_args);

        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(true, mesh_device, workload, core_to_rt_args, common_runtime_args);
    }
}

// Don't cover all cores of kernel with SetRuntimeArgs. Verify that correct offset used to access common runtime args.
TEST_F(MeshDeviceFixture, TensixSetRuntimeArgsSubsetOfCoresCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> initial_runtime_args = {0xfeadbeee, 0xabababab};
        std::vector<uint32_t> common_runtime_args = {0x11001100, 0x22002200, 0x33003300, 0x44004400};

        auto [workload, kernel] = unit_tests::runtime_args::initialize_program_compute(
            mesh_device, core_range_set, initial_runtime_args.size(), common_runtime_args.size());
        auto& program = workload.get_programs().at(device_range);
        SetRuntimeArgs(program, kernel, first_core_range, initial_runtime_args);  // First core range only.

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto x = first_core_range.start_coord.x; x <= first_core_range.end_coord.x; x++) {
            for (auto y = first_core_range.start_coord.y; y <= first_core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = initial_runtime_args;
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, kernel, common_runtime_args);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(true, mesh_device, workload, core_to_rt_args, common_runtime_args);
    }
}

// Different unique runtime args per core. Not overly special, but verify that it works.
TEST_F(MeshDeviceFixture, TensixSetRuntimeArgsUniqueValuesCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> common_runtime_args = {0x11001100, 0x22002200, 0x33003300, 0x44004400};
        auto [workload, kernel] = unit_tests::runtime_args::initialize_program_compute(
            mesh_device, core_range_set, 2, common_runtime_args.size());
        auto& program = workload.get_programs().at(device_range);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    // Generate an rt arg val based on x and y.
                    uint32_t val_offset = (x * 100) + (y * 10);
                    std::vector<uint32_t> initial_runtime_args = {101 + val_offset, 202 + val_offset};
                    SetRuntimeArgs(program, kernel, logical_core, initial_runtime_args);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, kernel, common_runtime_args);

        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(true, mesh_device, workload, core_to_rt_args, common_runtime_args);
    }
}

// Some cores have more unique runtime args than others. Unused in kernel, but API supports it, so verify it works and
// that common runtime args are appropriately offset by amount from core(s) with most unique runtime args.
TEST_F(MeshDeviceFixture, TensixSetRuntimeArgsVaryingLengthPerCore) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> common_runtime_args = {0x11001100, 0x22002200, 0x33003300, 0x44004400};

        // Figure out max number of unique runtime args across all cores, so kernel
        // can attempt to increment that many unique rt args per core. Kernels
        // with fewer will just increment unused memory, no big deal.
        uint32_t max_unique_rt_args = 0;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    uint32_t num_rt_args = 2 + x + y;
                    max_unique_rt_args = std::max(max_unique_rt_args, num_rt_args);
                }
            }
        }

        auto [workload, kernel] = unit_tests::runtime_args::initialize_program_compute(
            mesh_device, core_range_set, max_unique_rt_args, common_runtime_args.size());
        auto& program = workload.get_programs().at(device_range);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    // Generate rt args length and val based on x,y arbitrarily.
                    uint32_t val_offset = (x * 100) + (y * 10);
                    uint32_t num_rt_args = 2 + x + y;
                    std::vector<uint32_t> initial_runtime_args;
                    initial_runtime_args.reserve(num_rt_args);
                    for (uint32_t i = 0; i < num_rt_args; i++) {
                        initial_runtime_args.push_back(101 + val_offset + (i * 66));
                    }
                    SetRuntimeArgs(program, kernel, logical_core, initial_runtime_args);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, 0, common_runtime_args);

        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(true, mesh_device, workload, core_to_rt_args, common_runtime_args);
    }
}

// Too many unique and common runtime args, overflows allowed space and throws expected exception from both
// unique/common APIs.
TEST_F(MeshDeviceFixture, TensixIllegalTooManyRuntimeArgs) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        CoreRange first_core_range(CoreCoord(1, 1), CoreCoord(2, 2));
        CoreRangeSet core_range_set(first_core_range);
        auto [workload, kernel] = unit_tests::runtime_args::initialize_program_compute(
            mesh_device, core_range_set, 0, 0);  // Kernel isn't run here.
        auto& program = workload.get_programs().at(device_range);

        // Set 100 unique args, then try to set max_runtime_args + 1 common args and fail.
        std::vector<uint32_t> initial_runtime_args(100);
        SetRuntimeArgs(program, kernel, core_range_set, initial_runtime_args);
        std::vector<uint32_t> common_runtime_args(tt::tt_metal::max_runtime_args + 1);
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program, 0, common_runtime_args));

        // Set 100 common args, then try to set another tt::tt_metal::max_runtime_args + 1 unique args and fail.
        std::vector<uint32_t> more_common_runtime_args(100);
        SetCommonRuntimeArgs(program, kernel, more_common_runtime_args);
        std::vector<uint32_t> more_unique_args(tt::tt_metal::max_runtime_args + 1);
        EXPECT_ANY_THROW(SetRuntimeArgs(program, 0, core_range_set, more_unique_args));
    }
}

TEST_F(MeshDeviceFixture, TensixIllegallyModifyRTArgs) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        auto* device = mesh_device->get_devices()[0];
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        auto workload = unit_tests::runtime_args::initialize_program_data_movement_rta(mesh_device, core_range_set, 2);
        auto& program = workload.get_programs().at(device_range);
        ASSERT_TRUE(program.impl().num_kernels() == 1);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        detail::WriteRuntimeArgsToDevice(device, program);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(false, mesh_device, workload, core_to_rt_args);

        std::vector<uint32_t> invalid_runtime_args = {303, 404, 505};
        EXPECT_ANY_THROW(SetRuntimeArgs(program, 0, first_core_range, invalid_runtime_args));

        // Cannot modify number of common runtime args either.
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};
        SetCommonRuntimeArgs(program, 0, common_runtime_args);
        std::vector<uint32_t> illegal_common_runtime_args = {0, 1, 2, 3, 4, 5};
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program, 0, illegal_common_runtime_args));
    }
}

TEST_F(MeshDeviceFixture, TensixSetCommonRuntimeArgsMultipleCreateKernel) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        auto grid_size = this->devices_.at(id)->logical_grid_size();
        auto max_x = grid_size.x - 1;
        auto max_y = grid_size.y - 1;

        // Split into 4 quads
        // Slow dispatch test. All coords are available for use
        CoreRange core_range_0(CoreCoord(0, 0), CoreCoord(max_x / 2, max_y / 2));
        CoreRange core_range_1(CoreCoord((max_x / 2) + 1, 0), CoreCoord(max_x, max_y / 2));
        CoreRange core_range_2(CoreCoord(0, (max_y / 2) + 1), CoreCoord(max_x / 2, max_y));
        CoreRange core_range_3(CoreCoord((max_x / 2) + 1, (max_y / 2) + 1), CoreCoord(max_x, max_y));

        CoreRangeSet core_range_set_0(std::vector{core_range_0, core_range_1});
        CoreRangeSet core_range_set_1(std::vector{core_range_2, core_range_3});

        std::vector<uint32_t> common_rtas{0xdeadbeef, 0xabcd1234, 101};

        auto [workload, kernels] = unit_tests::runtime_args::initialize_program_compute_multi_range_sets(
            mesh_device, {core_range_set_0, core_range_set_1}, 0, common_rtas.size());
        auto& program = workload.get_programs().at(device_range);
        for (const auto& kernel : kernels) {
            SetCommonRuntimeArgs(program, kernel, common_rtas);
        }
        distributed::EnqueueMeshWorkload(cq, workload, false);
        unit_tests::runtime_args::verify_results(true, mesh_device, workload, {}, common_rtas);
    }
}

// Test that active ethernet cores correctly validate max runtime args
TEST_F(MeshDeviceFixture, ActiveEthIllegalTooManyRuntimeArgs) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t active_eth_max_runtime_args =
        hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::KERNEL_CONFIG) / sizeof(uint32_t);
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto* device = mesh_device->get_devices()[0];
        auto active_eth_cores = device->get_active_ethernet_cores(true);

        // Skip test if no active ethernet cores available
        if (active_eth_cores.empty()) {
            log_info(LogTest, "Skipping ActiveEthIllegalTooManyRuntimeArgs test - no active ethernet cores available");
            continue;
        }

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt::tt_metal::Program program = tt_metal::CreateProgram();

        // Create kernel on first active ethernet core
        CoreCoord eth_core = *active_eth_cores.begin();
        CoreRange eth_core_range(eth_core, eth_core);

        auto kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});

        workload.add_program(device_range, std::move(program));
        auto& program_ref = workload.get_programs().at(device_range);

        // Verify that setting exactly max common args works when no unique args are set (should not throw)
        // Note: We test this FIRST because even failed SetRuntimeArgs calls can pollute max_runtime_args_per_core_
        std::vector<uint32_t> max_common_args(active_eth_max_runtime_args);
        EXPECT_NO_THROW(SetCommonRuntimeArgs(program_ref, kernel, max_common_args));

        // Try to set too many unique runtime args (should fail)
        // Note: This must come after testing max common args because it pollutes the kernel state
        std::vector<uint32_t> too_many_args(active_eth_max_runtime_args + 1);
        EXPECT_ANY_THROW(SetRuntimeArgs(program_ref, kernel, eth_core, too_many_args));

        // Try to set too many common runtime args (should fail)
        // Create a new kernel for this test since common args can only be set once
        tt::tt_metal::Program program_common_test = tt_metal::CreateProgram();
        auto kernel_common_test = tt_metal::CreateKernel(
            program_common_test,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});

        auto device_range_common_test = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload_common_test;
        workload_common_test.add_program(device_range_common_test, std::move(program_common_test));
        auto& program_common_test_ref = workload_common_test.get_programs().at(device_range_common_test);

        std::vector<uint32_t> too_many_common_args(active_eth_max_runtime_args + 1);
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program_common_test_ref, kernel_common_test, too_many_common_args));

        // Verify that setting exactly max active eth unique runtime args works (should not throw)
        // However, we can't do this now because common args are already set to max, and unique+common must <= max
        // So we create a new program/kernel for this test
        tt::tt_metal::Program program2 = tt_metal::CreateProgram();
        auto kernel2 = tt_metal::CreateKernel(
            program2,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});

        auto device_range2 = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload2;
        workload2.add_program(device_range2, std::move(program2));
        auto& program2_ref = workload2.get_programs().at(device_range2);

        std::vector<uint32_t> max_unique_args(active_eth_max_runtime_args);
        EXPECT_NO_THROW(SetRuntimeArgs(program2_ref, kernel2, eth_core, max_unique_args));
    }
}

// Test that idle ethernet cores correctly validate max runtime args using IDLE_ETH kernel config size
TEST_F(MeshDeviceFixture, IdleEthIllegalTooManyRuntimeArgs) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t idle_eth_max_runtime_args =
        hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::KERNEL_CONFIG) / sizeof(uint32_t);
    for (unsigned int id = 0; id < num_devices_; id++) {
        auto mesh_device = this->devices_.at(id);
        auto* device = mesh_device->get_devices()[0];
        auto idle_eth_cores = device->get_inactive_ethernet_cores();

        // Skip test if no idle ethernet cores available
        if (idle_eth_cores.empty()) {
            log_info(LogTest, "Skipping IdleEthIllegalTooManyRuntimeArgs test - no idle ethernet cores available");
            continue;
        }

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt::tt_metal::Program program = tt_metal::CreateProgram();

        // Create kernel on first idle ethernet core
        CoreCoord eth_core = *idle_eth_cores.begin();
        CoreRange eth_core_range(eth_core, eth_core);

        auto kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});

        workload.add_program(device_range, std::move(program));
        auto& program_ref = workload.get_programs().at(device_range);

        // Verify that setting exactly max common args works when no unique args are set (should not throw)
        // Note: We test this FIRST because even failed SetRuntimeArgs calls can pollute max_runtime_args_per_core_
        std::vector<uint32_t> max_common_args(idle_eth_max_runtime_args);
        EXPECT_NO_THROW(SetCommonRuntimeArgs(program_ref, kernel, max_common_args));

        // Try to set too many unique runtime args (should fail)
        // Note: This must come after testing max common args because it pollutes the kernel state
        std::vector<uint32_t> too_many_args(idle_eth_max_runtime_args + 1);
        EXPECT_ANY_THROW(SetRuntimeArgs(program_ref, kernel, eth_core, too_many_args));

        // Try to set too many common runtime args (should fail)
        // Create a new kernel for this test since common args can only be set once
        tt::tt_metal::Program program_common_test = tt_metal::CreateProgram();
        auto kernel_common_test = tt_metal::CreateKernel(
            program_common_test,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});

        auto device_range_common_test = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload_common_test;
        workload_common_test.add_program(device_range_common_test, std::move(program_common_test));
        auto& program_common_test_ref = workload_common_test.get_programs().at(device_range_common_test);

        std::vector<uint32_t> too_many_common_args(idle_eth_max_runtime_args + 1);
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program_common_test_ref, kernel_common_test, too_many_common_args));

        // Verify that setting exactly max idle eth unique runtime args works (should not throw)
        // However, we can't do this now because common args are already set to max, and unique+common must <= max
        // So we create a new program/kernel for this test
        tt::tt_metal::Program program2 = tt_metal::CreateProgram();
        auto kernel2 = tt_metal::CreateKernel(
            program2,
            "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
            eth_core_range,
            tt_metal::EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});

        auto device_range2 = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload2;
        workload2.add_program(device_range2, std::move(program2));
        auto& program2_ref = workload2.get_programs().at(device_range2);

        std::vector<uint32_t> max_unique_args(idle_eth_max_runtime_args);
        EXPECT_NO_THROW(SetRuntimeArgs(program2_ref, kernel2, eth_core, max_unique_args));
    }
}

// Quasar only test: Single kernel running on all 8 DM processors with shared CRTAs.
// Verifies all DMs receive identical CRTA values at the same L1 address
// TODO: Once SW supports multiple quasar clusters/cores, expand to multiple CoreRangeSets
// to verify CRTA dispatch across different Kernel groups
TEST_F(MeshDeviceSingleCardFixture, QuasarCRTASharedL1Address) {
    if (arch_ != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "This test is only meant for Quasar";
    }

    auto mesh_device = devices_[0];
    constexpr CoreCoord core = {0, 0};
    CoreRange core_range(core);
    CoreRangeSet core_range_set(std::vector{core_range});

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    uint32_t num_dm = MetalContext::instance().hal().get_processor_types_count(
        HalProgrammableCoreType::TENSIX, ttsl::as_underlying_type(tt::tt_metal::HalProcessorClassType::DM));
    std::vector<uint32_t> common_rtas{0xdeadbeef, 0xabcd1234, 0x101};
    // Single kernel on all DMs: all DMs should receive identical CRTAs
    std::vector<std::vector<uint32_t>> all_crtas(num_dm, common_rtas);
    auto [workload, handles] = unit_tests::runtime_args::initialize_program_data_movement_rta_quasar(
        mesh_device, core_range_set, common_rtas.size(), true, /*num_kernels*/ 1, num_dm);
    auto& program = workload.get_programs().at(device_range);
    SetCommonRuntimeArgs(program, handles[0], common_rtas);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    // Verify all DMs share the same CRTA L1 address
    unit_tests::runtime_args::verify_quasar_crtas(mesh_device, core, all_crtas, /*expect_shared_address*/ true);
}

// Quasar only test: 8 separate kernels, each running on a unique DM processor, with unique CRTAs.
// Verifies each kernel receives its own distinct CRTA values at different L1 addresses
// TODO: Once SW supports multiple quasar clusters/cores, expand to multiple CoreRangeSets
// to verify CRTA dispatch across different Kernel groups
TEST_F(MeshDeviceSingleCardFixture, QuasarCRTAUniqueL1Addresses) {
    if (arch_ != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "This test is only meant for Quasar";
    }

    auto mesh_device = devices_[0];
    constexpr CoreCoord core = {0, 0};
    CoreRange core_range(core);
    CoreRangeSet core_range_set(std::vector{core_range});

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> base_crtas = {0x101, 0x202, 0x303};
    // One kernel per DM: each kernel gets its own CRTA offset
    uint32_t num_kernels = MetalContext::instance().hal().get_processor_types_count(
        HalProgrammableCoreType::TENSIX, ttsl::as_underlying_type(tt::tt_metal::HalProcessorClassType::DM));
    auto [workload, handles] = unit_tests::runtime_args::initialize_program_data_movement_rta_quasar(
        mesh_device, core_range_set, base_crtas.size(), true, num_kernels, /*dm_processors_per_kernel*/ 1);
    auto& program = workload.get_programs().at(device_range);
    std::vector<std::vector<uint32_t>> all_crtas(num_kernels);
    for (uint32_t i = 0; i < num_kernels; i++) {
        std::vector<uint32_t> kernel_crtas = {base_crtas[0] + i, base_crtas[1] + i, base_crtas[2] + i};
        all_crtas[i] = kernel_crtas;
        SetCommonRuntimeArgs(program, handles[i], kernel_crtas);
    }
    distributed::EnqueueMeshWorkload(cq, workload, false);
    // Verify each kernel has a unique CRTA L1 address
    unit_tests::runtime_args::verify_quasar_crtas(mesh_device, core, all_crtas, /*expect_shared_address*/ false);
}

}  // namespace tt::tt_metal
