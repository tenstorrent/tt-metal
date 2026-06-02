// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "debug_tools_fixture.hpp"

namespace tt::tt_metal {

// Fixture for RTA/CRTA watcher bounds check tests
class RTATestFixture : public MeshWatcherFixture {
protected:
    const std::string rta_crta_kernel_path =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp";
    bool is_quasar{false};
    uint32_t l1_unreserved_base{0};
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* device{nullptr};
    uint32_t num_dms_{0};

    void SetUp() override {
        bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
        if (watcher_assert_disabled) {
            GTEST_SKIP() << "This test requires watcher assert checks (RTA/CRTA) to be enabled";
        }
        MeshWatcherFixture::SetUp();
        mesh_device = devices_[0];
        device = mesh_device->get_devices()[0];
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        // On Quasar, DM0/DM1 are reserved for internal use; user kernels can only land on DM2..DM7.
        if (is_quasar) {
            constexpr uint32_t kQuasarReservedDmCores = 2;
            num_dms_ -= kQuasarReservedDmCores;
        }
        l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    }
    // Common test data
    const std::vector<uint32_t> default_rtas{0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> default_crtas{0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};

    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

    // Helper: Read and validate RTA/CRTA results from L1
    void ValidateArgResults(
        const CoreCoord& core,
        const std::vector<uint32_t>& expected_rtas,
        const std::vector<uint32_t>& expected_crtas) {
        const uint32_t total_size = (2 + expected_rtas.size() + expected_crtas.size()) * sizeof(uint32_t);
        std::vector<uint32_t> read_result;

        tt::tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, total_size, read_result);

        // Validate counts
        EXPECT_EQ(read_result[0], expected_rtas.size());
        EXPECT_EQ(read_result[1], expected_crtas.size());

        // Validate RTA payload
        for (uint32_t i = 0; i < expected_rtas.size(); i++) {
            EXPECT_EQ(read_result[i + 2], expected_rtas[i]);
        }

        // Validate CRTA payload
        for (uint32_t i = 0; i < expected_crtas.size(); i++) {
            EXPECT_EQ(read_result[i + expected_rtas.size() + 2], expected_crtas[i]);
        }
    }

    // Helper: Wait for watcher exception with specific message
    void ExpectWatcherException(
        const std::string& expected_message, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::string exception;
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < timeout) {
            exception = MetalContext::instance().watcher_server()->exception_message();
            if (!exception.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_FALSE(exception.empty()) << "Timeout (" << timeout.count() << "ms) waiting for watcher exception.\n"
                                        << "Expected: " << expected_message;

        EXPECT_TRUE(exception.find(expected_message) != std::string::npos)
            << "Watcher exception mismatch:\n"
            << "  Expected substring: " << expected_message << "\n"
            << "  Actual exception:   " << exception;
    }
};

// Test parameters structure
struct RTAAssertTestParams {
    std::string test_name;                  // For readable test names
    bool test_rta;                          // true = test RTA, false = test CRTA
    std::string expected_message;           // Expected watcher exception message
    HalProcessorClassType processor_class;  // DM or COMPUTE
};

// Parameterized test fixture for RTA/CRTA out-of-bounds assertions
class RTAAssertTest : public RTATestFixture, public ::testing::WithParamInterface<RTAAssertTestParams> {};

// Verifies watcher handles 0xFFFF sentinel pattern (used when no RTAs set):
// - Device interprets 0xBEEF#### as rta_count = 0 (validated on DM0/TRISC0)
// - Kernels not accessing args run successfully with sentinel pattern
// - Kernels accessing args when count = 0 trigger "out of bounds" assert
// - Catches bug: kernel placed on cores but SetRuntimeArgs() only called for subset
TEST_F(RTATestFixture, SentinelPatternHandlingAndMissingRTADetection) {
    if (IsSlowDispatch() || is_quasar) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    // Part A: Zero-arg kernels should see rta_count = 0, crta_count = 0
    {
        CoreRange core_range(CoreCoord(0, 0), CoreCoord(2, 2));
        CoreRangeSet core_range_set(std::vector{core_range});

        const uint32_t total_read_size = 2 * sizeof(uint32_t);
        // Separate L1 space for TRISC0 writeback (DM writes 2 words, so offset by 8 bytes)
        uint32_t compute_scratch_addr = l1_unreserved_base + total_read_size;

        // Zero-init the L1 scratch space on all cores (both DM and compute regions)
        std::vector<uint32_t> zero_init(4, 0);  // 2 words for DM + 2 words for TRISC0
        for (const auto& core : core_range) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
            tt::tt_metal::detail::WriteToDeviceL1(device, core, compute_scratch_addr, zero_init);
        }

        distributed::MeshWorkload workload;
        Program program;

        CreateKernel(
            program,
            rta_crta_kernel_path,
            core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {l1_unreserved_base}});

        CreateKernel(
            program, rta_crta_kernel_path, core_range_set, ComputeConfig{.compile_args = {compute_scratch_addr}});

        // No SetRuntimeArgs called - all cores have RTA/CRTA offset = 0xFFFF pattern
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);

        std::vector<uint32_t> read_result;

        for (const auto& core : core_range) {
            // DM: verify rta_count = 0, crta_count = 0
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, total_read_size, read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count

            // TRISC0: verify rta_count = 0, crta_count = 0
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, core, compute_scratch_addr, total_read_size, read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count
        }
    }

    // Part B: Accessing unset RTAs on a subset of cores should trigger assert
    {
        CoreRange cores_with_rtas(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange cores_without_rtas(CoreCoord(2, 2), CoreCoord(3, 3));
        CoreRangeSet core_range_set(std::vector{cores_with_rtas, cores_without_rtas});

        distributed::MeshWorkload workload;
        Program program;

        auto kernel = CreateKernel(
            program,
            rta_crta_kernel_path,
            core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {{"MAX_RTA_IDX", "0"}}});  // Kernel will try to read RTA[0]

        SetRuntimeArgs(program, kernel, cores_with_rtas, default_rtas);
        // cores_without_rtas has NO RTAs set - 0xBEEF#### pattern -> rta_count = 0 -> assert

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);
        ExpectWatcherException("unique runtime arg index out of bounds");
    }
}

// Correct RTA/CRTA Dispatch and Payload Validation
// Validates happy path: RTAs/CRTAs are correctly dispatched and read back
// Tests multiple core ranges with different RTA sizes, common RTAs shared across
// cores, and re-dispatching on subsequent runs. Ensures arg counts and payload
// values match what was set via SetRuntimeArgs/SetCommonRuntimeArgs
TEST_F(RTATestFixture, CorrectArgDispatchAndPayloadValidation) {
    // Quasar: single core; other archs: multi-core with two ranges
    CoreRange core_range1 =
        is_quasar ? CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)) : CoreRange(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange core_range2(CoreCoord(2, 2), CoreCoord(3, 3));
    CoreRangeSet core_range_set =
        is_quasar ? CoreRangeSet(std::vector{core_range1}) : CoreRangeSet(std::vector{core_range1, core_range2});

    // Use l1_unreserved space for RTA/CRTA write back to L1 for verification of correct dispatch payload
    const uint32_t total_word_count = 2 + default_rtas.size() + default_crtas.size();

    // Zero-init the L1 scratch space on all cores before running
    std::vector<uint32_t> zero_init(total_word_count, 0);
    for (const auto& core : core_range1) {
        tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
        }
    }

    distributed::MeshWorkload workload;
    Program program;
    constexpr const char* DM_KERNEL_NAME = "rta_dm";

    KernelHandle kernel = 0;
    std::vector<uint32_t> rtas_range2 = {0x1000, 0x1001, 0x1002};

    auto build_metal2_program = [&]() {
        experimental::metal2_host_api::KernelSpec dm_spec{
            .unique_id = DM_KERNEL_NAME,
            .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{rta_crta_kernel_path},
            .num_threads = static_cast<uint8_t>(num_dms_),
            .compile_time_arg_bindings = {{"dm_id", 0}, {"l1_scratch_addr", l1_unreserved_base}},
            .runtime_arguments_schema =
                {.num_runtime_varargs = default_rtas.size(), .num_common_runtime_varargs = default_crtas.size()},
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        };
        experimental::metal2_host_api::WorkUnitSpec wu{
            .unique_id = "main",
            .kernels = {DM_KERNEL_NAME},
            .target_nodes = experimental::metal2_host_api::NodeRangeSet{core_range_set},
        };
        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "rta_validation",
            .kernels = {dm_spec},
            .work_units = {wu},
        };
        program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);
    };

    // Helper to set runtime arguments (per-node varargs + common varargs) on the active program
    auto set_metal2_runtime_args = [&](Program& prog) {
        experimental::metal2_host_api::ProgramRunParams params;
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams krp{
            .kernel_spec_name = DM_KERNEL_NAME,
            .common_runtime_varargs = default_crtas,
        };
        for (const auto& c : core_range1) {
            krp.runtime_varargs.push_back({experimental::metal2_host_api::NodeCoord{c}, default_rtas});
        }
        params.kernel_run_params = {krp};
        experimental::metal2_host_api::SetProgramRunParameters(prog, params);
    };

    if (is_quasar) {
        build_metal2_program();
        set_metal2_runtime_args(program);
        workload.add_program(device_range, std::move(program));
    } else {
        // Compile args: [l1_scratch_addr]
        kernel = CreateKernel(
            program,
            rta_crta_kernel_path,
            core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {l1_unreserved_base}});
        // Range 1: 5 RTAs per core
        SetRuntimeArgs(program, kernel, core_range1, default_rtas);
        // Range 2: 3 RTAs per core (different size than core_range1)
        SetRuntimeArgs(program, kernel, core_range2, rtas_range2);
        // Common args
        SetCommonRuntimeArgs(program, kernel, default_crtas);
        workload.add_program(device_range, std::move(program));
    }

    RunProgram(mesh_device, workload);

    // Validate first run
    for (const auto& core : core_range1) {
        ValidateArgResults(core, default_rtas, default_crtas);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            ValidateArgResults(core, rtas_range2, default_crtas);
        }
    }

    // Zero-init again before second run
    for (const auto& core : core_range1) {
        tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
        }
    }

    // Second run: re-set runtime args. Tests that we can memcpy new data
    // directly into the command issue queue with the arg count.
    auto& program_from_workload = workload.get_programs().at(device_range);
    if (is_quasar) {
        set_metal2_runtime_args(program_from_workload);
    } else {
        SetRuntimeArgs(program_from_workload, kernel, core_range1, default_rtas);
        SetRuntimeArgs(program_from_workload, kernel, core_range2, rtas_range2);
    }
    RunProgram(mesh_device, workload);

    // Validate second run for both core ranges
    for (const auto& core : core_range1) {
        ValidateArgResults(core, default_rtas, default_crtas);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            ValidateArgResults(core, rtas_range2, default_crtas);
        }
    }
}

// Parameterized test: Out-of-Bounds Arg Access Detection
// Verifies watcher detects kernels accessing args beyond bounds (index >= count)
// Tests RTA/CRTA access on DM0 and TRISC0 (single processor per test)
TEST_P(RTAAssertTest, OutOfBoundsArgAccessDetection) {
    const auto& params = GetParam();

    // Dispatch mode validation:
    // - Quasar: SD only (FD not yet available)
    // - Other archs: FD only
    if (IsSlowDispatch() && !is_quasar) {
        GTEST_SKIP() << "This test requires fast dispatch mode (except on Quasar)";
    }

    // Quasar: single core; other archs: 5x5 grid
    CoreRange core_range =
        is_quasar ? CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)) : CoreRange(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];

    distributed::MeshWorkload workload;
    Program program;

    constexpr const char* OOB_KERNEL_NAME = "rta_oob";

    if (is_quasar) {
        experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines m2_defines;
        if (params.test_rta) {
            m2_defines.push_back({"MAX_RTA_IDX", std::to_string(default_rtas.size())});
        } else {
            m2_defines.push_back({"MAX_CRTA_IDX", std::to_string(default_crtas.size())});
        }

        // RTA test reads index == default_rtas.size() (one past the end), so the kernel must declare
        // exactly default_rtas.size() varargs to make that access OOB.
        experimental::metal2_host_api::KernelSpec::RuntimeArgSchema schema;
        if (params.test_rta) {
            schema.num_runtime_varargs = default_rtas.size();
        } else {
            schema.num_common_runtime_varargs = default_crtas.size();
        }

        experimental::metal2_host_api::KernelSpec kspec;
        if (params.processor_class == HalProcessorClassType::DM) {
            kspec = experimental::metal2_host_api::KernelSpec{
                .unique_id = OOB_KERNEL_NAME,
                .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{rta_crta_kernel_path},
                .num_threads = static_cast<uint8_t>(num_dms_),
                .compiler_options = {.defines = m2_defines},
                .compile_time_arg_bindings = {{"dm_id", 0}},
                .runtime_arguments_schema = schema,
                .config_spec =
                    experimental::metal2_host_api::DataMovementConfiguration{
                        .gen2_data_movement_config =
                            experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
            };
        } else if (params.processor_class == HalProcessorClassType::COMPUTE) {
            kspec = experimental::metal2_host_api::KernelSpec{
                .unique_id = OOB_KERNEL_NAME,
                .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{rta_crta_kernel_path},
                .num_threads = 1,  // Run only on 1 NEO Cluster
                .compiler_options = {.defines = m2_defines},
                .runtime_arguments_schema = schema,
                .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
            };
        } else {
            TT_THROW("Unsupported processor class");
        }

        experimental::metal2_host_api::WorkUnitSpec wu{
            .unique_id = "main",
            .kernels = {OOB_KERNEL_NAME},
            .target_nodes = experimental::metal2_host_api::NodeRangeSet{core_range_set},
        };
        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "rta_oob",
            .kernels = {kspec},
            .work_units = {wu},
        };
        program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

        experimental::metal2_host_api::ProgramRunParams params_m2;
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams krp{.kernel_spec_name = OOB_KERNEL_NAME};
        if (params.test_rta) {
            for (const auto& c : core_range) {
                krp.runtime_varargs.push_back({experimental::metal2_host_api::NodeCoord{c}, default_rtas});
            }
        } else {
            krp.common_runtime_varargs = default_crtas;
        }
        params_m2.kernel_run_params = {krp};
        experimental::metal2_host_api::SetProgramRunParameters(program, params_m2);
    } else {
        std::map<std::string, std::string> defines;
        if (params.test_rta) {
            defines["MAX_RTA_IDX"] = std::to_string(default_rtas.size());  // Out of bounds
        } else {
            defines["MAX_CRTA_IDX"] = std::to_string(default_crtas.size());  // Out of bounds
        }

        KernelHandle kernel;
        switch (params.processor_class) {
            case HalProcessorClassType::DM:
                kernel = CreateKernel(
                    program,
                    rta_crta_kernel_path,
                    core_range_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
                break;
            case HalProcessorClassType::COMPUTE:
                kernel = CreateKernel(program, rta_crta_kernel_path, core_range_set, ComputeConfig{.defines = defines});
                break;
            default: TT_THROW("Unsupported processor class");
        }

        if (params.test_rta) {
            SetRuntimeArgs(program, kernel, core_range, default_rtas);
        } else {
            SetCommonRuntimeArgs(program, kernel, default_crtas);
        }
    }

    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    ExpectWatcherException(params.expected_message);
}

// Multi-DM test: verifies all DMs running concurrently can trigger RTA bounds check on Quasar.
// Uses a sync barrier so all DMs hit the OOB access together, stress-testing
// watcher's first-writer-wins assert mechanism. Any DM can report the error.
TEST_F(RTATestFixture, QuasarMultiDMOutOfBoundsArgDetection) {
    if (!is_quasar) {
        GTEST_SKIP() << "Test only applicable to Quasar";
    }

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(0, 0));
    CoreRangeSet core_range_set(std::vector{core_range});

    distributed::MeshWorkload workload;
    Program program;

    constexpr const char* MULTI_DM_KERNEL_NAME = "multi_dm_oob";
    experimental::metal2_host_api::KernelSpec dm_spec{
        .unique_id = MULTI_DM_KERNEL_NAME,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{rta_crta_kernel_path},
        .num_threads = static_cast<uint8_t>(num_dms_),
        .compiler_options =
            {.defines = {{"MAX_RTA_IDX", std::to_string(default_rtas.size())}, {"TEST_MULTI_DM_RTA", "1"}}},
        .compile_time_arg_bindings = {{"num_dms", num_dms_}, {"l1_sync_addr", l1_unreserved_base}},
        .runtime_arguments_schema = {.num_runtime_varargs = default_rtas.size()},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };
    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {MULTI_DM_KERNEL_NAME},
        .target_nodes = experimental::metal2_host_api::NodeRangeSet{core_range_set},
    };
    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "multi_dm_oob",
        .kernels = {dm_spec},
        .work_units = {wu},
    };
    program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams krp{.kernel_spec_name = MULTI_DM_KERNEL_NAME};
    for (const auto& c : core_range) {
        krp.runtime_varargs.push_back({experimental::metal2_host_api::NodeCoord{c}, default_rtas});
    }
    params.kernel_run_params = {krp};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    workload.add_program(device_range, std::move(program));

    // Zero out sync counter before launch
    std::vector<uint32_t> zero_sync = {0, 0};
    tt::tt_metal::detail::WriteToDeviceL1(device, core_range.start_coord, l1_unreserved_base, zero_sync);

    RunProgram(mesh_device, workload);

    // Any DM can report the error; just verify the bounds-check message appears
    ExpectWatcherException("unique runtime arg index out of bounds");
}

INSTANTIATE_TEST_SUITE_P(
    WatcherArgAsserts,
    RTAAssertTest,
    ::testing::Values(
        // RTA tests on DM0
        RTAAssertTestParams{"RTA_DM0", true, "unique runtime arg index out of bounds", HalProcessorClassType::DM},
        // RTA tests on TRISC0
        RTAAssertTestParams{
            "RTA_TRISC0", true, "unique runtime arg index out of bounds", HalProcessorClassType::COMPUTE},
        // CRTA tests on DM0
        RTAAssertTestParams{"CRTA_DM0", false, "common runtime arg index out of bounds", HalProcessorClassType::DM},
        // CRTA tests on TRISC0
        RTAAssertTestParams{
            "CRTA_TRISC0", false, "common runtime arg index out of bounds", HalProcessorClassType::COMPUTE}),
    [](const ::testing::TestParamInfo<RTAAssertTestParams>& info) { return info.param.test_name; });
}  // namespace tt::tt_metal
