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

// Verifies watcher handles the dispatcher's sentinel pattern (used when no RTAs set):
// - Device interprets 0xBEEF#### as rta_count = 0 (validated on DM0/TRISC0)
// - Kernels not accessing args run successfully with sentinel pattern
// - Kernels accessing args when count = 0 trigger "out of bounds" assert
// - Catches bug: kernel placed on cores but runtime args only bound for a subset
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

        const experimental::KernelSpecName DM_KERNEL_NAME{"zero_arg_dm"};
        const experimental::KernelSpecName COMPUTE_KERNEL_NAME{"zero_arg_compute"};

        experimental::KernelSpec dm_spec{
            .unique_id = DM_KERNEL_NAME,
            .source = rta_crta_kernel_path,
            .num_threads = 1,
            .compile_time_args = {{"l1_scratch_addr", l1_unreserved_base}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen1_config =
                        experimental::DataMovementHardwareConfig::Gen1Config{
                            .processor = DataMovementProcessor::RISCV_0}},
        };
        experimental::KernelSpec compute_spec{
            .unique_id = COMPUTE_KERNEL_NAME,
            .source = rta_crta_kernel_path,
            .num_threads = 1,
            .compile_time_args = {{"l1_scratch_addr", compute_scratch_addr}},
            .hw_config = experimental::ComputeHardwareConfig{},
        };
        experimental::WorkUnitSpec wu{
            .name = "main",
            .kernels = {DM_KERNEL_NAME, COMPUTE_KERNEL_NAME},
            .target_nodes = experimental::NodeRangeSet{core_range_set},
        };
        experimental::ProgramSpec spec{
            .name = "zero_arg_schema",
            .kernels = {dm_spec, compute_spec},
            .work_units = {wu},
        };
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

        distributed::MeshWorkload workload;
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

    // Part B: Accessing unset RTAs on a subset of cores should trigger assert.
    // Uses the legacy host API because Metal 2.0's SetProgramRunArgs validates that every
    // targeted node has its varargs bound and rejects partial bindings up front — i.e. it
    // structurally prevents the bug class this sub-test exercises. The legacy dispatcher instead
    // writes a 0xBEEF#### sentinel on unbound cores, which the device interprets as
    // rta_count = 0, so the kernel's RTA[0] access is OOB and trips the watcher assert.
    {
        const std::string oob_kernel_path =
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_oob_legacy.cpp";

        CoreRange cores_with_rtas(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange cores_without_rtas(CoreCoord(2, 2), CoreCoord(3, 3));
        CoreRangeSet core_range_set(std::vector{cores_with_rtas, cores_without_rtas});

        distributed::MeshWorkload workload;
        Program program;

        auto kernel = CreateKernel(
            program,
            oob_kernel_path,
            core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program, kernel, cores_with_rtas, default_rtas);
        // cores_without_rtas has NO RTAs set — 0xBEEF#### pattern -> rta_count = 0 -> assert

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
    const experimental::KernelSpecName DM_KERNEL_NAME{"rta_dm"};

    // Pad to match default_rtas.size() (== num_runtime_varargs in the schema).
    // Metal 2.0 enforces a uniform per-node RTA count across all NodeCoords in a kernel run, so
    // every node binding under this kernel must provide exactly num_runtime_varargs values.
    std::vector<uint32_t> rtas_range2 = {0x1000, 0x1001, 0x1002, 0x1003, 0x1004};

    // Build a Metal 2.0 KernelSpec that works on both gen1 (single BRISC) and gen2 (all Quasar user DMs).
    // Provide both gen1 and gen2 configs so the runtime selects the one matching the current arch.
    experimental::DataMovementHardwareConfig dm_cfg{
        .gen1_config =
            experimental::DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_0},
        .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
    };

    experimental::KernelSpec dm_spec{
        .unique_id = DM_KERNEL_NAME,
        .source = rta_crta_kernel_path,
        .num_threads = is_quasar ? num_dms_ : 1u,
        .hw_config = dm_cfg,
        .advanced_options =
            experimental::KernelAdvancedOptions{
                .num_runtime_varargs = default_rtas.size(),
                .num_common_runtime_varargs = default_crtas.size(),
            },
    };
    if (is_quasar) {
        dm_spec.compile_time_args = {{"dm_id", 0}, {"l1_scratch_addr", l1_unreserved_base}};
    } else {
        dm_spec.compile_time_args = {{"l1_scratch_addr", l1_unreserved_base}};
    }
    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {DM_KERNEL_NAME},
        .target_nodes = experimental::NodeRangeSet{core_range_set},
    };
    experimental::ProgramSpec spec{
        .name = "rta_validation",
        .kernels = {dm_spec},
        .work_units = {wu},
    };
    program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Note: Quasar uses only core_range1; gen1 also covers core_range2 with a different RTA payload.
    auto set_metal2_runtime_args = [&](Program& prog) {
        experimental::ProgramRunArgs params;
        experimental::ProgramRunArgs::KernelRunArgs krp{
            .advanced_options =
                experimental::AdvancedKernelRunArgs{
                    .common_runtime_varargs = default_crtas,
                },
        };
        for (const auto& c : core_range1) {
            krp.advanced_options.runtime_varargs.emplace(experimental::NodeCoord{c}, default_rtas);
        }
        if (!is_quasar) {
            for (const auto& c : core_range2) {
                krp.advanced_options.runtime_varargs.emplace(experimental::NodeCoord{c}, rtas_range2);
            }
        }
        krp.kernel = DM_KERNEL_NAME;
        params.kernel_run_args = {krp};
        experimental::SetProgramRunArgs(prog, params);
    };

    set_metal2_runtime_args(program);
    workload.add_program(device_range, std::move(program));

    RunProgram(mesh_device, workload);

    // Validate first run
    auto validate_range = [&](const CoreRange& cr, const std::vector<uint32_t>& expected) {
        for (const auto& core : cr) {
            ValidateArgResults(core, expected, default_crtas);
        }
    };
    validate_range(core_range1, default_rtas);
    if (!is_quasar) {
        validate_range(core_range2, rtas_range2);
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

    // Second run: re-set runtime args. Tests that we can re-issue the same command stream with new values.
    auto& program_from_workload = workload.get_programs().at(device_range);
    set_metal2_runtime_args(program_from_workload);
    RunProgram(mesh_device, workload);

    // Validate second run for both core ranges
    validate_range(core_range1, default_rtas);
    if (!is_quasar) {
        validate_range(core_range2, rtas_range2);
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

    const experimental::KernelSpecName OOB_KERNEL_NAME{"rta_oob"};

    experimental::KernelSpec::CompilerOptions::Defines m2_defines;
    if (params.test_rta) {
        m2_defines.emplace("MAX_RTA_IDX", std::to_string(default_rtas.size()));
    } else {
        m2_defines.emplace("MAX_CRTA_IDX", std::to_string(default_crtas.size()));
    }

    // RTA test reads index == default_rtas.size() (one past the end), so the kernel must declare
    // exactly default_rtas.size() varargs to make that access OOB.
    experimental::KernelAdvancedOptions adv_opts;
    if (params.test_rta) {
        adv_opts.num_runtime_varargs = default_rtas.size();
    } else {
        adv_opts.num_common_runtime_varargs = default_crtas.size();
    }

    experimental::KernelSpec kspec{
        .unique_id = OOB_KERNEL_NAME,
        .source = rta_crta_kernel_path,
        .compiler_options = {.defines = m2_defines},
        .advanced_options = adv_opts,
    };
    if (params.processor_class == HalProcessorClassType::DM) {
        if (is_quasar) {
            kspec.num_threads = num_dms_;
            kspec.compile_time_args = {{"dm_id", 0}};
        } else {
            kspec.num_threads = 1;
        }
        // Provide both gen1 and gen2 configs so the same KernelSpec runs on either arch.
        kspec.hw_config = experimental::DataMovementHardwareConfig{
            .gen1_config =
                experimental::DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_0},
            .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
        };
    } else if (params.processor_class == HalProcessorClassType::COMPUTE) {
        kspec.num_threads = 1;  // On Quasar, only 1 NEO Cluster; gen1 has a single compute group.
        kspec.hw_config = experimental::ComputeHardwareConfig{};
    } else {
        TT_THROW("Unsupported processor class");
    }

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {OOB_KERNEL_NAME},
        .target_nodes = experimental::NodeRangeSet{core_range_set},
    };
    experimental::ProgramSpec spec{
        .name = "rta_oob",
        .kernels = {kspec},
        .work_units = {wu},
    };
    program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params_m2;
    experimental::ProgramRunArgs::KernelRunArgs krp{};
    if (params.test_rta) {
        for (const auto& c : core_range) {
            krp.advanced_options.runtime_varargs.emplace(experimental::NodeCoord{c}, default_rtas);
        }
    } else {
        krp.advanced_options.common_runtime_varargs = default_crtas;
    }
    krp.kernel = OOB_KERNEL_NAME;
    params_m2.kernel_run_args = {krp};
    experimental::SetProgramRunArgs(program, params_m2);

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

    const experimental::KernelSpecName MULTI_DM_KERNEL_NAME{"multi_dm_oob"};
    experimental::KernelSpec dm_spec{
        .unique_id = MULTI_DM_KERNEL_NAME,
        .source = rta_crta_kernel_path,
        .num_threads = num_dms_,
        .compiler_options =
            {.defines = {{"MAX_RTA_IDX", std::to_string(default_rtas.size())}, {"TEST_MULTI_DM_RTA", "1"}}},
        .compile_time_args = {{"num_dms", num_dms_}, {"l1_sync_addr", l1_unreserved_base}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
        .advanced_options =
            experimental::KernelAdvancedOptions{
                .num_runtime_varargs = default_rtas.size(),
            },
    };
    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {MULTI_DM_KERNEL_NAME},
        .target_nodes = experimental::NodeRangeSet{core_range_set},
    };
    experimental::ProgramSpec spec{
        .name = "multi_dm_oob",
        .kernels = {dm_spec},
        .work_units = {wu},
    };
    program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs krp{};
    for (const auto& c : core_range) {
        krp.advanced_options.runtime_varargs.emplace(experimental::NodeCoord{c}, default_rtas);
    }
    krp.kernel = MULTI_DM_KERNEL_NAME;
    params.kernel_run_args = {krp};
    experimental::SetProgramRunArgs(program, params);

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
