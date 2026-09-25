// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

// The one kernel source used by more than one test in this file: several tests print a
// per-iteration message from many RISCs at once and count the results.
constexpr const char* kPrintIterationsKernel = "tests/tt_metal/tt_metal/test_kernels/device_print/print_iterations.cpp";

// DM0 (ISR) and DM1 (remapper) are reserved for the runtime, so user kernels get DM2..DM7. The HAL
// registers all 8 DM processors and has no concept of the reservation, so this is the one number
// that cannot be queried.
constexpr uint32_t kQuasarReservedDmThreads = 2;

// Quasar Tensix node topology, derived from the HAL rather than hardcoded.
struct QuasarNodeTopology {
    uint32_t user_dm_threads;
    uint32_t compute_engines;
    uint32_t triscs_per_engine;

    uint32_t compute_processors() const { return compute_engines * triscs_per_engine; }
};

// Quasar only: the HAL flattens compute processors, registering 4 engines x 4 TRISCs as 16 entries,
// so the engine count is the quotient of the processor count and the per-engine firmware-binary
// count. Same derivation as tt_metal/impl/debug/debug_helpers.hpp:349.
QuasarNodeTopology GetQuasarNodeTopology() {
    const auto& hal = MetalContext::instance().hal();
    const uint32_t tensix_idx = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const auto dm_class = static_cast<uint32_t>(HalProcessorClassType::DM);
    const auto compute_class = static_cast<uint32_t>(HalProcessorClassType::COMPUTE);

    const uint32_t triscs_per_engine = hal.get_processor_class_num_fw_binaries(tensix_idx, compute_class);
    return QuasarNodeTopology{
        .user_dm_threads =
            hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, dm_class) - kQuasarReservedDmThreads,
        .compute_engines =
            hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, compute_class) / triscs_per_engine,
        .triscs_per_engine = triscs_per_engine,
    };
}

// The node these tests place kernels on. Any Tensix worker will do -- what is being tested is that
// every RISC *within one node* prints concurrently without interleaving.
constexpr experimental::NodeCoord kDefaultPrintNode{0, 0};

// A Quasar print-storm program: `dm_threads` user DM threads and `compute_engines` Tensix engines on
// `node`, all running `kernel_path`. Pass 0 for either count to omit that kernel.
//
// Kernel contract: `kernel_path` takes exactly one runtime vararg, its iteration count, read with
// get_arg_val<uint32_t>(0). Neither KernelSpec declares a named runtime-arg schema, so the vararg
// section starts at index 0 -- which is what lets the same kernel source also run unchanged on the
// classic Wormhole/Blackhole path.
experimental::ProgramSpec MakeQuasarPrintSpec(
    std::string_view kernel_path,
    uint32_t dm_threads,
    uint32_t compute_engines,
    const experimental::NodeCoord& node = kDefaultPrintNode) {
    experimental::ProgramSpec spec{.name = "dprint_concurrent"};
    experimental::Group<experimental::KernelSpecName> placed;

    if (dm_threads > 0) {
        spec.kernels.push_back(experimental::KernelSpec{
            .unique_id = experimental::KernelSpecName{"dm_print"},
            .source = std::filesystem::path{kernel_path},
            .num_threads = dm_threads,
            .hw_config = experimental::DataMovementHardwareConfig{},
            .advanced_options = experimental::KernelAdvancedOptions{.num_runtime_varargs = 1},
        });
        placed.push_back(experimental::KernelSpecName{"dm_print"});
    }
    if (compute_engines > 0) {
        spec.kernels.push_back(experimental::KernelSpec{
            .unique_id = experimental::KernelSpecName{"compute_print"},
            .source = std::filesystem::path{kernel_path},
            .num_threads = compute_engines,
            .hw_config = experimental::ComputeHardwareConfig{},
            .advanced_options = experimental::KernelAdvancedOptions{.num_runtime_varargs = 1},
        });
        placed.push_back(experimental::KernelSpecName{"compute_print"});
    }

    spec.work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = placed, .target_nodes = node}};
    return spec;
}

// A single compute kernel on `node`, portable across generations. Gen1 runs it on TRISC0/1/2; Gen2
// runs it on one Tensix engine's TRISCs. Both satisfy the tests that only assert on printed strings.
experimental::ProgramSpec MakeComputePrintSpec(
    tt::ARCH arch, std::string_view kernel_path, const experimental::NodeCoord& node = kDefaultPrintNode) {
    experimental::ComputeHardwareConfig hw_config;
    if (arch == tt::ARCH::QUASAR) {
        hw_config = experimental::ComputeHardwareConfig{};
    } else {
        hw_config = experimental::ComputeHardwareConfig{};
    }

    const experimental::KernelSpecName name{"compute_print"};
    return experimental::ProgramSpec{
        .name = "dprint_compute",
        .kernels = {experimental::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{kernel_path},
            .num_threads = 1,
            .hw_config = hw_config,
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {name}, .target_nodes = node}},
    };
}

// Gives every kernel in the spec its iteration count as vararg 0. The node is read back out of the
// spec so it cannot drift from the one the kernels were placed on.
experimental::ProgramRunArgs MakeIterationArgs(const experimental::ProgramSpec& spec, uint32_t iterations) {
    const auto& node = std::get<experimental::NodeCoord>(spec.work_units.front().target_nodes);

    experimental::ProgramRunArgs args;
    for (const auto& kernel : spec.kernels) {
        args.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = kernel.unique_id,
            .advanced_options = experimental::AdvancedKernelRunArgs{.runtime_varargs = {{node, {iterations}}}},
        });
    }
    return args;
}

}  // namespace

class DevicePrintOutputFixture : public DevicePrintFixture {
public:
    void TestOutput(
        const std::string& kernel_path,
        const std::vector<std::string>& expected_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        for (auto& mesh_device : this->devices_) {
            RunProgram(mesh_device, kernel_path, runtime_args);
            EXPECT_TRUE(FileContainsAllStrings(dprint_file_name, expected_messages));
        }
    }

    // Runs a callstack kernel and verifies the resolved frames: `present` must appear in the given
    // order, and every entry in `absent` must NOT appear (i.e. it was skipped or never reached).
    void TestCallstack(
        const std::string& kernel_path,
        const std::vector<std::string>& present,
        const std::vector<std::string>& absent = {}) {
        for (auto& mesh_device : this->devices_) {
            RunProgram(mesh_device, kernel_path);
            EXPECT_TRUE(FileContainsAllStringsInOrder(dprint_file_name, present));
            EXPECT_TRUE(FileContainsNoneOfStrings(dprint_file_name, absent));
        }
    }
};

TEST_F(DevicePrintOutputFixture, PrintSimpleString) {
    std::vector<std::string> messages = {
        "Hello world!",
        "First line.",
        "Second line.",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_simple_string.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintSingleUintArg) {
    std::vector<uint32_t> runtime_args = {42};
    std::vector<std::string> messages = {
        "Printing uint32_t from arg: 42",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", messages, runtime_args);
}

TEST_F(DevicePrintOutputFixture, PrintFactorial) {
    std::vector<uint32_t> runtime_args = {5};
    std::vector<std::string> messages = {
        "factorial(5) = 120",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_factorial.cpp", messages, runtime_args);
}

TEST_F(DevicePrintOutputFixture, PrintBasicTypes) {
    std::vector<std::string> messages = {
        "int8_t: -8",
        "uint8_t: 8",
        "int16_t: -16",
        "uint16_t: 16",
        "int32_t: -32",
        "uint32_t: 32",
        "int64_t: -64",
        "uint64_t: 64",
        "float: 3.14",
        "double: 6.28",
        "bool: true",
        "bf4_t: 0.5",
        "bf8_t: 0.375",
        "bf16_t: 0.122558594",
        "Reordered args: true -16 -32 -64",
        "Reordered args: true -16 -32 -64",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintWithFormatSpecified) {
    std::vector<std::string> messages = {
        "int8_t:         -8",
        "uint8_t: 0B1000",
        "int16_t: -16       ",
        "uint16_t: 0X10",
        "int32_t:    -32    ",
        "uint32_t: 0x20",
        "int64_t: -64",
        "uint64_t: 0X000040",
        "float: 3.14",
        "double: 6.28000",
        "bool: true",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_with_format_specified.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintManyIterations) {
    uint32_t iterations = 1000;
    std::vector<uint32_t> runtime_args = {iterations};
    std::vector<std::string> messages;

    messages.reserve(iterations);
    for (uint32_t i = 0; i < iterations; i++) {
        messages.push_back("Test iteration: " + std::to_string(i));
    }

    TestOutput(kPrintIterationsKernel, messages, runtime_args);
}

// Test that printing from multiple RISCs on the same core works and doesn't interleave messages.
// We detect interleaving by having garbage data in server output.
// If all messages are present and correctly formatted, we can be reasonably sure that there was no interleaving.
TEST_F(DevicePrintOutputFixture, PrintConcurrentAllRiscs) {
    size_t device_counter = 0;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->arch() != tt::ARCH::WORMHOLE_B0 && mesh_device->arch() != tt::ARCH::BLACKHOLE &&
            mesh_device->arch() != tt::ARCH::QUASAR) {
            // Test currently works on WH, BH, and Quasar
            continue;
        }
        device_counter++;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        // Quasar prints from many more concurrent RISCs (22 on this test) than WH/BH (5).
        // 1000 × 22 = 22k contended prints take well over 10 minutes on the RTL emulator,
        // so reduce the iteration count on Quasar. The race-detection signal does not require
        // a large count — even a small number of iterations with high concurrency exposes any
        // memory-ordering bugs immediately, and the assertion is per-iteration.
        const uint32_t iterations_count = (mesh_device->arch() == tt::ARCH::QUASAR) ? 10u : 1000u;
        std::vector<uint32_t> runtime_args = {iterations_count};

        // Per-arch expected number of distinct RISCs printing concurrently on the same core.
        //   WH/BH: 5  (BRISC + NCRISC + TRISC0/1/2)
        //   Quasar: 6 user DMs (DM2..DM7) + 16 TRISCs (4 Tensix engines × 4 TRISCs) = 22
        int risc_count_per_iter = 0;

        if (mesh_device->arch() == tt::ARCH::QUASAR) {
            // One DM kernel covering all 6 user DM threads in the cluster (DM0/DM1 reserved), plus one
            // compute kernel covering all 4 Tensix engines × 4 TRISCs = 16 baby-RISCs.
            const auto topology = GetQuasarNodeTopology();
            auto spec = MakeQuasarPrintSpec(kPrintIterationsKernel, topology.user_dm_threads, topology.compute_engines);
            Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
            experimental::SetProgramRunArgs(program, MakeIterationArgs(spec, iterations_count));
            workload.add_program(device_range, std::move(program));

            risc_count_per_iter = static_cast<int>(topology.user_dm_threads + topology.compute_processors());
        } else {
            Program program = Program();
            workload.add_program(device_range, std::move(program));
            auto& program_ = workload.get_programs().at(device_range);

            constexpr CoreCoord core = {0, 0};

            // WH/BH: BRISC
            auto kernel_handle = CreateKernel(
                program_,
                kPrintIterationsKernel,
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

            SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

            // NCRISC
            kernel_handle = CreateKernel(
                program_,
                kPrintIterationsKernel,
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

            SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

            // TRISC0 (Unpack), TRISC1 (Math), TRISC2 (Pack)
            kernel_handle = CreateKernel(program_, kPrintIterationsKernel, core, ComputeConfig{});

            SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

            // Every RISC on a WH/BH Tensix core: BRISC + NCRISC + TRISC0/1/2.
            risc_count_per_iter = static_cast<int>(
                MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::TENSIX));
        }

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        // Verify all risc_count_per_iter * N messages are correctly formatted.
        // Each iteration value must appear exactly risc_count_per_iter times — once per RISC.
        std::fstream log_file;
        ASSERT_TRUE(OpenFile(dprint_file_name, log_file, std::fstream::in));
        std::vector<int> counts(iterations_count, 0);
        std::string line;
        for (;;) {
            if (!getline(log_file, line)) {
                break;
            }
            int iter = -1;
            if (sscanf(line.c_str(), "Test iteration: %d", &iter) == 1 && iter >= 0 && iter < counts.size()) {
                counts[iter]++;
            }
        }
        const int expected_count = risc_count_per_iter * static_cast<int>(device_counter);
        for (int i = 0; i < static_cast<int>(counts.size()); i++) {
            EXPECT_EQ(counts[i], expected_count)
                << "Iteration " << i << " appeared " << counts[i] << " times (expected " << expected_count << " times)";
        }
    }
}

// All-workers variant of PrintConcurrentAllRiscs: run the same per-iteration print kernel on EVERY
// Tensix worker core (the full compute grid) to stress the DevicePrintDispatch worker-L1 -> DRAM
// aggregation under maximum concurrency. The single-core variant never exercises the multi-core
// aggregation path; this one does. Each iteration value must appear EXACTLY
// risc_count_per_iter * num_worker_cores times: fewer => the aggregation dropped data, more =>
// it duplicated data. Run with TT_METAL_DEVICE_PRINT=1 to route through the DRAM aggregation path.
TEST_F(DevicePrintOutputFixture, PrintConcurrentAllRiscsAllWorkers) {
    size_t device_counter = 0;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->arch() != tt::ARCH::WORMHOLE_B0 && mesh_device->arch() != tt::ARCH::BLACKHOLE) {
            continue;  // WH/BH only.
        }
        device_counter++;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // Cover the entire compute grid (every usable worker core).
        CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});
        const uint32_t num_cores = grid.x * grid.y;

        // 1000 iters x 5 riscs x ~110 cores produces enough output (~11 MB) to wrap the 1 MB DRAM
        // aggregation ring many times — this is what exercises the ring full/empty boundary and the
        // backpressure/drain path that previously dropped data. DPRINT_AW_ITERS overrides it for
        // heavier manual stress.
        uint32_t iterations_count = 1000u;
        if (const char* env = std::getenv("DPRINT_AW_ITERS")) {
            iterations_count = static_cast<uint32_t>(std::max(1, atoi(env)));
        }
        std::vector<uint32_t> runtime_args = {iterations_count};

        // BRISC
        auto kernel_handle = CreateKernel(
            program_,
            kPrintIterationsKernel,
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program_, kernel_handle, all_cores, runtime_args);

        // NCRISC
        kernel_handle = CreateKernel(
            program_,
            kPrintIterationsKernel,
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program_, kernel_handle, all_cores, runtime_args);

        // TRISC0 (Unpack), TRISC1 (Math), TRISC2 (Pack)
        kernel_handle = CreateKernel(program_, kPrintIterationsKernel, all_cores, ComputeConfig{});
        SetRuntimeArgs(program_, kernel_handle, all_cores, runtime_args);

        const int risc_count_per_iter =
            static_cast<int>(MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::TENSIX));

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        // Count how many times each iteration value appears across the whole grid. Each value must
        // appear exactly risc_count_per_iter * num_cores times: fewer => the DRAM aggregation dropped
        // data, more => it duplicated data.
        std::fstream log_file;
        ASSERT_TRUE(OpenFile(dprint_file_name, log_file, std::fstream::in));
        std::vector<int> counts(iterations_count, 0);
        std::string line;
        while (getline(log_file, line)) {
            int iter = -1;
            if (sscanf(line.c_str(), "Test iteration: %d", &iter) == 1 && iter >= 0 && iter < (int)counts.size()) {
                counts[iter]++;
            }
        }
        const int expected_count = risc_count_per_iter * static_cast<int>(num_cores) * static_cast<int>(device_counter);
        for (int i = 0; i < (int)counts.size(); i++) {
            EXPECT_EQ(counts[i], expected_count)
                << "Iteration " << i << " appeared " << counts[i] << " times (expected " << expected_count << ")";
        }
    }
}

// Quasar-only race-detection variant of PrintConcurrentAllRiscs that uses *only the rocket-core
// DMs* (DM2..DM7). Isolates whether the contention bug is within a single rocket-core group, or
// requires the mix of rocket + baby-risc contenders.
TEST_F(DevicePrintOutputFixture, PrintConcurrentRocketRiscs) {
    size_t device_counter = 0;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->arch() != tt::ARCH::QUASAR) {
            continue;  // Quasar-only: rocket cores are a Quasar-specific concept.
        }
        device_counter++;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        const uint32_t iterations_count = 5;

        const auto topology = GetQuasarNodeTopology();
        auto spec = MakeQuasarPrintSpec(kPrintIterationsKernel, topology.user_dm_threads, /*compute_engines=*/0);
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
        experimental::SetProgramRunArgs(program, MakeIterationArgs(spec, iterations_count));
        workload.add_program(device_range, std::move(program));

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        std::fstream log_file;
        ASSERT_TRUE(OpenFile(dprint_file_name, log_file, std::fstream::in));
        std::vector<int> counts(iterations_count, 0);
        std::string line;
        for (;;) {
            if (!getline(log_file, line)) {
                break;
            }
            int iter = -1;
            if (sscanf(line.c_str(), "Test iteration: %d", &iter) == 1 && iter >= 0 && iter < counts.size()) {
                counts[iter]++;
            }
        }
        const int expected_count = static_cast<int>(topology.user_dm_threads) * static_cast<int>(device_counter);
        for (int i = 0; i < static_cast<int>(counts.size()); i++) {
            EXPECT_EQ(counts[i], expected_count)
                << "Iteration " << i << " appeared " << counts[i] << " times (expected " << expected_count << " times)";
        }
    }
}

// Quasar-only race-detection variant of PrintConcurrentAllRiscs that uses *only baby-risc TRISCs*
// (4 Tensix engines × 4 TRISCs = 16 threads). Complements PrintConcurrentRocketRiscs so we can
// localize the locking bug to rocket-only, baby-risc-only, or the cross-group case.
TEST_F(DevicePrintOutputFixture, PrintConcurrentBabyRiscs) {
    size_t device_counter = 0;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->arch() != tt::ARCH::QUASAR) {
            continue;  // Quasar-only.
        }
        device_counter++;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        const uint32_t iterations_count = 5;

        const auto topology = GetQuasarNodeTopology();
        auto spec = MakeQuasarPrintSpec(kPrintIterationsKernel, /*dm_threads=*/0, topology.compute_engines);
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
        experimental::SetProgramRunArgs(program, MakeIterationArgs(spec, iterations_count));
        workload.add_program(device_range, std::move(program));

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        std::fstream log_file;
        ASSERT_TRUE(OpenFile(dprint_file_name, log_file, std::fstream::in));
        std::vector<int> counts(iterations_count, 0);
        std::string line;
        for (;;) {
            if (!getline(log_file, line)) {
                break;
            }
            int iter = -1;
            if (sscanf(line.c_str(), "Test iteration: %d", &iter) == 1 && iter >= 0 && iter < counts.size()) {
                counts[iter]++;
            }
        }
        const int expected_count = static_cast<int>(topology.compute_processors()) * static_cast<int>(device_counter);
        for (int i = 0; i < static_cast<int>(counts.size()); i++) {
            EXPECT_EQ(counts[i], expected_count)
                << "Iteration " << i << " appeared " << counts[i] << " times (expected " << expected_count << " times)";
        }
    }
}

TEST_F(DevicePrintOutputFixture, PrintAllArgumentSizes) {
    std::vector<std::string> messages = {
        "No arguments",
        "1 argument: 1",
        "2 arguments: 1 2",
        "3 arguments: 1 2 3",
        "4 arguments: 1 2 3 4",
        "5 arguments: 1 2 3 4 5",
        "6 arguments: 1 2 3 4 5 6",
        "7 arguments: 1 2 3 4 5 6 7",
        "8 arguments: 1 2 3 4 5 6 7 8",
        "9 arguments: 1 2 3 4 5 6 7 8 9",
        "10 arguments: 1 2 3 4 5 6 7 8 9 10",
        "11 arguments: 1 2 3 4 5 6 7 8 9 10 11",
        "12 arguments: 1 2 3 4 5 6 7 8 9 10 11 12",
        "13 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13",
        "14 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14",
        "15 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
        "16 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
        "17 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17",
        "18 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18",
        "19 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19",
        "20 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
        "21 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21",
        "22 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22",
        "23 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23",
        "24 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
        "25 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
        "26 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26",
        "27 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27",
        "28 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28",
        "29 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29",
        "30 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30",
        "31 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31",
        "32 arguments: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_all_argument_sizes.cpp", messages);
}

// When DWARF debug info is present in the ELF, enum values are printed as their symbolic names.
// Without '#' the output is just the value name; with '#' the full qualified type::value name is printed.
// Bit-field enums print each active flag separated by " | ".
// Unrecognised values are printed as (EnumType)integer.
TEST_F(DevicePrintOutputFixture, PrintEnumValue) {
    std::vector<std::string> messages = {
        // Plain format: only the value name
        "Enum1 value: Value2",
        // Alternate form (#): full qualified type name + value name
        "Enum1 full name value: test::deep::Enum1::Value3",
        "Enum1 unrecognized value: (test::deep::Enum1)100",
        "Enum1 full name unrecognized value: (test::deep::Enum1)100",
        "Enum2 value: ValueB",
        "Enum2 full name value: test_shallow::Enum2::ValueC",
        "EnumClass value: ValueY",
        "EnumClass full name value: EnumClass::ValueZ",
        // Bit-field enum: active flags joined by " | "
        "BitEnum value: Flag1 | Flag3",
        "BitEnum full name value: flags::BitEnum::Flag2 | flags::BitEnum::Flag3",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_enum_value.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintBuiltinTypes) {
    std::vector<std::string> messages = {
        "i=1",
        "unknown=5",
        "u=42",
        "ll=-123456789012345",
        "ull=123456789012345",
        "s=-12345",
        "us=12345",
        "cvllu=98765432109876",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_builtin_types.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintStringTypes) {
    std::vector<std::string> messages = {
        // Runtime const char* prints as hex address (we just check the prefix)
        "Sample string: 0x",
        // Compile-time CTSTR resolves to the actual string content from the ELF
        "Compile time string: Hello world!",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_string_types.cpp", messages);
}

// Type names are extracted from __PRETTY_FUNCTION__ at compile time and resolved from the ELF on the
// host, same as CTSTR. int/float are used over uint32_t/int8_t, whose spelling is typedef-dependent.
TEST_F(DevicePrintOutputFixture, PrintTypeName) {
    std::vector<std::string> messages = {
        "builtin type: int",
        "struct type: test::deep::Foo",
        "enum type: test::deep::Bar",
        "template type: test::deep::Wrapper<int>",
        "decltype: float",
        "with value: float = 1",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_type_name.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintReorder) {
    std::vector<std::string> messages = {
        "u16_1: 16 u16_2: 32 u32_1: 1 u32_2: 2 u32_3: 4 u32_4: 8",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_reorder.cpp", messages);
}

// Verify the DPRINT(format, ...) alias defined in dprint.h produces output identical to
// DEVICE_PRINT(format, ...).
TEST_F(DevicePrintOutputFixture, DprintAliasMatchesDevicePrint) {
    std::vector<std::string> messages = {
        "no args",
        "one arg: 42",
        "three args: 1 2 3",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_dprint_alias.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintInlineFunction) {
    std::vector<std::string> messages = {
        "BEFORE!!!",
        "INLINE!!!",
        "AFTER!!!",
    };

    for (auto& mesh_device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto spec = MakeComputePrintSpec(
            mesh_device->arch(), "tests/tt_metal/tt_metal/test_kernels/device_print/print_inline_function.cpp");
        workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));

        RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        EXPECT_TRUE(FileContainsAllStrings(dprint_file_name, messages));
    }
}

TEST_F(DevicePrintOutputFixture, PrintCallstackPcFullRaFirmware) {
    // The PC inline chain reaches the kernel_main terminal, so the unwind stops there and the
    // firmware return address is never resolved -- no continuation sentinel is emitted.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_pc_full_ra_fw.cpp",
        /* present */ {"CALLSTACK_BEGIN", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"..."});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackPcFull) {
    // PC alone unwinds its inline chain all the way to the kernel_main terminal (RA is poisoned and
    // trimmed once the terminal is found), so no continuation sentinel is emitted.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_pc_full.cpp",
        /* present */ {"CALLSTACK_BEGIN", "pc3", "pc2", "pc1", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"..."});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackPcRaFull) {
    // The PC and RA inline chains together unwind all the way to the kernel_main terminal, so no
    // continuation sentinel is emitted.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_pc_ra_full.cpp",
        /* present */
        {"CALLSTACK_BEGIN", "pc3", "pc2", "pc1", "ra3", "ra2", "ra1", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"..."});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackPcRaPartial) {
    // ra1 is noinline, so kernel_main sits above ra1's frame and cannot be reached from a single
    // return address: the unwind stops after ra1 and emits the "..." sentinel. kernel_main must not
    // appear, since it was never reached.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_pc_ra_partial.cpp",
        /* present */ {"CALLSTACK_BEGIN", "pc3", "pc2", "pc1", "ra3", "ra2", "ra1", "...", "CALLSTACK_END"},
        /* absent */ {"kernel_main"});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackSentinelSkip) {
    // Both PC and RA are the invalid-address sentinel, with a non-zero skip count. Neither address
    // resolves, so the callstack collapses to the "..." sentinel regardless of skip_frames.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_sentinel.cpp",
        /* present */ {"CALLSTACK_BEGIN", "...", "CALLSTACK_END"});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackSkipCrossPcRa) {
    // skip_frames = 4 crosses the PC/RA boundary: it removes the two PC frames (pc2, pc1) and the
    // first two RA frames (ra5, ra4), leaving ra3, ra2, ra1 and kernel_main. The skipped frames must
    // not appear.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_skip_cross.cpp",
        /* present */ {"CALLSTACK_BEGIN", "ra3", "ra2", "ra1", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"pc2", "pc1", "ra5", "ra4"});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackSkipUnderflow) {
    // skip_frames skips through the terminal, which would underflow the remaining-frame count. The
    // host clamps the skip, so the callstack collapses to the "..." sentinel instead of reading out
    // of bounds. None of the frames between the print site and the terminal should appear.
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_skip_underflow.cpp",
        /* present */ {"CALLSTACK_BEGIN", "...", "CALLSTACK_END"},
        /* absent */ {"inner", "middle", "kernel_main"});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackCurrent) {
    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_helper.cpp",
        /* present */ {"CALLSTACK_BEGIN", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"current"});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackTailCallUnambiguous) {
    GTEST_SKIP() << "Fix unwinding of tail calls #47666";

    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_tailcall_unambiguous.cpp",
        /* present */ {"CALLSTACK_BEGIN", "leaf", "mid", "top", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"..."});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackTailCallResolvable) {
    GTEST_SKIP() << "Fix unwinding of tail calls #47666";

    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_tailcall_resolvable.cpp",
        /* present */ {"CALLSTACK_BEGIN", "left_top", "left", "fork_func", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"right", "right_top", "..."});
}

TEST_F(DevicePrintOutputFixture, PrintCallstackTailCallUnresolvable) {
    GTEST_SKIP() << "Fix unwinding of tail calls #47666";

    TestCallstack(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_callstack_tailcall_unresolvable.cpp",
        /* present */ {"CALLSTACK_BEGIN", "leaf", "...", "kernel_main", "CALLSTACK_END"},
        /* absent */ {"left", "right"});
}
