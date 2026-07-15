// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

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
#include "impl/context/metal_context.hpp"
#include "common/mesh_dispatch_fixture.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

// These Tests verify built in gcc atomics on Tensix RISCV processors
// https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html

class RISCVAtomicsFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    static constexpr uint32_t SENTINEL{0xABABABAB};
    static constexpr uint32_t iterations{2000};
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/riscv_atomics.cpp";
    uint32_t l1_unreserved_base{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    uint32_t num_dms_{0};
    std::vector<uint32_t> result;
    uint32_t word_count_32b{0};

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        // No atomics support on WH
        if (arch_ == tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "RISCV Atomics not supported on Wormhole";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        l1_unreserved_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        if (is_quasar) {
            // Metal 2.0 reserves DM0/DM1 for runtime; user kernels get at most 6 threads
            num_dms_ = std::min(num_dms_, 6u);
        }
        // Number of uint32_t words per atomic_type: Quasar uses 64-bit atomics (2 words), BH uses 32-bit (1 word)
        word_count_32b = is_quasar ? 2 : 1;
    }

    void run_riscv_atomics_test(const std::map<std::string, std::string>& dm_defines) {
        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::KernelSpec::CompilerOptions::Defines defines_vec(dm_defines);

        // Quasar: one multi-threaded DM kernel spans the user DMs (DM2..DM7).
        // Gen1 (BH): one KernelSpec per DM processor (BRISC, NCRISC), each single-threaded with its
        // own NOC binding. The kernel itself picks its role via mhartid (Quasar) or COMPILE_FOR_BRISC
        // (Gen1) — see riscv_atomics.cpp.
        std::vector<experimental::KernelSpec> kernel_specs;
        std::vector<experimental::KernelSpecName> kernel_names;
        experimental::ProgramRunArgs params;
        const auto make_run_params = [&](const experimental::KernelSpecName& kernel_name) {
            return experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = kernel_name,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"l1_counter_addr", l1_unreserved_base}, {"increment_times", iterations}}),
            };
        };

        if (is_quasar) {
            const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = DM_KERNEL,
                .source = kernel_path,
                .num_threads = num_dms_,
                .compiler_options = {.defines = defines_vec},
                .runtime_arg_schema = {.runtime_arg_names = {"l1_counter_addr", "increment_times"}},
                .hw_config = experimental::DataMovementGen2Config{},
            });
            kernel_names.push_back(DM_KERNEL);
            params.kernel_run_args.push_back(make_run_params(DM_KERNEL));
        } else {
            for (uint32_t dm_id = 0; dm_id < num_dms_; dm_id++) {
                const experimental::KernelSpecName name{"dm_kernel_" + std::to_string(dm_id)};
                kernel_specs.push_back(experimental::KernelSpec{
                    .unique_id = name,
                    .source = kernel_path,
                    .num_threads = 1,
                    .compiler_options = {.defines = defines_vec},
                    .runtime_arg_schema = {.runtime_arg_names = {"l1_counter_addr", "increment_times"}},
                    .hw_config =
                        experimental::DataMovementGen1Config{
                            .processor = static_cast<tt_metal::DataMovementProcessor>(dm_id),
                            .noc = (dm_id == 1 ? NOC::RISCV_1_default : NOC::RISCV_0_default),
                        },
                });
                kernel_names.push_back(name);
                params.kernel_run_args.push_back(make_run_params(name));
            }
        }

        experimental::WorkUnitSpec main_wu{
            .name = "main",
            .kernels = kernel_names,
            .target_nodes = core,
        };
        experimental::ProgramSpec spec{
            .name = "riscv_atomics",
            .kernels = kernel_specs,
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);
    }
};

// The lowest-numbered DM thread (writer) atomically stores SENTINEL to L1,
// remaining DM thread(s) (readers) spin on atomic load until they see it.
// Host reads back each reader's result slot and verifies SENTINEL was observed.
TEST_F(RISCVAtomicsFixture, TestAtomicLoadStoreRISCV) {
    std::map<std::string, std::string> dm_defines = {
        {"TEST_ATOMIC_LOAD_STORE", "1"}, {"SENTINEL", std::to_string(SENTINEL)}};

    // Number of slots used in L1:
    // 1 shared_value slot + (num_dms - 1) reader result slots
    const uint32_t l1_slot_count = num_dms_;

    // Zero-Init the L1 atomics scratch space
    std::vector<uint32_t> initial_l1_words(l1_slot_count * word_count_32b, 0);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, initial_l1_words);

    // Run the test
    run_riscv_atomics_test(dm_defines);

    // Skip shared_value_ptr slot; read only consumer result slots
    constexpr uint32_t l1_offset = 1;
    const uint32_t read_addr = l1_unreserved_base + (l1_offset * sizeof(uint32_t) * word_count_32b);
    // Read back words written by all DMs to verify atomic load/store went through properly
    const uint32_t read_size_bytes = sizeof(uint32_t) * (num_dms_ - 1) * word_count_32b;

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, read_addr, read_size_bytes, result);
    ASSERT_EQ(result.size(), (num_dms_ - 1) * word_count_32b);

    // Quasar publishes a 64-bit result, while BH reads back a 32-bit result
    if (is_quasar) {
        // Read from all DMs
        for (uint32_t i = 0; i < result.size() - 1; i += word_count_32b) {
            const uint64_t observed = static_cast<uint64_t>(result[i]) | (static_cast<uint64_t>(result[i + 1]) << 32);
            log_info(LogTest, "Result: {} (expected: {})", observed, SENTINEL);
            EXPECT_EQ(observed, SENTINEL) << "Kernel should have observed the producer value";
        }
    } else {
        log_info(LogTest, "Result: {} (expected: {})", result[0], SENTINEL);
        EXPECT_EQ(result[0], SENTINEL) << "Kernel should have observed the producer value";
    }
}

// All DMs atomically increment a shared L1 counter 2000 times each.
// Host verifies final count == num_dms * 2000 to confirm atomic increments
TEST_F(RISCVAtomicsFixture, TestAtomicAddFetchRISCV) {
    std::map<std::string, std::string> dm_defines = {{"TEST_ATOMIC_ADD_FETCH", "1"}};

    // Zero-Init the L1 atomics scratch space
    std::vector<uint32_t> initial_l1_words(word_count_32b, 0);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, initial_l1_words);

    const uint32_t expected_result = num_dms_ * iterations;
    const uint32_t read_size_bytes = sizeof(uint32_t) * word_count_32b;

    // Run the test
    run_riscv_atomics_test(dm_defines);

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, l1_unreserved_base, read_size_bytes, result);
    ASSERT_EQ(result.size(), word_count_32b);

    // Quasar publishes a 64-bit result, while BH reads back a 32-bit result
    uint64_t observed = result[0];

    if (is_quasar) {
        observed |= static_cast<uint64_t>(result[1]) << 32;
    }

    log_info(LogTest, "Result: {} (expected: {})", observed, static_cast<uint64_t>(expected_result));
    EXPECT_EQ(observed, static_cast<uint64_t>(expected_result))
        << "Kernel should have performed " << expected_result << " atomic additions";
}

// All DMs CAS increment a shared L1 counter 2000 times each. Zalrsc extension only supported on Quasar.
// Host verifies final count == num_dms * 2000 to confirm atomic CAS increments
TEST_F(RISCVAtomicsFixture, TestAtomicCASRISCV) {
    if (!is_quasar) {
        GTEST_SKIP() << "RISCV CAS Atomics only supported on Quasar";
    }

    std::map<std::string, std::string> dm_defines = {{"TEST_ATOMIC_CAS", "1"}};

    // Zero-Init the L1 atomics scratch space
    std::vector<uint32_t> initial_l1_words(word_count_32b, 0);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, initial_l1_words);

    const uint64_t expected_result = num_dms_ * iterations;
    const uint32_t read_size_bytes = sizeof(uint32_t) * word_count_32b;

    // Run the test
    run_riscv_atomics_test(dm_defines);

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, l1_unreserved_base, read_size_bytes, result);
    ASSERT_EQ(result.size(), word_count_32b);

    // Quasar publishes a 64-bit result
    const uint64_t result64 = static_cast<uint64_t>(result[0]) | (static_cast<uint64_t>(result[1]) << 32);
    log_info(LogTest, "Result: {} (expected: {})", result64, expected_result);
    EXPECT_EQ(result64, expected_result) << "Kernel should have performed " << expected_result << " atomic additions";
}
}  // namespace tt::tt_metal
