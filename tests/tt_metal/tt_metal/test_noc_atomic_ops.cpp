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
// Tests that exercise the NoC atomic decrement and the 4-bit
// compare-and-swap (noc_fast_atomic_cas4) that EXTERNAL down() builds on.
class NocAtomicOpsFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    static constexpr uint32_t iterations{50};
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_atomic_ops_probe.cpp";
    uint32_t l1_unreserved_base{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    uint32_t num_dms_{0};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (arch_ == tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Probes target Blackhole/Quasar (Wormhole lacks RISC-V AMOs)";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        l1_unreserved_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        if (is_quasar) {
            num_dms_ = std::min(num_dms_, 6u);
        }
    }

    // Writes init_value to the shared word, runs the kernel in the given mode
    // on all user DM cores, and returns the final word.
    uint32_t run(const std::string& mode_define, uint32_t init_value) {
        std::vector<uint32_t> init{init_value};
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, init);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        std::vector<experimental::KernelSpec> kernel_specs;
        std::vector<experimental::KernelSpecName> kernel_names;
        experimental::ProgramRunArgs params;
        const auto make_run_params = [&](const experimental::KernelSpecName& name) {
            return experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = name,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"sem_addr", l1_unreserved_base}, {"increment_times", iterations}}),
            };
        };

        if (is_quasar) {
            const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = DM_KERNEL,
                .source = kernel_path,
                .num_threads = num_dms_,
                .compiler_options = {.defines = {{mode_define, "1"}}},
                .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "increment_times"}},
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
                    .compiler_options = {.defines = {{mode_define, "1"}}},
                    .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "increment_times"}},
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

        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = kernel_names, .target_nodes = core};
        experimental::ProgramSpec spec{.name = "noc_atomic_ops", .kernels = kernel_specs, .work_units = {main_wu}};
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, l1_unreserved_base, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        return result.empty() ? 0u : result[0];
    }
};

// All user DMs atomically decrement a word pre-set to num_dms*iterations; an
// exact zero proves atomic decrement works through the noc_semaphore_inc (INCR_GET) path.
TEST_F(NocAtomicOpsFixture, TestAtomicDecrementIncrGet) {
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run("PROBE_DECR_INCRGET", start);
    log_info(LogTest, "INCR_GET decrement: {} (expected 0; started at {})", observed, start);
    EXPECT_EQ(observed, 0u) << "Cross-domain atomic decrement via INCR_GET(-1) lost/added updates.";
}

// Same decrement shape, but through a raw RISCV_AMO emit.
TEST_F(NocAtomicOpsFixture, TestAtomicDecrementAmo) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC RISCV_AMO emit is Quasar-only";
    }
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run("PROBE_DECR_AMO", start);
    log_info(LogTest, "RISCV_AMO AMOADD decrement: {} (expected 0; started at {})", observed, start);
    EXPECT_EQ(observed, 0u) << "Raw NOC_AT_INS_RISCV_AMO AMOADD decrement did not produce exact 0.";
}

// A matching 4-bit CAS moves the word 5 -> 9, then a mismatched CAS must leave it at 9.
TEST_F(NocAtomicOpsFixture, TestAtomicCas) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    const uint32_t observed = run("PROBE_CAS", 5u);
    log_info(LogTest, "CAS result: {} (expected 9: 5->9 on match, then no-op on mismatch)", observed);
    EXPECT_EQ(observed, 9u) << "Raw NOC_AT_INS_CAS did not compare-and-swap as expected.";
}

// noc_fast_atomic_cas4 must return the pre-op word on success and on failure
TEST_F(NocAtomicOpsFixture, TestAtomicCasReturnsPreOpValue) {
    if (!is_quasar) {
        GTEST_SKIP() << "noc_fast_atomic_cas4 (RoCC builtin emit) is Quasar-only";
    }
    constexpr uint32_t WORD2_OFF = 16u;
    constexpr uint32_t REPORT_OFF = 128u;
    constexpr uint32_t REPORT_WORDS = 7u;

    std::vector<uint32_t> word2_init{0x15u};
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base + WORD2_OFF, word2_init);
    std::vector<uint32_t> report_init(REPORT_WORDS, 0u);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base + REPORT_OFF, report_init);

    run("PROBE_CAS_RET", 5u);

    tt::tt_metal::detail::ReadFromDeviceL1(
        device_, core, l1_unreserved_base + REPORT_OFF, REPORT_WORDS * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), REPORT_WORDS);
    log_info(
        LogTest,
        "CAS return probe: success imm/polled={:#x}/{:#x}, fail imm/polled={:#x}/{:#x}, word={:#x}, "
        "word2 polled/word={:#x}/{:#x}",
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6]);

    EXPECT_EQ(result[1], 5u)
        << "successful CAS did not return the pre-op word (expected 5): a CAS winner cannot confirm "
           "what it swapped out, so cas4's return path is unusable for a lock/down() upgrade.";
    EXPECT_EQ(result[3], 9u) << "FAILED CAS did not return the pre-op word (expected 9): a CAS loser cannot learn the "
                                "current value, so a CAS retry loop cannot be built on cas4 returns.";
    EXPECT_EQ(result[4], 9u) << "failed CAS modified the target word (expected 9 unchanged): CAS failure is not "
                                "side-effect-free, so any concurrent use of cas4 corrupts the word.";
    EXPECT_EQ(result[5], 0x15u)
        << "CAS on a word with upper-28 bits set did not return the pre-op word (expected 0x15): the "
           "return path is unreliable outside the [0,15] value range.";
    EXPECT_EQ(result[6], 0x15u)
        << "CAS(cmp=5) on 0x15 changed the word: HW ignored the word[31:4]==0 success condition, so "
           "the 4-bit CAS is NOT safe next to words that can exceed 15.";
    EXPECT_EQ(result[0], result[1])
        << "noc_async_atomic_barrier does NOT order the CAS return-value write; the sentinel-poll in "
           "any consumer of cas4 returns is REQUIRED, not optional.";
    EXPECT_EQ(result[2], result[3])
        << "noc_async_atomic_barrier does NOT order the CAS return-value write; the sentinel-poll in "
           "any consumer of cas4 returns is REQUIRED, not optional.";
}

}  // namespace tt::tt_metal
