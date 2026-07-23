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

// ============================================================================
// PROBE: NoC atomic operations beyond plain increment.
// ============================================================================
// Verifies that cross-domain atomic DECREMENT and compare-and-swap work on Quasar
// via NoC atomics. The NoC HW defines SWAP/CAS/ACC/RISCV_AMO opcodes but tt-metal
// SW only ever emits INCR_GET; if these pass, "all cross-domain ops atomic" is a
// software deliverable on the EXTERNAL semaphore tier (no register file needed).
//   - TestAtomicDecrementIncrGet : decrement via the EXISTING INCR_GET path
//     (noc_semaphore_inc with incr=-1, wrap=31). Highest-confidence.
//   - TestAtomicDecrementAmo     : decrement via a raw NOC_AT_INS_RISCV_AMO (AMOADD).
//   - TestAtomicCas              : 4-bit compare-and-swap via a raw NOC_AT_INS_CAS.
// ============================================================================
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
            GTEST_SKIP() << "No NoC atomics on Wormhole";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        l1_unreserved_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        if (is_quasar) {
            num_dms_ = std::min(num_dms_, 6u);  // Metal 2.0 reserves DM0/DM1
        }
    }

    // Write init_value to the shared word, run the probe kernel (mode via -D define)
    // on all user DM cores, return the final 32-bit word.
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

// Atomic cross-domain DECREMENT via the existing INCR_GET path (incr=-1, wrap=31).
// All user DMs decrement a word pre-set to num_dms*iterations -> exact 0 proves
// atomic decrement is already reachable through today's noc_semaphore_inc.
TEST_F(NocAtomicOpsFixture, TestAtomicDecrementIncrGet) {
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run("PROBE_DECR_INCRGET", start);
    log_info(LogTest, "INCR_GET decrement: {} (expected 0; started at {})", observed, start);
    EXPECT_EQ(observed, 0u) << "Cross-domain atomic decrement via INCR_GET(-1) lost/added updates.";
}

// Atomic decrement via a raw NOC_AT_INS_RISCV_AMO (AMOADD, operand -1).
// The raw emit uses Quasar-only RoCC builtins / NOC_AT_* symbols, so it is Quasar-only
// (Blackhole has a different NoC emit path). TestAtomicDecrementIncrGet stays portable.
TEST_F(NocAtomicOpsFixture, TestAtomicDecrementAmo) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC RISCV_AMO emit is Quasar-only";
    }
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run("PROBE_DECR_AMO", start);
    log_info(LogTest, "RISCV_AMO AMOADD decrement: {} (expected 0; started at {})", observed, start);
    EXPECT_EQ(observed, 0u) << "Raw NOC_AT_INS_RISCV_AMO AMOADD decrement did not produce exact 0.";
}

// 4-bit compare-and-swap via a raw NOC_AT_INS_CAS. Word starts at 5:
// CAS(cmp=5,swap=9) succeeds -> 9; CAS(cmp=5,swap=2) fails (word is 9) -> unchanged.
TEST_F(NocAtomicOpsFixture, TestAtomicCas) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    const uint32_t observed = run("PROBE_CAS", 5u);
    log_info(LogTest, "CAS result: {} (expected 9: 5->9 on match, then no-op on mismatch)", observed);
    EXPECT_EQ(observed, 9u) << "Raw NOC_AT_INS_CAS did not compare-and-swap as expected.";
}

}  // namespace tt::tt_metal
