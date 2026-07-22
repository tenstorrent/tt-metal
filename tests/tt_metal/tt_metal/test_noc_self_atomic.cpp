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
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "common/mesh_dispatch_fixture.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

// ============================================================================
// KEYSTONE TESTS for the auto-path-selecting Quasar semaphore design.
// ============================================================================
//
// The design routes every access to an externally-touched semaphore -- INCLUDING
// a DM core's own local increment -- through a NoC atomic (NOC_AT_INS_INCR_GET) so
// local and remote writers serialize at one NIU atomicity point. RISC-V AMOs
// cannot substitute for the local case (they hang on the uncached alias,
// dev_mem_map.h:34-35), so the self-targeted NoC atomic is the ONLY mechanism --
// with NO software fallback. The DM-local fast path, in turn, relies on a 32-bit
// RISC-V AMO on the cached alias.
//
// These tests establish the load-bearing hardware behaviors before any of the
// design is built:
//   1. TestSelfTargetedNocAtomicIncrement -- self-targeted (loopback) NoC atomic
//      works, does not deadlock, and serializes across same-node DM cores.
//   2. TestSelfVsRemoteNodeNocAtomic      -- a self-targeted (loopback) atomic and
//      a genuinely-remote (other-node) atomic to the same word are mutually atomic.
//   3. TestDmCachedAmo32                   -- a 32-bit RISC-V AMO (amoadd.w) on the
//      cached L1 alias is correct (the DM_LOCAL_CACHED fast-path keystone).
// ============================================================================
class NocSelfAtomicFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    // Keystone is a correctness/deadlock check, not a stress test. NoC atomic
    // round-trips are slow on emu RTL sim; bump this for silicon stress runs.
    static constexpr uint32_t iterations{100};
    const std::string kernel_path_noc =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_self_atomic.cpp";
    const std::string kernel_path_amo32 =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_amo32.cpp";
    uint32_t l1_unreserved_base{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    uint32_t num_dms_{0};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        // Wormhole has no NoC/RISC-V atomics; these are Quasar/Blackhole features.
        if (arch_ == tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "No atomics on Wormhole";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        l1_unreserved_base = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        if (is_quasar) {
            // Metal 2.0 reserves DM0/DM1 for runtime; user kernels get at most 6 threads.
            num_dms_ = std::min(num_dms_, 6u);
        }
    }

    // Launch one incrementer per user DM core, all targeting the word at `sem_addr`
    // on `core`. Quasar: a single multi-threaded DM kernel spans the user DMs
    // (DM2..DM7). Gen1 (BH): one single-threaded KernelSpec per DM processor.
    void run_single_node(const std::string& kernel_src, uint32_t sem_addr) {
        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        std::vector<experimental::KernelSpec> kernel_specs;
        std::vector<experimental::KernelSpecName> kernel_names;
        experimental::ProgramRunArgs params;
        const auto make_run_params = [&](const experimental::KernelSpecName& kernel_name) {
            return experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = kernel_name,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"sem_addr", sem_addr}, {"increment_times", iterations}}),
            };
        };

        if (is_quasar) {
            const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = DM_KERNEL,
                .source = kernel_src,
                .num_threads = num_dms_,
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
                    .source = kernel_src,
                    .num_threads = 1,
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

        experimental::WorkUnitSpec main_wu{
            .name = "main",
            .kernels = kernel_names,
            .target_nodes = core,
        };
        experimental::ProgramSpec spec{
            .name = "atomic_keystone",
            .kernels = kernel_specs,
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);
    }

    uint32_t read_counter(const experimental::NodeCoord& node, uint32_t addr) {
        tt::tt_metal::detail::ReadFromDeviceL1(device_, node, addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        return result.empty() ? 0u : result[0];
    }

    void zero_counter(const experimental::NodeCoord& node, uint32_t addr) {
        std::vector<uint32_t> initial_l1_words(1, 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_, node, addr, initial_l1_words);
    }
};

// (1) All user DM cores loopback-increment ONE shared word via self-targeted NoC
// atomic. Exact final count == num_dms * iterations proves the self-targeted
// INCR_GET does not deadlock and same-node NoC atomics from independent DM cores
// serialize atomically at the destination NIU (no lost updates).
TEST_F(NocSelfAtomicFixture, TestSelfTargetedNocAtomicIncrement) {
    zero_counter(core, l1_unreserved_base);

    run_single_node(kernel_path_noc, l1_unreserved_base);

    const uint32_t observed = read_counter(core, l1_unreserved_base);
    const uint32_t expected = num_dms_ * iterations;
    log_info(
        LogTest, "Self-targeted NoC atomic count: {} (expected {} = {} DMs x {})", observed, expected, num_dms_, iterations);
    EXPECT_EQ(observed, expected)
        << "Self-targeted NoC atomic increments lost updates: the loopback INCR_GET is NOT mutually "
           "atomic across same-node DM cores. The auto-path EXTERNAL semaphore mode has no software "
           "fallback if this fails (KEYSTONE FAILS).";
}

// (2) A self-targeted (loopback) atomic on node_0 and a genuinely-remote atomic
// from node_1 both increment the SAME word on node_0. Exact final count ==
// 2 * iterations proves the two arrive as independent atomic requests that
// serialize at node_0's NIU -- i.e. the local (loopback) and external (remote)
// paths converge on one atomicity point, which the EXTERNAL mode requires.
TEST_F(NocSelfAtomicFixture, TestSelfVsRemoteNodeNocAtomic) {
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    log_info(LogTest, "compute-with-storage grid: {} x {}", grid.x, grid.y);
    if (grid.x < 2 && grid.y < 2) {
        GTEST_SKIP() << "Requires >= 2 worker nodes for a genuinely-remote source (this device/emu exposes a 1x1 grid)";
    }
    const experimental::NodeCoord node_0{0, 0};
    // Pick the second node along whichever axis has room.
    const experimental::NodeCoord node_1 =
        (grid.x >= 2) ? experimental::NodeCoord{1, 0} : experimental::NodeCoord{0, 1};

    // Shared counter word lives on node_0.
    zero_counter(node_0, l1_unreserved_base);

    // node_1's kernel needs node_0's virtual NoC coords to target the word remotely.
    const CoreCoord node_0_virtual = mesh_device_->worker_core_from_logical_core(node_0);

    distributed::MeshWorkload workload;
    Program program;
    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

    const experimental::KernelSpecName SELF_KERNEL{"self_kernel"};
    const experimental::KernelSpecName REMOTE_KERNEL{"remote_kernel"};

    experimental::KernelSpec self_spec{
        .unique_id = SELF_KERNEL,
        .source = kernel_path_noc,
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "increment_times"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec remote_spec{
        .unique_id = REMOTE_KERNEL,
        .source = kernel_path_noc,
        .num_threads = 1,
        .compiler_options = {.defines = {{"REMOTE_TARGET", "1"}}},
        .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "increment_times", "remote_noc_x", "remote_noc_y"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec wu_0{.name = "wu_0", .kernels = {SELF_KERNEL}, .target_nodes = node_0};
    experimental::WorkUnitSpec wu_1{.name = "wu_1", .kernels = {REMOTE_KERNEL}, .target_nodes = node_1};

    experimental::ProgramSpec spec{
        .name = "noc_self_vs_remote_atomic",
        .kernels = {self_spec, remote_spec},
        .work_units = {wu_0, wu_1},
    };
    program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = SELF_KERNEL,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node_0, {{"sem_addr", l1_unreserved_base}, {"increment_times", iterations}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = REMOTE_KERNEL,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node_1,
                {{"sem_addr", l1_unreserved_base},
                 {"increment_times", iterations},
                 {"remote_noc_x", static_cast<uint32_t>(node_0_virtual.x)},
                 {"remote_noc_y", static_cast<uint32_t>(node_0_virtual.y)}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device_, workload);

    const uint32_t observed = read_counter(node_0, l1_unreserved_base);
    const uint32_t expected = 2 * iterations;  // node_0 self + node_1 remote
    log_info(LogTest, "Self+remote NoC atomic count: {} (expected {} = 2 x {})", observed, expected, iterations);
    EXPECT_EQ(observed, expected)
        << "A self-targeted (node_0) and a genuinely-remote (node_1) NoC atomic to the same word are "
           "NOT mutually atomic at the destination NIU.";
}

// (3) All user DM cores increment ONE shared cached word via a 32-bit RISC-V AMO
// (amoadd.w). Exact final count == num_dms * iterations proves 32-bit AMOs on L1
// are reliable -- the DM_LOCAL_CACHED fast-path keystone. (The only AMO width
// proven on Quasar today is 64-bit; semaphore words are 32-bit.)
TEST_F(NocSelfAtomicFixture, TestDmCachedAmo32) {
    zero_counter(core, l1_unreserved_base);

    run_single_node(kernel_path_amo32, l1_unreserved_base);

    const uint32_t observed = read_counter(core, l1_unreserved_base);
    const uint32_t expected = num_dms_ * iterations;
    log_info(
        LogTest, "32-bit cached AMO count: {} (expected {} = {} DMs x {})", observed, expected, num_dms_, iterations);
    EXPECT_EQ(observed, expected)
        << "32-bit RISC-V AMO (amoadd.w) on the cached L1 alias lost updates: the DM_LOCAL_CACHED fast "
           "path needs a 32-bit CAS loop or a 64-bit word instead.";
}

}  // namespace tt::tt_metal
