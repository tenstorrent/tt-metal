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

// Keystone tests for the auto-path-selecting Quasar semaphore design. On an
// externally-touched semaphore even a DM core's own increment goes through a
// self-targeted NoC atomic so local and remote writers serialize at one NIU --
// and that path has no software fallback: RISC-V AMOs hang on the uncached alias
// (dev_mem_map.h). The DM-local fast path is a 32-bit RISC-V AMO on the cached
// alias. Each test below pins one of these hardware assumptions.
class NocSelfAtomicFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    // A correctness/deadlock check, not a stress test: NoC atomic round-trips are
    // slow on the emu RTL sim. Bump for silicon stress runs.
    static constexpr uint32_t iterations{100};
    const std::string kernel_path_noc = "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_self_atomic.cpp";
    const std::string kernel_path_amo32 = "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_amo32.cpp";
    const std::string kernel_path_cas32 = "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_cas32.cpp";
    const std::string kernel_path_cacheline = "tests/tt_metal/tt_metal/test_kernels/dataflow/dm_cacheline_probe.cpp";
    const std::string kernel_path_cas_drain = "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_self_cas_drain.cpp";
    uint32_t l1_unreserved_base{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    uint32_t num_dms_{0};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        // No fixture-wide Wormhole skip: the pure-NoC tests use no RISC-V AMOs and
        // must run there too; the AMO/cacheline/CAS tests carry per-test skips.
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

    // Launch the given kernel on every user DM core, all targeting the word at
    // `sem_addr` on `core`.
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

    void set_counter(const experimental::NodeCoord& node, uint32_t addr, uint32_t value) {
        std::vector<uint32_t> initial_l1_words(1, value);
        tt::tt_metal::detail::WriteToDeviceL1(device_, node, addr, initial_l1_words);
    }

    void zero_counter(const experimental::NodeCoord& node, uint32_t addr) { set_counter(node, addr, 0); }

    // Launch the single-thread cache-line-width probe on `core`. The probe sweeps from
    // the line-aligned `base_addr` and writes 2 + 3*NUM_SEPS words at `report_addr`:
    // the residency control pair, then wc/wn_pre/wn_post per separation.
    void run_cacheline_probe(uint32_t base_addr, uint32_t report_addr, uint32_t residency_addr) {
        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        const experimental::KernelSpecName DM_KERNEL{"cacheline_probe"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path_cacheline,
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"base_addr", "report_addr", "residency_addr"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "dm_cacheline_probe",
            .kernels = {kernel_spec},
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"base_addr", base_addr}, {"report_addr", report_addr}, {"residency_addr", residency_addr}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);
    }

    // Launch the CAS-lock probe on all user DM threads (Quasar-only: raw CAS emit) and
    // return the final sem word. Scratch from l1_unreserved_base: sem word at +0
    // (preloaded to sem_init), lock word at +16 (its own 16B atom), 8 per-hart pre-op
    // return slots at +32; the lock and every ret slot are zeroed.
    uint32_t run_cas_lock(uint32_t mode, uint32_t pairs, uint32_t sem_init, uint32_t lock_offset = 16) {
        const uint32_t sem_addr = l1_unreserved_base;
        // lock_offset picks the CAS lane: the 4-bit CAS acts on the 32-bit lane
        // (addr>>2)&3 of a 16B row. Production lock words are 16B-spaced (always
        // lane 0); offsets 20/24/28 characterize the other lanes.
        const uint32_t lock_addr = l1_unreserved_base + lock_offset;
        const uint32_t ret_base = l1_unreserved_base + 32;
        std::vector<uint32_t> init(16, 0);  // sem word + lock word + 8 ret slots
        init[0] = sem_init;
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, l1_unreserved_base, init);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        const experimental::KernelSpecName DM_KERNEL{"cas_drain"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path_cas_drain,
            .num_threads = num_dms_,
            .runtime_arg_schema =
                {.runtime_arg_names = {"sem_addr", "lock_addr", "ret_base", "increment_times", "mode", "pairs"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "noc_self_cas_drain",
            .kernels = {kernel_spec},
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core,
                    {{"sem_addr", sem_addr},
                     {"lock_addr", lock_addr},
                     {"ret_base", ret_base},
                     {"increment_times", iterations},
                     {"mode", mode},
                     {"pairs", pairs}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        return read_counter(core, sem_addr);
    }
};

// (1) All user DM cores loopback-increment one shared word via a self-targeted NoC
// atomic. The exact final count proves the loopback INCR_GET neither deadlocks nor
// loses updates across same-node DM cores.
TEST_F(NocSelfAtomicFixture, TestSelfTargetedNocAtomicIncrement) {
    zero_counter(core, l1_unreserved_base);

    run_single_node(kernel_path_noc, l1_unreserved_base);

    const uint32_t observed = read_counter(core, l1_unreserved_base);
    const uint32_t expected = num_dms_ * iterations;
    log_info(
        LogTest,
        "Self-targeted NoC atomic count: {} (expected {} = {} DMs x {})",
        observed,
        expected,
        num_dms_,
        iterations);
    EXPECT_EQ(observed, expected)
        << "Self-targeted NoC atomic increments lost updates: the loopback INCR_GET is NOT mutually "
           "atomic across same-node DM cores. The auto-path EXTERNAL semaphore mode has no software "
           "fallback if this fails (KEYSTONE FAILS).";
}

// (2) A loopback atomic on node_0 and a genuinely-remote atomic from node_1 both
// increment the same word on node_0. Exact 2 * iterations proves the two paths
// serialize at one NIU atomicity point, which EXTERNAL mode requires.
TEST_F(NocSelfAtomicFixture, TestSelfVsRemoteNodeNocAtomic) {
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    log_info(LogTest, "compute-with-storage grid: {} x {}", grid.x, grid.y);
    if (grid.x < 2 && grid.y < 2) {
        GTEST_SKIP() << "Requires >= 2 worker nodes for a genuinely-remote source (this device/emu exposes a 1x1 grid)";
    }
    const experimental::NodeCoord node_0{0, 0};
    const experimental::NodeCoord node_1 =
        (grid.x >= 2) ? experimental::NodeCoord{1, 0} : experimental::NodeCoord{0, 1};

    zero_counter(node_0, l1_unreserved_base);

    // The remote kernel targets the word via node_0's virtual NoC coords.
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
    // Gen1 (Blackhole): a Gen2 config would FATAL in MakeProgramFromSpec; one kernel
    // per node, so RISCV_0 serves both.
    if (!is_quasar) {
        self_spec.hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        remote_spec.hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    }

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

// (3) All user DM cores increment one shared cached word via a 32-bit RISC-V AMO --
// the DM_LOCAL_CACHED fast-path keystone. Only the 64-bit AMO width was proven on
// Quasar before this; semaphore words are 32-bit.
TEST_F(NocSelfAtomicFixture, TestDmCachedAmo32) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Wormhole lacks RISC-V AMOs (amoadd.w)";
    }
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

// (4) All user DM cores conditionally decrement one shared cached word using the
// production down() shape (noc_semaphore.h, DM_LOCAL_CACHED). Draining exactly to 0
// proves no decrement was lost and no credit was consumed twice.
TEST_F(NocSelfAtomicFixture, TestDmCachedCas32) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Wormhole lacks RISC-V AMOs (lr.w/sc.w)";
    }
    if (!is_quasar) {
        GTEST_SKIP() << "lr.w/sc.w needs Zalrsc; Blackhole DM cores are Zaamo-only";
    }
    const uint32_t start = num_dms_ * iterations;
    set_counter(core, l1_unreserved_base, start);

    run_single_node(kernel_path_cas32, l1_unreserved_base);

    const uint32_t observed = read_counter(core, l1_unreserved_base);
    log_info(
        LogTest,
        "32-bit cached CAS drain: {} (expected 0; started at {} = {} DMs x {})",
        observed,
        start,
        num_dms_,
        iterations);
    EXPECT_EQ(observed, 0u) << "lr.w/sc.w conditional decrement LOST updates on the cached L1 alias -- the "
                               "DM_LOCAL_CACHED multi-consumer down() keystone failed. (An over-committed decrement "
                               "cannot show here: the guarded CAS never takes the word below 0, so that presents as "
                               "a HANG with surplus threads spinning.)";
}

// (5) Measures the DM write-back cache line width: the smallest separation at which a
// cached-AMO word's dirty write-back stops clobbering a NoC-atomic word -- the
// separation the DM_LOCAL_CACHED pool segregation must enforce. The controls make a
// platform that does not model the hazard fail rather than yield a bogus small width.
TEST_F(NocSelfAtomicFixture, TestDmCacheLineWidth) {
    if (!is_quasar) {
        GTEST_SKIP() << "cached/uncached alias + write-back cache is Quasar-only";
    }
    // Must match dm_cacheline_probe.cpp exactly.
    constexpr uint32_t NUM_SEPS = 6;
    const uint32_t seps[NUM_SEPS] = {4u, 8u, 16u, 32u, 64u, 128u};
    constexpr uint32_t CACHED_ADD = 5u;
    constexpr uint32_t NOC_ADD = 7u;
    constexpr uint32_t STRIDE = 512u;
    constexpr uint32_t RES_OLD = 0x1111u;
    constexpr uint32_t RES_NEW = 0x2222u;
    constexpr int EXPECTED_WIDTH = 64;  // documented L1 D$ / L2 line (risc_common.h)

    // Line-aligned base (base+sep shares a line with base iff sep < width); report and
    // residency scratch sit well past the swept region.
    const uint32_t base = (l1_unreserved_base + 511u) & ~511u;
    const uint32_t report = base + NUM_SEPS * STRIDE + 1024u;
    const uint32_t residency = report + 1024u;

    run_cacheline_probe(base, report, residency);

    const uint32_t total = 2u + 3u * NUM_SEPS;
    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report, total * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), total);

    // Control A: prove the platform actually models a write-back cache; otherwise the
    // clobber hazard cannot be measured here at all.
    const uint32_t res_noflush = result[0];
    const uint32_t res_flushed = result[1];
    ASSERT_EQ(res_flushed, RES_NEW)
        << "Control A: flush_l2_cache_line did not write the cached value back to TL1 -> cache/flush not "
           "functional on this platform.";
    ASSERT_EQ(res_noflush, RES_OLD)
        << "Control A: a cached-alias write was visible via the uncached alias WITHOUT a flush (saw 0x" << std::hex
        << res_noflush << ", expected 0x" << RES_OLD << std::dec
        << ") -> this platform is NOT modeling a write-back cache (flat/coherent/write-through). The "
           "write-back-clobber hazard cannot be measured here; verify the DM cache line width on silicon.";

    // Per-separation sweep.
    int width = -1;
    bool seen_clobber = false;
    bool seen_safe = false;
    bool monotonic = true;
    for (uint32_t i = 0; i < NUM_SEPS; i++) {
        const uint32_t wc_val = result[2 + 3 * i + 0];
        const uint32_t wn_pre = result[2 + 3 * i + 1];
        const uint32_t wn_post = result[2 + 3 * i + 2];
        const bool safe = (wn_post == NOC_ADD);
        const bool clobbered = (wn_post == 0u);
        log_info(
            LogTest,
            "sep={:>4}B  wc={} (exp {})  wn_pre={} (exp {})  wn_post={}  -> {}",
            seps[i],
            wc_val,
            CACHED_ADD,
            wn_pre,
            NOC_ADD,
            wn_post,
            safe ? "SAFE (different line)" : (clobbered ? "CLOBBERED (shared line)" : "UNEXPECTED"));
        EXPECT_EQ(wc_val, CACHED_ADD) << "sep=" << seps[i] << "B: kernel liveness (cached AMO) failed";
        // Control B: the atomic must have landed pre-flush, else a post-flush 0 is meaningless.
        ASSERT_EQ(wn_pre, NOC_ADD) << "sep=" << seps[i]
                                   << "B: NoC atomic did not land at TL1 pre-flush (wn_pre=" << wn_pre
                                   << ") -> sample invalid, cannot distinguish clobber from never-landed.";
        EXPECT_TRUE(safe || clobbered) << "sep=" << seps[i] << "B: wn_post=" << wn_post << " is neither " << NOC_ADD
                                       << " nor 0";
        if (clobbered) {
            seen_clobber = true;
            if (seen_safe) {
                monotonic = false;  // a clobber after a safe result breaks same-line-iff-sep<width
            }
        } else if (safe) {
            seen_safe = true;
            if (width < 0) {
                width = static_cast<int>(seps[i]);
            }
        }
    }

    // Positive control: the smallest separation shares wc's line and must clobber, else
    // the clobber mechanism is not live here and the width is invalid.
    ASSERT_TRUE(seen_clobber)
        << "POSITIVE CONTROL FAILED: no separation was clobbered across the sweep -> the DM write-back-clobber "
           "hazard is not live on this platform (coherent NIU, or the emu is not modeling write-back + "
           "NoC-incoherence). The width is INVALID; do NOT trust it -- verify on silicon.";
    ASSERT_EQ(result[2 + 3 * 0 + 2], 0u)
        << "POSITIVE CONTROL FAILED: sep=4B (same cache line) did NOT clobber -> the write-back-clobber hazard "
           "is not live here. Width INVALID; verify on silicon.";

    EXPECT_TRUE(monotonic)
        << "clobber->safe transition is non-monotonic across the sweep: the 'shared cache line iff sep < width' "
           "model does not hold -- investigate before trusting any width.";
    ASSERT_GE(width, 0)
        << "even sep=128B was clobbered: the DM write-back line width exceeds the FLUSH64-detectable range; use "
           "the natural-eviction variant or measure on silicon.";
    log_info(LogTest, "==> DM write-back cache line width (min NoC-safe separation) = {} B", width);
    EXPECT_EQ(width, EXPECTED_WIDTH)
        << "measured DM write-back line width (" << width
        << "B) != documented 64B (L1 D$/L2 line). If this is "
           "real (not an artifact), the pool segregation must use the measured value -- investigate.";
}

// (6a) All user DM threads drain one word through the CAS-lock-protected
// check-then-decrement (the production multi-consumer EXTERNAL down() body).
// Exact 0 proves the lock grants exclusively under full contention.
TEST_F(NocSelfAtomicFixture, TestSelfCasLockDrain) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run_cas_lock(0 /*mode: pure drain*/, 0 /*pairs unused*/, start);
    log_info(
        LogTest, "CAS-lock drain: {} (expected 0; started at {} = {} DMs x {})", observed, start, num_dms_, iterations);
    EXPECT_EQ(observed, 0u)
        << "The NoC CAS lock lost or double-granted around the decrement (a double grant on the LAST "
           "credit wraps the word to the ret-slot sentinel and presents as a HANG instead) -- EXTERNAL "
           "multi-consumer "
           "down() cannot be enabled.";
}

// (6b) Same drain with the lock word at CAS lanes 1 and 3 (see run_cas_lock). Not a
// production path (locks are 16B-spaced, lane 0 -- MEM_NOC_SEM_LOCK_SIZE); pins the
// lane encoding.
TEST_F(NocSelfAtomicFixture, TestSelfCasLockDrainLanes13) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    for (const uint32_t lane : {1u, 3u}) {
        const uint32_t start = num_dms_ * iterations;
        const uint32_t observed = run_cas_lock(0 /*mode: pure drain*/, 0, start, /*lock_offset=*/16 + lane * 4);
        log_info(LogTest, "CAS-lock drain lane {}: {} (expected 0; started at {})", lane, observed, start);
        EXPECT_EQ(observed, 0u) << "CAS lane " << lane << " (lock word at 16B-row offset " << lane * 4
                                << ") lost or double-granted";
    }
}

// (6c) Lane-2 anomaly, characterization only: with the lock at CAS lane 2 the drain
// makes no progress and the word reads 0x80000000 | initial_count on emu RTL (lanes
// 0/1/3 are clean) -- the reason production lock words are 16B-spaced. Run manually
// (--gtest_also_run_disabled_tests) to re-characterize on silicon; if all lanes prove
// clean there, the lock region can shrink back to 4B packing.
TEST_F(NocSelfAtomicFixture, DISABLED_TestSelfCasLockLane2Anomaly) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    const uint32_t start = num_dms_ * iterations;
    const uint32_t observed = run_cas_lock(0 /*mode: pure drain*/, 0, start, /*lock_offset=*/16 + 8);
    log_info(LogTest, "CAS-lock drain lane 2: {} (expected 0; started at {})", observed, start);
    EXPECT_EQ(observed, 0u) << "lane-2 CAS anomaly reproduced (observed " << observed << ")";
}

// (6d) Producer/consumer hart pairs on one word: even harts do plain INCR_GET up()s
// (no lock), odd harts do lock-protected decrements. Exact 0 proves CAS and INCR_GET
// serialize at the NIU -- the real up()/down() mix, since producers never take the lock.
TEST_F(NocSelfAtomicFixture, TestSelfCasLockVsIncr) {
    if (!is_quasar) {
        GTEST_SKIP() << "raw NoC CAS emit is Quasar-only";
    }
    const uint32_t pairs = num_dms_ / 2;
    if (pairs == 0) {
        GTEST_SKIP() << "needs at least one producer/consumer hart pair";
    }
    const uint32_t observed = run_cas_lock(1 /*mode: mixed*/, pairs, 0);
    log_info(LogTest, "CAS-lock vs INCR_GET mix: {} (expected 0; {} pairs x {})", observed, pairs, iterations);
    EXPECT_EQ(observed, 0u)
        << "CAS and INCR_GET did not serialize mutually at the NIU: a locked decrement raced a plain "
           "producer increment -- EXTERNAL multi-consumer down() cannot coexist with up().";
}

}  // namespace tt::tt_metal
