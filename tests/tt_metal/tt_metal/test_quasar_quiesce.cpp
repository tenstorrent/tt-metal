// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "llrt/hal.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device_quiesce.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/data_movement_hardware_config.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace tt::tt_metal {
namespace {

constexpr const char* kSimpleL1WriteKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp";
constexpr const char* kTaggedNocQuiesceKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/quasar_quiesce_tagged_noc.cpp";

class QuasarDeviceQuiesceFixture : public AnyDispatchMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        this->DetectDispatchMode();
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (this->arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Quasar-only quiesce test suite";
        }
        if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
            GTEST_SKIP() << "Device quiesce tests run under simulator";
        }
        if (MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
            GTEST_SKIP() << "Device quiesce tests exercise RISC-V firmware loops (not tt-emule)";
        }
        this->create_devices();
        init_max_dfbs();
    }

    void TearDown() override {
        const bool parked = cores_parked_;
        for (auto& [device_id, dev] : id_to_device_) {
            if (dev && dev->is_initialized()) {
                dev->close();
            }
            dev.reset();
        }
        id_to_device_.clear();
        devices_.clear();
        if (parked || HasFailure() ||
            MetalContext::instance().quiesce_context_state() != MetalContext::QuiesceContextState::None) {
            // Simulate external AI-IP soft-reset between GTest cases in the same simulator process:
            // clear QuiesceContextState so teardown() runs reset_cores() and leaves a clean slate.
            MetalContext::instance().set_quiesce_context_state(MetalContext::QuiesceContextState::None);
            MetalContext::instance().teardown();
        }
        cores_parked_ = false;
    }

    ChipId chip_id() { return this->device().get_device_ids()[0]; }

    HalProgrammableCoreType programmable_type_for_virtual_core(const CoreCoord& virtual_core) {
        return this->device().get_programmable_core_type(virtual_core);
    }

    dev_msgs::quiesce_msg_t read_quiesce_slot_virtual(const CoreCoord& virtual_core) {
        auto& ctx = MetalContext::instance();
        const auto prog_type = programmable_type_for_virtual_core(virtual_core);
        const uint64_t addr = ctx.hal().get_dev_addr(prog_type, HalL1MemAddrType::QUIESCE);
        auto msg = ctx.hal().get_dev_msgs_factory(prog_type).create<dev_msgs::quiesce_msg_t>();
        ctx.get_cluster().read_core(msg.data(), msg.size(), tt_cxy_pair(chip_id(), virtual_core), addr);
        ctx.get_cluster().advance_device_execution(chip_id());
        return msg;
    }

    dev_msgs::quiesce_msg_t read_worker_quiesce_slot(const CoreCoord& logical_worker) {
        const CoreCoord virtual_core = this->device().worker_core_from_logical_core(logical_worker);
        return read_quiesce_slot_virtual(virtual_core);
    }

    void write_quiesce_control_virtual(
        const CoreCoord& virtual_core,
        uint32_t control_val,
        std::optional<uint32_t> target_op_count_slot0 = std::nullopt) {
        auto& ctx = MetalContext::instance();
        const auto prog_type = programmable_type_for_virtual_core(virtual_core);
        const auto factory = ctx.hal().get_dev_msgs_factory(prog_type);
        const uint64_t base_addr = ctx.hal().get_dev_addr(prog_type, HalL1MemAddrType::QUIESCE);

        if (target_op_count_slot0.has_value()) {
            const uint64_t target_addr =
                base_addr + factory.offset_of<dev_msgs::quiesce_msg_t>(dev_msgs::quiesce_msg_t::Field::target_op_count);
            const uint32_t target_val = *target_op_count_slot0;
            ctx.get_cluster().write_core(
                &target_val, sizeof(target_val), tt_cxy_pair(chip_id(), virtual_core), target_addr);
            ctx.get_cluster().l1_barrier(chip_id());
        }

        const uint64_t ctrl_addr =
            base_addr + factory.offset_of<dev_msgs::quiesce_msg_t>(dev_msgs::quiesce_msg_t::Field::control);
        ctx.get_cluster().write_core(
            &control_val, sizeof(control_val), tt_cxy_pair(chip_id(), virtual_core), ctrl_addr);
        ctx.get_cluster().l1_barrier(chip_id());
        cores_parked_ = true;
    }

    void write_worker_quiesce_control(const CoreCoord& logical_worker, uint32_t control_val) {
        const CoreCoord virtual_core = this->device().worker_core_from_logical_core(logical_worker);
        write_quiesce_control_virtual(virtual_core, control_val);
    }

    void wait_for_quiesced_virtual(
        const CoreCoord& virtual_core, std::chrono::milliseconds timeout = std::chrono::milliseconds(15000)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            auto msg = read_quiesce_slot_virtual(virtual_core);
            if (msg.view().status() == dev_msgs::QUIESCE_STATUS_QUIESCED) {
                return;
            }
        }
        auto final_msg = read_quiesce_slot_virtual(virtual_core);
        FAIL() << "Timed out waiting for virtual core " << virtual_core.str() << " to reach QUIESCE_STATUS_QUIESCED; "
               << "last status=0x" << std::hex << final_msg.view().status() << " control=0x"
               << final_msg.view().control();
    }

    void wait_for_worker_quiesced(const CoreCoord& logical_worker) {
        wait_for_quiesced_virtual(this->device().worker_core_from_logical_core(logical_worker));
    }

    distributed::MeshWorkload make_simple_write_workload(
        const NodeCoord& node, uint32_t address, uint32_t value, const std::string& tag) {
        distributed::MeshWorkload wl;
        distributed::MeshCoordinateRange device_range(this->device().shape());
        const KernelSpecName kName{tag};
        KernelSpec spec_k{
            .unique_id = kName,
            .source = kSimpleL1WriteKernel,
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = DataMovementHardwareConfig{},
        };
        WorkUnitSpec wu{.name = "main", .kernels = {kName}, .target_nodes = node};
        ProgramSpec pspec{.name = "quiesce_write_" + tag, .kernels = {spec_k}, .work_units = {wu}};
        Program prog = experimental::MakeProgramFromSpec(this->device(), pspec);
        ProgramRunArgs run_args;
        run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
            .kernel = kName,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
            .common_runtime_arg_values = {{"value", value}},
        }};
        experimental::SetProgramRunArgs(prog, run_args);
        wl.add_program(device_range, std::move(prog));
        return wl;
    }

    void reset_and_reopen_device() {
        for (auto& [device_id, dev] : id_to_device_) {
            if (dev && dev->is_initialized()) {
                dev->close();
            }
            dev.reset();
        }
        id_to_device_.clear();
        devices_.clear();
        MetalContext::instance().teardown();
        cores_parked_ = false;
        this->create_devices();
    }

    bool cores_parked_ = false;
};

TEST_F(QuasarDeviceQuiesceFixture, BootInitializesQuiesceSlotRunning) {
    const auto& hal = MetalContext::instance().hal();
    EXPECT_EQ(hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::QUIESCE), 64U);
    EXPECT_EQ(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::QUIESCE) % 64U, 0U);
    EXPECT_EQ(hal.get_dev_size(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::QUIESCE), 64U);
    EXPECT_EQ(hal.get_dev_addr(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::QUIESCE) % 64U, 0U);

    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            auto msg = read_worker_quiesce_slot(CoreCoord{x, y});
            auto view = msg.view();
            EXPECT_EQ(view.status(), 0x51551001U) << "Worker (" << x << "," << y << ")";
            EXPECT_EQ(view.control(), 0U) << "Worker (" << x << "," << y << ")";
            EXPECT_EQ(view.target_op_count()[0], 0U);
            EXPECT_EQ(view.target_op_count()[1], 0U);
            EXPECT_EQ(view.op_started_count()[0], 0U);
            EXPECT_EQ(view.op_started_count()[1], 0U);
        }
    }
}

TEST_F(QuasarDeviceQuiesceFixture, IdleWorkersQuiesceParkAllSubordinatesAndReinitCleanly) {
    const NodeCoord worker0{0, 0};
    const NodeCoord worker1{1, 0};
    const uint32_t scratch_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Run a warmup op so DM and TRISC harts have executed at least one launch and triggered sync-register init.
    {
        auto wl = make_simple_write_workload(worker0, scratch_addr, 0x11223344U, "warmup");
        distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl, /*blocking=*/true);
        std::vector<uint32_t> readback(1, 0);
        slow_dispatch::ReadFromL1(this->device(), worker0, scratch_addr, sizeof(uint32_t), readback);
        ASSERT_EQ(readback[0], 0x11223344U);
    }

    if (!this->IsSlowDispatch()) {
        // Phase 1: Drain and park all dispatch cores first when Fast Dispatch is active.
        const auto dispatch_cores = MetalContext::instance().device_manager()->get_virtual_dispatch_cores(chip_id());
        for (const CoreCoord& disp_core : dispatch_cores) {
            write_quiesce_control_virtual(disp_core, dev_msgs::QUIESCE_CTRL_NEXT_OP);
        }
        for (const CoreCoord& disp_core : dispatch_cores) {
            wait_for_quiesced_virtual(disp_core);
        }
    }

    // Phase 2: Request quiesce on workers (0,0) and (1,0) and verify both park with all subordinates DONE (0).
    write_worker_quiesce_control(worker0, dev_msgs::QUIESCE_CTRL_NEXT_OP);
    write_worker_quiesce_control(worker1, dev_msgs::QUIESCE_CTRL_NEXT_OP);
    wait_for_worker_quiesced(worker0);
    wait_for_worker_quiesced(worker1);

    // Verify subordinate_sync map is completely zeroed (all subordinate DMs and Neo TRISCs acked DONE before nop spin).
    const auto& hal = MetalContext::instance().hal();
    const auto factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    const uint64_t sub_sync_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX) +
        factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::subordinate_sync);
    auto sub_sync = factory.create<dev_msgs::subordinate_sync_msg_t>();
    MetalContext::instance().get_cluster().read_core(
        sub_sync.data(),
        sub_sync.size(),
        tt_cxy_pair(chip_id(), this->device().worker_core_from_logical_core(worker0)),
        sub_sync_addr);
    for (size_t i = 0; i < sub_sync.view().map().size(); ++i) {
        EXPECT_EQ(sub_sync.view().map()[i], 0U) << "subordinate_sync byte " << i << " not DONE after park";
    }

    // Step 4 verification: reset & re-initialize device and confirm Workload 2 runs from a clean state.
    reset_and_reopen_device();
    EXPECT_EQ(read_worker_quiesce_slot(worker0).view().status(), 0x51551001U);

    auto wl2 = make_simple_write_workload(worker0, scratch_addr, 0x55667788U, "post_reinit");
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl2, /*blocking=*/true);
    std::vector<uint32_t> post_reinit_readback(1, 0);
    slow_dispatch::ReadFromL1(this->device(), worker0, scratch_addr, sizeof(uint32_t), post_reinit_readback);
    EXPECT_EQ(post_reinit_readback[0], 0x55667788U);
}

TEST_F(QuasarDeviceQuiesceFixture, SlowDispatchMidKernelQuiesceFinishesOpAndDrainsUserTrids) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Slow Dispatch mid-kernel quiesce test requires TT_METAL_SLOW_DISPATCH_MODE=1";
    }

    const NodeCoord src_worker{0, 0};
    const NodeCoord dst_worker{1, 0};
    const CoreCoord dst_noc =
        this->device().virtual_noc0_coordinate(NOC::NOC_0, this->device().worker_core_from_logical_core(dst_worker));

    const uint32_t base_l1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t handshake_addr = base_l1;          // 2 words: [0] = started, [1] = gate
    const uint32_t src_stage_addr = base_l1 + 64U;    // 7 * 16B staging slots on src_worker
    const uint32_t dst_output_addr = base_l1 + 256U;  // 7 * 16B destination slots on dst_worker

    std::vector<uint32_t> zero_handshake(2, 0U);
    std::vector<uint32_t> zero_dst(7 * 4, 0U);
    slow_dispatch::WriteToL1(this->device(), src_worker, handshake_addr, zero_handshake);
    slow_dispatch::WriteToL1(this->device(), dst_worker, dst_output_addr, zero_dst);

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange device_range(this->device().shape());
    const KernelSpecName kName{"tagged_noc_quiesce"};
    KernelSpec spec_k{
        .unique_id = kName,
        .source = kTaggedNocQuiesceKernel,
        .num_threads = 2,  // Runs concurrently on DM2 and DM3
        .runtime_arg_schema =
            {.runtime_arg_names = {"handshake_addr", "src_stage_addr", "dst_noc_x", "dst_noc_y", "dst_l1_addr"}},
        .hw_config = DataMovementHardwareConfig{},
    };
    WorkUnitSpec wu{.name = "main", .kernels = {kName}, .target_nodes = src_worker};
    ProgramSpec pspec{.name = "mid_kernel_quiesce", .kernels = {spec_k}, .work_units = {wu}};
    Program prog = experimental::MakeProgramFromSpec(this->device(), pspec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = kName,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            src_worker,
            {{"handshake_addr", handshake_addr},
             {"src_stage_addr", src_stage_addr},
             {"dst_noc_x", dst_noc.x},
             {"dst_noc_y", dst_noc.y},
             {"dst_l1_addr", dst_output_addr}}),
    }};
    experimental::SetProgramRunArgs(prog, run_args);
    wl.add_program(device_range, std::move(prog));

    // Launch non-blocking so host can write QUIESCE_CTRL_NEXT_OP while the kernel is mid-execution.
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl, /*blocking=*/false);

    auto& cluster = MetalContext::instance().get_cluster();
    const CoreCoord src_virtual = this->device().worker_core_from_logical_core(src_worker);

    // 1. Wait until the kernel has entered kernel_main() and published KERNEL_STARTED_MAGIC (0xB007ED01).
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    uint32_t started_word = 0U;
    while (std::chrono::steady_clock::now() < start_deadline) {
        cluster.read_core(&started_word, sizeof(started_word), tt_cxy_pair(chip_id(), src_virtual), handshake_addr);
        cluster.advance_device_execution(chip_id());
        if (started_word == 0xB007ED01U) {
            break;
        }
    }
    ASSERT_EQ(started_word, 0xB007ED01U) << "Worker kernel did not signal KERNEL_STARTED_MAGIC";

    // 2. Write QUIESCE_CTRL_NEXT_OP while the kernel is actively spinning inside kernel_main().
    write_worker_quiesce_control(src_worker, dev_msgs::QUIESCE_CTRL_NEXT_OP);

    // Advance execution several times and confirm the worker does NOT prematurely report QUIESCED mid-kernel.
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(read_worker_quiesce_slot(src_worker).view().status(), 0x51551001U);
    }

    // 3. Open the gate so DM2 and DM3 issue tagged NOC writes on TRIDs 1..7 and return from kernel_main().
    const uint32_t open_gate = 0x6A7E0001U;
    cluster.write_core(
        &open_gate, sizeof(open_gate), tt_cxy_pair(chip_id(), src_virtual), handshake_addr + sizeof(uint32_t));
    cluster.l1_barrier(chip_id());

    // 4. Finish() waits on RUN_MSG_DONE; afterwards DM0 loops back to GW, drains all TRIDs, parks, and sets QUIESCED.
    distributed::Finish(this->device().mesh_command_queue());
    wait_for_worker_quiesced(src_worker);

    // 5. Verify all 7 tagged NOC writes (TRIDs 1..7) landed intact on dst_worker.
    std::vector<uint32_t> dst_words(7 * 4, 0U);
    slow_dispatch::ReadFromL1(
        this->device(), dst_worker, dst_output_addr, dst_words.size() * sizeof(uint32_t), dst_words);
    EXPECT_EQ(dst_words[0 * 4], 0x721D0001U);
    EXPECT_EQ(dst_words[1 * 4], 0x721D0002U);
    EXPECT_EQ(dst_words[2 * 4], 0x721D0003U);
    EXPECT_EQ(dst_words[3 * 4], 0x721D0004U);
    EXPECT_EQ(dst_words[4 * 4], 0x721D0005U);
    EXPECT_EQ(dst_words[5 * 4], 0x721D0006U);
    EXPECT_EQ(dst_words[6 * 4], 0x721D0007U);
}

TEST_F(QuasarDeviceQuiesceFixture, FastDispatchMidStreamAtOpCountAndUnpoppedCompletionQueue) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "Fast Dispatch mid-stream quiesce test requires Fast Dispatch (e.g. Quasar RTL simulator)";
    }

    const NodeCoord worker0{0, 0};
    const uint32_t base_l1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    constexpr uint32_t kTotalEnqueuedOps = 10U;
    constexpr uint32_t kStopAtOpCount = 4U;

    std::vector<uint32_t> zero_slots(kTotalEnqueuedOps * 4, 0U);
    slow_dispatch::WriteToL1(this->device(), worker0, base_l1, zero_slots);

    // Preset QUIESCE_CTRL_AT_OP_COUNT with target_op_count[0] = 4 on the dispatch core(s) before streaming 10 ops.
    const auto dispatch_cores = MetalContext::instance().device_manager()->get_virtual_dispatch_cores(chip_id());
    ASSERT_FALSE(dispatch_cores.empty());
    for (const CoreCoord& disp_core : dispatch_cores) {
        write_quiesce_control_virtual(disp_core, dev_msgs::QUIESCE_CTRL_AT_OP_COUNT, kStopAtOpCount);
    }

    // Also allocate a small L1 buffer and enqueue non-blocking reads so cq_dispatch exercises the
    // un-popped completion-queue path while quiescing.
    distributed::DeviceLocalBufferConfig local_cfg{.page_size = 64, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig repl_cfg{.size = 64};
    auto mesh_buf = distributed::MeshBuffer::create(repl_cfg, local_cfg, &this->device());
    std::vector<uint32_t> host_sink(16, 0U);

    auto& cq = this->device().mesh_command_queue();
    for (uint32_t op = 0; op < kTotalEnqueuedOps; ++op) {
        auto wl = make_simple_write_workload(
            worker0, base_l1 + op * 16U, 0xABCD0001U + op, "fd_stream_" + std::to_string(op));
        distributed::EnqueueMeshWorkload(cq, wl, /*blocking=*/false);
        distributed::EnqueueReadMeshBuffer(cq, host_sink, mesh_buf, /*blocking=*/false);
    }

    // Phase 1: Wait for dispatch core(s) to stop at op 4, drain in-flight workers, and reach QUIESCED.
    for (const CoreCoord& disp_core : dispatch_cores) {
        wait_for_quiesced_virtual(disp_core);
        EXPECT_EQ(read_quiesce_slot_virtual(disp_core).view().op_started_count()[0], 4U);
    }

    // Phase 2: Park workers.
    write_worker_quiesce_control(worker0, dev_msgs::QUIESCE_CTRL_NEXT_OP);
    wait_for_worker_quiesced(worker0);

    // Verify ops 0..3 ran and ops 4..9 never executed.
    std::vector<uint32_t> slot_results(kTotalEnqueuedOps * 4, 0U);
    slow_dispatch::ReadFromL1(this->device(), worker0, base_l1, slot_results.size() * sizeof(uint32_t), slot_results);
    EXPECT_EQ(slot_results[0 * 4], 0xABCD0001U);
    EXPECT_EQ(slot_results[1 * 4], 0xABCD0002U);
    EXPECT_EQ(slot_results[2 * 4], 0xABCD0003U);
    EXPECT_EQ(slot_results[3 * 4], 0xABCD0004U);
    EXPECT_EQ(slot_results[4 * 4], 0U);
    EXPECT_EQ(slot_results[9 * 4], 0U);
}

TEST_F(QuasarDeviceQuiesceFixture, FastDispatchMidTraceExecBufQuiesce) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "Fast Dispatch trace quiesce test requires Fast Dispatch (e.g. Quasar RTL simulator)";
    }

    const NodeCoord worker0{0, 0};
    const uint32_t base_l1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    auto& cq = this->device().mesh_command_queue();
    auto wl0 = make_simple_write_workload(worker0, base_l1 + 0U, 0x10000001U, "trace_op0");
    auto wl1 = make_simple_write_workload(worker0, base_l1 + 16U, 0x10000002U, "trace_op1");
    auto wl2 = make_simple_write_workload(worker0, base_l1 + 32U, 0x10000003U, "trace_op2");

    // Warmup & capture 3-op trace (warmups advance op_started_count to 3).
    distributed::EnqueueMeshWorkload(cq, wl0, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(cq, wl1, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(cq, wl2, /*blocking=*/true);

    std::vector<uint32_t> zeros(12, 0U);
    slow_dispatch::WriteToL1(this->device(), worker0, base_l1, zeros);

    distributed::MeshTraceId trace_id = this->device().begin_mesh_trace(cq);
    distributed::EnqueueMeshWorkload(cq, wl0, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(cq, wl1, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(cq, wl2, /*blocking=*/false);
    this->device().end_mesh_trace(cq, trace_id);

    // Stop mid-trace right after the 2nd op of the trace replay (op_completed_count == 3 warmup + 2 trace = 5).
    const auto dispatch_cores = MetalContext::instance().device_manager()->get_virtual_dispatch_cores(chip_id());
    ASSERT_FALSE(dispatch_cores.empty());
    for (const CoreCoord& disp_core : dispatch_cores) {
        write_quiesce_control_virtual(disp_core, dev_msgs::QUIESCE_CTRL_AT_OP_COUNT, 5U);
    }

    this->device().replay_mesh_trace(cq, trace_id, /*blocking=*/false);

    for (const CoreCoord& disp_core : dispatch_cores) {
        wait_for_quiesced_virtual(disp_core);
        EXPECT_EQ(read_quiesce_slot_virtual(disp_core).view().op_started_count()[0], 5U);
    }

    write_worker_quiesce_control(worker0, dev_msgs::QUIESCE_CTRL_NEXT_OP);
    wait_for_worker_quiesced(worker0);

    std::vector<uint32_t> results(12, 0U);
    slow_dispatch::ReadFromL1(this->device(), worker0, base_l1, results.size() * sizeof(uint32_t), results);
    EXPECT_EQ(results[0], 0x10000001U);
    EXPECT_EQ(results[4], 0x10000002U);
    EXPECT_EQ(results[8], 0U);

    this->device().release_mesh_trace(trace_id);
}

TEST_F(QuasarDeviceQuiesceFixture, HostApiQuiescesMidKernelClosesAndReopensCleanly) {
    const NodeCoord src_worker{0, 0};
    const NodeCoord dst_worker{1, 0};
    const ChipId cid = chip_id();
    const CoreCoord src_virtual = this->device().worker_core_from_logical_core(src_worker);
    const CoreCoord dst_virtual = this->device().worker_core_from_logical_core(dst_worker);
    const CoreCoord dst_noc = this->device().virtual_noc0_coordinate(NOC::NOC_0, dst_virtual);

    const uint32_t base_l1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t handshake_addr = base_l1;
    const uint32_t src_stage_addr = base_l1 + 64U;
    const uint32_t dst_output_addr = base_l1 + 256U;

    std::vector<uint32_t> zero_handshake(2, 0U);
    std::vector<uint32_t> zero_dst(7 * 4, 0U);
    slow_dispatch::WriteToL1(this->device(), src_worker, handshake_addr, zero_handshake);
    slow_dispatch::WriteToL1(this->device(), dst_worker, dst_output_addr, zero_dst);

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange device_range(this->device().shape());
    const KernelSpecName kName{"host_api_tagged_noc_quiesce"};
    KernelSpec spec_k{
        .unique_id = kName,
        .source = kTaggedNocQuiesceKernel,
        .num_threads = 2,
        .runtime_arg_schema =
            {.runtime_arg_names = {"handshake_addr", "src_stage_addr", "dst_noc_x", "dst_noc_y", "dst_l1_addr"}},
        .hw_config = DataMovementHardwareConfig{},
    };
    WorkUnitSpec wu{.name = "main", .kernels = {kName}, .target_nodes = src_worker};
    ProgramSpec pspec{.name = "host_api_mid_kernel_quiesce", .kernels = {spec_k}, .work_units = {wu}};
    Program prog = experimental::MakeProgramFromSpec(this->device(), pspec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = kName,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            src_worker,
            {{"handshake_addr", handshake_addr},
             {"src_stage_addr", src_stage_addr},
             {"dst_noc_x", dst_noc.x},
             {"dst_noc_y", dst_noc.y},
             {"dst_l1_addr", dst_output_addr}}),
    }};
    experimental::SetProgramRunArgs(prog, run_args);
    wl.add_program(device_range, std::move(prog));

    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl, /*blocking=*/false);

    auto& cluster = MetalContext::instance().get_cluster();
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    uint32_t started_word = 0U;
    while (std::chrono::steady_clock::now() < start_deadline) {
        cluster.read_core(&started_word, sizeof(started_word), tt_cxy_pair(cid, src_virtual), handshake_addr);
        cluster.advance_device_execution(cid);
        if (started_word == 0xB007ED01U) {
            break;
        }
    }
    ASSERT_EQ(started_word, 0xB007ED01U) << "Worker kernel did not signal KERNEL_STARTED_MAGIC";

    // Release the spin gate without advancing simulator cycles or calling Finish() so the kernel is still
    // in-flight when quiesce_for_soft_reset() writes QUIESCE_CTRL_NEXT_OP and begins polling.
    const uint32_t open_gate = 0x6A7E0001U;
    cluster.write_core(&open_gate, sizeof(open_gate), tt_cxy_pair(cid, src_virtual), handshake_addr + sizeof(uint32_t));

    auto res = experimental::quiesce_for_soft_reset(
        this->device(), experimental::QuiesceOptions{.timeout = std::chrono::seconds{15}});
    EXPECT_TRUE(res.ok());
    EXPECT_EQ(res.status, experimental::QuiesceStatus::Quiesced);
    EXPECT_TRUE(res.unquiesced_cores.empty());

    // MeshDevice must be closed inline by quiesce_for_soft_reset(), and subsequent close() is an idempotent no-op
    // while calling quiesce_for_soft_reset() again on an already-closed MeshDevice fails fast.
    EXPECT_FALSE(this->device().is_initialized());
    EXPECT_TRUE(this->device().close());
    EXPECT_ANY_THROW((void)experimental::quiesce_for_soft_reset(this->device()));

    // Verify all 7 tagged NOC writes (TRIDs 1..7) landed intact before the worker parked.
    std::vector<uint32_t> dst_words(7 * 4, 0U);
    cluster.read_core(
        dst_words.data(), dst_words.size() * sizeof(uint32_t), tt_cxy_pair(cid, dst_virtual), dst_output_addr);
    EXPECT_EQ(dst_words[0 * 4], 0x721D0001U);
    EXPECT_EQ(dst_words[1 * 4], 0x721D0002U);
    EXPECT_EQ(dst_words[2 * 4], 0x721D0003U);
    EXPECT_EQ(dst_words[3 * 4], 0x721D0004U);
    EXPECT_EQ(dst_words[4 * 4], 0x721D0005U);
    EXPECT_EQ(dst_words[5 * 4], 0x721D0006U);
    EXPECT_EQ(dst_words[6 * 4], 0x721D0007U);

    // Reopen a fresh MeshDevice in the same process WITHOUT manually calling MetalContext::teardown():
    // QuiescedForSoftReset + force_reinit_ must silently tear down host state, reset cores, relaunch FW,
    // and execute Workload 2 cleanly.
    id_to_device_.clear();
    devices_.clear();
    this->create_devices();
    init_max_dfbs();

    const uint32_t reopen_addr = base_l1 + 512U;
    std::vector<uint32_t> zero_word{0U};
    slow_dispatch::WriteToL1(this->device(), src_worker, reopen_addr, zero_word);
    auto wl2 = make_simple_write_workload(src_worker, reopen_addr, 0xCAFEBEEFU, "post_reopen_workload");
    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl2, /*blocking=*/true);

    std::vector<uint32_t> reopen_out{0U};
    slow_dispatch::ReadFromL1(this->device(), src_worker, reopen_addr, sizeof(uint32_t), reopen_out);
    EXPECT_EQ(reopen_out[0], 0xCAFEBEEFU);
}

TEST_F(QuasarDeviceQuiesceFixture, HostApiTimesOutOnStuckKernelAndFailsSafely) {
    // Non-positive timeout must fail fast as caller misuse.
    EXPECT_ANY_THROW((void)experimental::quiesce_for_soft_reset(
        this->device(), experimental::QuiesceOptions{.timeout = std::chrono::milliseconds{0}}));

    const NodeCoord src_worker{0, 0};
    const NodeCoord dst_worker{1, 0};
    const ChipId cid = chip_id();
    const CoreCoord src_virtual = this->device().worker_core_from_logical_core(src_worker);
    const CoreCoord dst_virtual = this->device().worker_core_from_logical_core(dst_worker);
    const CoreCoord dst_noc = this->device().virtual_noc0_coordinate(NOC::NOC_0, dst_virtual);

    const uint32_t base_l1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t handshake_addr = base_l1;
    const uint32_t src_stage_addr = base_l1 + 64U;
    const uint32_t dst_output_addr = base_l1 + 256U;

    std::vector<uint32_t> zero_handshake(2, 0U);
    slow_dispatch::WriteToL1(this->device(), src_worker, handshake_addr, zero_handshake);

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange device_range(this->device().shape());
    const KernelSpecName kName{"stuck_kernel_timeout_quiesce"};
    KernelSpec spec_k{
        .unique_id = kName,
        .source = kTaggedNocQuiesceKernel,
        .num_threads = 2,
        .runtime_arg_schema =
            {.runtime_arg_names = {"handshake_addr", "src_stage_addr", "dst_noc_x", "dst_noc_y", "dst_l1_addr"}},
        .hw_config = DataMovementHardwareConfig{},
    };
    WorkUnitSpec wu{.name = "main", .kernels = {kName}, .target_nodes = src_worker};
    ProgramSpec pspec{.name = "stuck_kernel_quiesce", .kernels = {spec_k}, .work_units = {wu}};
    Program prog = experimental::MakeProgramFromSpec(this->device(), pspec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = kName,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            src_worker,
            {{"handshake_addr", handshake_addr},
             {"src_stage_addr", src_stage_addr},
             {"dst_noc_x", dst_noc.x},
             {"dst_noc_y", dst_noc.y},
             {"dst_l1_addr", dst_output_addr}}),
    }};
    experimental::SetProgramRunArgs(prog, run_args);
    wl.add_program(device_range, std::move(prog));

    distributed::EnqueueMeshWorkload(this->device().mesh_command_queue(), wl, /*blocking=*/false);

    auto& cluster = MetalContext::instance().get_cluster();
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    uint32_t started_word = 0U;
    while (std::chrono::steady_clock::now() < start_deadline) {
        cluster.read_core(&started_word, sizeof(started_word), tt_cxy_pair(cid, src_virtual), handshake_addr);
        cluster.advance_device_execution(cid);
        if (started_word == 0xB007ED01U) {
            break;
        }
    }
    ASSERT_EQ(started_word, 0xB007ED01U) << "Worker kernel did not signal KERNEL_STARTED_MAGIC";

    // Intentionally DO NOT open the gate — src_worker spins indefinitely inside kernel_main(), simulating a
    // deadlocked / infinite-loop workload.
    constexpr auto kShortTimeout = std::chrono::milliseconds{250};
    auto res =
        experimental::quiesce_for_soft_reset(this->device(), experimental::QuiesceOptions{.timeout = kShortTimeout});
    EXPECT_FALSE(res.ok());
    EXPECT_FALSE(static_cast<bool>(res));
    EXPECT_EQ(res.status, experimental::QuiesceStatus::TimedOutInWorkers);
    EXPECT_GE(res.elapsed, kShortTimeout);
    ASSERT_FALSE(res.unquiesced_cores.empty());

    bool found_stuck_worker = false;
    for (const auto& diag : res.unquiesced_cores) {
        if (diag.chip_id == cid && diag.virtual_core == src_virtual) {
            found_stuck_worker = true;
            EXPECT_EQ(diag.core_type, HalProgrammableCoreType::TENSIX);
            EXPECT_EQ(diag.raw_status, dev_msgs::QUIESCE_STATUS_RUNNING);
            EXPECT_EQ(diag.control, dev_msgs::QUIESCE_CTRL_NEXT_OP);
        }
    }
    EXPECT_TRUE(found_stuck_worker) << "Expected stuck worker core " << src_virtual.str() << " in unquiesced_cores";

    // On timeout, MeshDevice was not closed inline, yet closing it must not hang, and attempting to re-initialize
    // without an external SoC reset must fail fast.
    EXPECT_TRUE(this->device().is_initialized());
    EXPECT_TRUE(this->device().close());
    EXPECT_ANY_THROW(this->create_devices());
    cores_parked_ = true;
}

}  // namespace
}  // namespace tt::tt_metal
