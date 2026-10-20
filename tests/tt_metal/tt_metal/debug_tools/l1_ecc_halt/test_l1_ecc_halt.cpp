// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole silicon L1 ECC halt. The tests come in three kinds:
// 1. Emulated halt (EmulatedHaltIsReported, HaltMessageNamesTheCause, EventSynchronizeReportsHalt): Blackhole has
//    no L1 error injection, so the host writes the halt record the BRISC handler would write. These cover the host
//    side: every wait loop reports the halt, and the message names the cause.
// 2. Real ECC errors (UnpopulatedRowErrorsAreIgnored): a NoC read past the end of L1 raises real DBEs from the
//    unpopulated rows. The handler must run and return without halting.
// 3. Real interrupt (DISABLED_WatchdogHalt): the watchdog's software interrupt raises IRQ 3, and the real handler
//    halts the core. The BRISC stays in the handler until tt-smi -r, so this test runs on its own
//    (tests/scripts/run_l1_ecc_halt_watchdog_test.sh).

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "hal_types.hpp"
#include "hostdev/l1_ecc_halt_constants.h"
#include "impl/context/metal_context.hpp"
#include <internal/l1_ecc_halt.hpp>
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_dispatch_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

constexpr const char* hang_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/l1_ecc_halt_hang.cpp";
constexpr const char* overread_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/l1_ecc_overread.cpp";
const CoreCoord test_core{0, 0};
constexpr auto report_deadline = std::chrono::seconds(30);
// ECC_CTRL after firmware arms the halt.
constexpr uint32_t armed_ecc_ctrl =
    static_cast<uint32_t>(EccView::L1_COUNTS) | ECC_CTRL_L1_DBE_IRQ_EN | ECC_CTRL_RISC_MEM_IRQ_EN;

class L1EccHaltTest : public MeshDispatchFixture {
protected:
    void SetUp() override {
        auto& ctx = MetalContext::instance();
        if (ctx.get_cluster().arch() != tt::ARCH::BLACKHOLE ||
            ctx.get_cluster().get_target_device_type() != tt::TargetDevice::Silicon ||
            !ctx.rtoptions().get_l1_ecc_halt_enabled()) {
            GTEST_SKIP() << "L1 ECC halt runs only on Blackhole silicon with the feature enabled";
        }
        MeshDispatchFixture::SetUp();
        test_target_ =
            tt_cxy_pair(devices_[0]->get_device_ids()[0], devices_[0]->worker_core_from_logical_core(test_core));
    }

    distributed::MeshWorkload single_core_workload(const char* kernel_path, const std::vector<uint32_t>& runtime_args) {
        Program program = CreateProgram();
        KernelHandle kernel = CreateKernel(
            program,
            kernel_path,
            test_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, kernel, test_core, runtime_args);
        distributed::MeshWorkload workload;
        auto zero = distributed::MeshCoordinate(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
        return workload;
    }

    uint64_t halt_record_addr() {
        const auto& hal = MetalContext::instance().hal();
        return hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX) +
               hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX)
                   .offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::l1_ecc_halt);
    }

    // The record survives until the next firmware init, so it can be written before the launch.
    void write_halt_record() {
        MetalContext::instance().get_cluster().write_core(
            record_.data(), record_.size(), test_target_, halt_record_addr());
    }

    // A halt record stays on the shared device, so every test that reports one reopens it before the next test.
    void expect_check_reports(const std::string& text) {
        get_shared_devices().needs_recovery = true;
        EXPECT_THAT(
            [&] { internal::L1EccHaltWatch(devices_[0].get()).check_now(); },
            ThrowsMessage<std::runtime_error>(HasSubstr(text)));
    }

    void expect_halt_reported_by(const std::function<void()>& wait) {
        get_shared_devices().needs_recovery = true;
        auto start = std::chrono::steady_clock::now();
        EXPECT_THAT(wait, ThrowsMessage<std::runtime_error>(HasSubstr("tt-smi -r")));
        EXPECT_LT(std::chrono::steady_clock::now() - start, report_deadline);
    }

    dev_msgs::l1_ecc_halt_msg_t record_ = MetalContext::instance()
                                              .hal()
                                              .get_dev_msgs_factory(HalProgrammableCoreType::TENSIX)
                                              .create<dev_msgs::l1_ecc_halt_msg_t>();
    tt_cxy_pair test_target_{0, CoreCoord{}};
};

}  // namespace

TEST_F(L1EccHaltTest, EmulatedHaltIsReported) {
    record_.view().reason() = L1_ECC_HALT_REASON_UNEXPECTED;
    write_halt_record();
    auto workload = single_core_workload(hang_kernel, {0});
    auto& cq = devices_[0]->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    expect_halt_reported_by([&] { distributed::Finish(cq); });
    EXPECT_TRUE(MetalContext::instance().l1_ecc_halt_detected());
    // A retry after the caught error must not return as if the work had completed.
    EXPECT_THROW(distributed::Finish(cq), std::runtime_error);
}

// The L1 address is (row * 32 + bank) * 16. Banks 16-31 are in the second last-error word.
TEST_F(L1EccHaltTest, HaltMessageNamesTheCause) {
    record_.view().reason() = L1_ECC_HALT_REASON_RISC_MEMORY;
    record_.view().l1_and_local_mem_counts() = ECC_TRISC1_LOCAL_MEM_MASK;
    record_.view().icache_and_ncrisc_local_mem_counts() = ECC_NCRISC_ICACHE_MASK;
    write_halt_record();
    expect_check_reports("TRISC1 local memory DBE, NCRISC instruction cache parity error");

    MetalContext::instance().set_l1_ecc_halt_detected(false);
    record_.view().reason() = L1_ECC_HALT_REASON_DBE;
    record_.view().last_l1_error() = 0x8002000A;  // valid [31], bank 5 [17], row 10
    write_halt_record();
    expect_check_reports("bank 5 row 10 (L1 0x1450)");

    MetalContext::instance().set_l1_ecc_halt_detected(false);
    record_.view().last_l1_error() = 0x8000000A;           // valid [31], row 10
    record_.view().last_l1_error_banks_16_31() = 0x10000;  // bank 20 [16]
    write_halt_record();
    expect_check_reports("bank 20 row 10 (L1 0x1540)");
}

TEST_F(L1EccHaltTest, EventSynchronizeReportsHalt) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "Events are fast dispatch only";
    }
    record_.view().reason() = L1_ECC_HALT_REASON_UNEXPECTED;
    write_halt_record();
    auto workload = single_core_workload(hang_kernel, {0});
    auto& cq = devices_[0]->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    auto event = cq.enqueue_record_event_to_host();
    expect_halt_reported_by([&] { distributed::EventSynchronize(event); });
}

// The ECC scrubber reports false errors from the L1 rows past the populated end. A NoC read of the core's own L1
// past the end raises the same errors on demand: they must be detected and must not halt the core.
TEST_F(L1EccHaltTest, UnpopulatedRowErrorsAreIgnored) {
    auto& ctx = MetalContext::instance();
    if (ctx.rtoptions().get_watcher_enabled()) {
        GTEST_SKIP() << "Watcher NoC sanitize rejects the read past the end of L1";
    }
    const auto& hal = ctx.hal();
    constexpr uint32_t bytes = 1024;
    const uint32_t l1_end = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const uint32_t dst = devices_[0]->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto workload = single_core_workload(overread_kernel, {l1_end, dst, bytes});
    this->RunProgram(devices_[0], workload);

    uint32_t ctrl = 0, status = 0, int_no = 0, irq3 = 0;
    ctx.get_cluster().read_reg(&ctrl, test_target_, hal.get_tensix_debug_reg_addr(TensixDebugReg::L1_ECC_CTRL));
    ctx.get_cluster().read_reg(&status, test_target_, hal.get_tensix_debug_reg_addr(TensixDebugReg::L1_ECC_STATUS));
    ctx.get_cluster().read_reg(&int_no, test_target_, hal.get_tensix_debug_reg_addr(TensixDebugReg::BRISC_INT_NO));
    ctx.get_cluster().read_reg(&irq3, test_target_, hal.get_tensix_debug_reg_addr(TensixDebugReg::PIC_HW_INT_3));
    // Firmware arms the L1 DBE and RISC memory interrupts and leaves view 2 selected.
    EXPECT_EQ(ctrl, armed_ecc_ctrl) << "ECC interrupts not armed, ECC_CTRL 0x" << std::hex << ctrl;
    EXPECT_NE(status & ECC_DBE_COUNT_OVF_MASK, 0u) << "the read past the end of L1 raised no DBE";
    EXPECT_EQ(int_no, 0u) << "BRISC interrupt left latched";
    // The DBEs raised IRQ 3, and only the handler's clear drops it.
    EXPECT_EQ(irq3, 0u) << "IRQ 3 still raised: the handler did not run";
    EXPECT_NO_THROW(internal::L1EccHaltWatch(devices_[0].get()).check_now());
    EXPECT_FALSE(ctx.l1_ecc_halt_detected());
}

TEST_F(L1EccHaltTest, DISABLED_WatchdogHalt) {
    auto workload = single_core_workload(hang_kernel, {1});  // raises IRQ 3
    auto& cq = devices_[0]->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    expect_halt_reported_by([&] { distributed::Finish(cq); });

    // The real handler wrote the record and is still running (BRISC_INT_NO is 32 + the hardware IRQ index).
    auto& ctx = MetalContext::instance();
    uint32_t int_no = 0;
    ctx.get_cluster().read_reg(
        &int_no, test_target_, ctx.hal().get_tensix_debug_reg_addr(TensixDebugReg::BRISC_INT_NO));
    EXPECT_EQ(int_no, 32u + 3u);
    ctx.get_cluster().read_core(record_.data(), record_.size(), test_target_, halt_record_addr());
    EXPECT_EQ(record_.view().reason(), L1_ECC_HALT_REASON_UNEXPECTED);
    EXPECT_EQ(record_.view().ecc_ctrl(), armed_ecc_ctrl);
}
