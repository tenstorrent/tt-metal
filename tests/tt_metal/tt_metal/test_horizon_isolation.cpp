// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Negative isolation test for a Horizon build split into two devices (one per column, ATT on,
// TT_METAL_NOC_ATT=horizon_2x3). Two processes, one per device, coordinated through files in
// HORIZON_ISOLATION_SYNC_DIR:
//
//   HorizonIsolationVictim   (device 0): seeds a sentinel at TARGET on its Tensix (and Dispatch)
//                            L1, signals "victim_ready", waits for the attacker, then checks the
//                            sentinel is unchanged.
//   HorizonIsolationAttacker (device 1): waits for "victim_ready", runs a DM kernel that POSTS
//                            writes of a marker to every selector parked on a split device
//                            (1, 3, 5, 6, 63) at TARGET, plus a control write to its own Tensix
//                            (selector 0) at CONTROL before and after. Checks the control writes
//                            landed and its own TARGET is untouched, then signals "attacker_done".
//
// Both are skipped unless TT_METAL_QUASAR_VARIANT=horizon, TT_METAL_NOC_ATT=horizon_2x3 and
// HORIZON_ISOLATION_SYNC_DIR are set.

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace horizon_isolation {

constexpr uint32_t SENTINEL = 0x5e471e15u;           // victim TARGET
constexpr uint32_t ATTACKER_SENTINEL = 0x5e471e16u;  // attacker TARGET: distinct, so a mix-up of the two shows
constexpr uint32_t MARKER = 0xa77a0000u;             // the kernel ORs in a per-write tag below 0x1000
constexpr uint32_t WORDS = 4;
constexpr uint32_t BYTES = WORDS * sizeof(uint32_t);

std::filesystem::path sync_dir() { return std::getenv("HORIZON_ISOLATION_SYNC_DIR"); }

uint32_t env_seconds(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : fallback;
}

void notify(const std::string& name, const std::string& content = "") {
    std::ofstream(sync_dir() / name) << content << "\n";
    std::cout << "[ISOLATION] signalled " << name << std::endl;
}

// Waits for any of `names`; returns the one seen, or "" on timeout.
std::string wait_for(std::initializer_list<const char*> names, uint32_t timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        for (const char* n : names) {
            if (std::filesystem::exists(sync_dir() / n)) {
                return n;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return "";
}

// L1 addresses shared by both devices (same soc descriptor, same HAL).
struct IsolationLayout {
    uint32_t target;   // the victim's sentinel; the attacker's parked writes aim here
    uint32_t control;  // the attacker's own control writes (2 x BYTES)
    uint32_t src;      // the attacker kernel's source payloads
};

IsolationLayout isolation_layout() {
    const uint32_t base = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t aligned = (base + 0xfff) & ~0xfffu;
    return {aligned + 0x1000, aligned + 0x2000, aligned + 0x3000};
}

class HorizonIsolationFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        auto is = [](const char* name, const char* value) {
            const char* v = std::getenv(name);
            return v != nullptr && std::string(v) == value;
        };
        if (std::getenv("TT_METAL_SIMULATOR") == nullptr || !is("TT_METAL_QUASAR_VARIANT", "horizon") ||
            !is("TT_METAL_NOC_ATT", "horizon_2x3") || std::getenv("HORIZON_ISOLATION_SYNC_DIR") == nullptr) {
            GTEST_SKIP() << "Needs a Horizon split simulator build: set TT_METAL_SIMULATOR, "
                            "TT_METAL_QUASAR_VARIANT=horizon, TT_METAL_NOC_ATT=horizon_2x3 and "
                            "HORIZON_ISOLATION_SYNC_DIR.";
        }
        QuasarMeshDeviceSingleCardFixture::SetUp();
    }

    ChipId chip() { return slow_dispatch::physical_device_from_unit_mesh(this->device()).id(); }

    std::vector<uint32_t> read_tensix(uint32_t addr) {
        std::vector<uint32_t> r(WORDS, 0);
        slow_dispatch::ReadFromL1(this->device(), CoreCoord{0, 0}, addr, BYTES, r);
        return r;
    }
};

std::string hex(const std::vector<uint32_t>& v) {
    std::string s;
    char b[16];
    for (uint32_t w : v) {
        snprintf(b, sizeof(b), "0x%08x ", w);
        s += b;
    }
    return s;
}

}  // namespace horizon_isolation

using namespace horizon_isolation;

TEST_F(HorizonIsolationFixture, HorizonIsolationVictim) {
    const IsolationLayout l = isolation_layout();
    const std::vector<uint32_t> sentinel(WORDS, SENTINEL);
    const auto& cluster = MetalContext::instance().get_cluster();
    const tt_cxy_pair dispatch(chip(), CoreCoord{0, 1});  // Dispatch tile, device frame

    std::vector<uint32_t> seed = sentinel;
    slow_dispatch::WriteToL1(this->device(), CoreCoord{0, 0}, l.target, seed);
    ASSERT_EQ(read_tensix(l.target), sentinel) << "could not seed the victim's Tensix L1";

    // Best effort: the Dispatch tile too, checked only if the seed reads back.
    bool check_dispatch = false;
    std::vector<uint32_t> d(WORDS, 0);
    try {
        cluster.write_core(sentinel.data(), BYTES, dispatch, l.target);
        cluster.read_core(d, BYTES, dispatch, l.target);
        check_dispatch = (d == sentinel);
    } catch (const std::exception& e) {
        std::cout << "[ISOLATION] dispatch tile not accessible: " << e.what() << std::endl;
    }
    std::cout << "[ISOLATION] victim chip " << chip() << ": sentinel at 0x" << std::hex << l.target << std::dec
              << " on Tensix (0,0)" << (check_dispatch ? " and Dispatch (0,1)" : " (Dispatch not checked)")
              << std::endl;

    notify("victim_ready");
    const std::string seen =
        wait_for({"attacker_done", "attacker_wedged"}, env_seconds("HORIZON_ISOLATION_TIMEOUT_S", 3600));
    std::cout << "[ISOLATION] attacker status: " << (seen.empty() ? "TIMEOUT" : seen) << std::endl;

    // Check the victim's memory whatever the attacker's fate.
    const auto t = read_tensix(l.target);
    std::cout << "[ISOLATION] victim Tensix (0,0) @0x" << std::hex << l.target << std::dec << ": " << hex(t)
              << std::endl;
    EXPECT_EQ(t, sentinel) << "ISOLATION LEAK: the other device's writes reached this device's Tensix L1";
    if (check_dispatch) {
        cluster.read_core(d, BYTES, dispatch, l.target);
        std::cout << "[ISOLATION] victim Dispatch (0,1) @0x" << std::hex << l.target << std::dec << ": " << hex(d)
                  << std::endl;
        EXPECT_EQ(d, sentinel) << "ISOLATION LEAK: the other device's writes reached this device's Dispatch L1";
    }
    EXPECT_EQ(seen, "attacker_done") << "the attacker did not finish (see its log)";
}

TEST_F(HorizonIsolationFixture, HorizonIsolationAttacker) {
    const IsolationLayout l = isolation_layout();
    const std::vector<uint32_t> sentinel(WORDS, ATTACKER_SENTINEL);

    // Seed own TARGET (a parked write misrouted to self would show here) and clear CONTROL.
    std::vector<uint32_t> seed = sentinel;
    slow_dispatch::WriteToL1(this->device(), CoreCoord{0, 0}, l.target, seed);
    std::vector<uint32_t> zero(2 * WORDS, 0);
    slow_dispatch::WriteToL1(this->device(), CoreCoord{0, 0}, l.control, zero);

    const uint32_t ready_timeout = env_seconds("HORIZON_ISOLATION_TIMEOUT_S", 3600);
    ASSERT_EQ(wait_for({"victim_ready"}, ready_timeout), "victim_ready") << "the victim never became ready";

    const experimental::NodeCoord node{0, 0};
    const experimental::KernelSpecName DM_KERNEL{"attacker"};
    experimental::KernelSpec spec_k{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_isolation_attacker.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "target_addr", "control_addr", "marker"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "horizon_isolation_attacker", .kernels = {spec_k}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node, {{"src_addr", l.src}, {"target_addr", l.target}, {"control_addr", l.control}, {"marker", MARKER}})}};
    experimental::SetProgramRunArgs(program, params);

    // The simulator wait has no timeout: a watchdog reports a wedge and exits so the victim proceeds.
    std::atomic<bool> finished{false};
    std::thread watchdog([&finished, s = env_seconds("HORIZON_ISOLATION_KERNEL_TIMEOUT_S", 900)] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(s);
        while (!finished && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (!finished) {
            std::cout << "[ISOLATION] WEDGED: the attacker kernel did not finish within " << s << " s" << std::endl;
            notify("attacker_wedged");
            std::_Exit(3);
        }
    });
    std::cout << "[ISOLATION] attacker chip " << chip() << ": posting to selectors 1,3,5,6,63 at 0x" << std::hex
              << l.target << ", control at 0x" << l.control << std::dec << std::endl;
    LaunchProgram(this->device(), std::move(program));
    finished = true;
    watchdog.join();
    std::cout << "[ISOLATION] attacker kernel finished" << std::endl;

    std::vector<uint32_t> ctrl(2 * WORDS, 0);
    slow_dispatch::ReadFromL1(this->device(), CoreCoord{0, 0}, l.control, 2 * BYTES, ctrl);
    const auto own_target = read_tensix(l.target);
    std::cout << "[ISOLATION] attacker control @0x" << std::hex << l.control << std::dec << ": " << hex(ctrl)
              << std::endl;
    std::cout << "[ISOLATION] attacker own target @0x" << std::hex << l.target << std::dec << ": " << hex(own_target)
              << std::endl;
    notify("attacker_done");

    for (uint32_t i = 0; i < WORDS; i++) {
        EXPECT_EQ(ctrl[i], MARKER | 0xc00 | i) << "control write before the attack did not land";
        EXPECT_EQ(ctrl[WORDS + i], MARKER | 0xc10 | i) << "control write after the attack did not land";
    }
    EXPECT_EQ(own_target, sentinel) << "a parked-selector write landed on the attacker's own Tensix";
}
