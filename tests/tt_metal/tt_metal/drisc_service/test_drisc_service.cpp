// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs on chip 0, or the chip DRISC_SERVICE_TEST_CHIP names.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "impl/drisc_service/drisc_core.hpp"
#include "impl/drisc_service/service.hpp"
#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal {
namespace {

namespace abi = internal::drisc_service::abi;
using internal::drisc_service::DriscCore;
using internal::drisc_service::Seat;

const ChipId chip_id = [] {
    const char* chip = std::getenv("DRISC_SERVICE_TEST_CHIP");
    return chip != nullptr ? static_cast<ChipId>(std::atoi(chip)) : ChipId{0};
}();
constexpr uint32_t niu_cfg_noc2axi = 1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE;

constexpr std::chrono::milliseconds stop_timeout{2000};
constexpr std::chrono::milliseconds reply_timeout{2000};

// The chip under test through a bare UMD cluster, as a client process opens it.
class Device {
public:
    Device() : umd(tt::umd::ClusterOptions{.num_host_mem_ch_per_mmio_device = 0}) {
        if (umd.get_cluster_description()->get_arch(chip_id) == tt::ARCH::BLACKHOLE) {
            soc.emplace(internal::drisc_service::soc_descriptor(umd, chip_id));
        }
    }

    bool blackhole() const { return soc.has_value(); }
    uint32_t num_banks() const { return static_cast<uint32_t>(soc->get_num_dram_views()); }
    std::array<Seat, NUM_NOCS> seats(uint32_t bank) {
        return internal::drisc_service::bank_seats(umd, chip_id, *soc, bank);
    }
    DriscCore core(const Seat& seat) { return DriscCore(umd, chip_id, seat.translated); }

    tt::umd::Cluster umd;
    std::optional<metal_SocDescriptor> soc;
};

template <typename F>
bool wait_until(std::chrono::milliseconds timeout, F&& done) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Empty when the seat's kernel stops within stop_timeout and leaves its NIUs as DRISC firmware does.
std::string check_stopped(Device& device, const Seat& seat) {
    const DriscCore core = device.core(seat);
    const auto state = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::state_addr); };
    if (!wait_until(stop_timeout, [&] { return state() == abi::SERVICE_STOPPED; })) {
        return fmt::format(
            "bank {} NoC{} seat {} did not stop: state {:#x}", seat.bank, seat.noc, seat.translated.str(), state());
    }
    const std::array<uint32_t, NUM_NOCS> cfg = core.read_niu_cfg();
    for (uint32_t niu = 0; niu < cfg.size(); niu++) {
        const bool noc2axi = (cfg[niu] & niu_cfg_noc2axi) != 0;
        if (noc2axi != (((seat.noc2axi_mask >> niu) & 1u) != 0)) {
            return fmt::format(
                "bank {} NoC{} seat {} stopped with NIU{} cfg {:#x}, mask {:#x}",
                seat.bank,
                seat.noc,
                seat.translated.str(),
                niu,
                cfg[niu],
                seat.noc2axi_mask);
        }
    }
    return {};
}

// Loads the service on `seat` and starts it with `block`, after filling in the header fields the seat decides.
void launch(Device& device, const Seat& seat, abi::L1CommandBlock block) {
    const DriscCore core = device.core(seat);
    ll_api::ElfFile image;
    image.ReadImage(fmt::format(
        "{}runtime/hw/lib/blackhole/drisc_service_noc{}.elf",
        internal::drisc_service::runtime_options().get_root_dir(),
        seat.noc));
    core.load(image);
    block.header.state = abi::SERVICE_BOOTING;
    block.header.bank_size = seat.bank_size;
    std::copy(seat.bank_noc_xy_addr.begin(), seat.bank_noc_xy_addr.end(), block.header.bank_noc_xy_addr);
    block.header.noc2axi_niu_mask = seat.noc2axi_mask;
    core.write_l1(abi::L1CommandBlockLayout::base_addr, &block, sizeof(block));
    core.start();
}

// ---- Tests ---------------------------------------------------------------------------------------------------------

class DriscService : public ::testing::Test {
protected:
    static void SetUpTestSuite() { device_ = std::make_unique<Device>(); }
    static void TearDownTestSuite() { device_.reset(); }
    void SetUp() override {
        if (!device_->blackhole()) {
            GTEST_SKIP() << "The DRISC service is Blackhole-only";
        }
    }

    static Device& device() { return *device_; }

private:
    static inline std::unique_ptr<Device> device_;
};

// With no client, the kernel boots, sets its NIU modes, stamps its build block, scans its slots and stops. Both NIUs
// start in NOC2AXI mode, so the mode check fails unless the kernel set them.
TEST_F(DriscService, KernelWithNoClientBootsAndStops) {
    for (const Seat& seat : device().seats(0)) {
        const std::optional<internal::drisc_service::SeatLock> lock =
            internal::drisc_service::SeatLock::try_acquire_client(seat.id);
        ASSERT_TRUE(lock.has_value() && lock->sole()) << "bank 0's NoC" << seat.noc << " seat is in use";
        const DriscCore core = device().core(seat);
        core.force_stop();
        const std::array<uint32_t, NUM_NOCS> cfg = core.read_niu_cfg();
        for (uint32_t niu = 0; niu < cfg.size(); niu++) {
            const uint32_t noc2axi = cfg[niu] | niu_cfg_noc2axi;
            device().umd.write_to_device_reg(
                &noc2axi, sizeof(noc2axi), chip_id, seat.translated, NOC_CFG(NIU_CFG_0) + niu * NOC_INSTANCE_OFFSET);
        }
        launch(device(), seat, {});
        const std::string problem = check_stopped(device(), seat);
        EXPECT_TRUE(problem.empty()) << problem;
        abi::KernelBuildBlock build{};
        core.read_l1(abi::L1CommandBlockLayout::build_addr, &build, sizeof(build));
        EXPECT_EQ(build.magic, abi::BUILD_BLOCK_MAGIC) << "NoC" << seat.noc;
        EXPECT_EQ(build.abi_version, abi::ABI_VERSION) << "NoC" << seat.noc;
        EXPECT_GT(core.read_l1_u32(abi::L1CommandBlockLayout::heartbeat_addr), 0u)
            << "NoC" << seat.noc << " never scanned its slots";
    }
}

// A client written against the ABI alone. It attaches at launch, has a CMD_LOCAL and a CMD_SINGLE copy served, reads
// both completions from a control page in Tensix L1 rather than pinned host memory, and detaches.
TEST_F(DriscService, KernelServesSlotCommands) {
    const Seat seat = device().seats(0)[0];
    const std::optional<internal::drisc_service::SeatLock> lock =
        internal::drisc_service::SeatLock::try_acquire_client(seat.id);
    ASSERT_TRUE(lock.has_value() && lock->sole()) << "bank 0's NoC0 seat is in use";
    const DriscCore core = device().core(seat);
    core.force_stop();

    constexpr uint32_t bytes = 64 * 1024 + 48;  // several kernel chunks, and not a whole number of them
    constexpr uint64_t src_offset = 0;
    constexpr uint64_t local_dst_offset = 1024 * 1024;
    const CoreCoord view_endpoint = device().soc->get_preferred_worker_core_for_dram_view(seat.bank, 0);
    const tt::umd::CoreCoord bank{view_endpoint.x, view_endpoint.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
    const tt::umd::CoreCoord tensix = device().soc->get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED).at(0);
    const uint32_t tensix_xy = abi::noc_xy(tensix.x, tensix.y);
    constexpr uint64_t page_addr = MEM_L1_SIZE / 2;
    constexpr uint64_t single_dst_addr = page_addr + abi::CONTROL_PAGE_SIZE;

    std::vector<uint8_t> pattern(bytes);
    for (uint32_t i = 0; i < bytes; i++) {
        pattern[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    const std::vector<uint8_t> zeros(bytes);
    const abi::ControlPage empty_page{};
    device().umd.write_to_device(pattern.data(), bytes, chip_id, bank, src_offset);
    device().umd.write_to_device(zeros.data(), bytes, chip_id, bank, local_dst_offset);
    device().umd.write_to_device(zeros.data(), bytes, chip_id, tensix, single_dst_addr);
    device().umd.write_to_device(&empty_page, sizeof(empty_page), chip_id, tensix, page_addr);

    constexpr uint32_t slot = 0;
    constexpr uint32_t gen = 1;
    abi::L1CommandBlock block{};
    block.slots[slot].control_page = abi::NocEndpoint{page_addr, tensix_xy};
    block.slots[slot].request = abi::slot_word(abi::SLOT_ATTACH, gen);
    launch(device(), seat, block);
    const auto ack = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(slot)); };
    ASSERT_TRUE(wait_until(reply_timeout, [&] { return ack() == abi::slot_word(abi::SLOT_ATTACH, gen); }))
        << "not attached: ack " << ack();

    const std::array<abi::KernelCommand, 2> commands{
        abi::make_local_command(src_offset, local_dst_offset, bytes),
        abi::make_single_command(
            abi::NocEndpoint{src_offset, seat.bank_noc_xy_addr[0]}, abi::NocEndpoint{single_dst_addr, tensix_xy}, bytes),
    };
    core.write_l1(abi::L1CommandBlockLayout::command_queue_addr(slot), commands.data(), sizeof(commands));
    const uint32_t doorbell = commands.size();
    core.write_l1(abi::L1CommandBlockLayout::doorbell_addr(slot), &doorbell, sizeof(doorbell));
    const auto done_count = [&] {
        abi::ControlPage page{};
        device().umd.read_from_device(&page, chip_id, tensix, page_addr, sizeof(page));
        return page.done_count;
    };
    EXPECT_TRUE(wait_until(reply_timeout, [&] { return done_count() == doorbell; }))
        << "done_count " << done_count() << " of " << doorbell;

    std::vector<uint8_t> local_dst(bytes);
    device().umd.read_from_device(local_dst.data(), chip_id, bank, local_dst_offset, bytes);
    EXPECT_EQ(local_dst, pattern) << "CMD_LOCAL";
    std::vector<uint8_t> single_dst(bytes);
    device().umd.read_from_device(single_dst.data(), chip_id, tensix, single_dst_addr, bytes);
    EXPECT_EQ(single_dst, pattern) << "CMD_SINGLE";

    const uint32_t detach = abi::slot_word(abi::SLOT_DETACH, gen);
    core.write_l1(abi::L1CommandBlockLayout::exit_mailbox_addr(slot), &detach, sizeof(detach));
    EXPECT_TRUE(wait_until(reply_timeout, [&] { return ack() == abi::slot_word(abi::SLOT_FREED, gen); }))
        << "not freed: ack " << ack();
    const std::string problem = check_stopped(device(), seat);
    EXPECT_TRUE(problem.empty()) << problem;
}

}  // namespace
}  // namespace tt::tt_metal
