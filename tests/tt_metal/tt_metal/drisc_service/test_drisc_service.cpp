// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs the DRISC service on chip 0, or the chip DRISC_SERVICE_TEST_CHIP names, through a bare UMD cluster, as a client
// process would, with no metal device in this process (metal would take the seat locks itself). Copies are checked
// through UMD's own DRAM reads and writes, never through the service, with a pattern unique to each (nonce, bank,
// address).

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
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
using internal::drisc_service::Client;
using internal::drisc_service::DriscCore;
using internal::drisc_service::IssuePath;
using internal::drisc_service::Seat;
using internal::drisc_service::seat_of;
using internal::drisc_service::SeatClient;

const ChipId chip_id = [] {
    const char* chip = std::getenv("DRISC_SERVICE_TEST_CHIP");
    return chip != nullptr ? static_cast<ChipId>(std::atoi(chip)) : ChipId{0};
}();
constexpr uint64_t dram_test_offset = 1ull << 30;
constexpr uint32_t slot_size = 512 * 1024;
constexpr uint32_t dram_slots = 32;  // per bank, per region
constexpr uint64_t region_size = uint64_t{dram_slots} * slot_size;
constexpr uint32_t host_slots = 16;
constexpr uint32_t max_round_commands = 8;
constexpr std::array<uint32_t, 5> edge_sizes = {64, 4096, 28 * 1024, 28 * 1024 + 64, slot_size};
constexpr uint32_t niu_cfg_noc2axi = 1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE;

constexpr std::chrono::milliseconds drain_timeout{10000};
constexpr std::chrono::milliseconds stop_timeout{2000};
constexpr std::chrono::milliseconds reply_timeout{2000};
// Long enough for the kernel to run many slot scans.
constexpr std::chrono::milliseconds scan_wait{200};

enum class Op : uint32_t { h2d, d2h, d2d, local };
constexpr std::array<Op, 4> all_ops = {Op::h2d, Op::d2h, Op::d2d, Op::local};

std::string_view op_name(Op op) {
    switch (op) {
        case Op::h2d: return "host->DRAM";
        case Op::d2h: return "DRAM->host";
        case Op::d2d: return "DRAM->DRAM";
        case Op::local: return "local DMA";
    }
    return "?";
}

uint64_t next_nonce() {
    static uint64_t calls = 0;
    const auto now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return now ^ (++calls * 0xD1B54A32D192ED03ull) ^ (static_cast<uint64_t>(::getpid()) << 40);
}

uint32_t pattern_word(uint64_t nonce, uint32_t bank, uint64_t dram_addr) {
    const uint64_t word_index = (static_cast<uint64_t>(bank) << 40) | (dram_addr / sizeof(uint32_t));
    const uint64_t mixed = nonce ^ (word_index * 0x9E3779B97F4A7C15ull);
    return static_cast<uint32_t>(mixed ^ (mixed >> 32));
}

void fill_pattern(uint32_t* words, uint32_t size, uint64_t nonce, uint32_t bank, uint64_t dram_addr) {
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        words[i] = pattern_word(nonce, bank, dram_addr + i * sizeof(uint32_t));
    }
}

// Empty when `words` hold the pattern, else the first mismatch.
std::string pattern_mismatch(const uint32_t* words, uint32_t size, uint64_t nonce, uint32_t bank, uint64_t dram_addr) {
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        const uint32_t expected = pattern_word(nonce, bank, dram_addr + i * sizeof(uint32_t));
        if (words[i] != expected) {
            return fmt::format(
                "size {}, pattern of bank {} DRAM {:#x}: first mismatch at byte {:#x}, got {:#x}, expected {:#x}",
                size,
                bank,
                dram_addr,
                i * sizeof(uint32_t),
                words[i],
                expected);
        }
    }
    return {};
}

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
    uint64_t bank_offset(uint32_t bank) const { return soc->get_address_offset(static_cast<int>(bank)); }

    // `bank` at `addr` through its `noc` endpoint, the one a seat on that NoC must name.
    abi::NocEndpoint dram_endpoint(uint32_t bank, uint64_t addr, uint32_t noc) const {
        const CoreCoord endpoint = soc->get_preferred_worker_core_for_dram_view(static_cast<int>(bank), noc);
        return {bank_offset(bank) + addr, abi::noc_xy(endpoint.x, endpoint.y)};
    }

    void write_dram(uint32_t bank, uint64_t addr, const void* data, uint32_t size) {
        umd.write_to_device(data, size, chip_id, noc0_endpoint(bank), bank_offset(bank) + addr);
    }
    void read_dram(uint32_t bank, uint64_t addr, void* data, uint32_t size) {
        umd.read_from_device(data, chip_id, noc0_endpoint(bank), bank_offset(bank) + addr, size);
    }

    tt::umd::Cluster umd;
    std::optional<metal_SocDescriptor> soc;

private:
    tt::umd::CoreCoord noc0_endpoint(uint32_t bank) const {
        const CoreCoord endpoint = soc->get_preferred_worker_core_for_dram_view(static_cast<int>(bank), 0);
        return {endpoint.x, endpoint.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED};
    }
};

// Host memory the service copies to and from. Declare it before the Client that copies into it.
class HostMemory {
public:
    HostMemory(Device& device, size_t size) :
        buffer_(internal::drisc_service::allocate_host_buffer(device.umd, chip_id, size, device_addr_)) {}
    uint32_t* at(uint64_t offset) { return static_cast<uint32_t*>(buffer_->get_va()) + offset / sizeof(uint32_t); }
    uint64_t device_addr(uint64_t offset) const { return device_addr_ + offset; }

private:
    uint64_t device_addr_ = 0;
    std::unique_ptr<tt::umd::SysmemBuffer> buffer_;
};

// Issues verified copies through one Client, each command in its own DRAM and host slots. Its DRAM lives in region
// `region` of every bank, so clients with distinct regions never touch each other's data.
class Exerciser {
public:
    Exerciser(Device& device, Client& client, uint32_t region, uint64_t seed) :
        device_(device),
        client_(client),
        host_(device, uint64_t{host_slots} * slot_size),
        base_(dram_test_offset + region * region_size),
        rng_(seed),
        next_dram_(device.num_banks(), 0) {}

    // Stages and issues `count` commands of `op` on seat (`bank`, `noc`), one by one or as one batch.
    void start(uint32_t bank, uint32_t noc, Op op, uint32_t count, bool batch) {
        TT_FATAL(count <= max_round_commands, "a round runs at most {} commands", max_round_commands);
        TT_FATAL(started_ == nullptr, "finish() the started round first");
        started_ = &client_.seat(bank, noc);
        started_name_ = fmt::format("{} on bank {} NoC{} ({})", op_name(op), bank, noc, batch ? "batch" : "single");
        std::vector<abi::KernelCommand> commands;
        commands.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            commands.push_back(stage(*started_, op, bank, noc, checks_));
        }
        if (batch) {
            issue_with_room(*started_, [&] { return started_->issue_batch(commands); });
        } else {
            for (const auto& command : commands) {
                issue_with_room(*started_, [&] { return started_->issue(command); });
            }
        }
    }

    // A random round on seat (`bank`, `noc`).
    void start_random(uint32_t bank, uint32_t noc) {
        const Op op = all_ops[below(all_ops.size())];
        const auto count = 1 + static_cast<uint32_t>(below(max_round_commands));
        start(bank, noc, op, count, below(2) == 0);
    }

    // Drains the started round and checks it. Returns the first mismatch, or empty.
    std::string finish() {
        started_->drain(drain_timeout);
        started_ = nullptr;
        std::vector<Check> checks = std::move(checks_);
        checks_.clear();
        for (const Check& check : checks) {
            const std::string mismatch = verify(check);
            if (!mismatch.empty()) {
                return started_name_ + ": " + mismatch;
            }
        }
        return {};
    }

    std::string round(uint32_t bank, uint32_t noc, Op op, uint32_t count, bool batch) {
        start(bank, noc, op, count, batch);
        return finish();
    }

    std::string random_round() {
        const auto& banks = client_.banks();
        start_random(banks[below(banks.size())], static_cast<uint32_t>(below(2)));
        return finish();
    }

private:
    struct Check {
        bool host;  // the data lands in host memory, at `addr` = host offset; else in DRAM `bank` at `addr`
        uint32_t bank;
        uint64_t addr;
        uint32_t size;
        uint64_t nonce;
        uint32_t key_bank;  // where the source pattern was keyed
        uint64_t key_addr;
    };

    uint64_t below(uint64_t n) { return std::uniform_int_distribution<uint64_t>(0, n - 1)(rng_); }

    uint32_t random_size() {
        if (below(2) == 0) {
            return edge_sizes[below(edge_sizes.size())];
        }
        return static_cast<uint32_t>(64 * (1 + below(slot_size / 64)));
    }

    uint64_t next_dram(uint32_t bank) { return base_ + (next_dram_[bank]++ % dram_slots) * uint64_t{slot_size}; }
    uint64_t next_host() { return (next_host_++ % host_slots) * uint64_t{slot_size}; }

    void seed_dram(uint32_t bank, uint64_t addr, uint32_t size, uint64_t nonce) {
        std::vector<uint32_t> words(size / sizeof(uint32_t));
        fill_pattern(words.data(), size, nonce, bank, addr);
        device_.write_dram(bank, addr, words.data(), size);
    }

    abi::KernelCommand stage(SeatClient& seat, Op op, uint32_t bank, uint32_t noc, std::vector<Check>& checks) {
        const uint32_t size = random_size();
        const uint64_t nonce = next_nonce();
        abi::NocEndpoint src{};
        abi::NocEndpoint dst{};
        switch (op) {
            case Op::h2d: {
                const uint64_t host = next_host();
                const uint64_t addr = next_dram(bank);
                fill_pattern(host_.at(host), size, nonce, bank, addr);
                src = seat.host_endpoint(host_.device_addr(host));
                dst = device_.dram_endpoint(bank, addr, noc);
                checks.push_back({false, bank, addr, size, nonce, bank, addr});
                break;
            }
            case Op::d2h: {
                const uint64_t addr = next_dram(bank);
                const uint64_t host = next_host();
                seed_dram(bank, addr, size, nonce);
                std::memset(host_.at(host), 0, size);
                src = device_.dram_endpoint(bank, addr, noc);
                dst = seat.host_endpoint(host_.device_addr(host));
                checks.push_back({true, 0, host, size, nonce, bank, addr});
                break;
            }
            case Op::d2d: {
                const uint64_t src_addr = next_dram(bank);
                const auto dst_bank = static_cast<uint32_t>(below(device_.num_banks()));
                const uint64_t dst_addr = next_dram(dst_bank);
                seed_dram(bank, src_addr, size, nonce);
                src = device_.dram_endpoint(bank, src_addr, noc);
                dst = device_.dram_endpoint(dst_bank, dst_addr, noc);
                checks.push_back({false, dst_bank, dst_addr, size, nonce, bank, src_addr});
                break;
            }
            case Op::local: {
                const uint64_t src_addr = next_dram(bank);
                const uint64_t dst_addr = next_dram(bank);
                seed_dram(bank, src_addr, size, nonce);
                checks.push_back({false, bank, dst_addr, size, nonce, bank, src_addr});
                return abi::make_local_command(
                    device_.bank_offset(bank) + src_addr, device_.bank_offset(bank) + dst_addr, size);
            }
        }
        return abi::make_single_command(src, dst, size);
    }

    std::string verify(const Check& check) {
        if (check.host) {
            return pattern_mismatch(host_.at(check.addr), check.size, check.nonce, check.key_bank, check.key_addr);
        }
        std::vector<uint32_t> words(check.size / sizeof(uint32_t));
        device_.read_dram(check.bank, check.addr, words.data(), check.size);
        const std::string mismatch =
            pattern_mismatch(words.data(), check.size, check.nonce, check.key_bank, check.key_addr);
        return mismatch.empty() ? mismatch
                                : fmt::format("into bank {} DRAM {:#x}: {}", check.bank, check.addr, mismatch);
    }

    template <typename F>
    static void issue_with_room(const SeatClient& seat, F&& try_issue) {
        const auto deadline = std::chrono::steady_clock::now() + drain_timeout;
        while (!try_issue()) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "the queue stayed full for {} ms with {} in flight",
                drain_timeout.count(),
                seat.in_flight());
        }
    }

    Device& device_;
    Client& client_;
    HostMemory host_;
    uint64_t base_;
    std::mt19937_64 rng_;
    std::vector<uint32_t> next_dram_;
    uint32_t next_host_ = 0;
    SeatClient* started_ = nullptr;
    std::string started_name_;
    std::vector<Check> checks_;
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

bool heartbeat_moves(const SeatClient& seat) {
    const uint32_t before = seat.heartbeat();
    return wait_until(std::chrono::milliseconds(1000), [&] { return seat.heartbeat() != before; });
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
    static uint32_t last_bank() { return device().num_banks() - 1; }
    static std::vector<uint32_t> all_banks() {
        std::vector<uint32_t> banks(device().num_banks());
        std::iota(banks.begin(), banks.end(), 0);
        return banks;
    }
    // Client is neither copyable nor movable, so tests hold it in an optional and reset() it to detach.
    static void attach(
        std::optional<Client>& client, std::vector<uint32_t> banks, IssuePath issue_path = IssuePath::Window) {
        client.emplace(device().umd, chip_id, std::move(banks), internal::drisc_service::runtime_root(), issue_path);
    }

    static void expect_all_stopped(const std::vector<uint32_t>& banks) {
        for (const uint32_t bank : banks) {
            for (const Seat& seat : device().seats(bank)) {
                const std::string problem = check_stopped(device(), seat);
                EXPECT_TRUE(problem.empty()) << problem;
            }
        }
    }

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

TEST_F(DriscService, FirstClientLaunchesAndLastStops) {
    std::optional<Client> client;
    attach(client, {0});
    for (uint32_t noc = 0; noc < 2; noc++) {
        const SeatClient& seat = client->seat(0, noc);
        EXPECT_TRUE(seat.launched()) << "NoC" << noc;
        EXPECT_EQ(seat.state(), abi::SERVICE_RUNNING) << "NoC" << noc;
        EXPECT_TRUE(heartbeat_moves(seat)) << "NoC" << noc << " kernel loop is not running";
    }
    client.reset();
    expect_all_stopped({0});
}

// Every seat copies in every direction, and names its bank as the SoC descriptor does.
TEST_F(DriscService, CopiesOnBothSeatsOfTwoBanks) {
    std::optional<Client> client;
    attach(client, {0, last_bank()});
    Exerciser exerciser(device(), *client, 0, 1);
    for (const uint32_t bank : client->banks()) {
        for (uint32_t noc = 0; noc < 2; noc++) {
            const abi::NocEndpoint expected = device().dram_endpoint(bank, dram_test_offset, noc);
            const abi::NocEndpoint named = client->seat(bank, noc).bank_endpoint(dram_test_offset);
            EXPECT_EQ(named.local_addr, expected.local_addr) << "bank " << bank << " NoC" << noc;
            EXPECT_EQ(named.noc_xy_addr, expected.noc_xy_addr) << "bank " << bank << " NoC" << noc;
            for (const Op op : all_ops) {
                for (const bool batch : {false, true}) {
                    const std::string mismatch = exerciser.round(bank, noc, op, max_round_commands, batch);
                    EXPECT_TRUE(mismatch.empty()) << mismatch;
                }
            }
        }
    }
    client.reset();
    expect_all_stopped({0, last_bank()});
}

// One DRAM->host copy issued on every seat of two banks, none drained on its own: client.drain() returns only once
// every seat is idle and every copy has landed.
TEST_F(DriscService, ClientDrainWaitsForEverySeat) {
    constexpr uint32_t copy_size = 1 << 20;
    std::optional<Client> client;
    attach(client, {0, last_bank()});
    HostMemory host(device(), 2 * client->banks().size() * copy_size);
    const auto src = [](uint32_t noc) { return dram_test_offset + noc * copy_size; };
    std::vector<uint64_t> nonces;
    for (const uint32_t bank : client->banks()) {
        for (uint32_t noc = 0; noc < 2; noc++) {
            const uint64_t nonce = next_nonce();
            std::vector<uint32_t> words(copy_size / sizeof(uint32_t));
            fill_pattern(words.data(), copy_size, nonce, bank, src(noc));
            device().write_dram(bank, src(noc), words.data(), copy_size);
            SeatClient& seat = client->seat(bank, noc);
            EXPECT_TRUE(seat.issue(abi::make_single_command(
                device().dram_endpoint(bank, src(noc), noc),
                seat.host_endpoint(host.device_addr(nonces.size() * copy_size)),
                copy_size)));
            nonces.push_back(nonce);
        }
    }
    client->drain(drain_timeout);
    uint32_t index = 0;
    for (const uint32_t bank : client->banks()) {
        for (uint32_t noc = 0; noc < 2; noc++, index++) {
            EXPECT_EQ(client->seat(bank, noc).in_flight(), 0u);
            EXPECT_EQ(pattern_mismatch(host.at(index * copy_size), copy_size, nonces[index], bank, src(noc)), "");
        }
    }
    client.reset();
    expect_all_stopped({0, last_bank()});
}

TEST_F(DriscService, SecondClientJoinsAndOutlivesTheFirst) {
    std::optional<Client> first;
    std::optional<Client> second;
    attach(first, {0});
    attach(second, {0});
    for (uint32_t noc = 0; noc < 2; noc++) {
        EXPECT_FALSE(second->seat(0, noc).launched()) << "NoC" << noc << ": a second client relaunched the kernel";
        EXPECT_NE(second->seat(0, noc).slot(), first->seat(0, noc).slot()) << "NoC" << noc;
    }
    first.reset();
    Exerciser exerciser(device(), *second, 0, 2);
    for (uint32_t noc = 0; noc < 2; noc++) {
        EXPECT_EQ(second->seat(0, noc).state(), abi::SERVICE_RUNNING) << "NoC" << noc;
        for (const Op op : all_ops) {
            const std::string mismatch = exerciser.round(0, noc, op, 4, false);
            EXPECT_TRUE(mismatch.empty()) << mismatch;
        }
    }
    second.reset();
    expect_all_stopped({0});
}

TEST_F(DriscService, NinthClientIsRefused) {
    std::vector<std::optional<Client>> clients(abi::MAX_CLIENTS);
    for (auto& client : clients) {
        attach(client, {0});
    }
    std::optional<Client> ninth;
    try {
        attach(ninth, {0});
        ADD_FAILURE() << "a ninth client attached to a seat with " << abi::MAX_CLIENTS << " clients";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string_view(e.what()).find("client slots"), std::string_view::npos) << e.what();
    }
    Exerciser exerciser(device(), *clients.back(), 0, 3);
    for (uint32_t noc = 0; noc < 2; noc++) {
        const std::string mismatch = exerciser.round(0, noc, Op::local, 4, true);
        EXPECT_TRUE(mismatch.empty()) << "after the refusal: " << mismatch;
    }
    ninth.reset();
    clients.clear();
    expect_all_stopped({0});
}

TEST_F(DriscService, StaleDetachIsIgnored) {
    std::optional<Client> client;
    attach(client, {0});
    const SeatClient& seat = client->seat(0, 0);
    const DriscCore core = device().core(seat_of(seat));
    const uint32_t stale_gen = seat.gen() == abi::SLOT_GEN_MASK ? 1 : seat.gen() + 1;
    core.write_l1_u32(
        abi::L1CommandBlockLayout::exit_mailbox_addr(seat.slot()), abi::slot_word(abi::SLOT_DETACH, stale_gen));
    std::this_thread::sleep_for(scan_wait);
    EXPECT_EQ(
        core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(seat.slot())),
        abi::slot_word(abi::SLOT_ATTACH, seat.gen()))
        << "a DETACH of another generation detached the slot";
    EXPECT_EQ(seat.state(), abi::SERVICE_RUNNING);
    Exerciser exerciser(device(), *client, 0, 4);
    const std::string mismatch = exerciser.round(0, 0, Op::d2h, 4, false);
    EXPECT_TRUE(mismatch.empty()) << mismatch;
    client.reset();
    expect_all_stopped({0});
}

// A detach holds the seat's gate until the KMD's own DETACH is sent, so no client takes the freed slot before then.
TEST_F(DriscService, DetachWaitsForTheSeatGate) {
    std::optional<Client> client;
    attach(client, {0});
    const SeatClient& seat = client->seat(0, 0);
    const DriscCore core = device().core(seat_of(seat));
    const uint32_t ack_addr = abi::L1CommandBlockLayout::ack_addr(seat.slot());
    const uint32_t attached = abi::slot_word(abi::SLOT_ATTACH, seat.gen());
    std::optional<internal::drisc_service::SeatGate> gate(
        std::in_place, seat_of(seat).id, std::chrono::milliseconds(1000));
    std::thread detach([&] { client.reset(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(core.read_l1_u32(ack_addr), attached) << "the client detached without taking the seat's gate";
    gate.reset();
    detach.join();
    EXPECT_NE(core.read_l1_u32(ack_addr), attached) << "the slot was not freed once the gate was free";
    expect_all_stopped({0});
}

TEST_F(DriscService, MetalSeatLockRefusesClients) {
    const Seat seat = device().seats(0)[0];
    const std::optional<internal::drisc_service::SeatLock> metal =
        internal::drisc_service::SeatLock::try_acquire_exclusive(seat.id);
    ASSERT_TRUE(metal.has_value()) << "something else holds bank 0's NoC0 seat";
    std::optional<Client> client;
    try {
        attach(client, {0});
        ADD_FAILURE() << "a client attached to a seat metal holds";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string_view(e.what()).find("held exclusively"), std::string_view::npos) << e.what();
    }
}

// A client refused its last seat detaches the seats it already took, so their kernels stop.
TEST_F(DriscService, RefusedClientReleasesItsOtherSeats) {
    const Seat seat = device().seats(last_bank())[1];
    const std::optional<internal::drisc_service::SeatLock> metal =
        internal::drisc_service::SeatLock::try_acquire_exclusive(seat.id);
    ASSERT_TRUE(metal.has_value()) << "something else holds the last bank's NoC1 seat";
    std::optional<Client> client;
    EXPECT_ANY_THROW(attach(client, {0, last_bank()})) << "a client attached to a seat metal holds";
    expect_all_stopped({0});
    const std::string problem = check_stopped(device(), device().seats(last_bank())[0]);
    EXPECT_TRUE(problem.empty()) << problem;
}

// issue() and issue_batch() refuse what the kernel would run wrongly, issuing nothing, and the queue stays usable.
// A bank past the chip's last one fails the client before any seat attaches; seat() rejects a bank the client did not
// attach and a NoC other than 0 or 1.
// A seat that has not run the service since the last reset holds no service state, so the refused client is checked by
// comparing state words, not by expecting STOPPED.
TEST_F(DriscService, RejectsBadBanksAndSeats) {
    const auto states = [] {
        std::vector<uint32_t> words;
        for (const uint32_t bank : all_banks()) {
            for (const Seat& seat : device().seats(bank)) {
                words.push_back(device().core(seat).read_l1_u32(abi::L1CommandBlockLayout::state_addr));
            }
        }
        return words;
    };
    const std::vector<uint32_t> before = states();
    std::optional<Client> client;
    EXPECT_THROW(client.emplace(device().umd, chip_id, std::vector<uint32_t>{0, device().num_banks()}), std::exception);
    EXPECT_FALSE(client.has_value());
    EXPECT_EQ(states(), before) << "a refused client changed a seat's state";
    attach(client, {0});
    EXPECT_THROW(client->seat(last_bank(), 0), std::exception);
    EXPECT_THROW(client->seat(0, 2), std::exception);
    client.reset();
    expect_all_stopped({0});
}

TEST_F(DriscService, RejectsInvalidCommands) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    const uint64_t base = device().bank_offset(0) + dram_test_offset;
    // Far enough apart that only the check named by each case below can reject it.
    const abi::KernelCommand valid = abi::make_local_command(base, base + 2 * slot_size, slot_size);
    const auto with = [&](auto change) {
        abi::KernelCommand command = valid;
        change(command);
        return command;
    };
    const auto expect_rejected = [](auto issue, std::string_view reason, const std::string& what) {
        try {
            (void)issue();
            ADD_FAILURE() << what << " was accepted";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string_view(e.what()).find(reason), std::string_view::npos)
                << what << " was rejected for another reason: " << e.what();
        }
    };
    constexpr uint16_t wide_coords = 1u << abi::CMD_KIND_SHIFT;
    const std::vector<std::tuple<std::string, std::string_view, abi::KernelCommand>> invalid = {
        {"batch kind", "neither CMD_SINGLE nor CMD_LOCAL", abi::make_batch_command(seat.host_endpoint(0), 1)},
        {"size 0", "size is 0", with([](auto& c) { c.size = 0; })},
        {"wide src coords", "do not fit", with([&](auto& c) { c.src_noc_xy_addr = wide_coords; })},
        {"misaligned src", "B aligned", with([](auto& c) { c.src_addr += 16; })},
        {"misaligned dst", "B aligned", with([](auto& c) { c.dst_addr += 16; })},
        {"overlap", "overlap", with([](auto& c) { c.dst_addr = c.src_addr + 64; })},
        {"local size off the 16 B grid", "is not a multiple of", with([](auto& c) { c.size += 8; })},
        {"local past the bank", "runs past", with([&](auto& c) { c.dst_addr = seat_of(seat).bank_size - 64; })},
    };
    for (const auto& [what, reason, command] : invalid) {
        expect_rejected([&] { return seat.issue(command); }, reason, what);
        expect_rejected([&] { return seat.issue_batch(std::array{valid, command}); }, reason, "batch with " + what);
    }
    EXPECT_ANY_THROW((void)seat.issue_batch({})) << "empty batch";
    const std::vector<abi::KernelCommand> too_long(abi::BATCH_MAX_COMMANDS + 1, valid);
    EXPECT_ANY_THROW((void)seat.issue_batch(too_long)) << abi::BATCH_MAX_COMMANDS + 1 << " commands";
    EXPECT_EQ(seat.in_flight(), 0u) << "a rejected command was issued";

    // On this NoC0 seat a NoC1 endpoint reaches a stream-mode NIU, except as an own-bank source read by DMA.
    const abi::NocEndpoint noc0_own = device().dram_endpoint(0, dram_test_offset + 2 * slot_size, 0);
    const abi::NocEndpoint noc1_own = device().dram_endpoint(0, dram_test_offset, 1);
    const abi::NocEndpoint noc1_other = device().dram_endpoint(1, dram_test_offset, 1);
    const abi::NocEndpoint noc0_other = device().dram_endpoint(1, dram_test_offset + 2 * slot_size, 0);
    const std::vector<std::pair<std::string, abi::KernelCommand>> wrong_noc = {
        {"src at another bank's NoC1 endpoint", abi::make_single_command(noc1_other, noc0_own, slot_size)},
        {"dst at another bank's NoC1 endpoint", abi::make_single_command(noc0_own, noc1_other, slot_size)},
        {"dst at its own bank's NoC1 endpoint", abi::make_single_command(noc0_other, noc1_own, slot_size)},
        {"own-bank src off the 16 B grid at its NoC1 endpoint", abi::make_single_command(noc1_own, noc0_own, 72)},
    };
    for (const auto& [what, command] : wrong_noc) {
        expect_rejected([&] { return seat.issue(command); }, "stream mode", what);
        expect_rejected(
            [&] { return seat.issue_batch(std::array{valid, command}); }, "stream mode", "batch with " + what);
    }
    EXPECT_EQ(seat.in_flight(), 0u) << "a command naming a stream-mode NIU was issued";
    ASSERT_TRUE(seat.issue(abi::make_single_command(noc1_own, noc0_other, slot_size)))
        << "an own-bank source at its NoC1 endpoint, read by DMA, found the queue full";
    seat.drain(drain_timeout);

    Exerciser exerciser(device(), *client, 0, 8);
    const std::string mismatch = exerciser.round(0, 0, Op::h2d, 4, true);
    EXPECT_TRUE(mismatch.empty()) << "after the rejections: " << mismatch;
    client.reset();
    expect_all_stopped({0});
}

// A single command within the seat's bank runs as a local copy, so it lands whichever endpoints it names; one off
// the 16 B grid stays a single command.
TEST_F(DriscService, SameBankSingleRunsAsLocal) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    constexpr uint32_t size = 64 * 1024;
    const uint64_t src = dram_test_offset;
    const uint64_t dst = dram_test_offset + slot_size;
    std::vector<uint32_t> payload(size / sizeof(uint32_t));
    for (uint32_t i = 0; i < payload.size(); i++) {
        payload[i] = 0x5A5A0000u ^ i;
    }
    device().write_dram(0, src, payload.data(), size);
    for (const bool batched : {false, true}) {
        const std::vector<uint32_t> zeros(payload.size(), 0);
        device().write_dram(0, dst, zeros.data(), size);
        // NoC1 endpoints on a NoC0 seat: refused as a single command, fine as a local copy.
        const abi::KernelCommand command =
            abi::make_single_command(device().dram_endpoint(0, src, 1), device().dram_endpoint(0, dst, 1), size);
        ASSERT_TRUE(batched ? seat.issue_batch(std::array{command}) : seat.issue(command));
        seat.drain(drain_timeout);
        std::vector<uint32_t> landed(payload.size());
        device().read_dram(0, dst, landed.data(), size);
        EXPECT_EQ(landed, payload) << (batched ? "batched" : "single") << " same-bank copy";
    }
    EXPECT_ANY_THROW((void)seat.issue(
        abi::make_single_command(device().dram_endpoint(0, src, 1), device().dram_endpoint(0, dst, 1), size + 8)))
        << "a same-bank copy off the 16 B grid stayed a single command at stream-mode endpoints";
    client.reset();
    expect_all_stopped({0});
}

// The control page carries the heartbeat often enough for a host watchdog, and done_count() follows completed().
TEST_F(DriscService, ControlPageCarriesHeartbeatAndDoneCount) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    using clock = std::chrono::steady_clock;
    uint32_t last = seat.heartbeat();
    uint32_t changes = 0;
    auto last_change = clock::now();
    auto longest = clock::duration::zero();
    for (const auto end = last_change + std::chrono::seconds(1); clock::now() < end;) {
        if (const uint32_t value = seat.heartbeat(); value != last) {
            const auto now = clock::now();
            longest = std::max(longest, now - last_change);
            last_change = now;
            last = value;
            changes++;
        }
    }
    longest = std::max(longest, clock::now() - last_change);
    EXPECT_GT(changes, 10u) << "the control page heartbeat barely moved in 1 s";
    EXPECT_LT(longest, std::chrono::milliseconds(50))
        << "the control page heartbeat stood still for "
        << std::chrono::duration_cast<std::chrono::milliseconds>(longest).count() << " ms";

    Exerciser exerciser(device(), *client, 1, 9);
    for (uint32_t i = 0; i < 4; i++) {
        ASSERT_EQ(exerciser.round(0, 0, Op::local, 4, true), "");
    }
    EXPECT_EQ(seat.done_count(), 4u) << "one batch per round";
    EXPECT_EQ(seat.done_count(), static_cast<uint32_t>(seat.completed()));
}

// A seat's NIUs keep whatever mode they were left in, such as NOC2AXI on both after a chip reset before any tt-metal
// open: the kernel sets the seat's layout when it boots.
TEST_F(DriscService, KernelSetsNiuModesAtBoot) {
    for (const Seat& seat : device().seats(0)) {
        const std::array<uint32_t, NUM_NOCS> cfg = device().core(seat).read_niu_cfg();
        for (uint32_t niu = 0; niu < cfg.size(); niu++) {
            const uint32_t noc2axi = cfg[niu] | niu_cfg_noc2axi;
            device().umd.write_to_device_reg(
                &noc2axi, sizeof(noc2axi), chip_id, seat.translated, NOC_CFG(NIU_CFG_0) + niu * NOC_INSTANCE_OFFSET);
        }
    }
    std::optional<Client> client;
    attach(client, {0});
    Exerciser exerciser(device(), *client, 1, 11);
    for (uint32_t noc = 0; noc < 2; noc++) {
        EXPECT_EQ(exerciser.round(0, noc, Op::d2h, 4, true), "") << "NoC" << noc;
    }
    client.reset();
    expect_all_stopped({0});
}

// A kernel that stops beating while it serves a client is not relaunched under it by a joining client, and is
// relaunched by the next client that holds the seat alone.
TEST_F(DriscService, HungKernelIsRelaunchedOnlyBySoleClient) {
    std::optional<Client> first;
    attach(first, {0});
    const Seat seat = seat_of(first->seat(0, 0));
    device().core(seat).force_stop();
    ASSERT_EQ(first->seat(0, 0).state(), abi::SERVICE_RUNNING) << "the forced stop changed the kernel's state word";
    ASSERT_FALSE(heartbeat_moves(first->seat(0, 0))) << "the kernel kept running through a forced stop";

    std::optional<Client> joiner;
    try {
        attach(joiner, {0});
        ADD_FAILURE() << "a client joined a hung kernel that serves another client";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string_view(e.what()).find("and serves other clients"), std::string_view::npos) << e.what();
    }
    // Its slot is never freed, so this logs and leaks the client's host buffers.
    first.reset();

    std::optional<Client> sole;
    attach(sole, {0});
    EXPECT_TRUE(sole->seat(0, 0).launched()) << "the sole client did not relaunch the hung kernel";
    EXPECT_EQ(sole->seat(0, 0).state(), abi::SERVICE_RUNNING);
    EXPECT_TRUE(heartbeat_moves(sole->seat(0, 0))) << "the relaunched kernel is not running";
    Exerciser exerciser(device(), *sole, 0, 9);
    for (const Op op : all_ops) {
        const std::string mismatch = exerciser.round(0, 0, op, 4, false);
        EXPECT_TRUE(mismatch.empty()) << "after the relaunch: " << mismatch;
    }
    sole.reset();
    expect_all_stopped({0});
}

// issue() and issue_batch() return consecutive tickets (a batch takes one), and done() turns true for each, without
// blocking, once it completes.
TEST_F(DriscService, Tickets) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    constexpr uint32_t big = 64 << 20;
    const uint64_t big_base = device().bank_offset(0) + (3ull << 30);
    const abi::KernelCommand slow = abi::make_local_command(big_base, big_base + big, big);
    std::vector<uint64_t> tickets;
    for (uint32_t i = 0; i + 1 < abi::CLIENT_QUEUE_CAPACITY; i++) {
        const std::optional<uint64_t> ticket = seat.issue(slow);
        ASSERT_TRUE(ticket) << "command " << i << " found the queue full";
        tickets.push_back(*ticket);
    }
    const std::optional<uint64_t> batch_ticket = seat.issue_batch(std::array{slow, slow});
    ASSERT_TRUE(batch_ticket) << "the batch found the queue full";
    tickets.push_back(*batch_ticket);
    for (size_t i = 1; i < tickets.size(); i++) {
        EXPECT_EQ(tickets[i], tickets[0] + i) << "ticket " << i;
    }
    EXPECT_FALSE(seat.done(tickets.back())) << "the last of 9 x 64 MB copies was done right after issue";
    const auto deadline = std::chrono::steady_clock::now() + drain_timeout;
    for (const uint64_t ticket : tickets) {
        while (!seat.done(ticket)) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                << "ticket " << ticket << " not done; completed " << seat.completed();
        }
    }
    EXPECT_EQ(seat.completed(), tickets.back());
    EXPECT_EQ(seat.in_flight(), 0u);
}

// A DETACH with commands in flight frees the slot only once the kernel is done with them, and after that the kernel
// never writes the client's host memory again: the window in which the KMD unpins a dead client's pages.
TEST_F(DriscService, DetachWithWorkInFlight) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    constexpr uint32_t big = 64 << 20;
    const uint64_t big_base = device().bank_offset(0) + (3ull << 30);
    ASSERT_TRUE(seat.issue(abi::make_local_command(big_base, big_base + big, big)));
    HostMemory host(device(), slot_size);
    std::memset(host.at(0), 0xA5, slot_size);
    const abi::NocEndpoint src = device().dram_endpoint(0, dram_test_offset, 0);
    const abi::NocEndpoint dst = seat.host_endpoint(host.device_addr(0));
    ASSERT_TRUE(seat.issue(abi::make_single_command(src, dst, slot_size)));

    const DriscCore core = device().core(seat_of(seat));
    core.write_l1_u32(
        abi::L1CommandBlockLayout::exit_mailbox_addr(seat.slot()), abi::slot_word(abi::SLOT_DETACH, seat.gen()));
    const uint32_t freed = abi::slot_word(abi::SLOT_FREED, seat.gen());
    const auto ack = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(seat.slot())); };
    ASSERT_TRUE(wait_until(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS), [&] { return ack() == freed; }))
        << "ack " << ack() << ", want " << freed;
    const std::vector<uint32_t> at_free(host.at(0), host.at(0) + slot_size / sizeof(uint32_t));
    const uint32_t in_flight_at_free = seat.in_flight();
    const uint32_t heartbeat_at_free = seat.heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(std::equal(at_free.begin(), at_free.end(), host.at(0))) << "the kernel wrote host memory after freeing";
    EXPECT_EQ(seat.in_flight(), in_flight_at_free) << "the kernel published a done_count after freeing";
    EXPECT_EQ(seat.heartbeat(), heartbeat_at_free) << "the kernel published a heartbeat after freeing";
    // Drains in vain for abi::DETACH_TIMEOUT_MS, as its done_count stays behind, then finds the slot already freed.
    client.reset();
    expect_all_stopped({0});
}

// Three clients run random verified rounds (singles, batches, local DMA) on one seat at the same time.
TEST_F(DriscService, SharedSeatKeepsEveryByte) {
    constexpr uint32_t num_clients = 3;
    constexpr uint32_t rounds = 24;
    std::array<std::optional<Client>, num_clients> clients;
    std::array<std::optional<Exerciser>, num_clients> exercisers;
    for (uint32_t i = 0; i < num_clients; i++) {
        attach(clients[i], {0});
        exercisers[i].emplace(device(), *clients[i], 1 + i, 20 + i);
    }
    for (uint32_t r = 0; r < rounds; r++) {
        for (auto& exerciser : exercisers) {
            exerciser->start_random(0, r % 2);
        }
        for (uint32_t i = 0; i < num_clients; i++) {
            const std::string mismatch = exercisers[i]->finish();
            EXPECT_TRUE(mismatch.empty()) << "round " << r << ", client " << i << ": " << mismatch;
        }
    }
    for (auto& client : clients) {
        client.reset();
    }
    expect_all_stopped({0});
}

// A client on IssuePath::Shared runs random verified rounds on both seats of two banks.
TEST_F(DriscService, SharedIssueKeepsEveryByte) {
    std::optional<Client> client;
    attach(client, {0, last_bank()}, IssuePath::Shared);
    Exerciser exerciser(device(), *client, 1, 40);
    for (uint32_t r = 0; r < 24; r++) {
        const std::string mismatch = exerciser.round(
            r % 2 == 0 ? 0 : last_bank(), (r / 2) % 2, all_ops[r % all_ops.size()], max_round_commands, r % 3 == 0);
        EXPECT_TRUE(mismatch.empty()) << "round " << r << ": " << mismatch;
    }
    client.reset();
    expect_all_stopped({0, last_bank()});
}

// A client on every bank maps windows for its first max_windows_per_chip seats only; the rest issue through UMD's
// shared window, and every seat copies every byte right.
TEST_F(DriscService, WindowsAreCappedPerChip) {
    std::optional<Client> client;
    attach(client, all_banks());
    Exerciser exerciser(device(), *client, 1, 40);
    uint32_t windows = 0;
    for (const uint32_t bank : all_banks()) {
        for (uint32_t noc = 0; noc < 2; noc++) {
            const std::string mismatch =
                exerciser.round(bank, noc, all_ops[(2 * bank + noc) % all_ops.size()], max_round_commands, noc == 1);
            EXPECT_TRUE(mismatch.empty()) << "bank " << bank << " NoC" << noc << ": " << mismatch;
            windows += client->seat(bank, noc).has_window() ? 1 : 0;
        }
    }
    EXPECT_EQ(windows, std::min(2 * device().num_banks(), internal::drisc_service::max_windows_per_chip));
    client.reset();
    expect_all_stopped(all_banks());
}

// With every slot carving batches, clients detached through their exit mailbox (as a dead client's KMD exit write
// does) are freed with batch entries left to carve and window pulls possibly in flight, the kernel never writes their
// host memory after that, and the client that stays gets every byte right, before and after.
TEST_F(DriscService, DetachMidBatch) {
    constexpr uint32_t victims = abi::MAX_CLIENTS - 1;
    // The 512 KiB entries keep every victim busy for milliseconds, so none finishes before its detach.
    constexpr uint32_t host_span = 1024 * 1024;
    constexpr std::array<uint32_t, 4> entry_sizes = {64, 4096, 28 * 1024 + 64, 512 * 1024};
    std::optional<Client> survivor;
    attach(survivor, {0});
    Exerciser exerciser(device(), *survivor, 1, 30);
    std::vector<std::unique_ptr<HostMemory>> hosts;
    std::array<std::optional<Client>, victims> victim;
    for (uint32_t v = 0; v < victims; v++) {
        hosts.push_back(std::make_unique<HostMemory>(device(), host_span));
        attach(victim[v], {0});
    }

    const uint64_t dram = dram_test_offset + 4 * region_size;
    for (uint32_t v = 0; v < victims; v++) {
        SeatClient& seat = victim[v]->seat(0, 0);
        std::memset(hosts[v]->at(0), 0, host_span);
        std::vector<abi::KernelCommand> batch;
        batch.reserve(abi::BATCH_MAX_COMMANDS);
        for (uint32_t i = 0; i < abi::BATCH_MAX_COMMANDS; i++) {
            batch.push_back(abi::make_single_command(
                device().dram_endpoint(0, dram + uint64_t{i % 8} * (host_span / 2), 0),
                seat.host_endpoint(hosts[v]->device_addr(uint64_t{i % 2} * (host_span / 2))),
                entry_sizes[i % entry_sizes.size()]));
        }
        for (uint32_t b = 0; b < abi::CLIENT_QUEUE_CAPACITY; b++) {
            ASSERT_TRUE(seat.issue_batch(batch)) << "victim " << v << " batch " << b;
        }
    }
    exerciser.start(0, 0, Op::d2h, max_round_commands, true);

    const DriscCore core = device().core(seat_of(survivor->seat(0, 0)));
    std::mt19937 rng(31);
    for (uint32_t v = 0; v < victims; v++) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(rng() % 50);
        while (std::chrono::steady_clock::now() < until) {
        }
        const SeatClient& seat = victim[v]->seat(0, 0);
        core.write_l1_u32(
            abi::L1CommandBlockLayout::exit_mailbox_addr(seat.slot()), abi::slot_word(abi::SLOT_DETACH, seat.gen()));
    }
    std::vector<std::vector<uint32_t>> at_free;
    std::vector<uint32_t> in_flight_at_free;
    for (uint32_t v = 0; v < victims; v++) {
        const SeatClient& seat = victim[v]->seat(0, 0);
        const uint32_t freed = abi::slot_word(abi::SLOT_FREED, seat.gen());
        const auto ack = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(seat.slot())); };
        ASSERT_TRUE(wait_until(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS), [&] { return ack() == freed; }))
            << "victim " << v << ": ack " << ack() << ", want " << freed;
        at_free.emplace_back(hosts[v]->at(0), hosts[v]->at(0) + host_span / sizeof(uint32_t));
        in_flight_at_free.push_back(seat.in_flight());
        EXPECT_GT(in_flight_at_free.back(), 0u) << "victim " << v << " finished before its detach";
    }
    const std::string mismatch = exerciser.finish();
    EXPECT_TRUE(mismatch.empty()) << "across the detaches: " << mismatch;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (uint32_t v = 0; v < victims; v++) {
        EXPECT_TRUE(std::equal(at_free[v].begin(), at_free[v].end(), hosts[v]->at(0)))
            << "the kernel wrote victim " << v << "'s host memory after freeing it";
        EXPECT_EQ(victim[v]->seat(0, 0).in_flight(), in_flight_at_free[v])
            << "the kernel published a done_count for victim " << v << " after freeing it";
    }
    for (uint32_t r = 0; r < 2 * abi::MAX_CLIENTS; r++) {
        const std::string after = exerciser.round(0, 0, all_ops[r % all_ops.size()], max_round_commands, true);
        EXPECT_TRUE(after.empty()) << "after the detaches, round " << r << ": " << after;
    }
    const auto free_pull_trids = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::free_pull_trid_count_addr); };
    EXPECT_TRUE(wait_until(
        std::chrono::milliseconds(abi::ALIVE_PROBE_MS), [&] { return free_pull_trids() == abi::PULL_TRIDS; }))
        << "free pull TRIDs " << free_pull_trids() << " of " << abi::PULL_TRIDS;
    // Each drains in vain for abi::DETACH_TIMEOUT_MS, then finds its slot already freed.
    for (auto& v : victim) {
        v.reset();
    }
    survivor.reset();
    expect_all_stopped({0});
}

}  // namespace
}  // namespace tt::tt_metal
