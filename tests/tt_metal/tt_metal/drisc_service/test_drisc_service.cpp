// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs the DRISC service on chip 0, or the chip DRISC_SERVICE_TEST_CHIP names, through a bare UMD cluster, as a client
// process would, with no metal device in this process (metal would take the seat locks itself). Copies are checked
// through UMD's own DRAM reads and writes, never through the service, with a pattern unique to each (nonce, bank,
// address). The cross-process tests re-run this binary as DriscServiceChild.Run.

#include <gtest/gtest.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd-workload/cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/drisc_service/service.hpp"
#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"

namespace tt::tt_metal {
namespace {

namespace abi = drisc_service::abi;
using drisc_service::Client;
using drisc_service::DriscCore;
using drisc_service::IssuePath;
using drisc_service::PeerFlag;
using drisc_service::Seat;
using drisc_service::SeatClient;

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
// Long enough for the kernel to run many slot scans.
constexpr std::chrono::milliseconds scan_wait{200};
constexpr std::chrono::seconds child_timeout{180};

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
            soc.emplace(drisc_service::soc_descriptor(umd, chip_id));
        }
    }

    bool blackhole() const { return soc.has_value(); }
    uint32_t num_banks() const { return static_cast<uint32_t>(soc->get_num_dram_views()); }
    std::array<Seat, 2> seats(uint32_t bank) { return drisc_service::bank_seats(umd, chip_id, *soc, bank); }
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
        buffer_(drisc_service::allocate_host_buffer(device.umd, chip_id, size, device_addr_)) {}
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
        rounds++;
        commands_run += count;
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

    uint64_t rounds = 0;
    uint64_t commands_run = 0;

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
    const std::array<uint32_t, 2> cfg = core.read_niu_cfg();
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

// ---- Child processes -----------------------------------------------------------------------------------------------

std::string child_out_path() {
    static uint32_t calls = 0;
    return (std::filesystem::temp_directory_path() / fmt::format("drisc_service_child_{}_{}", ::getpid(), ++calls))
        .string();
}

pid_t spawn_child(const std::string& role, const std::string& out, std::vector<std::string> env) {
    env.push_back("DRISC_SERVICE_CHILD_ROLE=" + role);
    env.push_back("DRISC_SERVICE_CHILD_OUT=" + out);
    for (char** var = environ; *var != nullptr; var++) {
        const std::string_view entry(*var);
        // The parent's gtest output, filter and sharding settings would clobber its XML report or skip Run.
        if (entry.starts_with("GTEST_") || entry.starts_with("XML_OUTPUT_FILE=")) {
            continue;
        }
        env.emplace_back(entry);
    }
    std::vector<char*> envp;
    for (auto& var : env) {
        envp.push_back(var.data());
    }
    envp.push_back(nullptr);
    std::string exe = std::filesystem::read_symlink("/proc/self/exe").string();
    std::string filter = "--gtest_filter=DriscServiceChild.Run";
    std::array<char*, 3> argv = {exe.data(), filter.data(), nullptr};
    pid_t pid = 0;
    const int rc = ::posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), envp.data());
    TT_FATAL(rc == 0, "posix_spawn {} failed ({})", exe, std::strerror(rc));
    return pid;
}

// Kills and reaps the child when it goes out of scope still running.
class ChildGuard {
public:
    explicit ChildGuard(pid_t pid) : pid_(pid) {}
    ~ChildGuard() {
        if (::waitpid(pid_, nullptr, WNOHANG) == 0) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
    }
    ChildGuard(const ChildGuard&) = delete;
    ChildGuard& operator=(const ChildGuard&) = delete;

private:
    pid_t pid_;
};

int exit_code(int status) { return WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status); }

// The child's exit code, or -(signal) when a signal ended it. Kills it and throws when it runs past child_timeout.
int wait_child(pid_t pid) {
    const auto deadline = std::chrono::steady_clock::now() + child_timeout;
    int status = 0;
    while (true) {
        const pid_t done = ::waitpid(pid, &status, WNOHANG);
        TT_FATAL(done >= 0, "waitpid {} failed ({})", pid, std::strerror(errno));
        if (done == pid) {
            return exit_code(status);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
            TT_THROW("child {} ran past {} s and was killed", pid, child_timeout.count());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::optional<std::string> take_report(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    std::ifstream in(path);
    std::string report((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::filesystem::remove(path);
    return report;
}

// The first report of a child that keeps running after writing it. Nullopt when the child exits first or `timeout`
// passes.
std::optional<std::string> wait_for_report(pid_t pid, const std::string& path, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path)) {
            std::ifstream in(path);
            return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        }
        if (::waitpid(pid, nullptr, WNOHANG) == pid) {
            return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return std::nullopt;
}

void write_report(const std::string& text) {
    const std::string path = std::getenv("DRISC_SERVICE_CHILD_OUT");
    {
        std::ofstream out(path + ".tmp");
        out << text;
    }
    std::filesystem::rename(path + ".tmp", path);
}

std::string child_env(const char* name) {
    const char* value = std::getenv(name);
    TT_FATAL(value != nullptr, "the child needs {}", name);
    return value;
}

// "launched <n>" (seats this client booted) then one "seat <bank> <noc> <slot> <gen>" line per seat.
std::string describe_attachments(Client& client) {
    std::string seats;
    uint32_t launched = 0;
    for (const uint32_t bank : client.banks()) {
        for (uint32_t noc = 0; noc < 2; noc++) {
            const SeatClient& seat = client.seat(bank, noc);
            launched += seat.launched() ? 1 : 0;
            seats += fmt::format("seat {} {} {} {}\n", bank, noc, seat.slot(), seat.gen());
        }
    }
    return fmt::format("launched {}\n{}", launched, seats);
}

// client: attaches to DRISC_SERVICE_CHILD_BANKS ("0 3 5") and runs DRISC_SERVICE_CHILD_ROUNDS random rounds in DRAM
// region DRISC_SERVICE_CHILD_REGION, then ends as DRISC_SERVICE_CHILD_END says:
//   clean: destroys its Client.  abort: std::abort().  exit: _exit(3), skipping every destructor.
//   loop:  ignores the round count and runs until killed.
// Reports "ready\n" + its attachments once attached, "done <rounds> <commands>\n" + its attachments after a clean end,
// "refused" (exit 0) when every client slot of a seat stays taken, and "error: <what>" (exit 1) on anything else,
// a data mismatch included.
void run_client_child() {
    const rlimit no_core{0, 0};
    ::setrlimit(RLIMIT_CORE, &no_core);
    std::vector<uint32_t> banks;
    std::istringstream bank_list(child_env("DRISC_SERVICE_CHILD_BANKS"));
    for (uint32_t bank = 0; bank_list >> bank;) {
        banks.push_back(bank);
    }
    const auto region = static_cast<uint32_t>(std::stoul(child_env("DRISC_SERVICE_CHILD_REGION")));
    const uint64_t rounds = std::stoull(child_env("DRISC_SERVICE_CHILD_ROUNDS"));
    const std::string end = child_env("DRISC_SERVICE_CHILD_END");
    try {
        Device device;
        std::optional<Client> client;
        try {
            client.emplace(device.umd, chip_id, banks);
        } catch (const std::exception& e) {
            if (std::string_view(e.what()).find("client slots") != std::string_view::npos) {
                write_report("refused");
                std::_Exit(0);
            }
            throw;
        }
        const std::string attachments = describe_attachments(*client);
        write_report("ready\n" + attachments);
        Exerciser exerciser(device, *client, region, next_nonce());
        for (uint64_t i = 0; end == "loop" || i < rounds; i++) {
            const std::string mismatch = exerciser.random_round();
            if (!mismatch.empty()) {
                write_report("error: " + mismatch);
                std::_Exit(1);
            }
        }
        if (end == "abort") {
            std::abort();
        }
        if (end == "exit") {
            std::_Exit(3);
        }
        client.reset();
        write_report(fmt::format("done {} {}\n{}", exerciser.rounds, exerciser.commands_run, attachments));
    } catch (const std::exception& e) {
        write_report(fmt::format("error: {}", e.what()));
        std::_Exit(1);
    }
}

// metal_cycle: opens and closes a unit mesh on chip 0, then reports whether its DRAM programmable cores were on
// ("1" or "0"), or "error: <what>" when the open throws.
void run_metal_cycle_child() {
    std::shared_ptr<distributed::MeshDevice> mesh;
    try {
        mesh = distributed::MeshDevice::create_unit_mesh(chip_id);
    } catch (const std::exception& e) {
        write_report(fmt::format("error: {}", e.what()));
        return;
    }
    const bool dram_cores = MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
    mesh->close();
    write_report(dram_cores ? "1" : "0");
}

TEST(DriscServiceChild, Run) {
    const char* role = std::getenv("DRISC_SERVICE_CHILD_ROLE");
    if (role == nullptr) {
        GTEST_SKIP() << "Runs only as the child of a cross-process DRISC service test";
    }
    if (std::string_view(role) == "client") {
        run_client_child();
    } else {
        ASSERT_EQ(std::string_view(role), "metal_cycle");
        run_metal_cycle_child();
    }
}

struct Attachment {
    uint32_t bank;
    uint32_t noc;
    uint32_t slot;
    uint32_t gen;
};

std::vector<Attachment> parse_attachments(const std::string& report) {
    std::vector<Attachment> attachments;
    std::istringstream in(report);
    for (std::string line; std::getline(in, line);) {
        std::istringstream words(line);
        std::string tag;
        Attachment a{};
        if (words >> tag >> a.bank >> a.noc >> a.slot >> a.gen && tag == "seat") {
            attachments.push_back(a);
        }
    }
    return attachments;
}

// The number after `key` in `report`, or 0.
uint64_t report_number(const std::string& report, std::string_view key) {
    const size_t at = report.find(key);
    return at == std::string::npos ? 0 : std::stoull(report.substr(at + key.size()));
}

std::string metal_cycle(std::vector<std::string> env = {}) {
    const std::string out = child_out_path();
    const pid_t pid = spawn_child("metal_cycle", out, std::move(env));
    const int code = wait_child(pid);
    const auto report = take_report(out);
    TT_FATAL(report.has_value(), "the metal open/close child wrote no report (exit code {})", code);
    return *report;
}

std::vector<std::string> client_child_env(
    const std::vector<uint32_t>& banks, uint32_t region, uint64_t rounds, const std::string& end) {
    std::string bank_list;
    for (const uint32_t bank : banks) {
        bank_list += fmt::format("{} ", bank);
    }
    return {
        "DRISC_SERVICE_CHILD_BANKS=" + bank_list,
        fmt::format("DRISC_SERVICE_CHILD_REGION={}", region),
        fmt::format("DRISC_SERVICE_CHILD_ROUNDS={}", rounds),
        "DRISC_SERVICE_CHILD_END=" + end};
}

struct BenchLane {
    SeatClient* seat;
    std::vector<abi::KernelCommand> batch;  // one queue entry; size 1 with batched = false issues a single command
    uint64_t issued = 0;
};
constexpr auto bench_warmup = std::chrono::milliseconds(20);
constexpr auto bench_interval = std::chrono::milliseconds(200);

// Keeps every lane's queue full for bench_warmup + bench_interval; returns {GB/s, M commands/s} over the interval.
std::pair<double, double> bench_run(std::vector<BenchLane>& lanes, bool batched, uint32_t size) {
    using clock = std::chrono::steady_clock;
    const auto fill_until = [&](clock::time_point end) {
        while (clock::now() < end) {
            for (BenchLane& lane : lanes) {
                for (uint32_t k = 0; k < abi::CLIENT_QUEUE_CAPACITY; k++) {
                    const bool ok =
                        (batched ? lane.seat->issue_batch(lane.batch) : lane.seat->issue(lane.batch[0])).has_value();
                    if (!ok) {
                        break;
                    }
                    lane.issued++;
                }
            }
        }
    };
    const auto done = [&] {
        uint64_t entries = 0;
        for (const BenchLane& lane : lanes) {
            entries += lane.issued - lane.seat->in_flight();
        }
        return entries;
    };
    fill_until(clock::now() + bench_warmup);
    const uint64_t before = done();
    const auto start = clock::now();
    fill_until(start + bench_interval);
    const uint64_t after = done();
    const double seconds = std::chrono::duration<double>(clock::now() - start).count();
    for (BenchLane& lane : lanes) {
        lane.seat->drain(drain_timeout);
    }
    const double commands = double(after - before) * lanes[0].batch.size();
    return {commands * size / seconds / 1e9, commands / seconds / 1e6};
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
        client.emplace(device().umd, chip_id, std::move(banks), drisc_service::runtime_root(), issue_path);
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

TEST_F(DriscService, CopiesOnBothSeatsOfTwoBanks) {
    std::optional<Client> client;
    attach(client, {0, last_bank()});
    Exerciser exerciser(device(), *client, 0, 1);
    for (const uint32_t bank : client->banks()) {
        for (uint32_t noc = 0; noc < 2; noc++) {
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

// A small command from one client completes within latency_limit while another client's 64 MiB batch or single
// command runs on the same seat, rather than waiting for it to finish.
TEST_F(DriscService, SmallCommandIsNotHeldBehindAnotherClientsLargeOne) {
    constexpr uint32_t bulk_entry = 1 << 20;
    constexpr uint32_t bulk_size = abi::BATCH_MAX_COMMANDS * bulk_entry;
    constexpr uint32_t small_size = 64 * 1024;
    constexpr auto bulk_head_start = std::chrono::microseconds(300);
    constexpr auto latency_limit = std::chrono::microseconds(50);
    constexpr uint32_t trials = 8;
    std::optional<Client> bulk_client;
    std::optional<Client> small_client;
    attach(bulk_client, {0});
    attach(small_client, {0});
    SeatClient& bulk = bulk_client->seat(0, 0);
    SeatClient& small = small_client->seat(0, 0);
    HostMemory host(device(), small_size);
    const uint64_t bulk_src = dram_test_offset;
    const uint64_t bulk_dst = bulk_src + bulk_size;
    const uint64_t small_src = bulk_dst + bulk_size;
    std::vector<abi::KernelCommand> bulk_batch;
    for (uint32_t i = 0; i < abi::BATCH_MAX_COMMANDS; i++) {
        bulk_batch.push_back(abi::make_single_command(
            device().dram_endpoint(0, bulk_src + i * bulk_entry, 0),
            device().dram_endpoint(0, bulk_dst + i * bulk_entry, 0),
            bulk_entry));
    }
    const abi::KernelCommand bulk_single = abi::make_single_command(
        device().dram_endpoint(0, bulk_src, 0), device().dram_endpoint(0, bulk_dst, 0), bulk_size);

    // Seeds the small command's source with a fresh pattern and returns its nonce.
    const auto stage_small = [&] {
        const uint64_t nonce = next_nonce();
        std::vector<uint32_t> words(small_size / sizeof(uint32_t));
        fill_pattern(words.data(), small_size, nonce, 0, small_src);
        device().write_dram(0, small_src, words.data(), small_size);
        std::memset(host.at(0), 0, small_size);
        return nonce;
    };
    // Issue-to-done time of the staged DRAM->host copy on `small`, checked.
    const auto small_latency = [&](uint64_t nonce) {
        const abi::KernelCommand command = abi::make_single_command(
            device().dram_endpoint(0, small_src, 0), small.host_endpoint(host.device_addr(0)), small_size);
        const auto start = std::chrono::steady_clock::now();
        EXPECT_TRUE(small.issue(command));
        small.drain(drain_timeout);
        const auto latency = std::chrono::steady_clock::now() - start;
        EXPECT_EQ(pattern_mismatch(host.at(0), small_size, nonce, 0, small_src), "");
        return std::chrono::duration_cast<std::chrono::microseconds>(latency);
    };
    const auto median = [](std::vector<std::chrono::microseconds> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // Also keeps the small command clear of the posted writes that seed its source.
    const auto wait_head_start = [&] {
        const auto end = std::chrono::steady_clock::now() + bulk_head_start;
        while (std::chrono::steady_clock::now() < end);
    };

    std::vector<std::chrono::microseconds> alone;
    for (uint32_t trial = 0; trial < trials; trial++) {
        const uint64_t nonce = stage_small();
        wait_head_start();
        alone.push_back(small_latency(nonce));
    }
    for (const bool batch : {true, false}) {
        SCOPED_TRACE(batch ? "behind a batch" : "behind a single command");
        std::vector<std::chrono::microseconds> behind;
        for (uint32_t trial = 0; trial < trials; trial++) {
            const uint64_t nonce = stage_small();
            ASSERT_TRUE(batch ? bulk.issue_batch(bulk_batch) : bulk.issue(bulk_single));
            wait_head_start();
            ASSERT_GT(bulk.in_flight(), 0u) << "the bulk command finished before the small one was issued";
            behind.push_back(small_latency(nonce));
            bulk.drain(drain_timeout);
        }
        fmt::print(
            "{} MiB {}: small command alone {} us, behind it {} us (medians of {})\n",
            bulk_size >> 20,
            batch ? "batch" : "single",
            median(alone).count(),
            median(behind).count(),
            trials);
        EXPECT_LT(median(behind), latency_limit);
    }
    small_client.reset();
    bulk_client.reset();
    expect_all_stopped({0});
}

// Clients that keep one seat's queues full share its bytes equally, whatever their command sizes, and together keep the
// seat's bandwidth to within 5%.
TEST_F(DriscService, SharesSeatBandwidthEqually) {
    constexpr auto warmup = std::chrono::milliseconds(5);
    constexpr auto interval = std::chrono::milliseconds(50);
    constexpr uint64_t batch_bytes = 4 << 20;
    struct Mix {
        std::vector<double> shares;  // of the bytes the seat copied, per client
        double gbps;
    };
    // One client per entry of `command_sizes` on seat (bank 0, NoC0), each keeping its queue full of batches of DRAM
    // copies of that size in the bank. Batches, so that the host's per-issue cost cannot leave a queue empty; of
    // batch_bytes each, so that every client's progress is counted in the same small steps.
    const auto measure = [&](const std::vector<uint32_t>& command_sizes) {
        const size_t n = command_sizes.size();
        std::vector<std::optional<Client>> clients(n);
        std::vector<std::vector<abi::KernelCommand>> batches;
        uint64_t src = dram_test_offset;
        for (size_t i = 0; i < n; i++) {
            attach(clients[i], {0});
            TT_FATAL(batch_bytes % command_sizes[i] == 0, "a batch holds whole commands");
            batches.emplace_back(
                batch_bytes / command_sizes[i],
                abi::make_single_command(
                    device().dram_endpoint(0, src, 0),
                    device().dram_endpoint(0, src + command_sizes[i], 0),
                    command_sizes[i]));
            src += 2 * command_sizes[i];
        }
        std::vector<uint64_t> issued(n);
        const auto keep_full_until = [&](std::chrono::steady_clock::time_point end) {
            while (std::chrono::steady_clock::now() < end) {
                for (size_t i = 0; i < n; i++) {
                    for (uint32_t k = 0;
                         k < abi::CLIENT_QUEUE_CAPACITY && clients[i]->seat(0, 0).issue_batch(batches[i]);
                         k++) {
                        issued[i]++;
                    }
                }
            }
        };
        const auto completed_bytes = [&] {
            std::vector<uint64_t> bytes;
            for (size_t i = 0; i < n; i++) {
                bytes.push_back((issued[i] - clients[i]->seat(0, 0).in_flight()) * batch_bytes);
            }
            return bytes;
        };
        keep_full_until(std::chrono::steady_clock::now() + warmup);
        const std::vector<uint64_t> before = completed_bytes();
        const auto start = std::chrono::steady_clock::now();
        keep_full_until(start + interval);
        const std::vector<uint64_t> after = completed_bytes();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Mix mix{};
        uint64_t total = 0;
        for (size_t i = 0; i < n; i++) {
            total += after[i] - before[i];
        }
        for (size_t i = 0; i < n; i++) {
            mix.shares.push_back(double(after[i] - before[i]) / total);
        }
        mix.gbps = double(total) / seconds / 1e9;
        for (std::optional<Client>& client : clients) {
            client->seat(0, 0).drain(drain_timeout);
            client.reset();
        }
        return mix;
    };
    const auto expect_equal_shares = [](const Mix& mix, const char* what) {
        const double fair = 1.0 / mix.shares.size();
        for (size_t i = 0; i < mix.shares.size(); i++) {
            EXPECT_NEAR(mix.shares[i], fair, 0.05 * fair) << what << ", client " << i;
        }
    };

    constexpr uint32_t large = 1 << 20;
    constexpr uint32_t small = 64 << 10;
    const double alone = measure({large}).gbps;
    fmt::print("1 client: {:.2f} GB/s\n", alone);

    const Mix two = measure({large, large});
    fmt::print("2 clients: {:.3f} / {:.3f} of {:.2f} GB/s\n", two.shares[0], two.shares[1], two.gbps);
    expect_equal_shares(two, "2 clients");
    EXPECT_GT(two.gbps, 0.95 * alone);

    const Mix three = measure({large, large, large});
    fmt::print(
        "3 clients: {:.3f} / {:.3f} / {:.3f} of {:.2f} GB/s\n",
        three.shares[0],
        three.shares[1],
        three.shares[2],
        three.gbps);
    expect_equal_shares(three, "3 clients");
    EXPECT_GT(three.gbps, 0.95 * alone);

    // Turns are bytes, not commands: 64 KiB commands get as many bytes as 1 MiB ones. A 64 KiB command ends in a short
    // chunk, so its turns would run past CHUNK_SIZE unless the piece after that chunk is cut to fit.
    const Mix sizes = measure({large, small});
    fmt::print(
        "1 MiB + 64 KiB commands: {:.3f} / {:.3f} of {:.2f} GB/s\n", sizes.shares[0], sizes.shares[1], sizes.gbps);
    expect_equal_shares(sizes, "1 MiB + 64 KiB commands");
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
    const DriscCore core = device().core(seat.seat());
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

TEST_F(DriscService, MetalSeatLockRefusesClients) {
    const Seat seat = device().seats(0)[0];
    const std::optional<drisc_service::SeatLock> metal = drisc_service::SeatLock::try_acquire(seat.id);
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
    const std::optional<drisc_service::SeatLock> metal = drisc_service::SeatLock::try_acquire(seat.id);
    ASSERT_TRUE(metal.has_value()) << "something else holds the last bank's NoC1 seat";
    std::optional<Client> client;
    EXPECT_ANY_THROW(attach(client, {0, last_bank()})) << "a client attached to a seat metal holds";
    expect_all_stopped({0});
    const std::string problem = check_stopped(device(), device().seats(last_bank())[0]);
    EXPECT_TRUE(problem.empty()) << problem;
}

// Another process opens and closes metal while a client holds bank 0. Metal must turn its DRAM programmable cores off
// rather than take the seats, refuse to open when they are forced on, and leave the client's kernels running.
TEST_F(DriscService, MetalElsewhereLeavesClientsRunning) {
    ASSERT_EQ(metal_cycle(), "1") << "metal turned its DRAM programmable cores off with no client attached, so the "
                                     "checks below would pass without any seat lock";
    std::optional<Client> client;
    attach(client, {0});
    EXPECT_EQ(metal_cycle(), "0") << "metal enabled its DRAM programmable cores while a client held a seat";
    const std::string forced = metal_cycle({"TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1"});
    EXPECT_TRUE(forced.starts_with("error:") && forced.find("holds the seat of") != std::string::npos)
        << "metal opened with its DRAM programmable cores forced on while a client held a seat: " << forced;
    Exerciser exerciser(device(), *client, 0, 5);
    for (uint32_t noc = 0; noc < 2; noc++) {
        EXPECT_EQ(client->seat(0, noc).state(), abi::SERVICE_RUNNING) << "NoC" << noc;
        EXPECT_TRUE(heartbeat_moves(client->seat(0, noc))) << "NoC" << noc;
        const std::string mismatch = exerciser.round(0, noc, Op::h2d, 4, true);
        EXPECT_TRUE(mismatch.empty()) << mismatch;
    }
    client.reset();
    expect_all_stopped({0});
}

// A client child dies (SIGKILL mid-copy, abort, or _exit) while this process's client shares its seats: the KMD's exit
// write must free the dead child's slots, and this client must keep working. Then a child dies as the sole client: its
// kernels must stop.
class DriscServiceDeadClient : public DriscService, public ::testing::WithParamInterface<std::string> {};

TEST_P(DriscServiceDeadClient, IsDetached) {
    const std::string end = GetParam();
    for (const bool shared : {true, false}) {
        SCOPED_TRACE(shared ? "shared seats" : "sole client");
        std::optional<Client> mine;
        if (shared) {
            attach(mine, {0});
        }
        const std::string out = child_out_path();
        const pid_t pid = spawn_child("client", out, client_child_env({0}, 1, 20, end));
        const ChildGuard guard(pid);
        const auto ready = wait_for_report(pid, out, child_timeout);
        ASSERT_TRUE(ready.has_value() && ready->starts_with("ready")) << "the child did not attach";
        const std::vector<Attachment> attachments = parse_attachments(*ready);
        ASSERT_EQ(attachments.size(), 2u) << *ready;
        if (end == "loop") {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            ASSERT_EQ(::kill(pid, SIGKILL), 0);
        }
        const int code = wait_child(pid);
        const int want = end == "exit" ? 3 : end == "abort" ? -SIGABRT : -SIGKILL;
        const std::string last = take_report(out).value_or("");
        EXPECT_EQ(code, want) << last;

        const std::array<Seat, 2> seats = device().seats(0);
        for (const Attachment& a : attachments) {
            const DriscCore core = device().core(seats[a.noc]);
            const auto ack = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(a.slot)); };
            const uint32_t freed = abi::slot_word(abi::SLOT_FREED, a.gen);
            EXPECT_TRUE(wait_until(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS), [&] { return ack() == freed; }))
                << "NoC" << a.noc << " slot " << a.slot << ": ack " << ack() << ", want " << freed;
        }
        if (shared) {
            Exerciser exerciser(device(), *mine, 0, 6);
            for (uint32_t noc = 0; noc < 2; noc++) {
                EXPECT_EQ(mine->seat(0, noc).state(), abi::SERVICE_RUNNING) << "NoC" << noc;
                const std::string mismatch = exerciser.round(0, noc, Op::d2d, 4, true);
                EXPECT_TRUE(mismatch.empty()) << mismatch;
            }
        }
        mine.reset();
        expect_all_stopped({0});
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ends, DriscServiceDeadClient, ::testing::Values("loop", "abort", "exit"), [](const auto& info) {
        return info.param == "loop" ? std::string("Sigkill") : info.param;
    });

// Rolling client children on random banks, so all 16 DRISCs, for DRISC_SERVICE_STRESS_SECONDS (skipped when unset),
// at most DRISC_SERVICE_STRESS_CHILDREN (default 12) at once. Each child attaches, runs random verified copies in its
// own DRAM region, and ends cleanly, by abort, by _exit, or by SIGKILL at a random time from before attach to
// mid-copy. No child may report a mismatch or hang; once the last is gone every kernel must be stopped with its NIUs
// restored, and a fresh client must run on every seat. DRISC_SERVICE_STRESS_REPORT names a file for the stats.
TEST_F(DriscService, Stress) {
    const char* seconds_env = std::getenv("DRISC_SERVICE_STRESS_SECONDS");
    if (seconds_env == nullptr) {
        GTEST_SKIP() << "Set DRISC_SERVICE_STRESS_SECONDS to run";
    }
    const auto duration = std::chrono::seconds(std::stoul(seconds_env));
    const char* children_env = std::getenv("DRISC_SERVICE_STRESS_CHILDREN");
    const uint32_t max_children = children_env != nullptr ? std::stoul(children_env) : 12;
    constexpr uint32_t num_regions = 64;
    std::mt19937_64 rng(next_nonce());
    const auto below = [&](uint64_t n) { return std::uniform_int_distribution<uint64_t>(0, n - 1)(rng); };
    using clock = std::chrono::steady_clock;

    struct Live {
        pid_t pid;
        std::string out;
        std::string end;
        uint32_t region;
        clock::time_point started;
        clock::time_point kill_at;
    };
    std::vector<Live> live;
    // FIFO, so a region is reused only after every other free one: a killed child's in-flight DMA cannot land in data
    // a new child is checking.
    std::deque<uint32_t> free_regions;
    for (uint32_t r = 1; r <= num_regions; r++) {
        free_regions.push_back(r);
    }
    std::map<std::string, uint64_t> stats;
    std::vector<std::string> failures;

    const auto reap = [&](const Live& child, int code) {
        const std::string text = take_report(child.out).value_or("");
        const double secs = std::chrono::duration<double>(clock::now() - child.started).count();
        free_regions.push_back(child.region);
        if (text.starts_with("error")) {
            failures.push_back(fmt::format("child {} ({}, {:.1f} s): {}", child.pid, child.end, secs, text));
            return;
        }
        const bool refused = text.starts_with("refused");
        const bool expected = refused                ? code == 0
                              : child.end == "clean" ? code == 0
                              : child.end == "abort" ? code == -SIGABRT
                              : child.end == "exit"  ? code == 3
                                                     : code == -SIGKILL;
        if (!expected) {
            failures.push_back(fmt::format("child {} ({}) ended with code {}: {}", child.pid, child.end, code, text));
        }
        stats["ended_" + child.end]++;
        if (refused) {
            stats["refused"]++;
        } else if (text.starts_with("done")) {
            stats["done"]++;
            stats["rounds"] += report_number(text, "done ");
            stats["commands"] += std::stoull(text.substr(text.find(' ', 5) + 1));
        } else if (text.starts_with("ready")) {
            stats["died_attached"]++;
        } else {
            stats["died_before_attach"]++;
        }
        stats["seat_launches"] += report_number(text, "launched ");
    };

    const auto deadline = clock::now() + duration;
    auto last_log = clock::now();
    while (clock::now() < deadline || !live.empty()) {
        const auto now = clock::now();
        for (auto it = live.begin(); it != live.end();) {
            if (it->end == "kill" && now >= it->kill_at) {
                ::kill(it->pid, SIGKILL);
            }
            int status = 0;
            if (::waitpid(it->pid, &status, WNOHANG) == it->pid) {
                reap(*it, exit_code(status));
                it = live.erase(it);
            } else if (now - it->started > child_timeout) {
                ::kill(it->pid, SIGKILL);
                ::waitpid(it->pid, nullptr, 0);
                failures.push_back(
                    fmt::format("child {} ({}) hung past {} s", it->pid, it->end, child_timeout.count()));
                free_regions.push_back(it->region);
                it = live.erase(it);
            } else {
                ++it;
            }
        }
        if (now < deadline && live.size() < max_children) {
            std::vector<uint32_t> banks = all_banks();
            std::shuffle(banks.begin(), banks.end(), rng);
            banks.resize(below(10) == 0 ? banks.size() : 1 + below(4));
            // 5% attach only, 30% clean, 15% abort, 15% _exit, 35% SIGKILL within 6 s of spawn.
            const uint64_t pick = below(100);
            const std::string end = pick < 35 ? "clean" : pick < 50 ? "abort" : pick < 65 ? "exit" : "kill";
            const uint64_t rounds = pick < 5 ? 0 : 1 + below(40);
            const uint32_t region = free_regions.front();
            free_regions.pop_front();
            const std::string out = child_out_path();
            const pid_t pid =
                spawn_child("client", out, client_child_env(banks, region, rounds, end == "kill" ? "loop" : end));
            stats["spawned"]++;
            live.push_back({pid, out, end, region, now, now + std::chrono::milliseconds(below(6000))});
        }
        if (now - last_log > std::chrono::seconds(30)) {
            last_log = now;
            log_info(
                tt::LogTest,
                "stress: {} spawned, {} live, {} rounds, {} failures",
                stats["spawned"],
                live.size(),
                stats["rounds"],
                failures.size());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::vector<std::string> stuck;
    for (const uint32_t bank : all_banks()) {
        for (const Seat& seat : device().seats(bank)) {
            std::string problem = check_stopped(device(), seat);
            if (!problem.empty()) {
                stuck.push_back(std::move(problem));
            }
        }
    }
    std::string final_mismatch;
    {
        std::optional<Client> client;
        attach(client, all_banks());
        Exerciser exerciser(device(), *client, 0, 7);
        for (const uint32_t bank : client->banks()) {
            for (uint32_t noc = 0; noc < 2 && final_mismatch.empty(); noc++) {
                final_mismatch = exerciser.round(bank, noc, all_ops[(bank + noc) % all_ops.size()], 4, true);
            }
        }
    }
    expect_all_stopped(all_banks());

    std::string summary = fmt::format(
        "duration_s {}\nmax_children {}\nfailures {}\nstuck_seats {}\n",
        duration.count(),
        max_children,
        failures.size(),
        stuck.size());
    for (const auto& [key, value] : stats) {
        summary += fmt::format("{} {}\n", key, value);
    }
    for (const auto& failure : failures) {
        summary += "FAIL " + failure + "\n";
    }
    for (const auto& problem : stuck) {
        summary += "STUCK " + problem + "\n";
    }
    log_info(tt::LogTest, "stress summary:\n{}", summary);
    if (const char* path = std::getenv("DRISC_SERVICE_STRESS_REPORT")) {
        std::ofstream(path) << summary;
    }
    EXPECT_TRUE(failures.empty()) << failures.size() << " failures, first: " << failures.front();
    EXPECT_TRUE(stuck.empty()) << stuck.size() << " seats did not stop, first: " << stuck.front();
    EXPECT_TRUE(final_mismatch.empty()) << "after the churn: " << final_mismatch;
}

// issue() and issue_batch() refuse what the kernel would run wrongly, issuing nothing, and the queue stays usable.
// A bank past the chip's last one fails the client before any seat attaches; seat() rejects a bank the client did not
// attach and a NoC other than 0 or 1.
TEST_F(DriscService, RejectsBadBanksAndSeats) {
    std::optional<Client> client;
    EXPECT_THROW(client.emplace(device().umd, chip_id, std::vector<uint32_t>{0, device().num_banks()}), std::exception);
    EXPECT_FALSE(client.has_value());
    expect_all_stopped(all_banks());
    attach(client, {0});
    EXPECT_THROW(client->seat(last_bank(), 0), std::exception);
    EXPECT_THROW(client->seat(0, 2), std::exception);
    client.reset();
    expect_all_stopped(all_banks());
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
        {"local past the bank", "runs past", with([&](auto& c) { c.dst_addr = seat.seat().bank_size - 64; })},
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

// A kernel that stops beating while it serves a client is not relaunched under it by a joining client, and is
// relaunched by the next client that holds the seat alone.
TEST_F(DriscService, HungKernelIsRelaunchedOnlyBySoleClient) {
    std::optional<Client> first;
    attach(first, {0});
    const Seat seat = first->seat(0, 0).seat();
    device().core(seat).force_stop(seat.noc2axi_mask);
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

// A full queue refuses the next command without issuing it. Host copies of sizes off the 64 B grid write exactly
// their bytes: the guard bands on both sides of the destination keep their sentinel.
TEST_F(DriscService, QueueFullAndOddSizes) {
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    constexpr uint32_t big = 64 << 20;
    const uint64_t big_base = device().bank_offset(0) + (3ull << 30);
    const abi::KernelCommand slow = abi::make_local_command(big_base, big_base + big, big);
    for (uint32_t i = 0; i < abi::CLIENT_QUEUE_CAPACITY; i++) {
        ASSERT_TRUE(seat.issue(slow)) << "command " << i << " found the queue full";
    }
    EXPECT_FALSE(seat.issue(slow)) << "the queue took more than " << abi::CLIENT_QUEUE_CAPACITY << " commands";
    EXPECT_EQ(seat.in_flight(), abi::CLIENT_QUEUE_CAPACITY);
    seat.drain(drain_timeout);
    EXPECT_EQ(seat.in_flight(), 0u);

    constexpr uint32_t chunk = 28 * 1024;
    constexpr uint32_t guard = 64;
    constexpr std::array<uint32_t, 6> odd_sizes = {1, 63, 65, chunk - 16, chunk + 16, 3 * chunk + 1};
    constexpr uint8_t sentinel = 0xA5;
    const auto round_up = [](uint32_t size) { return (size + 63) / 64 * 64; };
    HostMemory host(device(), 2 * guard + round_up(odd_sizes.back()));
    auto* host_bytes = reinterpret_cast<uint8_t*>(host.at(0));
    std::mt19937 rng(10);
    const uint64_t dram = dram_test_offset;
    const auto random_payload = [&](uint32_t size) {
        std::vector<uint8_t> payload(round_up(size));
        std::generate(payload.begin(), payload.end(), [&] { return static_cast<uint8_t>(rng()); });
        return payload;
    };
    const auto expect_guarded =
        [&](const std::vector<uint8_t>& landed, const std::vector<uint8_t>& payload, uint32_t size) {
            std::vector<uint8_t> expected(landed.size(), sentinel);
            std::copy_n(payload.begin(), size, expected.begin() + guard);
            const auto diff = std::mismatch(landed.begin(), landed.end(), expected.begin());
            EXPECT_TRUE(diff.first == landed.end())
                << "byte " << (diff.first - landed.begin()) << " of the span (payload at " << guard << "): got "
                << static_cast<uint32_t>(*diff.first) << ", expected " << static_cast<uint32_t>(*diff.second);
        };
    for (uint32_t noc = 0; noc < 2; noc++) {
        SeatClient& on = client->seat(0, noc);
        for (const uint32_t size : odd_sizes) {
            for (const bool to_host : {false, true}) {
                SCOPED_TRACE(fmt::format("NoC{} {} B {}", noc, size, to_host ? "DRAM->host" : "host->DRAM"));
                const uint32_t span = 2 * guard + round_up(size);
                const std::vector<uint8_t> payload = random_payload(size);
                std::vector<uint8_t> landed(span, sentinel);
                abi::NocEndpoint src{};
                abi::NocEndpoint dst{};
                if (to_host) {
                    device().write_dram(0, dram, payload.data(), round_up(size));
                    std::memset(host_bytes, sentinel, span);
                    src = device().dram_endpoint(0, dram, noc);
                    dst = on.host_endpoint(host.device_addr(guard));
                } else {
                    std::memcpy(host_bytes, payload.data(), payload.size());
                    device().write_dram(0, dram, landed.data(), span);
                    src = on.host_endpoint(host.device_addr(0));
                    dst = device().dram_endpoint(0, dram + guard, noc);
                }
                ASSERT_TRUE(on.issue(abi::make_single_command(src, dst, size)));
                on.drain(drain_timeout);
                if (to_host) {
                    std::memcpy(landed.data(), host_bytes, span);
                } else {
                    device().read_dram(0, dram, landed.data(), span);
                }
                expect_guarded(landed, payload, size);
            }
        }
    }

    // CMD_LOCAL sizes on the 16 B grid but off the 64 B one.
    constexpr std::array<uint32_t, 6> local_sizes = {16, 48, 80, chunk - 16, chunk + 16, 3 * chunk + 16};
    const uint64_t local_dst = dram + slot_size;
    for (uint32_t noc = 0; noc < 2; noc++) {
        for (const uint32_t size : local_sizes) {
            SCOPED_TRACE(fmt::format("NoC{} {} B DRAM->DRAM local", noc, size));
            const uint32_t span = 2 * guard + round_up(size);
            const std::vector<uint8_t> payload = random_payload(size);
            std::vector<uint8_t> landed(span, sentinel);
            device().write_dram(0, dram, payload.data(), payload.size());
            device().write_dram(0, local_dst, landed.data(), span);
            const uint64_t offset = device().bank_offset(0);
            ASSERT_TRUE(
                client->seat(0, noc).issue(abi::make_local_command(offset + dram, offset + local_dst + guard, size)));
            client->seat(0, noc).drain(drain_timeout);
            device().read_dram(0, local_dst, landed.data(), span);
            expect_guarded(landed, payload, size);
        }
    }
    client.reset();
    expect_all_stopped({0});
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

    const DriscCore core = device().core(seat.seat());
    core.write_l1_u32(
        abi::L1CommandBlockLayout::exit_mailbox_addr(seat.slot()), abi::slot_word(abi::SLOT_DETACH, seat.gen()));
    const uint32_t freed = abi::slot_word(abi::SLOT_FREED, seat.gen());
    const auto ack = [&] { return core.read_l1_u32(abi::L1CommandBlockLayout::ack_addr(seat.slot())); };
    ASSERT_TRUE(wait_until(std::chrono::milliseconds(abi::DETACH_TIMEOUT_MS), [&] { return ack() == freed; }))
        << "ack " << ack() << ", want " << freed;
    const std::vector<uint32_t> at_free(host.at(0), host.at(0) + slot_size / sizeof(uint32_t));
    const uint32_t in_flight_at_free = seat.in_flight();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(std::equal(at_free.begin(), at_free.end(), host.at(0))) << "the kernel wrote host memory after freeing";
    EXPECT_EQ(seat.in_flight(), in_flight_at_free) << "the kernel published a done_count after freeing";
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
    EXPECT_EQ(windows, std::min(2 * device().num_banks(), drisc_service::max_windows_per_chip));
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

    const DriscCore core = device().core(survivor->seat(0, 0).seat());
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

// Bandwidth and command-rate bench for the design doc's tables. Opt-in: runs only with DRISC_SERVICE_BENCH set. Prints
// "BENCH <key> <value>" lines; values are GB/s, M commands/s, or us per issue() call. Queues are kept full; nothing is
// verified (the correctness tests above do that).
// ---- Peer copies: the chip under test writes into another chip's DRAM over PCIe -----------------------------------

class DriscServicePeer : public DriscService {
protected:
    void SetUp() override {
        DriscService::SetUp();
        if (!IsSkipped() && peers().empty()) {
            GTEST_SKIP() << "Needs a second chip with its own PCIe link";
        }
    }

    static std::vector<ChipId> peers() {
        std::vector<ChipId> chips;
        for (const ChipId chip : device().umd.get_target_mmio_device_ids()) {
            if (chip != chip_id) {
                chips.push_back(chip);
            }
        }
        return chips;
    }

    // Bank `bank` of chip `peer` through its NoC0 endpoint, as UMD's own DRAM reads name it.
    struct PeerBank {
        tt::umd::CoreCoord core;
        uint64_t offset;
    };
    static PeerBank peer_bank(ChipId peer, uint32_t bank) {
        const metal_SocDescriptor soc = drisc_service::soc_descriptor(device().umd, peer);
        const CoreCoord endpoint = soc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank), 0);
        return {
            {endpoint.x, endpoint.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED},
            soc.get_address_offset(static_cast<int>(bank))};
    }
};

// 1 MiB from bank 0 into bank 0 of every other chip, from both seats: the data, then a flag right after it,
// which this process waits for through a PeerFlag as the receiver would, then checks every byte.
TEST_F(DriscServicePeer, CopiesToEveryPeer) {
    constexpr uint32_t size = 1 << 20;
    std::optional<Client> client;
    attach(client, {0});
    HostMemory flags(device(), 4096);
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    uint32_t seq = static_cast<uint32_t>(next_nonce());
    for (const ChipId peer : peers()) {
        const PeerBank bank = peer_bank(peer, 0);
        const uint64_t dst = bank.offset + dram_test_offset;
        const abi::NocEndpoint target = client->peer_endpoint(peer, bank.core, dst, size + 64);
        PeerFlag flag(device().umd, peer, bank.core, dst + size);
        for (uint32_t noc = 0; noc < 2; noc++) {
            const uint64_t nonce = next_nonce();
            fill_pattern(words.data(), size, nonce, 0, dram_test_offset);
            device().write_dram(0, dram_test_offset, words.data(), size);
            // The flag starts one short of what this copy writes, so a stale value cannot satisfy the wait.
            const uint32_t before = seq++;
            device().umd.write_to_device(&before, sizeof(before), peer, bank.core, dst + size);
            *flags.at(0) = seq;
            SeatClient& seat = client->seat(0, noc);
            ASSERT_TRUE(seat.issue_batch(std::array{
                abi::make_single_command(device().dram_endpoint(0, dram_test_offset, noc), target, size),
                abi::make_single_command(seat.host_endpoint(flags.device_addr(0)), target.at_offset(size), 64)}));
            flag.wait(seq, drain_timeout);
            std::fill(words.begin(), words.end(), 0);
            device().umd.read_from_device(words.data(), peer, bank.core, dst, size);
            const std::string mismatch = pattern_mismatch(words.data(), size, nonce, 0, dram_test_offset);
            EXPECT_TRUE(mismatch.empty()) << "chip " << peer << " NoC" << noc << ": " << mismatch;
            seat.drain(drain_timeout);
        }
    }
    client.reset();
    expect_all_stopped({0});
}

// A copy into a Posted window, then its flag: chip B's own seat samples the flag and the data's last line together,
// with one 128 B DMA read from its DRAM, hundreds of times a round while chip A writes. A sample holding the new flag
// next to an old line is a flag that passed its data. A calibration that writes the line well after the flag must be
// caught first, or a clean result proves nothing. The receiving bank is loaded throughout: another chip writing into it
// over PCIe, and chip B copying within it.
TEST_F(DriscServicePeer, PostedFlagNeverPassesItsData) {
    if (peers().size() < 2) {
        GTEST_SKIP() << "Needs two other chips with their own PCIe links";
    }
    constexpr uint64_t window = 2 << 20;
    constexpr uint32_t slices = 16;
    constexpr uint32_t sample_bytes = 128;
    constexpr uint32_t sample_batches = 4;
    constexpr uint32_t samples = sample_batches * abi::BATCH_MAX_COMMANDS;
    constexpr uint32_t filler_bytes = 64 << 10;
    constexpr uint64_t filler_offset = window / 2 + window / 4;
    constexpr int rounds = 2000;
    const ChipId receiver = peers()[0];
    const ChipId other = peers()[1];
    const PeerBank dst_bank = peer_bank(receiver, 0);
    const uint64_t data_base = dram_test_offset + (64ull << 20);
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    HostMemory flags(device(), 4096);
    const abi::NocEndpoint target = client->peer_endpoint(receiver, dst_bank.core, dst_bank.offset + data_base, window);
    Client receiver_client(device().umd, receiver, {0});
    SeatClient& sampler = receiver_client.seat(0, 0);
    uint64_t samples_addr = 0;
    const auto samples_buffer =
        drisc_service::allocate_host_buffer(device().umd, receiver, samples * sample_bytes, samples_addr);
    const auto* sample_words = static_cast<const volatile uint32_t*>(samples_buffer->get_va());

    Client other_client(device().umd, other, {0});
    std::atomic<bool> stop{false};
    std::thread load([&] {
        const PeerBank other_src = peer_bank(other, 0);
        const abi::NocEndpoint src{
            other_src.offset + dram_test_offset, abi::noc_xy(other_src.core.x, other_src.core.y)};
        const abi::NocEndpoint dst = other_client.peer_endpoint(
            receiver, dst_bank.core, dst_bank.offset + dram_test_offset + (128ull << 20), window);
        const uint64_t local = dst_bank.offset + dram_test_offset + (256ull << 20);
        const std::array pushes{
            abi::make_single_command(src, dst, 1u << 20),
            abi::make_single_command(src, dst.at_offset(1u << 20), 1u << 20)};
        const std::array copies{abi::make_local_command(local, local + (64ull << 20), 1u << 20)};
        while (!stop) {
            while (other_client.seat(0, 0).issue_batch(pushes)) {
            }
            while (receiver_client.seat(0, 1).issue_batch(copies)) {
            }
        }
        other_client.drain(drain_timeout);
        receiver_client.seat(0, 1).drain(drain_timeout);
    });

    for (const uint32_t size : {64u << 10, 1u << 20}) {
        const uint64_t nonce = next_nonce();
        std::vector<uint32_t> words(uint64_t{slices} * size / sizeof(uint32_t));
        fill_pattern(words.data(), slices * size, nonce, 0, dram_test_offset);
        device().write_dram(0, dram_test_offset, words.data(), slices * size);
        // Layout in the window: data [0, size), flag [size, size + 64), the data's last line again [size + 64, size +
        // 128).
        const abi::NocEndpoint flag_and_line{
            dst_bank.offset + data_base + size, abi::noc_xy(dst_bank.core.x, dst_bank.core.y)};
        std::array<std::vector<abi::KernelCommand>, sample_batches> sample_cmds;
        for (uint32_t b = 0; b < sample_batches; b++) {
            for (uint32_t i = 0; i < abi::BATCH_MAX_COMMANDS; i++) {
                const uint64_t slot = uint64_t{b * abi::BATCH_MAX_COMMANDS + i} * sample_bytes;
                sample_cmds[b].push_back(
                    abi::make_single_command(flag_and_line, sampler.host_endpoint(samples_addr + slot), sample_bytes));
            }
        }
        for (const bool calibrate : {true, false}) {
            uint32_t stale_rounds = 0;
            uint32_t covered_rounds = 0;
            for (int r = 0; r < rounds; r++) {
                const uint32_t k = r % slices;
                const uint32_t flag =
                    static_cast<uint32_t>(nonce) ^ static_cast<uint32_t>(r + 1) ^ (calibrate ? 0x80000000u : 0);
                *flags.at(0) = flag;
                const abi::NocEndpoint src = device().dram_endpoint(0, dram_test_offset + uint64_t{k} * size, 0);
                const abi::NocEndpoint flag_src = seat.host_endpoint(flags.device_addr(0));
                const auto data = abi::make_single_command(src, target, size);
                const auto line = abi::make_single_command(src.at_offset(size - 64), target.at_offset(size + 64), 64);
                if (calibrate) {
                    ASSERT_TRUE(seat.issue_batch(std::array{
                        data,
                        abi::make_single_command(flag_src, target.at_offset(size), 64),
                        abi::make_single_command(src, target.at_offset(filler_offset), filler_bytes),
                        line}));
                } else {
                    ASSERT_TRUE(seat.issue_batch(
                        std::array{data, line, abi::make_single_command(flag_src, target.at_offset(size), 64)}));
                }
                for (const auto& batch : sample_cmds) {
                    ASSERT_TRUE(sampler.issue_batch(batch));
                }
                sampler.drain(drain_timeout);
                seat.drain(drain_timeout);
                bool saw_old = false;
                bool saw_new = false;
                bool stale = false;
                const uint64_t line_addr = dram_test_offset + uint64_t{k} * size + size - 64;
                for (uint32_t s = 0; s < samples; s++) {
                    const volatile uint32_t* w = sample_words + s * sample_bytes / sizeof(uint32_t);
                    if (w[0] != flag) {
                        saw_old = true;
                        continue;
                    }
                    saw_new = true;
                    uint32_t got[16];
                    for (uint32_t i = 0; i < 16; i++) {
                        got[i] = w[16 + i];
                    }
                    stale |= !pattern_mismatch(got, 64, nonce, 0, line_addr).empty();
                }
                covered_rounds += saw_old && saw_new ? 1 : 0;
                stale_rounds += stale ? 1 : 0;
            }
            log_info(
                tt::LogTest,
                "{} {} B: flag ahead of its line in {} of {} rounds; samples saw the flag change in {}",
                calibrate ? "calibration (line written after the flag)" : "flag written last",
                size,
                stale_rounds,
                rounds,
                covered_rounds);
            if (calibrate) {
                EXPECT_GT(stale_rounds, 0u) << size << " B: the samples never caught a line written after its flag";
            } else {
                EXPECT_EQ(stale_rounds, 0u) << size << " B: the flag passed its data in " << stale_rounds << " rounds";
                EXPECT_GT(covered_rounds, uint32_t{rounds / 10})
                    << size << " B: the samples rarely saw the flag change";
            }
        }
    }
    stop = true;
    load.join();
    client.reset();
    expect_all_stopped({0});
}

TEST_F(DriscServicePeer, RejectsBadPeerRangesAndCommands) {
    constexpr uint32_t size = 4096;
    constexpr uint64_t two_mib = 2 << 20;
    std::optional<Client> client;
    attach(client, {0});
    SeatClient& seat = client->seat(0, 0);
    const ChipId peer = peers().front();
    const PeerBank bank = peer_bank(peer, 0);
    const auto expect_rejected = [](auto call, std::string_view reason, const std::string& what) {
        try {
            (void)call();
            ADD_FAILURE() << what << " was accepted";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string_view(e.what()).find(reason), std::string_view::npos)
                << what << " was rejected for another reason: " << e.what();
        }
    };
    expect_rejected(
        [&] { return client->peer_endpoint(chip_id, bank.core, 0, size); }, "cannot be its own peer", "own chip");
    expect_rejected(
        [&] { return client->peer_endpoint(peer, bank.core, 0, 2 * two_mib); }, "crosses a 2 MiB block", "4 MiB");
    expect_rejected(
        [&] { return client->peer_endpoint(peer, bank.core, two_mib - size, 2 * size); },
        "crosses a 2 MiB block",
        "range across a 2 MiB block");

    HostMemory host(device(), 4096);
    const abi::NocEndpoint target = client->peer_endpoint(peer, bank.core, bank.offset + dram_test_offset, size);
    expect_rejected(
        [&] { return seat.issue(abi::make_single_command(target, seat.host_endpoint(host.device_addr(0)), size)); },
        "must never be read through",
        "a peer window as a source");
    EXPECT_EQ(seat.in_flight(), 0u) << "a rejected command was issued";
    client.reset();
    expect_all_stopped({0});
}

// Into every other chip, from one seat and from all 16, each seat on A's bank b writing B's bank b: CMD_SINGLE entries
// and a 64 B flag ending each batch (the rate counts data only).
TEST_F(DriscServicePeer, PeerBench) {
    if (std::getenv("DRISC_SERVICE_BENCH") == nullptr) {
        GTEST_SKIP() << "Set DRISC_SERVICE_BENCH to run";
    }
    constexpr uint64_t dram_base = 0xA0000000ull;
    constexpr uint64_t window = 2 << 20;
    const std::vector<std::tuple<uint32_t, uint32_t, const char*>> shapes = {
        {1u << 20, 2, "1m_b2"}, {24u << 10, 64, "24k_b64"}, {4352, 64, "4352b_b64"}};
    std::optional<Client> client;
    attach(client, all_banks());
    HostMemory flags(device(), 4096);
    for (const ChipId peer : peers()) {
        fmt::print(
            "BENCH peer{}_pci_device {}\n", peer, device().umd.get_tt_device(peer)->get_pci_device()->get_device_num());
        std::vector<abi::NocEndpoint> targets;
        for (uint32_t bank = 0; bank < device().num_banks(); bank++) {
            const PeerBank peer_dram = peer_bank(peer, bank);
            targets.push_back(client->peer_endpoint(peer, peer_dram.core, peer_dram.offset + dram_base, window));
        }
        const auto lane = [&](uint32_t bank, uint32_t noc, uint32_t size, uint32_t count) {
            SeatClient& seat = client->seat(bank, noc);
            BenchLane l{&seat, {}};
            for (uint32_t i = 0; i + 1 < count; i++) {
                const uint64_t off = (uint64_t{i} * size) % window;
                l.batch.push_back(abi::make_single_command(
                    device().dram_endpoint(bank, dram_base + off, noc), targets[bank].at_offset(off), size));
            }
            l.batch.push_back(abi::make_single_command(
                seat.host_endpoint(flags.device_addr(0)), targets[bank].at_offset(window - 64), 64));
            return l;
        };
        // bench_run counts every entry at `size`; a batch has one 64 B flag among `count` entries.
        const auto data_gbps = [&](std::vector<BenchLane>& lanes, uint32_t size, uint32_t count) {
            return bench_run(lanes, true, size).first * (count - 1) / count;
        };
        for (const auto& [size, count, unit] : shapes) {
            for (uint32_t noc = 0; noc < 2; noc++) {
                std::vector<BenchLane> lanes{lane(0, noc, size, count)};
                fmt::print("BENCH peer{}_seat1_noc{}_{}_gbps {:.3f}\n", peer, noc, unit, data_gbps(lanes, size, count));
            }
            std::vector<BenchLane> lanes;
            for (uint32_t bank = 0; bank < device().num_banks(); bank++) {
                for (uint32_t noc = 0; noc < 2; noc++) {
                    lanes.push_back(lane(bank, noc, size, count));
                }
            }
            fmt::print("BENCH peer{}_seat16_{}_gbps {:.3f}\n", peer, unit, data_gbps(lanes, size, count));
        }
    }
    client.reset();
    expect_all_stopped(all_banks());
}

TEST_F(DriscService, PerfBench) {
    if (std::getenv("DRISC_SERVICE_BENCH") == nullptr) {
        GTEST_SKIP() << "Set DRISC_SERVICE_BENCH to run";
    }
    using clock = std::chrono::steady_clock;
    constexpr auto interval = bench_interval;
    constexpr uint64_t dram_base = 0xA0000000ull;  // 2.5 GiB: past every test region
    constexpr uint64_t dram_span = 64ull << 20;
    constexpr uint64_t host_size = 32ull << 20;
    const uint32_t banks = device().num_banks();
    HostMemory host(device(), host_size);

    enum class Op { local, d2h_own, d2h_other, d2d_own, d2d_src_other, d2d_dst_other, h2d_own };
    const std::vector<std::pair<Op, const char*>> ops = {
        {Op::local, "local"},
        {Op::d2h_own, "d2h_own"},
        {Op::d2h_other, "d2h_other"},
        {Op::d2d_own, "d2d_own"},
        {Op::d2d_src_other, "d2d_src_other"},
        {Op::d2d_dst_other, "d2d_dst_other"},
        {Op::h2d_own, "h2d_own"}};

    // Command i of `op` on seat (bank, noc), `size` bytes.
    const auto command = [&](SeatClient& seat, Op op, uint32_t bank, uint32_t noc, uint32_t size, uint32_t i) {
        const uint32_t other = (bank + 1) % banks;
        const uint64_t a = dram_base + (uint64_t{i} * size) % dram_span;
        const uint64_t b = a + dram_span;
        const uint64_t h = (uint64_t{i} * size) % (host_size / 2);
        switch (op) {
            case Op::local:
                return abi::make_local_command(device().bank_offset(bank) + a, device().bank_offset(bank) + b, size);
            case Op::d2h_own:
                return abi::make_single_command(
                    device().dram_endpoint(bank, a, noc), seat.host_endpoint(host.device_addr(h)), size);
            case Op::d2h_other:
                return abi::make_single_command(
                    device().dram_endpoint(other, a, noc), seat.host_endpoint(host.device_addr(h)), size);
            case Op::d2d_own:
                return abi::make_single_command(
                    device().dram_endpoint(bank, a, noc), device().dram_endpoint(bank, b, noc), size);
            case Op::d2d_src_other:
                return abi::make_single_command(
                    device().dram_endpoint(other, a, noc), device().dram_endpoint(bank, b, noc), size);
            case Op::d2d_dst_other:
                return abi::make_single_command(
                    device().dram_endpoint(bank, a, noc), device().dram_endpoint(other, b, noc), size);
            case Op::h2d_own:
                return abi::make_single_command(
                    seat.host_endpoint(host.device_addr(h + host_size / 2)),
                    device().dram_endpoint(bank, a, noc),
                    size);
        }
        return abi::KernelCommand{};
    };

    std::string prefix;
    const auto print = [&prefix](const std::string& key, double value) {
        fmt::print("BENCH {}{} {:.3f}\n", prefix, key, value);
    };

    {
        std::optional<Client> client;
        attach(client, {0});
        // One seat, batches of 64 (24 KiB and 64 B), and 8 x 1 MiB.
        for (uint32_t noc = 0; noc < 2; noc++) {
            SeatClient& seat = client->seat(0, noc);
            for (const auto& [op, name] : ops) {
                for (const auto& [size, count, unit] : std::vector<std::tuple<uint32_t, uint32_t, const char*>>{
                         {24u << 10, 64, "24k_b64"}, {1u << 20, 8, "1m_b8"}, {64, 64, "64b_b64"}}) {
                    std::vector<BenchLane> lanes{{&seat, {}}};
                    for (uint32_t i = 0; i < count; i++) {
                        lanes[0].batch.push_back(command(seat, op, 0, noc, size, i));
                    }
                    const auto [gbps, mcps] = bench_run(lanes, true, size);
                    print(fmt::format("seat1_noc{}_{}_{}_gbps", noc, name, unit), gbps);
                    print(fmt::format("seat1_noc{}_{}_{}_mcps", noc, name, unit), mcps);
                }
            }
        }
        // Single (unbatched) commands and the host cost of one issue() call, on each issue path (Window keys: win_*).
        for (const IssuePath path : {IssuePath::Shared, IssuePath::Window}) {
            std::optional<Client> path_client;
            attach(path_client, {0}, path);
            prefix = path == IssuePath::Window ? "win_" : "";
            SeatClient& seat = path_client->seat(0, 0);
            for (const auto& [op, name] : std::vector<std::pair<Op, const char*>>{
                     {Op::d2h_own, "d2h_own"}, {Op::h2d_own, "h2d_own"}, {Op::d2d_own, "d2d_own"}}) {
                for (const uint32_t size : {64u, 24u << 10}) {
                    std::vector<BenchLane> lanes{{&seat, {command(seat, op, 0, 0, size, 0)}}};
                    const auto [gbps, mcps] = bench_run(lanes, false, size);
                    print(fmt::format("single_noc0_{}_{}b_mcps", name, size), mcps);
                    print(fmt::format("single_noc0_{}_{}b_gbps", name, size), gbps);
                }
            }
            {
                const abi::KernelCommand cmd = command(seat, Op::d2h_own, 0, 0, 64, 0);
                const std::vector<abi::KernelCommand> batch(64, cmd);
                for (const bool batched : {false, true}) {
                    double total_us = 0;
                    uint64_t calls = 0;
                    const auto end = clock::now() + interval;
                    while (clock::now() < end) {
                        const auto t0 = clock::now();
                        const bool ok = (batched ? seat.issue_batch(batch) : seat.issue(cmd)).has_value();
                        const auto t1 = clock::now();
                        if (ok) {
                            total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
                            calls++;
                        }
                    }
                    seat.drain(drain_timeout);
                    print(
                        batched ? "issue_batch64_call_us" : "issue_single_call_us",
                        total_us / std::max<uint64_t>(calls, 1));
                }
                // Unbatched commands with the KV comparison harness's address pattern: bank i % banks, the DRAM offset
                // advancing every `banks` commands across 1 GiB, the host side across 16 MiB. Pre-built, so the loop
                // costs only issue(). From a client alone on bank 1's seat, so in slot 0.
                {
                    std::optional<Client> spread_client;
                    attach(spread_client, {1}, path);
                    SeatClient& sseat = spread_client->seat(1, 0);
                    constexpr uint32_t size = 4352;
                    constexpr uint64_t n = 16384;
                    const uint64_t dram_slots = (1ull << 30) / size;
                    const uint64_t host_slots = (16ull << 20) / size;
                    for (const char* name : {"d2h", "h2d", "d2d"}) {
                        std::vector<abi::KernelCommand> spread;
                        spread.reserve(n);
                        for (uint64_t i = 0; i < n; i++) {
                            const uint64_t dram_off = (1ull << 30) + ((i / banks) % dram_slots) * size;
                            const uint64_t host_off = (i % host_slots) * size;
                            const auto bank = static_cast<uint32_t>(i % banks);
                            if (std::string_view(name) == "d2h") {
                                spread.push_back(abi::make_single_command(
                                    device().dram_endpoint(bank, dram_off, 0),
                                    sseat.host_endpoint(host.device_addr(host_off)),
                                    size));
                            } else if (std::string_view(name) == "h2d") {
                                spread.push_back(abi::make_single_command(
                                    sseat.host_endpoint(host.device_addr(host_off)),
                                    device().dram_endpoint(bank, dram_off, 0),
                                    size));
                            } else {
                                const uint64_t off = (1ull << 30) + (i % dram_slots) * size;
                                spread.push_back(abi::make_single_command(
                                    device().dram_endpoint(0, off, 0), device().dram_endpoint(1, off, 0), size));
                            }
                        }
                        uint64_t posted = 0;
                        const auto start = clock::now();
                        while (posted < n) {
                            if (sseat.issue(spread[posted])) {
                                posted++;
                            }
                        }
                        sseat.drain(drain_timeout);
                        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
                        print(fmt::format("spread_slot0_{}_4352b_mcps", name), n / seconds / 1e6);
                    }
                }
                // Unbatched commands with at most `depth` in flight: a rate that grows with depth means the round trip
                // limits it, a flat one means a per-command cost does. Depth 1 gives the round trip itself.
                for (const uint32_t bytes : {64u, 18432u}) {
                    const abi::KernelCommand one = command(seat, Op::d2h_own, 0, 0, bytes, 0);
                    for (const uint32_t depth : {1u, 2u, 4u, 8u}) {
                        uint64_t posted = 0;
                        const auto start = clock::now();
                        const auto end = start + interval;
                        while (clock::now() < end) {
                            if (seat.in_flight() < depth && seat.issue(one)) {
                                posted++;
                            }
                        }
                        const uint64_t done = posted - seat.in_flight();
                        const double seconds = std::chrono::duration<double>(clock::now() - start).count();
                        seat.drain(drain_timeout);
                        print(fmt::format("depth{}_d2h_{}b_mcps", depth, bytes), done / seconds / 1e6);
                        if (depth == 1) {
                            print(
                                fmt::format("depth1_d2h_{}b_roundtrip_us", bytes),
                                seconds * 1e6 / std::max<uint64_t>(done, 1));
                        }
                    }
                }
            }
            prefix.clear();
        }
        client.reset();
    }
    {
        // All 16 seats at once, from this one host thread.
        std::optional<Client> client;
        attach(client, all_banks());
        for (const auto& [op, name] : ops) {
            for (const auto& [size, count, unit] : std::vector<std::tuple<uint32_t, uint32_t, const char*>>{
                     {1u << 20, 8, "1m_b8"}, {24u << 10, 64, "24k_b64"}}) {
                std::vector<BenchLane> lanes;
                for (uint32_t bank = 0; bank < banks; bank++) {
                    for (uint32_t noc = 0; noc < 2; noc++) {
                        BenchLane lane{&client->seat(bank, noc), {}};
                        for (uint32_t i = 0; i < count; i++) {
                            lane.batch.push_back(command(*lane.seat, op, bank, noc, size, i));
                        }
                        lanes.push_back(std::move(lane));
                    }
                }
                const auto [gbps, mcps] = bench_run(lanes, true, size);
                print(fmt::format("seat16_{}_{}_gbps", name, unit), gbps);
            }
        }
        client.reset();
    }
    expect_all_stopped(all_banks());
}

}  // namespace
}  // namespace tt::tt_metal
