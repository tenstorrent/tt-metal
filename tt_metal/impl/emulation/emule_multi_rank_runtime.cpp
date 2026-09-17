// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_multi_rank_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>

#include "emule_fiber_scheduler.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "tt_emule/chip_store.hpp"
#include "tt_emule/l1_pool.hpp"
#include "tt_emule/rank_state.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::emule::multi_rank {
namespace {

constexpr uint32_t kNocLocalBits = 36;
constexpr uint32_t kNocNodeIdBits = 6;
constexpr uint64_t kNocLocalMask = (1ULL << kNocLocalBits) - 1;
constexpr uint32_t kNocNodeMask = (1 << kNocNodeIdBits) - 1;
constexpr uint32_t kL1SlotMask = static_cast<uint32_t>(tt_emule::L1Pool::SLOT_SIZE) - 1;

struct GlobalChip {
    uint64_t asic_id = 0;
    int owner_rank = -1;
    bool local = false;
};

std::mutex global_chip_mutex;
std::vector<GlobalChip> global_chips;
std::unordered_map<uint64_t, uint32_t> asic_to_global_chip;
std::map<tt::tt_fabric::FabricNodeId, uint32_t> node_to_global_chip;
std::unordered_map<uint32_t, tt::tt_fabric::FabricNodeId> global_chip_to_node;
bool global_chips_built = false;

// TopologyMapper is globally complete after its host exchange, so this registry can name chips that
// the current rank does not own without introducing another collective.
void build_global_chip_registry(tt::tt_fabric::ControlPlane& control_plane) {
    if (global_chips_built && !global_chips.empty()) {
        return;
    }
    global_chips_built = true;
    uint32_t next_synthetic = 0;
    try {
        const auto& topology_mapper = control_plane.get_topology_mapper();
        const auto all_meshes = control_plane.get_mesh_graph().get_mesh_ids();

        // Assign the real local chip ids first so later synthetic peer ids cannot collide with them.
        for (auto mesh_id : all_meshes) {
            uint32_t chip_count = 0;
            try {
                chip_count = static_cast<uint32_t>(control_plane.get_mesh_graph().get_mesh_shape(mesh_id).mesh_size());
            } catch (...) {
                continue;
            }
            for (uint32_t chip = 0; chip < chip_count; ++chip) {
                const tt::tt_fabric::FabricNodeId node(mesh_id, chip);
                uint64_t asic_id = 0;
                try {
                    asic_id = *topology_mapper.get_asic_id_from_fabric_node_id(node);
                } catch (...) {
                    continue;
                }
                int local_chip = -1;
                try {
                    local_chip = static_cast<int>(control_plane.get_physical_chip_id_from_fabric_node_id(node));
                } catch (...) {
                    continue;
                }

                const uint32_t id = static_cast<uint32_t>(local_chip);
                next_synthetic = std::max(next_synthetic, id + 1);
                if (global_chips.size() <= id) {
                    global_chips.resize(id + 1);
                }
                global_chips[id] = GlobalChip{asic_id, -1, true};
                asic_to_global_chip[asic_id] = id;
                node_to_global_chip[node] = id;
                global_chip_to_node.emplace(id, node);
            }
        }

        for (auto mesh_id : all_meshes) {
            uint32_t chip_count = 0;
            try {
                chip_count = static_cast<uint32_t>(control_plane.get_mesh_graph().get_mesh_shape(mesh_id).mesh_size());
            } catch (...) {
                continue;
            }
            for (uint32_t chip = 0; chip < chip_count; ++chip) {
                const tt::tt_fabric::FabricNodeId node(mesh_id, chip);
                if (node_to_global_chip.contains(node)) {
                    continue;
                }
                uint64_t asic_id = 0;
                try {
                    asic_id = *topology_mapper.get_asic_id_from_fabric_node_id(node);
                } catch (...) {
                    continue;
                }
                int owner_rank = -1;
                try {
                    const auto host_rank =
                        topology_mapper.get_host_rank_for_chip(mesh_id, static_cast<tt::ChipId>(chip));
                    if (host_rank.has_value()) {
                        owner_rank = topology_mapper.get_mpi_rank_for_mesh_host_rank(mesh_id, *host_rank);
                    }
                } catch (...) {
                    owner_rank = -1;
                }

                const uint32_t id = next_synthetic++;
                if (global_chips.size() <= id) {
                    global_chips.resize(id + 1);
                }
                global_chips[id] = GlobalChip{asic_id, owner_rank, false};
                asic_to_global_chip[asic_id] = id;
                node_to_global_chip[node] = id;
                global_chip_to_node.emplace(id, node);
            }
        }
    } catch (...) {
        // No control plane or fabric. Single-rank runs do not need the registry.
    }
}

std::optional<GlobalChip> global_chip_info(uint32_t chip) {
    std::lock_guard<std::mutex> lock(global_chip_mutex);
    if (chip >= global_chips.size() || global_chips[chip].asic_id == 0) {
        return std::nullopt;
    }
    return global_chips[chip];
}

struct PeerSegment {
    uint8_t* base = nullptr;
    std::unique_ptr<tt::umd::SocDescriptor> soc;
    std::unordered_map<tt_xy_pair, size_t> slot_of;
};

std::mutex peer_segment_mutex;
std::unordered_map<uint32_t, std::unique_ptr<PeerSegment>> peer_segments;

std::atomic<bool> probes_installed{false};
std::atomic<bool> fixed_point_confirmed{false};

bool peer_may_still_deliver();

void install_peer_probes() {
    static std::once_flag once;
    std::call_once(once, [] {
        tt::tt_metal::emule_fiber::set_peer_progress_probe(&peer_may_still_deliver);
        tt::tt_metal::emule_fiber::set_peer_liveness_probe(&peer_liveness);
        probes_installed.store(true, std::memory_order_release);
    });
}

bool peer_may_still_deliver() {
    auto& state = rank_state();
    if (!state.valid() || state.any_faulted() || fixed_point_confirmed.load(std::memory_order_acquire)) {
        return false;
    }
    state.publish_quiesced(true);
    return !state.global_fixed_point();
}

std::chrono::seconds peer_driver_timeout() {
    static const uint64_t seconds = [] {
        const char* value = std::getenv("TT_EMULE_PEER_DRIVER_TIMEOUT_SEC");
        if (value == nullptr || value[0] == '\0') {
            return uint64_t{120};
        }
        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        constexpr uint64_t kMaxSeconds = 86400;
        if (end == value || *end != '\0' || errno == ERANGE || std::strchr(value, '-') != nullptr || parsed == 0 ||
            static_cast<uint64_t>(parsed) > kMaxSeconds) {
            log_warning(
                tt::LogMetal,
                "TT_EMULE_PEER_DRIVER_TIMEOUT_SEC='{}' is not an integer in 1..{} seconds; using 120",
                value,
                kMaxSeconds);
            return uint64_t{120};
        }
        return static_cast<uint64_t>(parsed);
    }();
    return std::chrono::seconds(seconds);
}

[[noreturn]] void abort_peer_driver(const char* reason, const std::exception_ptr& fault) {
    std::string detail = "unknown exception";
    try {
        if (fault) {
            std::rethrow_exception(fault);
        }
    } catch (const std::exception& error) {
        detail = error.what();
    } catch (...) {
        detail = "non-std exception";
    }
    std::fprintf(stderr, "[EMULE] peer-wait driver: %s\n%s\n", reason, detail.c_str());
    rank_state().dump(reason);
    std::fflush(stderr);
    try {
        using tt::tt_metal::distributed::multihost::DistributedContext;
        if (DistributedContext::is_initialized()) {
            DistributedContext::get_world_context()->abort(EXIT_FAILURE);
        }
    } catch (...) {
    }
    std::abort();
}

class PeerWaitDriver {
public:
    explicit PeerWaitDriver(PeerDriverCallbacks callbacks) : callbacks_(std::move(callbacks)) {
        TT_FATAL(callbacks_.run_mutex != nullptr, "PeerWaitDriver requires the runner mutex");
        thread_ = std::thread([this] { run(); });
    }

    ~PeerWaitDriver() noexcept {
        driver.store(nullptr, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(wait_mutex_);
            stop_ = true;
        }
        condition_.notify_one();
        if (thread_.joinable()) {
            try {
                thread_.join();
            } catch (...) {
                std::terminate();
            }
        }
    }

    void notify() { condition_.notify_one(); }

    static std::atomic<PeerWaitDriver*> driver;

private:
    bool stopping() {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        return stop_;
    }

    void back_off() {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(1), [this] { return stop_; });
    }

    [[noreturn]] void fail_locked(const char* reason, std::exception_ptr fault) {
        rank_state().note_faulted();
        callbacks_.invalidate_run_sequence();
        abort_peer_driver(reason, fault);
    }

    void abandon_locked(const char* reason, const std::string& diagnostic) {
        try {
            tt::tt_metal::emule_fiber::FiberScheduler::instance().abandon_host_wait(diagnostic);
        } catch (...) {
            callbacks_.clear_suspended_state();
            fail_locked(reason, std::current_exception());
        }
        callbacks_.clear_suspended_state();
    }

    void run() {
        uint64_t sequence = 0;
        uint64_t progress = 0;
        std::chrono::steady_clock::time_point deadline;
        while (!stopping()) {
            if (!callbacks_.needs_peer_pump()) {
                std::unique_lock<std::mutex> lock(wait_mutex_);
                condition_.wait_for(
                    lock, std::chrono::milliseconds(50), [this] { return stop_ || callbacks_.needs_peer_pump(); });
                if (stop_) {
                    return;
                }
                if (!callbacks_.needs_peer_pump()) {
                    continue;
                }
            }

            const uint64_t current_sequence = callbacks_.run_sequence();
            const uint64_t current_progress = rank_state().delivery_sum();
            if (sequence != current_sequence || progress != current_progress) {
                sequence = current_sequence;
                progress = current_progress;
                deadline = std::chrono::steady_clock::now() + peer_driver_timeout();
            }

            std::unique_lock<std::mutex> run_lock(*callbacks_.run_mutex, std::try_to_lock);
            if (!run_lock.owns_lock()) {
                back_off();
                continue;
            }
            if (!callbacks_.needs_peer_pump() || sequence != callbacks_.run_sequence()) {
                continue;
            }
            if (rank_state().any_faulted()) {
                abandon_locked(
                    "peer rank faulted",
                    "EMULE fiber engine: peer-wait driver observed a faulted rank; the peer cannot deliver.");
                continue;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                abandon_locked(
                    "peer-wait deadline expired",
                    fmt::format(
                        "EMULE fiber engine: peer-wait driver saw no cross-rank delivery for {}s "
                        "(TT_EMULE_PEER_DRIVER_TIMEOUT_SEC).",
                        peer_driver_timeout().count()));
                continue;
            }
            if (!peer_liveness()) {
                run_lock.unlock();
                std::unique_lock<std::mutex> lock(wait_mutex_);
                condition_.wait_until(
                    lock,
                    std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(250)),
                    [this, sequence] {
                        return stop_ || !callbacks_.needs_peer_pump() || sequence != callbacks_.run_sequence();
                    });
                continue;
            }
            try {
                callbacks_.pump_locked();
            } catch (...) {
                fail_locked("background pump failed", std::current_exception());
            }
            run_lock.unlock();
            back_off();
        }
    }

    PeerDriverCallbacks callbacks_;
    std::mutex wait_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
    std::thread thread_;
};

std::atomic<PeerWaitDriver*> PeerWaitDriver::driver{nullptr};

}  // namespace

int global_chip_for_node(tt::tt_fabric::ControlPlane& control_plane, const tt::tt_fabric::FabricNodeId& node) {
    if (tt_emule::chip_store_job_is_multi_rank()) {
        std::lock_guard<std::mutex> lock(global_chip_mutex);
        build_global_chip_registry(control_plane);
        auto entry = node_to_global_chip.find(node);
        if (entry != node_to_global_chip.end()) {
            return static_cast<int>(entry->second);
        }
    }
    try {
        return static_cast<int>(control_plane.get_physical_chip_id_from_fabric_node_id(node));
    } catch (...) {
    }
    std::lock_guard<std::mutex> lock(global_chip_mutex);
    build_global_chip_registry(control_plane);
    auto entry = node_to_global_chip.find(node);
    return entry == node_to_global_chip.end() ? -1 : static_cast<int>(entry->second);
}

bool node_for_global_chip(
    tt::tt_fabric::ControlPlane& control_plane, uint32_t chip, tt::tt_fabric::FabricNodeId& node) {
    {
        std::lock_guard<std::mutex> lock(global_chip_mutex);
        auto entry = global_chip_to_node.find(chip);
        if (entry != global_chip_to_node.end()) {
            node = entry->second;
            return true;
        }
        if (chip < global_chips.size() && !global_chips[chip].local && global_chips[chip].asic_id != 0) {
            return false;
        }
    }
    try {
        node = control_plane.get_fabric_node_id_from_physical_chip_id(static_cast<tt::ChipId>(chip));
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<uint32_t> global_chip_for_asic(tt::tt_fabric::ControlPlane& control_plane, uint64_t asic_id) {
    std::lock_guard<std::mutex> lock(global_chip_mutex);
    build_global_chip_registry(control_plane);
    auto entry = asic_to_global_chip.find(asic_id);
    return entry == asic_to_global_chip.end() ? std::nullopt : std::optional<uint32_t>{entry->second};
}

uint8_t* resolve_peer_l1(uint32_t destination_chip, uint64_t noc_address, tt::umd::SWEmuleChip& local_chip) {
    if (!tt_emule::chip_store_shared()) {
        return nullptr;
    }
    const auto info = global_chip_info(destination_chip);
    if (!info.has_value() || info->local) {
        return nullptr;
    }

    PeerSegment* peer = nullptr;
    {
        std::lock_guard<std::mutex> lock(peer_segment_mutex);
        auto entry = peer_segments.find(destination_chip);
        if (entry != peer_segments.end()) {
            peer = entry->second.get();
        } else {
            const std::string prefix =
                "tt_emule." + tt_emule::chip_store_job_id() + ".u" + std::to_string(info->asic_id) + ".h";
            std::optional<uint64_t> peer_mask;
            try {
                for (const auto& path : std::filesystem::directory_iterator("/dev/shm")) {
                    const std::string name = path.path().filename().string();
                    if (!name.starts_with(prefix)) {
                        continue;
                    }
                    const std::string suffix = name.substr(prefix.size());
                    if (suffix.empty() ||
                        !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
                        continue;
                    }
                    const uint64_t mask = std::stoull(suffix);
                    if (peer_mask.has_value() && *peer_mask != mask) {
                        peer_mask.reset();
                        break;
                    }
                    peer_mask = mask;
                }
            } catch (...) {
                peer_mask.reset();
            }

            if (peer_mask.has_value()) {
                constexpr uint64_t kHarvestMask = (1ULL << 20) - 1;
                tt::ChipInfo chip_info;
                chip_info.noc_translation_enabled = local_chip.get_soc_descriptor().noc_translation_enabled;
                chip_info.harvesting_masks.tensix_harvesting_mask = *peer_mask & kHarvestMask;
                chip_info.harvesting_masks.dram_harvesting_mask = (*peer_mask >> 20) & kHarvestMask;
                chip_info.harvesting_masks.eth_harvesting_mask = (*peer_mask >> 40) & kHarvestMask;
                auto soc = std::make_unique<tt::umd::SocDescriptor>(
                    std::make_shared<tt::umd::SocArchDescriptor>(local_chip.get_soc_descriptor().arch), chip_info);
                auto slot_of = tt::umd::build_worker_slot_map(*soc);
                const size_t bytes = slot_of.size() * tt_emule::L1Pool::SLOT_SIZE;
                auto* base = static_cast<uint8_t*>(
                    tt_emule::chip_store_attach_peer(info->asic_id, *peer_mask, bytes, tt_emule::L1Pool::SLOT_SIZE));
                if (base != nullptr) {
                    auto segment = std::make_unique<PeerSegment>();
                    segment->base = base;
                    segment->soc = std::move(soc);
                    segment->slot_of = std::move(slot_of);
                    peer = peer_segments.emplace(destination_chip, std::move(segment)).first->second.get();
                }
            }
            if (peer == nullptr) {
                static std::set<uint32_t> warned;
                if (warned.insert(destination_chip).second) {
                    std::fprintf(
                        stderr,
                        "[EMULE_FABRIC] WARNING: no shared segment yet for peer chip=%u (asic=0x%llx, "
                        "rank=%d); will retry. A persistent miss drops deliveries to it.\n",
                        destination_chip,
                        static_cast<unsigned long long>(info->asic_id),
                        info->owner_rank);
                }
            }
        }
    }
    if (peer == nullptr) {
        return nullptr;
    }

    const uint32_t noc_x = (noc_address >> kNocLocalBits) & kNocNodeMask;
    const uint32_t noc_y = (noc_address >> (kNocLocalBits + kNocNodeIdBits)) & kNocNodeMask;
    auto verbatim = peer->slot_of.find(tt_xy_pair(noc_x, noc_y));
    size_t slot = verbatim == peer->slot_of.end() ? SIZE_MAX : verbatim->second;
    try {
        auto logical = local_chip.get_soc_descriptor().translate_coord_to(
            tt_xy_pair(noc_x, noc_y), tt::CoordSystem::TRANSLATED, tt::CoordSystem::LOGICAL);
        auto destination = peer->soc->translate_coord_to(
            tt_xy_pair(logical.x, logical.y), tt::CoordSystem::LOGICAL, tt::CoordSystem::TRANSLATED);
        auto translated = peer->slot_of.find(destination);
        if (translated != peer->slot_of.end()) {
            slot = translated->second;
        }
    } catch (...) {
        // Preserve a valid verbatim worker mapping when coordinate translation is unavailable.
    }
    if (slot == SIZE_MAX) {
        return nullptr;
    }

    const uint64_t raw_offset = noc_address & kNocLocalMask;
    if (raw_offset >= static_cast<uint64_t>(peer->soc->worker_l1_size)) {
        return nullptr;
    }
    return peer->base + slot * tt_emule::L1Pool::SLOT_SIZE + (static_cast<uint32_t>(raw_offset) & kL1SlotMask);
}

tt_emule::RankState& rank_state() {
    static tt_emule::RankState state = [] {
        using tt::tt_metal::distributed::multihost::DistributedContext;
        if (!tt_emule::chip_store_shared() || !DistributedContext::is_initialized()) {
            return tt_emule::RankState(nullptr, 0, 0);
        }
        const auto& context = DistributedContext::get_current_world();
        const auto world = static_cast<uint32_t>(*context->size());
        if (world <= 1) {
            return tt_emule::RankState(nullptr, 0, 0);
        }
        tt_emule::RankState result = tt_emule::rank_state_attach(world, static_cast<uint32_t>(*context->rank()));
        if (result.valid()) {
            result.join();
            install_peer_probes();
        }
        return result;
    }();
    static const struct RankDeparture {
        tt_emule::RankState* state;
        ~RankDeparture() { state->note_departed(); }
    } departure{&state};
    (void)departure;
    return state;
}

void begin_dispatch() {
    rank_state().begin_dispatch();
    fixed_point_confirmed.store(false, std::memory_order_release);
}

bool peer_liveness() {
    auto& state = rank_state();
    return state.valid() && !state.any_faulted() && !fixed_point_confirmed.load(std::memory_order_acquire) &&
           state.any_peer_unfinished() && !state.global_fixed_point();
}

bool peer_probes_installed() { return probes_installed.load(std::memory_order_acquire); }

void note_deliveries(uint32_t count) {
    auto& state = rank_state();
    for (uint32_t index = 0; index < count; ++index) {
        state.note_delivery();
    }
}

void ensure_peer_wait_driver(PeerDriverCallbacks callbacks) {
    if (!peer_probes_installed()) {
        return;
    }
    static PeerWaitDriver peer_driver(std::move(callbacks));
    PeerWaitDriver::driver.store(&peer_driver, std::memory_order_release);
}

void notify_peer_wait_driver() {
    if (auto* peer_driver = PeerWaitDriver::driver.load(std::memory_order_acquire)) {
        peer_driver->notify();
    }
}

}  // namespace tt::tt_metal::emule::multi_rank

extern "C" int __emule_gchip_for_node(uint32_t mesh_id, uint32_t chip_id) {
    try {
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        return tt::tt_metal::emule::multi_rank::global_chip_for_node(
            control_plane, tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{mesh_id}, chip_id));
    } catch (...) {
        return -1;
    }
}
