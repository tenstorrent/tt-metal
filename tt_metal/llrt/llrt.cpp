// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "hal.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "hal_types.hpp"
#include "llrt.hpp"
#include <umd/device/driver_atomics.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <llrt/tt_cluster.hpp>

// llrt = lower-level runtime
namespace tt::llrt {

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

namespace {

struct BinaryWriteRecord {
    std::vector<uint32_t> data;
    uint64_t address;
    uint64_t generation;  // incremented on each write, used to detect stale snapshots
};

struct CoreKey {
    tt::ChipId chip_id;
    CoreCoord core;
    uint64_t address;

    bool operator==(const CoreKey& other) const {
        return chip_id == other.chip_id && core == other.core && address == other.address;
    }
};

struct CoreKeyHash {
    std::size_t operator()(const CoreKey& k) const {
        auto h1 = std::hash<tt::ChipId>{}(k.chip_id);
        auto h2 = std::hash<std::size_t>{}(k.core.x);
        auto h3 = std::hash<std::size_t>{}(k.core.y);
        auto h4 = std::hash<uint64_t>{}(k.address);
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h4 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

struct PerChipValidationThread {
    std::thread thread;
    std::atomic<bool> stop_requested{false};
};

struct BinaryWriteTracker {
    std::mutex mutex;
    std::unordered_map<CoreKey, BinaryWriteRecord, CoreKeyHash> records;
    std::atomic<uint64_t> global_generation{0};

    std::mutex threads_mutex;
    std::unordered_map<tt::ChipId, std::unique_ptr<PerChipValidationThread>> validation_threads;

    static BinaryWriteTracker& instance() {
        static BinaryWriteTracker tracker;
        return tracker;
    }

    void record_write(
        tt::ChipId chip_id, const CoreCoord& core, uint64_t address, const uint32_t* data, uint32_t len_words) {
        std::lock_guard<std::mutex> lock(mutex);
        CoreKey key{chip_id, core, address};
        uint64_t gen = global_generation.fetch_add(1, std::memory_order_relaxed);
        records[key] = BinaryWriteRecord{
            std::vector<uint32_t>(data, data + len_words),
            address,
            gen,
        };
    }

    void clear_chip(tt::ChipId chip_id) {
        std::lock_guard<std::mutex> lock(mutex);
        std::erase_if(records, [chip_id](const auto& pair) { return pair.first.chip_id == chip_id; });
    }

    // Take a snapshot of all records for a chip (under lock), so validation can run without holding the lock.
    std::vector<std::pair<CoreKey, BinaryWriteRecord>> snapshot_chip(tt::ChipId chip_id) {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::pair<CoreKey, BinaryWriteRecord>> result;
        for (const auto& [key, record] : records) {
            if (key.chip_id == chip_id) {
                result.emplace_back(key, record);
            }
        }
        return result;
    }

    // Check if a record's generation still matches (i.e., it hasn't been overwritten since the snapshot).
    bool is_record_current(const CoreKey& key, uint64_t snapshot_generation) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(key);
        return it != records.end() && it->second.generation == snapshot_generation;
    }
};

bool binary_write_tracking_enabled() {
    static bool enabled = std::getenv("TT_METAL_BINARY_READBACK_ON_CLOSE") != nullptr;
    return enabled;
}

uint32_t get_validation_interval_ms() {
    const char* env = std::getenv("TT_METAL_BINARY_READBACK_INTERVAL_MS");
    if (env) {
        return static_cast<uint32_t>(std::stoul(env));
    }
    return 5000;
}

// Core validation logic, works on a snapshot. Returns (checked, failed, skipped).
// When check_generation is true, records that have been overwritten since the snapshot are skipped
// (avoids false positives when programs are actively being dispatched).
struct ValidationResult {
    uint32_t checked;
    uint32_t failed;
    uint32_t skipped;
};

ValidationResult validate_snapshot(
    tt::ChipId chip_id,
    const std::vector<std::pair<CoreKey, BinaryWriteRecord>>& snapshot,
    const char* context,
    bool check_generation) {
    uint32_t checked = 0;
    uint32_t failed = 0;
    uint32_t skipped = 0;

    for (const auto& [key, record] : snapshot) {
        uint32_t size_bytes = record.data.size() * sizeof(uint32_t);
        std::vector<uint32_t> read_data(record.data.size());
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            read_data.data(), size_bytes, tt_cxy_pair(chip_id, key.core), key.address);

        if (read_data != record.data) {
            // Before reporting, check if this record was overwritten by a newer write.
            if (check_generation && !BinaryWriteTracker::instance().is_record_current(key, record.generation)) {
                skipped++;
                continue;
            }

            checked++;
            failed++;

            uint32_t first_mismatch_idx = 0;
            for (uint32_t i = 0; i < record.data.size(); i++) {
                if (read_data[i] != record.data[i]) {
                    first_mismatch_idx = i;
                    break;
                }
            }

            log_error(
                tt::LogLLRuntime,
                "[{}] Binary readback MISMATCH on chip {} core {} addr 0x{:x}: "
                "size={} words, first mismatch at word {} (expected 0x{:08x}, got 0x{:08x})",
                context,
                chip_id,
                key.core.str(),
                key.address,
                record.data.size(),
                first_mismatch_idx,
                record.data[first_mismatch_idx],
                read_data[first_mismatch_idx]);
        } else {
            checked++;
        }
    }

    return {checked, failed, skipped};
}

void validation_thread_fn(tt::ChipId chip_id, PerChipValidationThread* state) {
    uint32_t interval_ms = get_validation_interval_ms();
    uint64_t iteration = 0;

    log_info(
        tt::LogLLRuntime,
        "Binary readback validation thread started for chip {} (interval={}ms)",
        chip_id,
        interval_ms);

    while (!state->stop_requested.load(std::memory_order_relaxed)) {
        // Sleep in small increments so we can respond to stop quickly
        for (uint32_t elapsed = 0; elapsed < interval_ms && !state->stop_requested.load(std::memory_order_relaxed);
             elapsed += 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (state->stop_requested.load(std::memory_order_relaxed)) {
            break;
        }

        iteration++;
        auto snapshot = BinaryWriteTracker::instance().snapshot_chip(chip_id);
        if (snapshot.empty()) {
            continue;
        }

        auto result = validate_snapshot(chip_id, snapshot, "periodic", true);

        if (result.failed > 0) {
            log_error(
                tt::LogLLRuntime,
                "Periodic binary validation iteration {} on chip {}: {}/{} regions FAILED ({} skipped as stale)",
                iteration,
                chip_id,
                result.failed,
                result.checked,
                result.skipped);
        } else {
            log_debug(
                tt::LogLLRuntime,
                "Periodic binary validation iteration {} on chip {}: {}/{} regions OK ({} skipped as stale)",
                iteration,
                chip_id,
                result.checked,
                result.checked,
                result.skipped);
        }
    }

    log_info(tt::LogLLRuntime, "Binary readback validation thread stopped for chip {}", chip_id);
}

}  // anonymous namespace

bool is_binary_write_tracking_enabled() { return binary_write_tracking_enabled(); }

void start_binary_validation_thread(tt::ChipId chip_id) {
    auto& tracker = BinaryWriteTracker::instance();
    std::lock_guard<std::mutex> lock(tracker.threads_mutex);
    if (tracker.validation_threads.count(chip_id)) {
        return;
    }
    auto ctx = std::make_unique<PerChipValidationThread>();
    auto* raw = ctx.get();
    ctx->thread = std::thread(validation_thread_fn, chip_id, raw);
    tracker.validation_threads[chip_id] = std::move(ctx);
}

void stop_binary_validation_thread(tt::ChipId chip_id) {
    auto& tracker = BinaryWriteTracker::instance();
    std::unique_ptr<PerChipValidationThread> ctx;
    {
        std::lock_guard<std::mutex> lock(tracker.threads_mutex);
        auto it = tracker.validation_threads.find(chip_id);
        if (it == tracker.validation_threads.end()) {
            return;
        }
        ctx = std::move(it->second);
        tracker.validation_threads.erase(it);
    }
    ctx->stop_requested.store(true, std::memory_order_relaxed);
    if (ctx->thread.joinable()) {
        ctx->thread.join();
    }
}

bool validate_binary_writes_on_device(tt::ChipId chip_id) {
    if (!binary_write_tracking_enabled()) {
        return true;
    }

    // Stop the periodic thread first so we don't race with the final validation
    stop_binary_validation_thread(chip_id);

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(chip_id);

    auto snapshot = BinaryWriteTracker::instance().snapshot_chip(chip_id);
    auto result = validate_snapshot(chip_id, snapshot, "close", false);

    if (result.checked > 0) {
        log_info(
            tt::LogLLRuntime,
            "Final binary readback validation on chip {}: checked {} regions, {} passed, {} failed",
            chip_id,
            result.checked,
            result.checked - result.failed,
            result.failed);
    }

    return result.failed == 0;
}

void clear_tracked_binary_writes(tt::ChipId chip_id) {
    stop_binary_validation_thread(chip_id);
    BinaryWriteTracker::instance().clear_chip(chip_id);
}

static thread_local int suppress_tracking_depth = 0;

SuppressBinaryTracking::SuppressBinaryTracking() { suppress_tracking_depth++; }
SuppressBinaryTracking::~SuppressBinaryTracking() { suppress_tracking_depth--; }
bool SuppressBinaryTracking::is_suppressed() { return suppress_tracking_depth > 0; }

const ll_api::memory& get_risc_binary(
    const std::string& path,
    ll_api::memory::Loading loading,
    const std::function<void(ll_api::memory&)>& update_callback) {
    static struct {
        std::unordered_map<std::string, std::unique_ptr<const ll_api::memory>> map;
        std::mutex mutex;
        std::condition_variable cvar;
    } cache;

    std::unique_lock lock(cache.mutex);
    auto [slot, inserted] = cache.map.try_emplace(path);
    const ll_api::memory* ptr = nullptr;
    if (inserted) {
        // We're the first with PATH. Create and insert.
        lock.unlock();
        ll_api::memory* mutable_ptr = new ll_api::memory(path, loading);
        if (update_callback) {
            update_callback(*mutable_ptr);
        }

        lock.lock();
        // maps have iterator stability, so SLOT is still valid.
        slot->second = decltype(slot->second)(mutable_ptr);
        ptr = mutable_ptr;
        // We can't wake just those waiting on this slot, so wake them
        // all. Should be a rare event anyway.
        cache.cvar.notify_all();
    } else {
        if (!slot->second) {
            // Someone else is creating the initial entry, wait for them.
            cache.cvar.wait(lock, [=] { return bool(slot->second); });
        }
        ptr = slot->second.get();
        TT_ASSERT(ptr->get_loading() == loading);
    }

    return *ptr;
}

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
CoreCoord logical_core_from_ethernet_core(tt::ChipId chip_id, const CoreCoord& ethernet_core) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
        chip_id, ethernet_core);
}

tt_metal::HalProgrammableCoreType get_core_type(tt::ChipId chip_id, const CoreCoord& virtual_core) {
    bool is_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, chip_id);
    bool is_active_eth_core = false;
    bool is_inactive_eth_core = false;

    // Determine whether an ethernet core is active or idle. Their host handshake interfaces are different.
    if (is_eth_core) {
        auto active_eth_cores =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(chip_id);
        auto inactive_eth_cores =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(chip_id);
        is_active_eth_core = active_eth_cores.contains(logical_core_from_ethernet_core(chip_id, virtual_core));
        is_inactive_eth_core = inactive_eth_cores.contains(logical_core_from_ethernet_core(chip_id, virtual_core));
        // we should not be operating on any reserved cores here.
        TT_ASSERT(is_active_eth_core or is_inactive_eth_core);
    }

    if (is_active_eth_core) {
        return tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    }
    if (is_inactive_eth_core) {
        return tt_metal::HalProgrammableCoreType::IDLE_ETH;
    }
    return tt_metal::HalProgrammableCoreType::TENSIX;
}

void send_reset_go_signal(tt::ChipId chip, const CoreCoord& virtual_core) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip, virtual_core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt_metal::MetalContext::instance().get_cluster();
    uint64_t go_signal_adrr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);
    auto reset_msg = hal.get_dev_msgs_factory(dispatch_core_type).create<tt_metal::dev_msgs::go_msg_t>();

    reset_msg.view().signal() = tt_metal::dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST;
    cluster.write_core_immediate(
        reset_msg.data(), reset_msg.size(), {static_cast<size_t>(chip), virtual_core}, go_signal_adrr);
    cluster.l1_barrier(chip);
    uint32_t go_message_index_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG_INDEX);
    uint32_t zero = 0;
    cluster.write_core_immediate(
        &zero, sizeof(uint32_t), {static_cast<size_t>(chip), virtual_core}, go_message_index_addr);
}

void write_launch_msg_to_core(
    tt::ChipId chip,
    CoreCoord core,
    tt_metal::dev_msgs::launch_msg_t::View msg,
    tt_metal::dev_msgs::go_msg_t::ConstView go_msg,
    bool send_go) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip, core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt_metal::MetalContext::instance().get_cluster();

    msg.kernel_config().mode() = tt_metal::dev_msgs::DISPATCH_MODE_HOST;

    uint64_t launch_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::LAUNCH);
    uint64_t go_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

    cluster.write_core_immediate(msg.data(), msg.size(), {static_cast<size_t>(chip), core}, launch_addr);
    tt_driver_atomics::sfence();
    if (send_go) {
        cluster.write_core_immediate(go_msg.data(), go_msg.size(), {static_cast<size_t>(chip), core}, go_addr);
    }
}

ll_api::memory read_mem_from_core(
    tt::ChipId chip, const CoreCoord& core, const ll_api::memory& mem, uint64_t local_init_addr) {
    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            &*mem_ptr, len * sizeof(uint32_t), tt_cxy_pair(chip, core), relo_addr);
    });
    return read_mem;
}

bool test_load_write_read_risc_binary(
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx) {
    TT_ASSERT(
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(core, chip_id) or
        tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(core, chip_id));

    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(core_type_idx, processor_class_idx, processor_type_idx)
                                   .local_init_addr;

    auto core_type = get_core_type(chip_id, core);
    // Depending on the arch, active ethernet may be shared local memory with the base firmware
    // Primary risc is shared
    // TODO: Move this query into the HAL
    bool local_mem_offset = processor_type_idx == 0 && core_type == tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr =
            tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr, local_mem_offset);

        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), relo_addr);
    });

    log_debug(tt::LogLLRuntime, "wrote hex to core {}", core.str().c_str());

    if (std::getenv("TT_METAL_KERNEL_READBACK_ENABLE") != nullptr) {
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(chip_id);
        ll_api::memory read_mem = read_mem_from_core(chip_id, core, mem, local_init_addr);
        log_debug(tt::LogLLRuntime, "read hex back from the core");
        return mem == read_mem;
    }

    return true;
}

bool test_load_multicast_write_risc_binary(
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const CoreCoord& start_core,
    const CoreCoord& end_core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx) {
    TT_ASSERT(
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(start_core, chip_id) and
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(end_core, chip_id));

    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(core_type_idx, processor_class_idx, processor_type_idx)
                                   .local_init_addr;

    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr =
            tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr, false);

        tt::tt_metal::MetalContext::instance().get_cluster().noc_multicast_write(
            &*mem_ptr, len_words * sizeof(uint32_t), chip_id, start_core, end_core, relo_addr);
    });

    log_debug(tt::LogLLRuntime, "multicast hex to cores {} - {}", start_core.str().c_str(), end_core.str().c_str());

    if (std::getenv("TT_METAL_KERNEL_READBACK_ENABLE")) {
        log_info(tt::LogLLRuntime, "WARNING: Readback after multicast write is not yet supported");
    }

    return true;
}

void write_binary_to_address(const ll_api::memory& mem, tt::ChipId chip_id, const CoreCoord& core, uint32_t address) {
    log_debug(tt::LogLLRuntime, "vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t /*addr*/, uint32_t len_words) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), address);
        if (binary_write_tracking_enabled() && !SuppressBinaryTracking::is_suppressed()) {
            BinaryWriteTracker::instance().record_write(chip_id, core, address, &*mem_ptr, len_words);
            start_binary_validation_thread(chip_id);
        }
    });
}

namespace internal_ {

namespace {

bool check_if_riscs_on_specified_core_done(tt::ChipId chip_id, const CoreCoord& core, int run_state) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip_id, core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    auto dev_msgs_factory = hal.get_dev_msgs_factory(dispatch_core_type);

    uint64_t go_msg_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

    auto get_mailbox_is_done = [&](uint64_t go_msg_addr) {
        auto core_status = dev_msgs_factory.create<tt_metal::dev_msgs::go_msg_t>();
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            core_status.data(), core_status.size(), {static_cast<size_t>(chip_id), core}, go_msg_addr & ~0x3);
        uint8_t run = core_status.view().signal();
        if (run != run_state && run != tt_metal::dev_msgs::RUN_MSG_DONE) {
            fprintf(
                stderr,
                "Read unexpected run_mailbox value: 0x%x (expected 0x%x or 0x%x)\n",
                run,
                run_state,
                tt_metal::dev_msgs::RUN_MSG_DONE);
            TT_FATAL(
                run == run_state || run == tt_metal::dev_msgs::RUN_MSG_DONE,
                "Read unexpected run_mailbox value from core {}",
                core.str());
        }

        return run == tt_metal::dev_msgs::RUN_MSG_DONE;
    };
    return get_mailbox_is_done(go_msg_addr);
}

void print_aerisc_training_status(tt::ChipId device_id, const CoreCoord& virtual_core) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        return;
    }
    if (get_core_type(device_id, virtual_core) != tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH) {
        return;
    }
    const auto port_status_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::PORT_STATUS);
    const auto retrain_count_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::RETRAIN_COUNT);
    const auto rx_link_up_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::RX_LINK_UP);
    uint32_t port_status = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, port_status_addr, sizeof(uint32_t))[0];
    uint32_t retrain_count = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, retrain_count_addr, sizeof(uint32_t))[0];
    uint32_t rx_link_up = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, rx_link_up_addr, sizeof(uint32_t))[0];
    log_critical(
        tt::LogMetal,
        "Device {}: Virtual core {}, Port status: {:#x}, Retrain count: {:#x}, Rx link up: {:#x}",
        device_id,
        virtual_core.str(),
        port_status,
        retrain_count,
        rx_link_up);
}

}  // namespace

void wait_until_cores_done(
    tt::ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms) {
    // poll the cores until the set of not done cores is empty
    [[maybe_unused]] int loop_count = 1;
    auto start = std::chrono::high_resolution_clock::now();
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    bool is_simulator = rtoptions.get_simulator_enabled();
    // For simulators, always disable timeout (infinite wait). For non-simulators, a 0
    // timeout means: use the configured timeout for operations.
    if (is_simulator) {
        timeout_ms = 0;
    } else if (timeout_ms == 0) {
        timeout_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(rtoptions.get_timeout_duration_for_operations())
                .count();
    }
    while (!not_done_phys_cores.empty()) {
        if (timeout_ms > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                for (const auto& core : not_done_phys_cores) {
                    // only prints if the core is an active ethernet core
                    print_aerisc_training_status(device_id, core);
                }
                std::string cores = fmt::format("{}", fmt::join(not_done_phys_cores, ", "));

                tt::tt_metal::MetalContext::instance().on_dispatch_timeout_detected();

                TT_THROW(
                    "Device {}: Timeout ({} ms) waiting for physical cores to finish: {}.",
                    device_id,
                    timeout_ms,
                    cores);
            }
        }

#ifdef DEBUG
        // Print not-done cores
        if (loop_count % 1000 == 0) {
            log_debug(
                tt::LogMetal, "Device {}: Not done phys cores: {}", device_id, fmt::join(not_done_phys_cores, " "));
            usleep(100000);
        }
#endif

        for (auto it = not_done_phys_cores.begin(); it != not_done_phys_cores.end();) {
            const auto& phys_core = *it;

            bool is_done = llrt::internal_::check_if_riscs_on_specified_core_done(device_id, phys_core, run_state);
            if (is_done) {
                log_debug(tt::LogMetal, "Device {}: Phys cores just done: {}", device_id, phys_core.str());
                it = not_done_phys_cores.erase(it);
            } else {
                ++it;
            }
        }
        loop_count++;

        // Continuously polling cores here can cause other host-driven noc transactions (dprint, watcher) to drastically
        // slow down for remote devices. So when debugging with these features, add a small delay to allow other
        // host-driven transactions through.
        if (rtoptions.get_watcher_enabled() || rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

void wait_for_idle(ChipId device_id, const std::vector<std::vector<CoreCoord>>& logical_cores) {
    std::unordered_set<CoreCoord> not_done_cores;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        const auto& logical_cores_of_type = logical_cores[index];
        for (const auto& logical_core : logical_cores_of_type) {
            auto physical_core =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, core_type);
            not_done_cores.insert(physical_core);
        }
    }
    wait_until_cores_done(device_id, tt_metal::dev_msgs::RUN_MSG_GO, not_done_cores);
}

void send_msg_to_eth_mailbox(
    tt::ChipId device_id,
    const CoreCoord& virtual_core,
    tt_metal::FWMailboxMsg msg_type,
    int mailbox_index,
    std::vector<uint32_t> args,
    bool wait_for_ack,
    int timeout_ms) {
    constexpr auto k_sleep_time = std::chrono::nanoseconds{5};
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }

    TT_ASSERT(
        get_core_type(device_id, virtual_core) == tt_metal::HalProgrammableCoreType::ACTIVE_ETH,
        "target core for send_msg_to_eth_mailbox {} (virtual) must be an active ethernet core",
        virtual_core.str());

    const auto max_args = hal.get_eth_fw_mailbox_arg_count();
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(mailbox_index);
    const auto status_mask = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto call = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_CALL);
    const auto done_message = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_DONE);

    // Check mailbox is empty/ready
    uint32_t msg_status = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                              device_id, virtual_core, mailbox_addr, sizeof(uint32_t))[0] &
                          status_mask;
    {
        const auto start_time = std::chrono::steady_clock::now();
        while (msg_status != done_message && msg_status != 0) {
            uint32_t mailbox_val = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                device_id, virtual_core, mailbox_addr, sizeof(uint32_t))[0];
            msg_status = mailbox_val & status_mask;
            const auto timenow = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timenow - start_time).count();
            if (elapsed > timeout_ms) {
                log_debug(
                    tt::LogMetal,
                    "Device {}: Timed out while waiting for ack when trying to launch Metal ethernet firmware on "
                    "ethernet core {}. Last message status: {:#x}",
                    device_id,
                    virtual_core.str(),
                    mailbox_val);

                TT_THROW("Device {} Firmware update is required. Minimum tt-firmware version is 18.10.0", device_id);
            }
            std::this_thread::sleep_for(k_sleep_time);
        }
    }

    // Must write args first.
    // Pad remaining args to zero
    args.resize(max_args, 0);
    uint32_t first_arg_addr = hal.get_eth_fw_mailbox_arg_addr(mailbox_index, 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        args.data(), tt_cxy_pair(device_id, virtual_core), first_arg_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);

    const auto msg_val = hal.get_eth_fw_mailbox_val(msg_type);
    const uint32_t msg = call | msg_val;
    log_debug(
        tt::LogLLRuntime,
        "Device {}: Eth {} Mailbox {:#x} Command {:#x}, {}",
        device_id,
        virtual_core.str(),
        mailbox_addr,
        msg,
        fmt::join(args, ", "));
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        std::vector<uint32_t>{msg}.data(), tt_cxy_pair(device_id, virtual_core), mailbox_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);

    // Wait for ack
    tt_cxy_pair target{static_cast<size_t>(device_id), virtual_core};
    if (wait_for_ack) {
        const auto start_time = std::chrono::steady_clock::now();
        do {
            uint32_t mailbox_val = 0;
            tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&mailbox_val, target, mailbox_addr);
            msg_status = mailbox_val & status_mask;
            const auto timenow = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timenow - start_time).count();
            if (elapsed > timeout_ms) {
                log_debug(
                    tt::LogMetal,
                    "Device {}: Timed out while waiting for ack when trying to launch Metal ethernet firmware on "
                    "ethernet core {}. Last message status: {:#x}",
                    device_id,
                    virtual_core.str(),
                    mailbox_val);

                TT_THROW("Device {} Firmware update is required. Minimum tt-firmware version is 18.10.0", device_id);
            }
            std::this_thread::sleep_for(k_sleep_time);
        } while (msg_status != done_message);
    }
}

void return_to_base_firmware_and_wait_for_heartbeat(
    tt::ChipId device_id, const CoreCoord& virtual_core, int timeout_ms) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }

    tt_cxy_pair target{static_cast<size_t>(device_id), virtual_core};
    const auto heartbeat_addr = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::HEARTBEAT);

    uint32_t heartbeat_val = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&heartbeat_val, target, heartbeat_addr);

    constexpr auto k_sleep_time = std::chrono::nanoseconds{5};
    std::this_thread::sleep_for(k_sleep_time);

    uint32_t previous_heartbeat_val = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&previous_heartbeat_val, target, heartbeat_addr);

    const auto start = std::chrono::steady_clock::now();

    // Below steps can be skipped if we already have a heartbeat from the base firmware
    while (heartbeat_val == previous_heartbeat_val) {
        std::this_thread::sleep_for(k_sleep_time);
        // Try sending the stop message again
        tt::llrt::internal_::set_metal_eth_fw_run_flag(device_id, virtual_core, false);
        previous_heartbeat_val = heartbeat_val;
        tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&heartbeat_val, target, heartbeat_addr);
        if (timeout_ms > 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                print_aerisc_training_status(device_id, virtual_core);
                TT_THROW(
                    "Device {}: Timed out while waiting for active ethernet core {} to become active again. "
                    "Try resetting the board. Minimum tt-firmware version is 18.10.0",
                    device_id,
                    virtual_core.str());
            }
        }
    }
}

void set_metal_eth_fw_run_flag(tt::ChipId device_id, const CoreCoord& virtual_core, bool enable) {
    constexpr auto k_CoreType = tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }
    tt::tt_metal::DeviceAddr mailbox_addr = hal.get_dev_addr(k_CoreType, tt::tt_metal::HalL1MemAddrType::MAILBOX);
    tt::tt_metal::DeviceAddr run_flag_addr =
        mailbox_addr + hal.get_dev_msgs_factory(k_CoreType)
                           .offset_of<tt::tt_metal::dev_msgs::mailboxes_t>(
                               tt::tt_metal::dev_msgs::mailboxes_t::Field::aerisc_run_flag);
    std::vector<uint32_t> en = {enable};
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        en.data(), tt_cxy_pair(device_id, virtual_core), run_flag_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);
}

}  // namespace internal_

}  // namespace tt::llrt
