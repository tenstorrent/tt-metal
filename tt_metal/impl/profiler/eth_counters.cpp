// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_counters.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_paths.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "device.hpp"

namespace tt::tt_metal::eth_counters {

namespace ec = llk::perf::eth;

namespace {

constexpr int kCalibrationPairs = 3;
constexpr auto kCalibrationGap = std::chrono::milliseconds(50);

std::optional<ec::Arch> arch_of(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return ec::Arch::BLACKHOLE;
        case tt::ARCH::WORMHOLE_B0: return ec::Arch::WORMHOLE;
        case tt::ARCH::QUASAR: return ec::Arch::QUASAR;
        default: return std::nullopt;
    }
}

// The first ethernet firmware that keeps the link counters in L1.
tt::umd::SemVer min_l1_firmware(ec::Arch arch) {
    return arch == ec::Arch::BLACKHOLE ? tt::umd::SemVer(1, 5, 0) : tt::umd::SemVer(6, 14, 0);
}

struct Pending {
    Reading before;
    uint64_t program_id = 0;
    bool overlapped = false;
};

struct DeviceState {
    bool disabled = false;
    bool prepared = false;
    bool calibrated = false;
    Plan plan;
    std::vector<std::pair<uint32_t, uint32_t>> cores;  // translated coordinates
    std::vector<int> channels;
    std::vector<double> idle_per_second;
    std::optional<Reading> run_start;
    std::optional<Pending> pending;
    uint64_t launch_seq = 0;
};

std::mutex g_mutex;
std::unordered_map<ChipId, DeviceState> g_devices;
bool g_traffic_started = false;
bool g_run_started = false;
bool g_idle_started = false;
uint64_t g_run_id = 0;

Reading read_all(MetalContext& ctx, ChipId chip, const DeviceState& st) {
    const Plan& plan = st.plan;
    Reading reading;
    reading.values.reserve(plan.reading_size());
    reading.time = std::chrono::steady_clock::now();
    std::vector<uint32_t> block(plan.l1_bytes / sizeof(uint32_t));
    for (size_t c = 0; c < st.cores.size(); c++) {
        const tt_cxy_pair core(chip, st.cores[c].first, st.cores[c].second);
        if (!block.empty()) {
            ctx.get_cluster().read_core(block.data(), plan.l1_bytes, core, plan.l1_base);
        }
        for (const Item& item : plan.items) {
            if (item.kind == Kind::L1) {
                const size_t word = (item.address - plan.l1_base) / sizeof(uint32_t);
                const uint64_t first = block[word];
                if (item.words == 1) {
                    reading.values.push_back(first);
                } else {
                    const uint64_t second = block[word + 1];
                    reading.values.push_back(item.high_first ? (first << 32 | second) : (second << 32 | first));
                }
            } else {
                uint32_t value = 0;
                ctx.get_cluster().read_reg(&value, core, item.address);
                reading.values.push_back(value);
            }
        }
    }
    return reading;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Counts per second that the queue registers move with no program running: keepalives and fabric heartbeats.
void calibrate_idle(MetalContext& ctx, ChipId chip, DeviceState& st) {
    std::vector<std::vector<double>> rates(st.plan.reading_size());
    for (int i = 0; i < kCalibrationPairs; i++) {
        const Reading a = read_all(ctx, chip, st);
        std::this_thread::sleep_for(kCalibrationGap);
        const Reading b = read_all(ctx, chip, st);
        const double seconds = std::chrono::duration<double>(b.time - a.time).count();
        for (size_t v = 0; v < rates.size(); v++) {
            const Item& item = st.plan.items[v % st.plan.items.size()];
            const double moved = is_register(item) ? double(uint32_t(b.values[v] - a.values[v])) : 0.0;
            rates[v].push_back(moved / seconds);
        }
    }
    st.idle_per_second.assign(rates.size(), 0.0);
    for (size_t v = 0; v < rates.size(); v++) {
        st.idle_per_second[v] = median(rates[v]);
    }
}

std::filesystem::path log_path(std::string_view name) {
    const std::filesystem::path dir = get_profiler_logs_dir();
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    return dir / name;
}

// The first write of a process starts a fresh file.
std::ofstream open_log(std::string_view name, std::string_view header, bool& started) {
    std::ofstream out(log_path(name), started ? std::ios::app : std::ios::trunc);
    if (!started) {
        out << header << "\n";
    }
    started = true;
    return out;
}

void write_idle_log(ChipId chip, const DeviceState& st) {
    std::ofstream out = open_log(IDLE_LOG, IDLE_HEADER, g_idle_started);
    const size_t per_core = st.plan.items.size();
    for (size_t v = 0; v < st.idle_per_second.size(); v++) {
        if (st.idle_per_second[v] == 0.0) {
            continue;
        }
        const size_t core = v / per_core;
        const Item& item = st.plan.items[v % per_core];
        out << chip << "," << st.cores[core].first << "," << st.cores[core].second << "," << st.channels[core] << ","
            << counter_name(item.counter) << "," << int(item.queue) << "," << st.idle_per_second[v] << "\n";
    }
}

bool prepare(MetalContext& ctx, IDevice& device, DeviceState& st) {
    if (st.disabled) {
        return false;
    }
    if (st.prepared) {
        return true;
    }
    const ChipId chip = device.id();
    const auto arch = arch_of(ctx.get_cluster().arch());
    const auto& soc = ctx.get_cluster().get_soc_desc(chip);
    std::vector<int> channels;
    for (const CoreCoord& logical : device.get_active_ethernet_cores(false)) {
        const auto it = soc.logical_eth_core_to_chan_map.find(logical);
        if (it != soc.logical_eth_core_to_chan_map.end()) {
            channels.push_back(it->second);
        }
    }
    std::sort(channels.begin(), channels.end());
    bool include_l1 = false;
    if (arch) {
        const auto version = ctx.get_cluster().get_ethernet_firmware_version();
        include_l1 = version.has_value() && *version >= min_l1_firmware(*arch);
        if (!include_l1) {
            log_warning(
                tt::LogMetal,
                "TT_METAL_PROFILE_ETH_COUNTERS: the ethernet firmware is older than the first one that keeps the "
                "link counters in L1, only the queue counters are captured");
        }
    }
    const auto plan = arch ? make_plan(*arch, static_cast<uint32_t>(channels.size()), include_l1) : std::nullopt;
    if (!plan || channels.empty()) {
        log_warning(
            tt::LogMetal,
            "TT_METAL_PROFILE_ETH_COUNTERS: no known ethernet counter map or no active ethernet core on device {}, "
            "nothing is captured",
            chip);
        st.disabled = true;
        return false;
    }
    st.plan = *plan;
    st.channels = std::move(channels);
    for (const int channel : st.channels) {
        const CoreCoord core = ctx.get_cluster().get_virtual_eth_core_from_channel(chip, channel);
        st.cores.emplace_back(core.x, core.y);
    }
    st.prepared = true;
    return true;
}

void disable_after_failure(DeviceState& st, const std::exception& e) {
    log_warning(
        tt::LogMetal,
        "TT_METAL_PROFILE_ETH_COUNTERS: reading the ethernet cores failed ({}), capture is off",
        e.what());
    st.disabled = true;
    st.pending.reset();
    st.run_start.reset();
}

uint64_t elapsed_ns(const Reading& a, const Reading& b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b.time - a.time).count();
}

void write_launch(DeviceState& st, ChipId chip, const Pending& pending, const Reading& after, uint32_t extra_flags) {
    uint32_t flags = (pending.overlapped ? FLAG_OVERLAPPED : 0u) | extra_flags;
    const std::vector<Delta> deltas = diff(st.plan, pending.before, after, &flags);
    const uint64_t ns = elapsed_ns(pending.before, after);
    std::ofstream out = open_log(TRAFFIC_LOG, TRAFFIC_HEADER, g_traffic_started);
    const uint64_t seq = st.launch_seq++;
    for (const Delta& d : deltas) {
        const Item& item = st.plan.items[d.item];
        out << chip << "," << pending.program_id << "," << seq << "," << flags << "," << ns << ","
            << st.cores[d.core].first << "," << st.cores[d.core].second << "," << st.channels[d.core] << ","
            << counter_name(item.counter) << "," << int(item.queue) << "," << d.count << "\n";
    }
    if (deltas.empty()) {
        out << chip << "," << pending.program_id << "," << seq << "," << flags << "," << ns << ",-1,-1,-1,NONE,-1,0\n";
    }
}

void write_run(DeviceState& st, ChipId chip, const Reading& before, const Reading& after) {
    uint32_t flags = 0;
    const std::vector<Delta> deltas = diff(st.plan, before, after, &flags);
    const uint64_t ns = elapsed_ns(before, after);
    std::ofstream out = open_log(RUN_LOG, RUN_HEADER, g_run_started);
    const uint64_t run = g_run_id++;
    for (const Delta& d : deltas) {
        const Item& item = st.plan.items[d.item];
        out << chip << "," << run << "," << flags << "," << ns << "," << st.cores[d.core].first << ","
            << st.cores[d.core].second << "," << st.channels[d.core] << "," << counter_name(item.counter) << ","
            << int(item.queue) << "," << d.count << "\n";
    }
    if (deltas.empty()) {
        out << chip << "," << run << "," << flags << "," << ns << ",-1,-1,-1,NONE,-1,0\n";
    }
}

}  // namespace

std::optional<Plan> make_plan(ec::Arch arch, uint32_t num_cores, bool include_l1) {
    const ec::Layout layout = ec::layout_for(arch);
    const ec::QueueTable tx = ec::tx_queue_table(arch);
    const ec::QueueTable rx = ec::rx_queue_table(arch);
    if (layout.txq_base == 0 || layout.rxq_base == 0 || layout.txq_word_bytes == 0 || tx.size == 0 || rx.size == 0) {
        return std::nullopt;
    }
    Plan plan;
    plan.arch = arch;
    plan.num_cores = num_cores;
    for (uint32_t q = 0; q < layout.num_txq; q++) {
        for (size_t i = 0; i < tx.size; i++) {
            plan.items.push_back(
                {tx.data[i].first,
                 Kind::TX_QUEUE,
                 static_cast<int8_t>(q),
                 1,
                 false,
                 ec::tx_queue_address(arch, q, tx.data[i].second)});
        }
    }
    for (uint32_t q = 0; q < layout.num_rxq; q++) {
        for (size_t i = 0; i < rx.size; i++) {
            plan.items.push_back(
                {rx.data[i].first,
                 Kind::RX_QUEUE,
                 static_cast<int8_t>(q),
                 1,
                 false,
                 ec::rx_queue_address(arch, q, rx.data[i].second)});
        }
    }
    const ec::L1Table l1 = ec::l1_table(arch);
    if (include_l1 && l1.size != 0) {
        uint32_t low = UINT32_MAX, high = 0;
        for (size_t i = 0; i < l1.size; i++) {
            const ec::L1Entry& e = l1.data[i];
            plan.items.push_back(
                {e.counter,
                 Kind::L1,
                 e.queue == ec::NO_QUEUE ? NO_QUEUE : static_cast<int8_t>(e.queue),
                 e.words,
                 e.order == ec::WordOrder::HIGH_FIRST,
                 e.address});
            low = std::min(low, e.address);
            high = std::max(high, e.address + 4u * e.words);
        }
        plan.l1_base = low;
        plan.l1_bytes = high - low;
    }
    return plan;
}

bool is_register(const Item& item) { return item.kind != Kind::L1; }

std::vector<Delta> diff(const Plan& plan, const Reading& before, const Reading& after, uint32_t* flags) {
    std::vector<Delta> out;
    const size_t n = std::min(before.values.size(), after.values.size());
    const size_t per_core = plan.items.size();
    bool stale = false;
    for (size_t i = 0; i < per_core; i++) {
        if (plan.items[i].counter == ec::EthCounter::FW_LINK_POLL_ALIVE_COUNT) {
            for (size_t v = i; v < n; v += per_core) {
                stale = stale || uint32_t(after.values[v] - before.values[v]) == 0;
            }
        }
    }
    if (stale) {
        *flags |= FLAG_STALE;
    }
    for (size_t v = 0; v < n; v++) {
        const Item& item = plan.items[v % per_core];
        if (stale && item.kind == Kind::L1) {
            continue;
        }
        uint64_t count = 0;
        if (item.words == 1) {
            count = uint32_t(after.values[v] - before.values[v]);
        } else if (after.values[v] >= before.values[v]) {
            count = after.values[v] - before.values[v];
        } else {
            *flags |= FLAG_RESET;
            continue;
        }
        if (count != 0) {
            out.push_back({static_cast<uint32_t>(v / per_core), v % per_core, count});
        }
    }
    return out;
}

std::string counter_name(ec::EthCounter counter) {
    using C = ec::EthCounter;
    switch (counter) {
        case C::TXQ_TRANSFER_CNT: return "TXQ_TRANSFER_CNT";
        case C::TXQ_PKT_START_CNT: return "TXQ_PKT_START_CNT";
        case C::TXQ_PKT_END_CNT: return "TXQ_PKT_END_CNT";
        case C::TXQ_WORD_CNT: return "TXQ_WORD_CNT";
        case C::TXQ_RESEND_CNT: return "TXQ_RESEND_CNT";
        case C::RXQ_WORD_CNT: return "RXQ_WORD_CNT";
        case C::RXQ_PKT_START_CNT: return "RXQ_PKT_START_CNT";
        case C::RXQ_PKT_END_CNT: return "RXQ_PKT_END_CNT";
        case C::RXQ_PKT_DROP_CNT: return "RXQ_PKT_DROP_CNT";
        case C::RXQ_BYTE_CNT: return "RXQ_BYTE_CNT";
        case C::FW_RETRAIN_COUNT: return "FW_RETRAIN_COUNT";
        case C::FW_LINK_FLAP_COUNT: return "FW_LINK_FLAP_COUNT";
        case C::FW_LINK_POLL_ALIVE_COUNT: return "FW_LINK_POLL_ALIVE_COUNT";
        case C::FW_FRAMES_TXD: return "FW_FRAMES_TXD";
        case C::FW_FRAMES_TXD_OK: return "FW_FRAMES_TXD_OK";
        case C::FW_FRAMES_TXD_BADFCS: return "FW_FRAMES_TXD_BADFCS";
        case C::FW_BYTES_TXD: return "FW_BYTES_TXD";
        case C::FW_BYTES_TXD_OK: return "FW_BYTES_TXD_OK";
        case C::FW_BYTES_TXD_BADFCS: return "FW_BYTES_TXD_BADFCS";
        case C::FW_FRAMES_RXD: return "FW_FRAMES_RXD";
        case C::FW_FRAMES_RXD_OK: return "FW_FRAMES_RXD_OK";
        case C::FW_FRAMES_RXD_BADFCS: return "FW_FRAMES_RXD_BADFCS";
        case C::FW_FRAMES_RXD_DROPPED: return "FW_FRAMES_RXD_DROPPED";
        case C::FW_BYTES_RXD: return "FW_BYTES_RXD";
        case C::FW_BYTES_RXD_OK: return "FW_BYTES_RXD_OK";
        case C::FW_BYTES_RXD_BADFCS: return "FW_BYTES_RXD_BADFCS";
        case C::FW_BYTES_RXD_DROPPED: return "FW_BYTES_RXD_DROPPED";
        case C::FW_CORR_CW: return "FW_CORR_CW";
        case C::FW_UNCORR_CW: return "FW_UNCORR_CW";
        case C::FW_TXQ_RESEND_CNT: return "FW_TXQ_RESEND_CNT";
        case C::FW_RXQ_PKT_DROP: return "FW_RXQ_PKT_DROP";
        case C::FW_CRC_ERR: return "FW_CRC_ERR";
        case C::UNDEF: break;
    }
    return "UNDEF";
}

void begin_run(MetalContext& ctx, IDevice& device) {
    if (!ctx.rtoptions().get_profiler_eth_counters_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceState& st = g_devices[device.id()];
    try {
        if (prepare(ctx, device, st)) {
            st.run_start = read_all(ctx, device.id(), st);
        }
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
}

void end_run(MetalContext& ctx, IDevice& device) {
    if (!ctx.rtoptions().get_profiler_eth_counters_enabled()) {
        return;
    }
    const ChipId chip = device.id();
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_devices.find(chip);
    if (it == g_devices.end()) {
        return;
    }
    DeviceState& st = it->second;
    try {
        if (!st.disabled && st.run_start) {
            write_run(st, chip, *st.run_start, read_all(ctx, chip, st));
        }
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
    g_devices.erase(it);
}

void begin_program(MetalContext& ctx, IDevice& device, const Program& program) {
    if (!ctx.rtoptions().get_profiler_eth_counters_enabled()) {
        return;
    }
    const ChipId chip = device.id();
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceState& st = g_devices[chip];
    try {
        if (!prepare(ctx, device, st)) {
            return;
        }
        if (!st.calibrated && ctx.rtoptions().get_profiler_eth_counters_calibrate()) {
            calibrate_idle(ctx, chip, st);
            write_idle_log(chip, st);
        }
        st.calibrated = true;
        Reading now = read_all(ctx, chip, st);
        const bool overlapped = st.pending.has_value();
        if (overlapped) {
            // The earlier program is still in flight: close it at this instant, its number includes the new program.
            write_launch(st, chip, *st.pending, now, FLAG_OVERLAPPED);
        }
        st.pending = Pending{std::move(now), program.get_runtime_id(), overlapped};
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
}

void end_pending(MetalContext& ctx, ChipId chip) {
    if (!ctx.rtoptions().get_profiler_eth_counters_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_devices.find(chip);
    if (it == g_devices.end() || !it->second.pending) {
        return;
    }
    DeviceState& st = it->second;
    const Pending pending = *st.pending;
    st.pending.reset();
    try {
        write_launch(st, chip, pending, read_all(ctx, chip, st), 0u);
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
}

}  // namespace tt::tt_metal::eth_counters
