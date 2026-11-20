// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_counters.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_paths.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "device.hpp"

namespace tt::tt_metal::dram_counters {

namespace dc = llk::perf::dram;

namespace {

constexpr int kSettleMaxReads = 8;
constexpr int kCalibrationPairs = 3;
constexpr int kCouplingReads = 64;
constexpr auto kCalibrationGap = std::chrono::milliseconds(20);

// One read counter and the two write data counters are enough for a traffic number; fewer registers means fewer
// reads, and every read of a tile is itself counted by it.
constexpr dc::DramCounter kCaptured[] = {
    dc::DramCounter::NIU_SLV_RD_DATA_WORD_SENT,
    dc::DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED,
    dc::DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED,
};

std::optional<dc::Arch> arch_of(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return dc::Arch::BLACKHOLE;
        case tt::ARCH::WORMHOLE_B0: return dc::Arch::WORMHOLE;
        case tt::ARCH::QUASAR: return dc::Arch::QUASAR;
        default: return std::nullopt;
    }
}

struct Pending {
    Reading before;
    uint64_t program_id = 0;
    bool overlapped = false;
};

struct DeviceState {
    bool disabled = false;
    bool initialized = false;
    Plan plan;
    Coupling coupling;
    std::vector<std::pair<uint32_t, uint32_t>> tiles;  // translated x, y
    std::vector<uint32_t> tile_noc0_y;                 // selects the NIU register instance of the tile
    std::vector<uint32_t> tile_reads;  // register reads issued to each tile so far
    std::vector<double> idle_per_pair;
    std::vector<double> idle_per_second;
    std::optional<Pending> pending;
    uint64_t launch_seq = 0;
};

std::mutex g_mutex;
// Contexts can share chip ids, so the state is per context and chip.
std::map<std::pair<int, ChipId>, DeviceState> g_devices;
bool g_traffic_started = false;
bool g_idle_started = false;

uint32_t read_one(MetalContext& ctx, ChipId chip, DeviceState& st, size_t tile, uint64_t addr, uint32_t& index) {
    uint32_t value = 0;
    ctx.get_cluster().read_reg(&value, tt_cxy_pair(chip, st.tiles[tile].first, st.tiles[tile].second), addr);
    index = ++st.tile_reads[tile];
    return value;
}

Reading read_all(MetalContext& ctx, ChipId chip, DeviceState& st) {
    const Plan& plan = st.plan;
    Reading reading;
    reading.values.reserve(plan.reading_size());
    reading.tile_reads.reserve(plan.reading_size());
    reading.time = std::chrono::steady_clock::now();
    for (size_t t = 0; t < st.tiles.size(); t++) {
        for (uint32_t noc = 0; noc < plan.num_nocs; noc++) {
            for (size_t c = 0; c < plan.counters.size(); c++) {
                uint32_t index = 0;
                reading.values.push_back(read_one(
                    ctx, chip, st, t, dc::niu_address(plan.arch, noc, st.tile_noc0_y[t], plan.offsets[c]), index));
                reading.tile_reads.push_back(index);
            }
        }
    }
    return reading;
}

// Only the write counters: what the settle loop watches. It never waits on a read counter.
std::vector<uint32_t> read_writes(MetalContext& ctx, ChipId chip, DeviceState& st) {
    const Plan& plan = st.plan;
    std::vector<uint32_t> values;
    for (size_t t = 0; t < st.tiles.size(); t++) {
        for (uint32_t noc = 0; noc < plan.num_nocs; noc++) {
            for (size_t c = 0; c < plan.counters.size(); c++) {
                if (is_write_counter(plan.counters[c])) {
                    uint32_t index = 0;
                    values.push_back(read_one(
                        ctx, chip, st, t, dc::niu_address(plan.arch, noc, st.tile_noc0_y[t], plan.offsets[c]), index));
                }
            }
        }
    }
    return values;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Words per own read that a back-to-back read of the read counter of the first tile adds, per NoC.
Coupling measure_coupling(MetalContext& ctx, ChipId chip, DeviceState& st) {
    const Plan& plan = st.plan;
    size_t read_slot = 0;
    for (size_t c = 0; c < plan.counters.size(); c++) {
        if (is_read_counter(plan.counters[c])) {
            read_slot = c;
        }
    }
    Coupling coupling(plan.num_nocs, 0);
    for (uint32_t noc = 0; noc < plan.num_nocs; noc++) {
        const uint64_t addr = dc::niu_address(plan.arch, noc, st.tile_noc0_y[0], plan.offsets[read_slot]);
        uint32_t first_index = 0, last_index = 0;
        const uint32_t first = read_one(ctx, chip, st, 0, addr, first_index);
        uint32_t last = first;
        for (int i = 0; i < kCouplingReads; i++) {
            last = read_one(ctx, chip, st, 0, addr, last_index);
        }
        const double per_read = double(uint32_t(last - first)) / double(last_index - first_index);
        coupling[noc] = per_read >= 0.5 ? 1u : 0u;
        if (per_read > 0.05 && std::abs(per_read - 1.0) > 0.05) {
            log_warning(
                tt::LogMetal,
                "TT_METAL_PROFILE_DRAM_COUNTERS: reading a DRAM tile NIU read counter on NoC {} adds {:.2f} words per "
                "read, expected 0 or 1; other traffic probably ran during the measurement",
                noc,
                per_read);
        }
    }
    return coupling;
}

std::vector<double> window_words(const DeviceState& st, const Reading& a, const Reading& b) {
    std::vector<double> out(a.values.size());
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = double(uint32_t(b.values[i] - a.values[i])) - self_words(st.plan, st.coupling, a, b, i);
    }
    return out;
}

// Two terms of the drift left after the own reads are removed: a cost per snapshot pair, and a rate in time that is
// the traffic of other agents.
void calibrate_idle(MetalContext& ctx, ChipId chip, DeviceState& st) {
    std::vector<std::vector<double>> tight, spaced;
    std::vector<double> tight_seconds, spaced_seconds;
    for (int i = 0; i < kCalibrationPairs; i++) {
        const Reading a = read_all(ctx, chip, st);
        const Reading b = read_all(ctx, chip, st);
        tight.push_back(window_words(st, a, b));
        tight_seconds.push_back(std::chrono::duration<double>(b.time - a.time).count());
    }
    for (int i = 0; i < kCalibrationPairs; i++) {
        const Reading a = read_all(ctx, chip, st);
        std::this_thread::sleep_for(kCalibrationGap);
        const Reading b = read_all(ctx, chip, st);
        spaced.push_back(window_words(st, a, b));
        spaced_seconds.push_back(std::chrono::duration<double>(b.time - a.time).count());
    }
    const size_t n = tight[0].size();
    st.idle_per_pair.assign(n, 0.0);
    st.idle_per_second.assign(n, 0.0);
    const double tight_s = median(tight_seconds);
    const double spaced_s = median(spaced_seconds);
    for (size_t i = 0; i < n; i++) {
        std::vector<double> t, s;
        for (int k = 0; k < kCalibrationPairs; k++) {
            t.push_back(tight[k][i]);
            s.push_back(spaced[k][i]);
        }
        const IdleFit fit = fit_idle(median(t), tight_s, median(s), spaced_s);
        st.idle_per_pair[i] = fit.per_pair;
        st.idle_per_second[i] = fit.per_second;
    }
}

std::filesystem::path log_path(std::string_view name) {
    const std::filesystem::path dir = get_profiler_logs_dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
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

struct Coordinates {
    uint32_t tile;
    int noc;
    size_t slot;
};

// Inverse of the Reading layout.
Coordinates locate(const Plan& plan, size_t index) {
    const size_t per_tile = size_t(plan.num_nocs) * plan.counters.size();
    const size_t rest = index % per_tile;
    return {
        static_cast<uint32_t>(index / per_tile),
        static_cast<int>(rest / plan.counters.size()),
        rest % plan.counters.size()};
}

void write_idle_log(ChipId chip, const DeviceState& st) {
    std::ofstream out = open_log(IDLE_LOG, IDLE_HEADER, g_idle_started);
    for (size_t i = 0; i < st.idle_per_pair.size(); i++) {
        if (st.idle_per_pair[i] == 0.0 && st.idle_per_second[i] == 0.0) {
            continue;
        }
        const Coordinates at = locate(st.plan, i);
        out << chip << "," << st.tiles[at.tile].first << "," << st.tiles[at.tile].second << "," << at.noc << ","
            << counter_name(st.plan.counters[at.slot]) << "," << st.idle_per_pair[i] << "," << st.idle_per_second[i]
            << "\n";
    }
}

bool prepare(MetalContext& ctx, ChipId chip, DeviceState& st) {
    if (st.disabled) {
        return false;
    }
    if (st.initialized) {
        return true;
    }
    const auto& rt = ctx.rtoptions();
    const auto arch = arch_of(ctx.get_cluster().arch());
    std::vector<std::pair<uint32_t, uint32_t>> tiles;
    std::vector<uint32_t> noc0_y;
    if (arch) {
        const auto& soc_desc = ctx.get_cluster().get_soc_desc(chip);
        for (const auto& core : soc_desc.get_cores(tt::CoreType::DRAM, tt::CoordSystem::NOC0)) {
            const auto translated = soc_desc.translate_coord_to(core, tt::CoordSystem::TRANSLATED);
            tiles.emplace_back(translated.x, translated.y);
            noc0_y.push_back(core.y);
        }
    }
    const auto plan = arch ? make_plan(*arch, static_cast<uint32_t>(tiles.size())) : std::nullopt;
    if (!plan || tiles.empty()) {
        log_warning(
            tt::LogMetal,
            "TT_METAL_PROFILE_DRAM_COUNTERS: no known DRAM tile counter map for this architecture, nothing is "
            "captured");
        st.disabled = true;
        return false;
    }
    st.plan = *plan;
    st.tiles = std::move(tiles);
    st.tile_noc0_y = std::move(noc0_y);
    st.tile_reads.assign(st.tiles.size(), 0);
    try {
        st.coupling = measure_coupling(ctx, chip, st);
        if (rt.get_profiler_dram_counters_calibrate()) {
            calibrate_idle(ctx, chip, st);
        }
        // Raw captures still restart the file, so the decoder never applies an older calibration to them.
        write_idle_log(chip, st);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "TT_METAL_PROFILE_DRAM_COUNTERS: reading the DRAM tiles failed ({}), capture is off",
            e.what());
        st.disabled = true;
        return false;
    }
    st.initialized = true;
    return true;
}

void write_launch(
    DeviceState& st, ChipId chip, const Pending& pending, const Reading& after, bool settled, uint32_t extra_flags) {
    const uint64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(after.time - pending.before.time).count();
    const uint32_t flags = (pending.overlapped ? FLAG_OVERLAPPED : 0u) | (settled ? 0u : FLAG_UNSETTLED) | extra_flags;
    const std::vector<Delta> deltas = diff(st.plan, st.coupling, pending.before, after);
    std::ofstream out = open_log(TRAFFIC_LOG, TRAFFIC_HEADER, g_traffic_started);
    const uint64_t seq = st.launch_seq++;
    for (const Delta& d : deltas) {
        out << chip << "," << pending.program_id << "," << seq << "," << flags << "," << elapsed_ns << ","
            << st.tiles[d.tile].first << "," << st.tiles[d.tile].second << "," << d.noc << ","
            << counter_name(d.counter) << "," << d.words << "," << d.self_words << "\n";
    }
    if (deltas.empty()) {
        out << chip << "," << pending.program_id << "," << seq << "," << flags << "," << elapsed_ns
            << ",-1,-1,-1,NONE,0,0\n";
    }
}

void disable_after_failure(DeviceState& st, const std::exception& e) {
    log_warning(
        tt::LogMetal, "TT_METAL_PROFILE_DRAM_COUNTERS: reading the DRAM tiles failed ({}), capture is off", e.what());
    st.disabled = true;
    st.pending.reset();
}

}  // namespace

std::optional<Plan> make_plan(dc::Arch arch, uint32_t num_tiles) {
    const dc::Layout layout = dc::layout_for(arch);
    const dc::Table niu = dc::niu_table(arch);
    if (layout.niu_base == 0 || layout.word_bytes == 0 || niu.size == 0) {
        return std::nullopt;
    }
    Plan plan;
    plan.arch = arch;
    plan.num_tiles = num_tiles;
    plan.num_nocs = layout.num_nocs;
    for (const dc::DramCounter wanted : kCaptured) {
        for (size_t i = 0; i < niu.size; i++) {
            if (niu.data[i].first == wanted) {
                plan.counters.push_back(wanted);
                plan.offsets.push_back(niu.data[i].second);
            }
        }
    }
    if (plan.counters.size() != std::size(kCaptured)) {
        return std::nullopt;
    }
    return plan;
}

bool is_read_counter(dc::DramCounter counter) {
    return counter == dc::DramCounter::NIU_SLV_RD_DATA_WORD_SENT || counter == dc::DramCounter::NIU_SLV_RD_REQ_RECEIVED;
}

bool is_write_counter(dc::DramCounter counter) {
    return counter == dc::DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED ||
           counter == dc::DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED;
}

uint32_t self_words(const Plan& plan, const Coupling& coupling, const Reading& before, const Reading& after, size_t i) {
    const Coordinates at = locate(plan, i);
    if (!is_read_counter(plan.counters[at.slot]) || static_cast<size_t>(at.noc) >= coupling.size()) {
        return 0;
    }
    return coupling[at.noc] * (after.tile_reads[i] - before.tile_reads[i]);
}

std::vector<Delta> diff(const Plan& plan, const Coupling& coupling, const Reading& before, const Reading& after) {
    std::vector<Delta> out;
    const size_t n = std::min(before.values.size(), after.values.size());
    for (size_t i = 0; i < n; i++) {
        const uint32_t raw = after.values[i] - before.values[i];
        const uint32_t own = self_words(plan, coupling, before, after, i);
        const uint32_t words = raw > own ? raw - own : 0;
        if (words != 0) {
            const Coordinates at = locate(plan, i);
            out.push_back({at.tile, at.noc, plan.counters[at.slot], words, own});
        }
    }
    return out;
}

IdleFit fit_idle(double tight_words, double tight_seconds, double spaced_words, double spaced_seconds) {
    IdleFit fit;
    if (spaced_seconds > tight_seconds) {
        fit.per_second = std::max(0.0, (spaced_words - tight_words) / (spaced_seconds - tight_seconds));
    }
    fit.per_pair = std::max(0.0, tight_words - fit.per_second * tight_seconds);
    return fit;
}

std::string counter_name(dc::DramCounter counter) {
    switch (counter) {
        case dc::DramCounter::NIU_SLV_RD_REQ_RECEIVED: return "NIU_SLV_RD_REQ_RECEIVED";
        case dc::DramCounter::NIU_SLV_RD_DATA_WORD_SENT: return "NIU_SLV_RD_DATA_WORD_SENT";
        case dc::DramCounter::NIU_SLV_NONPOSTED_WR_REQ_RECEIVED: return "NIU_SLV_NONPOSTED_WR_REQ_RECEIVED";
        case dc::DramCounter::NIU_SLV_POSTED_WR_REQ_RECEIVED: return "NIU_SLV_POSTED_WR_REQ_RECEIVED";
        case dc::DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED: return "NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED";
        case dc::DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED: return "NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED";
        case dc::DramCounter::UNDEF: break;
    }
    return "UNDEF";
}

void begin_program(MetalContext& ctx, IDevice& device, const Program& program) {
    if (!ctx.rtoptions().get_profiler_dram_counters_enabled()) {
        return;
    }
    const ChipId chip = device.id();
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceState& st = g_devices[{ctx.get_context_id().get(), chip}];
    try {
        if (!prepare(ctx, chip, st)) {
            return;
        }
        Reading now = read_all(ctx, chip, st);
        const bool overlapped = st.pending.has_value();
        if (overlapped) {
            // The earlier program is still in flight: close it at this instant, its number includes the new program.
            write_launch(st, chip, *st.pending, now, true, FLAG_OVERLAPPED);
        }
        st.pending = Pending{std::move(now), program.get_runtime_id(), overlapped};
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
}

void end_pending(MetalContext& ctx, ChipId chip) {
    if (!ctx.rtoptions().get_profiler_dram_counters_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_devices.find({ctx.get_context_id().get(), chip});
    if (it == g_devices.end() || !it->second.pending) {
        return;
    }
    DeviceState& st = it->second;
    const Pending pending = *st.pending;
    st.pending.reset();
    try {
        // Posted writes of the kernel can still be in flight when the cores report done: wait until the write
        // counters stop moving. The read counters keep moving with other agents' traffic and are never waited on.
        std::vector<uint32_t> last = read_writes(ctx, chip, st);
        bool settled = false;
        for (int i = 1; i < kSettleMaxReads && !settled; i++) {
            std::vector<uint32_t> next = read_writes(ctx, chip, st);
            settled = next == last;
            last = std::move(next);
        }
        write_launch(st, chip, pending, read_all(ctx, chip, st), settled, 0u);
    } catch (const std::exception& e) {
        disable_after_failure(st, e);
    }
}

void release(MetalContext& ctx) {
    const int context = ctx.get_context_id().get();
    std::lock_guard<std::mutex> lock(g_mutex);
    std::erase_if(g_devices, [context](const auto& entry) { return entry.first.first == context; });
}

}  // namespace tt::tt_metal::dram_counters
