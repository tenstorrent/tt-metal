// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_counters.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-metalium/program.hpp>
#include <internal/service/service_core_manager.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_paths.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "device.hpp"

namespace tt::tt_metal::dram_counters {

namespace dc = llk::perf::dram;
using llk::perf::noc::NocCounter;

namespace {

constexpr int kSettleMaxReads = 8;
constexpr int kCalibrationPairs = 3;
constexpr int kCouplingReads = 64;
constexpr auto kCalibrationGap = std::chrono::milliseconds(20);

// One read counter and the two write data counters are enough for a traffic number; fewer registers means fewer
// reads, and every read of a tile is itself counted by it.
struct Captured {
    NocCounter counter;
    const char* name;
    bool read;
};
constexpr Captured kCaptured[] = {
    {NocCounter::SLV_RD_DATA_WORD_SENT, "SLV_RD_DATA_WORD_SENT", true},
    {NocCounter::SLV_NONPOSTED_WR_DATA_WORD_RECEIVED, "SLV_NONPOSTED_WR_DATA_WORD_RECEIVED", false},
    {NocCounter::SLV_POSTED_WR_DATA_WORD_RECEIVED, "SLV_POSTED_WR_DATA_WORD_RECEIVED", false},
};

const Captured* captured(NocCounter counter) {
    for (const Captured& c : kCaptured) {
        if (c.counter == counter) {
            return &c;
        }
    }
    return nullptr;
}

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
    uint32_t flags = 0;
};

struct Controller {
    uint32_t channel = 0;
    std::pair<uint32_t, uint32_t> tile;  // translated x, y of a tile whose controller registers answer
    uint32_t saved_interval = 0;
    uint32_t saved_axi_control = 0;
    bool open = false;  // its firmware values are saved and not yet written back
};

struct DeviceState {
    std::mutex mutex;
    bool disabled = false;
    bool initialized = false;
    std::vector<Controller> controllers;  // empty unless the controller capture is on and found its registers
    bool controller_open = false;
    std::filesystem::path controller_state;  // where the firmware values wait while the controllers are open
    Plan plan;
    Coupling coupling;
    std::vector<std::pair<uint32_t, uint32_t>> tiles;  // translated x, y
    std::vector<uint32_t> tile_noc0_y;                 // selects the NIU register instance of the tile
    std::vector<uint32_t> tile_reads;                  // register reads issued to each tile so far
    std::vector<double> idle_per_pair;
    std::vector<double> idle_per_second;
    std::optional<Pending> pending;
};

std::mutex g_devices_mutex;
// Contexts can share chip ids, so the state is per context and chip.
std::map<std::pair<int, ChipId>, std::unique_ptr<DeviceState>> g_devices;
// The files and the launch sequence are shared by every device and context of the process.
std::mutex g_log_mutex;
bool g_traffic_started = false;
bool g_idle_started = false;
bool g_controller_started = false;
std::atomic<uint64_t> g_launch_seq{0};

DeviceState* state_of(MetalContext& ctx, ChipId chip, bool create) {
    std::lock_guard<std::mutex> lock(g_devices_mutex);
    const auto key = std::make_pair(ctx.get_context_id().get(), chip);
    auto it = g_devices.find(key);
    if (it == g_devices.end()) {
        if (!create) {
            return nullptr;
        }
        it = g_devices.emplace(key, std::make_unique<DeviceState>()).first;
    }
    return it->second.get();
}

uint32_t read_one(MetalContext& ctx, ChipId chip, DeviceState& st, size_t tile, uint64_t addr, uint32_t& index) {
    uint32_t value = 0;
    ctx.get_cluster().read_reg(&value, tt_cxy_pair(chip, st.tiles[tile].first, st.tiles[tile].second), addr);
    index = ++st.tile_reads[tile];
    return value;
}

// The captured counters that keep(slot) selects, in Reading order; the read index of each goes to tile_reads if given.
template <typename Keep>
std::vector<uint32_t> read_counters(
    MetalContext& ctx, ChipId chip, DeviceState& st, Keep keep, std::vector<uint32_t>* tile_reads = nullptr) {
    const Plan& plan = st.plan;
    std::vector<uint32_t> values;
    values.reserve(plan.reading_size());
    for (size_t t = 0; t < st.tiles.size(); t++) {
        for (uint32_t noc = 0; noc < plan.num_nocs; noc++) {
            for (size_t c = 0; c < plan.counters.size(); c++) {
                if (!keep(c)) {
                    continue;
                }
                uint32_t index = 0;
                values.push_back(read_one(
                    ctx, chip, st, t, dc::niu_address(plan.arch, noc, st.tile_noc0_y[t], plan.regs[c]), index));
                if (tile_reads != nullptr) {
                    tile_reads->push_back(index);
                }
            }
        }
    }
    return values;
}

Reading read_all(MetalContext& ctx, ChipId chip, DeviceState& st) {
    Reading reading;
    reading.time = std::chrono::steady_clock::now();
    reading.values = read_counters(ctx, chip, st, [](size_t) { return true; }, &reading.tile_reads);
    return reading;
}

// What the settle loop watches. It never waits on a read counter.
std::vector<uint32_t> read_writes(MetalContext& ctx, ChipId chip, DeviceState& st) {
    return read_counters(ctx, chip, st, [&](size_t c) { return is_write_counter(st.plan.counters[c]); });
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Words per own read that back-to-back reads of a tile's read counter add, per tile and NoC.
Coupling measure_coupling(MetalContext& ctx, ChipId chip, DeviceState& st) {
    const Plan& plan = st.plan;
    size_t read_slot = 0;
    for (size_t c = 0; c < plan.counters.size(); c++) {
        if (is_read_counter(plan.counters[c])) {
            read_slot = c;
        }
    }
    Coupling coupling(st.tiles.size() * plan.num_nocs, 0);
    for (size_t t = 0; t < st.tiles.size(); t++) {
        for (uint32_t noc = 0; noc < plan.num_nocs; noc++) {
            const uint64_t addr = dc::niu_address(plan.arch, noc, st.tile_noc0_y[t], plan.regs[read_slot]);
            uint32_t first_index = 0, last_index = 0;
            const uint32_t first = read_one(ctx, chip, st, t, addr, first_index);
            uint32_t last = first;
            for (int i = 0; i < kCouplingReads; i++) {
                last = read_one(ctx, chip, st, t, addr, last_index);
            }
            const double per_read = double(uint32_t(last - first)) / double(last_index - first_index);
            coupling[t * plan.num_nocs + noc] = per_read >= 0.5 ? 1u : 0u;
            if (per_read > 0.05 && std::abs(per_read - 1.0) > 0.05) {
                log_warning(
                    tt::LogMetal,
                    "TT_METAL_PROFILE_DRAM_COUNTERS: reading the read counter of DRAM tile {} on NoC {} adds {:.2f} "
                    "words per read, expected 0 or 1; other traffic probably ran during the measurement",
                    t,
                    noc,
                    per_read);
            }
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

// The drift left after the own reads are removed: a cost per snapshot pair and a rate in time.
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

// The first write of a process starts a fresh file. Callers hold g_log_mutex.
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
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::ofstream out = open_log(IDLE_LOG, dc::IDLE_HEADER, g_idle_started);
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

uint32_t controller_read(MetalContext& ctx, ChipId chip, const Controller& c, uint64_t addr) {
    uint32_t value = 0;
    ctx.get_cluster().read_reg(&value, tt_cxy_pair(chip, c.tile.first, c.tile.second), addr);
    return value;
}

void controller_write(MetalContext& ctx, ChipId chip, const Controller& c, uint64_t addr, uint32_t value) {
    ctx.get_cluster().write_reg(&value, tt_cxy_pair(chip, c.tile.first, c.tile.second), addr);
}

// One file per card in the temp directory, so a later process finds what a process that died with the controllers
// open could not put back.
std::filesystem::path controller_state_path(MetalContext& ctx, ChipId chip) {
    const auto& ids = ctx.get_cluster().get_unique_chip_ids();
    const auto it = ids.find(chip);
    if (it == ids.end()) {
        return {};
    }
    return std::filesystem::temp_directory_path() / fmt::format("tt_metal_dram_controller_{:016x}.state", it->second);
}

void save_controller_state(const DeviceState& st) {
    if (st.controller_state.empty()) {
        return;
    }
    std::ofstream out(st.controller_state, std::ios::trunc);
    for (const Controller& c : st.controllers) {
        out << c.tile.first << " " << c.tile.second << " " << c.saved_interval << " " << c.saved_axi_control << "\n";
    }
}

// Puts back the firmware values a crashed process left in the state file. A controller whose interval is no longer
// the capture's was reinitialized since (a card reset) and is left alone.
void recover_controllers(MetalContext& ctx, ChipId chip, dc::Arch arch, const std::filesystem::path& path) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::exists(path, ec)) {
        return;
    }
    const dc::ControllerLayout l = dc::controller_layout_for(arch);
    std::ifstream in(path);
    Controller c;
    uint32_t restored = 0;
    while (in >> c.tile.first >> c.tile.second >> c.saved_interval >> c.saved_axi_control) {
        if (controller_read(ctx, chip, c, l.register_window | l.interval) == CONTROLLER_INTERVAL) {
            controller_write(ctx, chip, c, l.register_window | l.interval, c.saved_interval);
            controller_write(ctx, chip, c, l.register_window | l.axi_control, c.saved_axi_control);
            restored++;
        }
    }
    std::filesystem::remove(path, ec);
    if (restored != 0) {
        log_warning(
            tt::LogMetal,
            "TT_METAL_PROFILE_DRAM_COUNTERS: restored {} memory controllers of device {} that an earlier process left "
            "in the capture state",
            restored,
            chip);
    }
}

// Not every tile of a channel reaches the controller registers, so each channel uses the first tile that does.
std::vector<Controller> find_controllers(MetalContext& ctx, ChipId chip, dc::Arch arch) {
    const dc::ControllerLayout l = dc::controller_layout_for(arch);
    std::vector<Controller> out;
    if (l.register_window == 0) {
        return out;
    }
    const auto& soc_desc = ctx.get_cluster().get_soc_desc(chip);
    const auto channels = soc_desc.get_dram_cores();
    for (uint32_t channel = 0; channel < channels.size(); channel++) {
        std::vector<Controller> ports;
        std::vector<uint32_t> axi_control;
        for (const auto& core : channels[channel]) {
            const auto translated = soc_desc.translate_coord_to(core, tt::CoordSystem::TRANSLATED);
            Controller c{channel, {translated.x, translated.y}};
            axi_control.push_back(controller_read(ctx, chip, c, l.register_window | l.axi_control));
            ports.push_back(c);
        }
        if (const auto port = answering_port(axi_control, l)) {
            out.push_back(ports[*port]);
        }
    }
    return out;
}

void disable_controllers(DeviceState& st, const std::exception& e) {
    log_warning(
        tt::LogMetal,
        "TT_METAL_PROFILE_DRAM_CONTROLLER: accessing the memory controllers failed ({}), controller capture is off",
        e.what());
    st.controllers.clear();
    st.controller_open = false;
}

// Starts one interval per controller with its clock gating off. The firmware values go to the state file before
// anything is written, and a controller is marked open once its values are saved.
void open_controllers(MetalContext& ctx, ChipId chip, DeviceState& st) {
    const dc::ControllerLayout l = dc::controller_layout_for(st.plan.arch);
    for (Controller& c : st.controllers) {
        c.saved_interval = controller_read(ctx, chip, c, l.register_window | l.interval);
        c.saved_axi_control = controller_read(ctx, chip, c, l.register_window | l.axi_control);
    }
    save_controller_state(st);
    st.controller_open = true;
    for (Controller& c : st.controllers) {
        c.open = true;
        controller_write(
            ctx, chip, c, l.register_window | l.axi_control, c.saved_axi_control & ~l.axi_dynamic_gate_bits);
        controller_write(ctx, chip, c, l.register_window | l.interval, 0);
        controller_write(ctx, chip, c, l.register_window | l.interval, CONTROLLER_INTERVAL);
    }
}

void restore_controller(MetalContext& ctx, ChipId chip, const dc::ControllerLayout& l, Controller& c) {
    c.open = false;
    controller_write(ctx, chip, c, l.register_window | l.interval, c.saved_interval);
    controller_write(ctx, chip, c, l.register_window | l.axi_control, c.saved_axi_control);
}

void forget_controller_state(DeviceState& st) {
    std::error_code ec;
    if (!st.controller_state.empty()) {
        std::filesystem::remove(st.controller_state, ec);
    }
}

// Puts the firmware's values back on every open controller without reading the counts, after a failure.
void abandon_controllers(MetalContext& ctx, ChipId chip, DeviceState& st) {
    if (!st.controller_open) {
        return;
    }
    st.controller_open = false;
    const dc::ControllerLayout l = dc::controller_layout_for(st.plan.arch);
    bool all_restored = true;
    for (Controller& c : st.controllers) {
        if (!c.open) {
            continue;
        }
        try {
            restore_controller(ctx, chip, l, c);
        } catch (const std::exception& e) {
            all_restored = false;
            log_warning(
                tt::LogMetal,
                "TT_METAL_PROFILE_DRAM_CONTROLLER: restoring memory controller {} of device {} failed ({})",
                c.channel,
                chip,
                e.what());
        }
    }
    if (all_restored) {
        forget_controller_state(st);
    }
}

// Writing a zero interval latches the counts of the interval in progress. The completion flag is the firmware's and
// is never read; a window as long as the shortest interval is flagged as possibly wrapped.
void close_controllers(
    MetalContext& ctx,
    ChipId chip,
    DeviceState& st,
    uint64_t program_id,
    uint64_t seq,
    uint32_t flags,
    uint64_t elapsed_ns) {
    if (!st.controller_open) {
        return;
    }
    const dc::ControllerLayout l = dc::controller_layout_for(st.plan.arch);
    const dc::ControllerTable table = dc::controller_table(st.plan.arch);
    const uint32_t row_flags = flags | (elapsed_ns >= CONTROLLER_INTERVAL_MIN_NS ? FLAG_INTERVAL_WRAPPED : 0u);
    std::vector<std::pair<const Controller*, std::vector<uint32_t>>> counts;
    for (Controller& c : st.controllers) {
        if (!c.open) {
            continue;
        }
        controller_write(ctx, chip, c, l.register_window | l.interval, 0);
        std::vector<uint32_t> values(table.size);
        for (size_t i = 0; i < table.size; i++) {
            values[i] = controller_read(ctx, chip, c, l.register_window | table.data[i].address);
        }
        restore_controller(ctx, chip, l, c);
        counts.emplace_back(&c, std::move(values));
    }
    st.controller_open = false;
    forget_controller_state(st);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::ofstream out = open_log(CONTROLLER_LOG, dc::CONTROLLER_HEADER, g_controller_started);
    for (const auto& [c, values] : counts) {
        for (size_t i = 0; i < table.size; i++) {
            if (values[i] != 0) {
                out << chip << "," << program_id << "," << seq << "," << row_flags << "," << elapsed_ns << ","
                    << c->channel << "," << c->tile.first << "," << c->tile.second << "," << table.data[i].name << ","
                    << values[i] << "\n";
            }
        }
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
    if (dc::controller_layout_for(st.plan.arch).register_window != 0) {
        st.controller_state = controller_state_path(ctx, chip);
        try {
            recover_controllers(ctx, chip, st.plan.arch, st.controller_state);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "TT_METAL_PROFILE_DRAM_COUNTERS: restoring the memory controllers of device {} failed ({})",
                chip,
                e.what());
        }
    }
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
    if (rt.get_profiler_dram_controller_enabled()) {
        try {
            st.controllers = find_controllers(ctx, chip, st.plan.arch);
        } catch (const std::exception& e) {
            disable_controllers(st, e);
        }
        if (st.controllers.empty()) {
            log_warning(
                tt::LogMetal,
                "TT_METAL_PROFILE_DRAM_CONTROLLER: no memory controller registers answer on device {}, only the DRAM "
                "tile counters are captured",
                chip);
        }
    }
    st.initialized = true;
    return true;
}

bool service_active(MetalContext& ctx, ChipId chip) {
    return !ctx.get_service_core_manager().claimed_cores(chip).empty();
}

// Writes the traffic rows of one launch and closes its controller interval under the same sequence number.
void close_launch(
    MetalContext& ctx, ChipId chip, DeviceState& st, const Pending& pending, const Reading& after, uint32_t flags) {
    const uint64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(after.time - pending.before.time).count();
    const std::vector<Delta> deltas = diff(st.plan, st.coupling, pending.before, after);
    const uint64_t seq = g_launch_seq++;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::ofstream out = open_log(TRAFFIC_LOG, dc::TRAFFIC_HEADER, g_traffic_started);
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
    try {
        close_controllers(ctx, chip, st, pending.program_id, seq, flags, elapsed_ns);
    } catch (const std::exception& e) {
        abandon_controllers(ctx, chip, st);
        disable_controllers(st, e);
    }
}

void disable_after_failure(DeviceState& st, const std::exception& e) {
    log_warning(
        tt::LogMetal, "TT_METAL_PROFILE_DRAM_COUNTERS: reading the DRAM tiles failed ({}), capture is off", e.what());
    st.disabled = true;
    st.pending.reset();
}

}  // namespace

uint32_t launch_flags(bool overlapped, bool settled, uint32_t extra_flags) {
    return (overlapped ? FLAG_OVERLAPPED : 0u) | (settled ? 0u : FLAG_UNSETTLED) | extra_flags;
}

std::optional<Plan> make_plan(dc::Arch arch, uint32_t num_tiles) {
    const dc::Layout layout = dc::layout_for(arch);
    if (layout.niu_base == 0 || layout.word_bytes == 0) {
        return std::nullopt;
    }
    Plan plan;
    plan.arch = arch;
    plan.num_tiles = num_tiles;
    plan.num_nocs = layout.num_nocs;
    for (const Captured& wanted : kCaptured) {
        for (const auto& entry : llk::perf::noc::noc_counters) {
            if (entry.counter == wanted.counter) {
                plan.counters.push_back(wanted.counter);
                plan.regs.push_back(entry.reg);
            }
        }
    }
    if (plan.counters.size() != std::size(kCaptured)) {
        return std::nullopt;
    }
    return plan;
}

bool is_read_counter(NocCounter counter) {
    const Captured* c = captured(counter);
    return c != nullptr && c->read;
}

bool is_write_counter(NocCounter counter) {
    const Captured* c = captured(counter);
    return c != nullptr && !c->read;
}

std::string counter_name(NocCounter counter) {
    const Captured* c = captured(counter);
    return c != nullptr ? c->name : "UNDEF";
}

uint32_t self_words(const Plan& plan, const Coupling& coupling, const Reading& before, const Reading& after, size_t i) {
    const Coordinates at = locate(plan, i);
    const size_t link = size_t(at.tile) * plan.num_nocs + at.noc;
    if (!is_read_counter(plan.counters[at.slot]) || link >= coupling.size()) {
        return 0;
    }
    return coupling[link] * (after.tile_reads[i] - before.tile_reads[i]);
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

std::optional<size_t> answering_port(const std::vector<uint32_t>& axi_control, const dc::ControllerLayout& l) {
    for (size_t i = 0; i < axi_control.size(); i++) {
        if (l.axi_reset_bits != 0 && (axi_control[i] & l.axi_reset_bits) == l.axi_reset_bits) {
            return i;
        }
    }
    return std::nullopt;
}

IdleFit fit_idle(double tight_words, double tight_seconds, double spaced_words, double spaced_seconds) {
    IdleFit fit;
    if (spaced_seconds > tight_seconds) {
        fit.per_second = std::max(0.0, (spaced_words - tight_words) / (spaced_seconds - tight_seconds));
    }
    fit.per_pair = std::max(0.0, tight_words - fit.per_second * tight_seconds);
    return fit;
}

void begin_program(MetalContext& ctx, IDevice& device, const Program& program) {
    if (!ctx.rtoptions().get_profiler_dram_counters_enabled()) {
        return;
    }
    const ChipId chip = device.id();
    DeviceState& st = *state_of(ctx, chip, true);
    std::lock_guard<std::mutex> lock(st.mutex);
    try {
        if (!prepare(ctx, chip, st)) {
            return;
        }
        const bool overlapped = st.pending.has_value();
        if (overlapped) {
            // The earlier program is still in flight: close it at this instant, its number includes the new program.
            const Pending earlier = *st.pending;
            st.pending.reset();
            close_launch(ctx, chip, st, earlier, read_all(ctx, chip, st), earlier.flags | FLAG_OVERLAPPED);
        }
        // The controller register writes go before the tile snapshot so the tile counters never see them.
        if (!st.controllers.empty()) {
            try {
                open_controllers(ctx, chip, st);
            } catch (const std::exception& e) {
                abandon_controllers(ctx, chip, st);
                disable_controllers(st, e);
            }
        }
        const uint32_t flags = launch_flags(overlapped, true, service_active(ctx, chip) ? FLAG_SERVICE_ACTIVE : 0u);
        st.pending = Pending{read_all(ctx, chip, st), program.get_runtime_id(), flags};
    } catch (const std::exception& e) {
        abandon_controllers(ctx, chip, st);
        disable_after_failure(st, e);
    }
}

void end_pending(MetalContext& ctx, ChipId chip, std::optional<uint64_t> program_id) {
    if (!ctx.rtoptions().get_profiler_dram_counters_enabled()) {
        return;
    }
    DeviceState* found = state_of(ctx, chip, false);
    if (found == nullptr) {
        return;
    }
    DeviceState& st = *found;
    std::lock_guard<std::mutex> lock(st.mutex);
    if (!st.pending || (program_id && st.pending->program_id != *program_id)) {
        return;
    }
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
        const uint32_t flags =
            pending.flags | launch_flags(false, settled, service_active(ctx, chip) ? FLAG_SERVICE_ACTIVE : 0u);
        close_launch(ctx, chip, st, pending, read_all(ctx, chip, st), flags);
    } catch (const std::exception& e) {
        abandon_controllers(ctx, chip, st);
        disable_after_failure(st, e);
    }
}

void release(MetalContext& ctx) {
    const int context = ctx.get_context_id().get();
    std::lock_guard<std::mutex> devices_lock(g_devices_mutex);
    for (auto& [key, state] : g_devices) {
        if (key.first != context || !state->pending) {
            continue;
        }
        DeviceState& st = *state;
        std::lock_guard<std::mutex> lock(st.mutex);
        const Pending pending = *st.pending;
        st.pending.reset();
        // A program nobody waited for: close it now, so its rows and the controllers end like any other launch.
        try {
            close_launch(
                ctx, key.second, st, pending, read_all(ctx, key.second, st), pending.flags | FLAG_NOT_FINISHED);
        } catch (const std::exception&) {
            abandon_controllers(ctx, key.second, st);
        }
    }
    std::erase_if(g_devices, [context](const auto& entry) { return entry.first.first == context; });
}

}  // namespace tt::tt_metal::dram_counters
