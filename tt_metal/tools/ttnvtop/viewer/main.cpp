// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop: TUI viewer. Reads one /dev/shm/tt_device_<asic>_util file per
// chip published by ttnvtop-collector and renders a live per-core
// utilization display. Zero UMD dependency — safe to run as many instances
// as desired on the same box.
//
// See tt_metal/tools/ttnvtop/PLAN.md §4 and §7 for the architecture.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <deque>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../collector/shm_publisher.hpp"
#include "../common/program_registry.hpp"
#include "../common/shm_schema.hpp"

namespace {

// 10 Hz: each frame reflects ~100 ms of activity. The collector publishes
// at 100 Hz (every 10 ms) and the LLK Hook B period is 5 ms, so the data
// underneath is fresh every frame. Higher render rates (>30 Hz) start to
// hit terminal redraw artifacts on most systems; lower rates (4 Hz) drop
// most kernels — Llama decode kernels run at sub-ms cadence and 4 Hz blurs
// 200+ kernels per frame into a single snapshot.
constexpr int kRenderHz = 10;
constexpr int kStaleThresholdMs = 2000;
constexpr int kBarWidth = 4;    // bar is N pipes wide inside the box
constexpr int kBoxInnerW = 10;  // 1 letter + bar + 1 sp + 4 pct = 10 visible chars
constexpr int kBoxOuterW = 12;  // +2 for | | borders

// ANSI color escapes. We use color ONLY to convey meaning:
//   - saturation (green/yellow/red) for F/S/D percentage
//   - per-program color on the #ID label so you can visually track a program
constexpr const char* kAnsiReset = "\x1b[0m";
constexpr const char* kAnsiDim = "\x1b[2m";
constexpr const char* kAnsiBold = "\x1b[1m";
constexpr const char* kPctGray = "\x1b[38;5;240m";    // idle (0%)
constexpr const char* kPctGreen = "\x1b[38;5;46m";    // 1..33%
constexpr const char* kPctYellow = "\x1b[38;5;220m";  // 34..66%
constexpr const char* kPctRed = "\x1b[38;5;196m";     // 67..100%

const char* pct_color(uint32_t pct) {
    if (pct == 0) {
        return kPctGray;
    }
    if (pct < 34) {
        return kPctGreen;
    }
    if (pct < 67) {
        return kPctYellow;
    }
    return kPctRed;
}

// Terminal width, for laying meters out in as many columns as fit.
int term_cols() {
    struct winsize ws{};
    if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return ws.ws_col;
    }
    return 120;
}

// Raw-ish stdin so single keypresses (f/s/d/g/q) switch views without Enter.
// Restored by the destructor so a Ctrl-C or normal exit leaves the terminal sane.
class RawMode {
public:
    RawMode() {
        if (!::isatty(STDIN_FILENO)) {
            return;
        }
        if (::tcgetattr(STDIN_FILENO, &saved_) != 0) {
            return;
        }
        active_ = true;
        struct termios raw = saved_;
        raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON) | static_cast<tcflag_t>(ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
    ~RawMode() { restore(); }
    void restore() {
        if (active_) {
            ::tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
            active_ = false;
        }
    }
    RawMode(const RawMode&) = delete;
    RawMode& operator=(const RawMode&) = delete;

private:
    struct termios saved_{};
    bool active_ = false;
};

// Drain pending keypresses; returns the last one seen (0 if none).
char poll_key() {
    char last = 0;
    char c = 0;
    while (::read(STDIN_FILENO, &c, 1) == 1) {
        last = c;
    }
    return last;
}

// Which per-core metric the htop-style meters display. F is the default
// because FPU compute% is the closest analogue of htop's CPU%.
enum class Metric { Fpu, Sfpu, Dispatch, All };

const char* metric_name(Metric m) {
    switch (m) {
        case Metric::Fpu: return "FPU";
        case Metric::Sfpu: return "SFPU";
        case Metric::Dispatch: return "DISPATCH";
        case Metric::All: return "ALL (F/S/D)";
    }
    return "?";
}

// Pull the selected metric out of a PerCoreView, in per-mille.
uint16_t metric_p1000(const ttnvtop::PerCoreView& v, Metric m) {
    switch (m) {
        case Metric::Fpu: return v.compute_busy_p1000;
        case Metric::Sfpu: return v.sfpu_busy_p1000;
        case Metric::Dispatch: return v.dispatch_busy_p1000;
        case Metric::All: return v.compute_busy_p1000;  // unused in All mode
    }
    return 0;
}

// htop-style bar meter:  NNN[|||||  xx.x%]
// The inner field is kMeterInnerW wide. The percentage text is right-aligned
// within it and the bar fills from the left; where they overlap the text wins,
// which is exactly what htop does (Meter.c BarMeterMode_draw).
constexpr int kMeterInnerW = 9;
constexpr int kMeterOuterW = 15;  // "NNN[" + inner + "]" + gap

std::string render_meter(uint32_t idx, uint16_t p1000) {
    const double pct = static_cast<double>(p1000) / 10.0;
    char pctbuf[16];
    std::snprintf(pctbuf, sizeof(pctbuf), "%.1f%%", pct);
    std::string text(pctbuf);
    if (static_cast<int>(text.size()) > kMeterInnerW) {
        text = text.substr(0, kMeterInnerW);
    }
    // Bar length proportional to the full inner width, min 1 pipe for any
    // non-zero reading so low-but-live cores stay visible (htop does this too).
    int bars = static_cast<int>((pct / 100.0) * kMeterInnerW + 0.5);
    if (p1000 > 0 && bars < 1) {
        bars = 1;
    }
    if (bars > kMeterInnerW) {
        bars = kMeterInnerW;
    }
    std::string inner(kMeterInnerW, ' ');
    for (int i = 0; i < bars; ++i) {
        inner[i] = '|';
    }
    // Overlay the right-aligned percentage.
    const int off = kMeterInnerW - static_cast<int>(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        inner[off + i] = text[i];
    }
    const uint32_t pct_i = static_cast<uint32_t>(pct + 0.5);
    std::ostringstream o;
    o << kAnsiDim << std::setw(3) << idx << kAnsiReset << "[" << pct_color(pct_i) << inner << kAnsiReset << "]";
    return o.str();
}

// Per-metric hues for the combined view. Deliberately NOT the saturation ramp
// (green/yellow/red) -- here colour identifies WHICH unit, and bar length
// carries the magnitude.
constexpr const char* kMetricFpuCol = "\x1b[38;5;82m";    // green   F
constexpr const char* kMetricSfpuCol = "\x1b[38;5;51m";   // cyan    S
constexpr const char* kMetricDispCol = "\x1b[38;5;213m";  // magenta D
constexpr const char* kMetricDramCol = "\x1b[38;5;214m";  // orange  DRAM

// DRAM bandwidth for a chip header row. Measured at the DRAM NOC endpoints,
// validated against exact byte counts to 0.07% (BH) / 0.80% (WH).
//
// Rendered only when the collector reports non-zero traffic: a chip genuinely
// idle and a chip whose DRAM registers were mis-addressed both publish 0, and
// showing "0.0 GB/s" would present the second as if it were the first.
std::string render_dram(const ttnvtop::UtilShmHeader* h) {
    if (h == nullptr || (h->dram_rd_mbps == 0 && h->dram_wr_mbps == 0)) {
        return "";
    }
    const double rd = static_cast<double>(h->dram_rd_mbps) / 1000.0;
    const double wr = static_cast<double>(h->dram_wr_mbps) / 1000.0;
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << "  " << kMetricDramCol << "DRAM " << (rd + wr) << " GB/s"
      << " (R " << rd << " W " << wr << ")";
    if (h->dram_peak_mbps > 0) {
        const double pct = 100.0 * (rd + wr) * 1000.0 / static_cast<double>(h->dram_peak_mbps);
        o << " " << std::setprecision(0) << pct << "% of " << (h->dram_peak_mbps / 1000) << "G";
    }
    o << kAnsiReset;
    return o.str();
}

// Combined meter: NNN[FFFSSSDDD] -- three 3-wide segments, one per unit.
// Same outer width as render_meter() so the column layout is unchanged; the
// numeric readout is traded for seeing all three units at once. F/S/D are
// independent occupancies (each 0-100% of wall time, and they overlap), so
// they are shown side by side rather than stacked -- stacking would imply they
// sum to 100%, which they do not: a mixed workload measured F 7.8 / S 27.4 /
// D 69.0 on the same core.
constexpr int kSegW = 3;

std::string render_meter_all(uint32_t idx, uint16_t f_p1000, uint16_t s_p1000, uint16_t d_p1000) {
    const auto seg = [](uint16_t p1000, const char* col) {
        int bars = static_cast<int>((static_cast<double>(p1000) / 1000.0) * kSegW + 0.5);
        if (p1000 > 0 && bars < 1) {
            bars = 1;
        }
        if (bars > kSegW) {
            bars = kSegW;
        }
        std::string b(static_cast<size_t>(bars), '|');
        b.append(static_cast<size_t>(kSegW - bars), ' ');
        return std::string(col) + b + kAnsiReset;
    };
    std::ostringstream o;
    o << kAnsiDim << std::setw(3) << idx << kAnsiReset << "[" << seg(f_p1000, kMetricFpuCol)
      << seg(s_p1000, kMetricSfpuCol) << seg(d_p1000, kMetricDispCol) << "]";
    return o.str();
}

// Stable per-program color. Intentionally disjoint from the saturation ramp
// (no pure green/yellow/red here) so you can tell "bar heat" from "program
// identity" at a glance.
const char* prog_color(uint32_t prog_id) {
    static constexpr const char* kPalette[] = {
        "\x1b[38;5;51m",   // bright cyan
        "\x1b[38;5;201m",  // magenta
        "\x1b[38;5;33m",   // blue
        "\x1b[38;5;208m",  // orange
        "\x1b[38;5;129m",  // purple
        "\x1b[38;5;37m",   // teal
        "\x1b[38;5;213m",  // pink
        "\x1b[38;5;99m",   // violet
    };
    return kPalette[prog_id % (sizeof(kPalette) / sizeof(kPalette[0]))];
}

// Render a bar of `filled` pipes + `width-filled` spaces. No brackets
// (brackets live on the box border). Caller wraps with color codes.
void append_bar(std::string& s, uint32_t pct, int width) {
    if (pct > 100) {
        pct = 100;
    }
    int filled = static_cast<int>((pct * static_cast<uint32_t>(width)) / 100u);
    for (int i = 0; i < width; ++i) {
        s.push_back(i < filled ? '|' : ' ');
    }
}

struct MappedShm {
    int fd = -1;
    void* map = nullptr;
    size_t map_size = 0;
    const ttnvtop::UtilShmHeader* header = nullptr;
    const ttnvtop::PerCoreView* cores = nullptr;
    std::string path;
    // WHICH INODE this mapping is of, not just which path.
    //
    // A collector restart replaces the file at the same path with a NEW inode: the old one
    // is unlinked, and an unlinked inode that someone still has mapped stays alive and
    // frozen forever. Matching on path alone kept that dead mapping and rendered its last
    // frame indefinitely while /dev/shm was perfectly fresh -- the whole "the TUI is hung,
    // values look stale, restarting the viewer fixes it" symptom.
    dev_t st_dev = 0;
    ino_t st_ino = 0;
};

// Program name registry mapping. Read-only view of the writer-side circular
// buffer at /dev/shm/tt_program_registry. If the file is absent or malformed,
// we silently fall back to "unnamed" display.
struct RegistryMap {
    int fd = -1;
    void* map = nullptr;
    size_t map_size = 0;
    const ttnvtop::RegistryHeader* header = nullptr;
    const ttnvtop::RegistryEntry* entries = nullptr;
    uint32_t last_cursor = 0;  // total writes observed so far (capped-summed)
    uint32_t last_writer_pid = 0;
};

bool map_registry(RegistryMap& out) {
    out.fd = ::open(ttnvtop::kRegistryShmPath, O_RDONLY);
    if (out.fd < 0) {
        return false;
    }
    struct stat st{};
    if (::fstat(out.fd, &st) != 0 || st.st_size < static_cast<off_t>(ttnvtop::registry_file_size())) {
        ::close(out.fd);
        out.fd = -1;
        return false;
    }
    out.map_size = ttnvtop::registry_file_size();
    out.map = ::mmap(nullptr, out.map_size, PROT_READ, MAP_SHARED, out.fd, 0);
    if (out.map == MAP_FAILED) {
        out.map = nullptr;
        ::close(out.fd);
        out.fd = -1;
        return false;
    }
    out.header = static_cast<const ttnvtop::RegistryHeader*>(out.map);
    if (std::memcmp(out.header->magic, ttnvtop::kRegistryMagic, 4) != 0 ||
        out.header->version != ttnvtop::kRegistryVersion || out.header->entry_size != sizeof(ttnvtop::RegistryEntry) ||
        out.header->capacity != ttnvtop::kRegistryCapacity) {
        ::munmap(out.map, out.map_size);
        out.map = nullptr;
        ::close(out.fd);
        out.fd = -1;
        out.header = nullptr;
        return false;
    }
    out.entries = reinterpret_cast<const ttnvtop::RegistryEntry*>(
        static_cast<const char*>(out.map) + sizeof(ttnvtop::RegistryHeader));
    out.last_cursor = 0;
    out.last_writer_pid = out.header->writer_pid;
    return true;
}

void unmap_registry(RegistryMap& r) {
    if (r.map != nullptr) {
        ::munmap(r.map, r.map_size);
    }
    if (r.fd >= 0) {
        ::close(r.fd);
    }
    r = RegistryMap{};
}

std::atomic<bool> g_stop{false};

void handle_sigint(int) { g_stop.store(true); }

const char* arch_label(uint32_t arch_id) {
    switch (arch_id) {
        case 1: return "Grayskull";
        case 2: return "Wormhole";
        case 3: return "Blackhole";
        case 4: return "Quasar";
        default: return "?";
    }
}

bool map_shm(const std::string& path, MappedShm& out) {
    out.fd = ::open(path.c_str(), O_RDONLY);
    if (out.fd < 0) {
        return false;
    }
    struct stat st{};
    if (::fstat(out.fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(ttnvtop::UtilShmHeader))) {
        ::close(out.fd);
        out.fd = -1;
        return false;
    }
    out.map_size = static_cast<size_t>(st.st_size);
    out.st_dev = st.st_dev;
    out.st_ino = st.st_ino;
    out.map = ::mmap(nullptr, out.map_size, PROT_READ, MAP_SHARED, out.fd, 0);
    if (out.map == MAP_FAILED) {
        out.map = nullptr;
        ::close(out.fd);
        out.fd = -1;
        return false;
    }
    out.header = static_cast<const ttnvtop::UtilShmHeader*>(out.map);
    if (std::memcmp(out.header->magic, ttnvtop::kShmMagic, 4) != 0 || out.header->version != ttnvtop::kShmVersion ||
        out.header->struct_size != sizeof(ttnvtop::PerCoreView)) {
        ::munmap(out.map, out.map_size);
        out.map = nullptr;
        ::close(out.fd);
        out.fd = -1;
        return false;
    }
    out.cores = reinterpret_cast<const ttnvtop::PerCoreView*>(
        static_cast<const char*>(out.map) + sizeof(ttnvtop::UtilShmHeader));
    out.path = path;
    return true;
}

void unmap_shm(MappedShm& m) {
    if (m.map != nullptr) {
        ::munmap(m.map, m.map_size);
    }
    if (m.fd >= 0) {
        ::close(m.fd);
    }
    m = MappedShm{};
}

uint64_t monotonic_us() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

}  // namespace

int main(int argc, char* argv[]) {
    std::set<uint64_t> chip_filter;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "-h" || a == "--help") {
            std::cout << "ttnvtop — TUI viewer of ttnvtop-collector SHM.\n\n"
                         "Usage: ttnvtop [--chip N]...\n\n"
                         "  --chip N         Only show chip with asic_id N. Repeatable; default shows all.\n\n"
                         "Each Tensix core is drawn as a bordered box containing F/S/D bars\n"
                         "(matrix/vector/dispatch), a percentage per bar, the (x,y) noc coord,\n"
                         "and the last 3 digits of the host_assigned_id of the program running.\n";
            return 0;
        }
        if (a == "--chip") {
            if (i + 1 >= argc) {
                std::cerr << "ttnvtop: --chip requires an integer argument\n";
                return 2;
            }
            char* endp = nullptr;
            const char* nstr = argv[++i];
            unsigned long long v = std::strtoull(nstr, &endp, 10);
            if (endp == nstr || *endp != '\0') {
                std::cerr << "ttnvtop: --chip got non-numeric value '" << nstr << "'\n";
                return 2;
            }
            chip_filter.insert(static_cast<uint64_t>(v));
            continue;
        }
    }
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::vector<MappedShm> maps;

    // Program-name registry. Lazily (re)opened each frame if not yet mapped
    // or if the writer process restarted. Missing file is fine — names just
    // display as "unnamed".
    RegistryMap registry;
    std::unordered_map<uint32_t, std::string> name_cache;
    // Phase 2.1.c: per-program cycles attributed in the last 1s drain window.
    // Keyed identically to name_cache (raw runtime_id, possibly encoded form).
    // Updated each frame by replaying every registry slot — the writer rewrites
    // the same slot with refreshed cycles_in_window each drain.
    std::unordered_map<uint32_t, uint64_t> cycles_cache;

    // Stage 1 (history pane): accumulate every program seen since the viewer
    // started. The PROGRAMS table only shows what's *currently* dispatched on
    // cores; this captures programs that ran briefly between viewer frames
    // and would otherwise scroll past invisible. Each registrar slot
    // ingested is one dispatch event; we count those plus track first/last
    // observation timestamps and peak cycles_total seen.
    struct HistoryEntry {
        std::string name;
        uint64_t first_seen_ms;
        uint64_t last_seen_ms;
        uint32_t dispatch_count;     // number of registry ingestions (= dispatches mod wrap)
        uint64_t peak_cycles_total;  // monotonic upper bound of registry's cycles_total field
    };
    std::unordered_map<uint32_t, HistoryEntry> history_cache;
    const auto session_start = std::chrono::steady_clock::now();
    auto session_ms_now = [&]() -> uint64_t {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - session_start)
                .count());
    };

    auto ingest_registry_entry = [&](const ttnvtop::RegistryEntry& e) {
        // The writer side (dispatch hooks) registers with the value it just
        // wrote into host_assigned_id. tt-metal has two encodings:
        //   - dispatch.cpp (single-device path):  raw runtime_id (e.g. 29539)
        //   - mesh / tt_metal.cpp:                EncodePerDeviceProgramID
        //                                         = (runtime_id << 10 | dev_id)
        // We store under e.runtime_id verbatim. Lookup tries multiple forms.
        const size_t max_len = ttnvtop::kRegistryNameMax;
        size_t n = 0;
        while (n < max_len && e.name[n] != '\0') {
            ++n;
        }
        name_cache[e.runtime_id] = std::string(e.name, n);
        cycles_cache[e.runtime_id] = e.cycles_in_window;

        // Stage 1: history accumulator. Each registrar-side fetch_add of
        // write_cursor produces one fresh entry per `register_program` call,
        // so seeing a slot under a runtime_id == one dispatch occurrence
        // (modulo the 16k circular-buffer wrap).
        const uint64_t now_ms = session_ms_now();
        auto& h = history_cache[e.runtime_id];
        if (h.dispatch_count == 0) {
            h.name = std::string(e.name, n);
            h.first_seen_ms = now_ms;
        }
        h.last_seen_ms = now_ms;
        ++h.dispatch_count;
        if (e.cycles_total > h.peak_cycles_total) {
            h.peak_cycles_total = e.cycles_total;
        }
    };

    auto refresh_registry = [&]() {
        if (registry.fd < 0) {
            if (!map_registry(registry)) {
                return;  // silently skip — collector/writer may not be running
            }
        }
        // Detect writer restart: if pid changed, start over.
        const uint32_t cur_pid = registry.header->writer_pid;
        if (cur_pid != registry.last_writer_pid) {
            name_cache.clear();
            history_cache.clear();
            registry.last_cursor = 0;
            registry.last_writer_pid = cur_pid;
        }
        const uint32_t cursor = registry.header->write_cursor.load(std::memory_order_acquire);
        if (cursor == registry.last_cursor) {
            return;
        }
        const uint32_t capacity = registry.header->capacity;
        uint32_t new_writes = cursor - registry.last_cursor;
        // If writer has produced more than `capacity` entries since last frame,
        // we've lapped the buffer and the slots before the current window are
        // overwritten — just rescan everything.
        if (new_writes >= capacity) {
            const uint32_t scan = cursor < capacity ? cursor : capacity;
            for (uint32_t i = 0; i < scan; ++i) {
                ingest_registry_entry(registry.entries[i]);
            }
        } else {
            for (uint32_t k = 0; k < new_writes; ++k) {
                const uint32_t total_written = registry.last_cursor + k;
                const uint32_t slot = total_written % capacity;
                ingest_registry_entry(registry.entries[slot]);
            }
        }
        registry.last_cursor = cursor;
    };

    // Refresh the set of SHM files: add any new ones, drop any that vanished or
    // whose collector PID is no longer alive.
    auto refresh_maps = [&]() {
        auto entries = ttnvtop::list_shm_files();
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.asic_id < b.asic_id; });

        // Apply --chip filter (matches the asic_id encoded in the filename,
        // which also equals UtilShmHeader::asic_id).
        if (!chip_filter.empty()) {
            entries.erase(
                std::remove_if(
                    entries.begin(), entries.end(), [&](const auto& e) { return chip_filter.count(e.asic_id) == 0; }),
                entries.end());
        }

        // Drop maps whose path no longer exists.
        maps.erase(
            std::remove_if(
                maps.begin(),
                maps.end(),
                [&](MappedShm& m) {
                    bool still_there = false;
                    for (const auto& e : entries) {
                        if (e.path != m.path) {
                            continue;
                        }
                        // Same path is NOT the same file. Compare the inode: if the
                        // collector was restarted, the name now points somewhere else and
                        // what we hold is a frozen orphan. Treat it as gone so the loop
                        // below re-maps the live one.
                        struct stat st{};
                        still_there =
                            ::stat(e.path.c_str(), &st) == 0 && st.st_dev == m.st_dev && st.st_ino == m.st_ino;
                        break;
                    }
                    if (!still_there) {
                        unmap_shm(m);
                    }
                    return !still_there;
                }),
            maps.end());

        // Add entries we don't have yet.
        for (const auto& e : entries) {
            bool already = false;
            for (const auto& m : maps) {
                if (m.path == e.path) {
                    already = true;
                    break;
                }
            }
            if (already) {
                continue;
            }
            MappedShm m;
            if (map_shm(e.path, m)) {
                maps.push_back(std::move(m));
            }
        }
        std::sort(maps.begin(), maps.end(), [](const MappedShm& a, const MappedShm& b) {
            return a.header->asic_id < b.header->asic_id;
        });
    };

    // Per-chip Gantt timeline: ring buffer of the last N frames' program sets.
    // Each frame appends the set of distinct programs running anywhere on the
    // chip (decoded raw runtime_id, ignoring kid==0). The TIMELINE pane below
    // the live grid renders one row per program × N columns of presence
    // markers so you can see when each program ran and for how many frames.
    // 80 frames × 100 ms = 8 s of visible history.
    constexpr size_t kTimelineWidth = 80;
    std::vector<std::deque<std::set<uint32_t>>> chip_timeline;

    const auto render_period = std::chrono::milliseconds(1000 / kRenderHz);
    RawMode raw_mode;             // restores the terminal on scope exit
    Metric metric = Metric::Fpu;  // f/s/d switch; FPU is the htop-CPU% analogue
    bool meter_view = true;       // 'g' toggles back to the spatial NoC grid
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(render_period);
        switch (poll_key()) {
            case 'f': metric = Metric::Fpu; break;
            case 's': metric = Metric::Sfpu; break;
            case 'd': metric = Metric::Dispatch; break;
            case 'a': metric = Metric::All; break;
            case 'g': meter_view = !meter_view; break;
            case 'q': g_stop.store(true, std::memory_order_relaxed); continue;
            default: break;
        }
        refresh_maps();
        refresh_registry();

        // Helper to resolve a program name from the cache. Returns nullptr
        // if no mapping is known.
        //
        // The argument is the decoded program id (bits 30:10 of host_assigned_id),
        // which is what callers display as "#NNN". The registry, however, may
        // be keyed by either:
        //   (a) the raw runtime_id (= same number as the decoded prog_id), or
        //   (b) the encoded value (runtime_id << 10 | device_id).
        // depending on which dispatch path registered it. Try (a) first, then
        // fall back by walking a small set of likely encoded forms.
        auto lookup_name = [&](uint32_t prog_id) -> const std::string* {
            auto it = name_cache.find(prog_id);
            if (it != name_cache.end()) {
                return &it->second;
            }
            // Try encoded forms — runtime_id << 10 | device_id for any
            // device_id this build of tt-metal might use. Mesh boards typically
            // have a small number of devices; bound the probe at 8 to keep this
            // O(1)-ish per render.
            for (uint32_t dev = 0; dev < 8; ++dev) {
                const uint32_t encoded = (prog_id << 10) | dev;
                it = name_cache.find(encoded);
                if (it != name_cache.end()) {
                    return &it->second;
                }
            }
            return nullptr;
        };

        // Phase 2.1.c: parallel lookup for cycles_in_window. Same key search
        // as lookup_name. Returns 0 when the registry has no value yet — that
        // signals "no Phase 2.1.c data" to the renderer (column shows --).
        auto lookup_cycles = [&](uint32_t prog_id) -> uint64_t {
            auto it = cycles_cache.find(prog_id);
            if (it != cycles_cache.end()) {
                return it->second;
            }
            for (uint32_t dev = 0; dev < 8; ++dev) {
                const uint32_t encoded = (prog_id << 10) | dev;
                it = cycles_cache.find(encoded);
                if (it != cycles_cache.end()) {
                    return it->second;
                }
            }
            return 0;
        };

        if (maps.empty()) {
            std::cout << "\x1b[H\x1b[2J"
                      << "ttnvtop   waiting for ttnvtop-collector ...\n\n"
                      << "  No /dev/shm/tt_device_*_util files published yet.\n"
                      << "  Start the collector in another terminal:\n"
                      << "    ttnvtop-collector\n"
                      << "\n[Ctrl-C to exit]\n";
            std::cout.flush();
            continue;
        }

        std::ostringstream out;
        out << "\x1b[H\x1b[2J";
        size_t total_cores = 0;
        for (const auto& m : maps) {
            total_cores += m.header->num_cores;
        }
        out << kAnsiBold << "ttnvtop" << kAnsiReset << "   " << maps.size() << " chip" << (maps.size() == 1 ? "" : "s")
            << "   " << total_cores << " cores   " << kRenderHz << " Hz"
            << "   showing " << kAnsiBold << metric_name(metric) << "%" << kAnsiReset << "\n"
            << kAnsiDim << "  [f] FPU  [s] SFPU  [d] dispatch  [a] all   [g] " << (meter_view ? "NoC grid" : "meters")
            << "   [q] quit     saturation: " << kAnsiReset << kPctGray << "idle" << kAnsiReset << " " << kPctGreen
            << "low" << kAnsiReset << " " << kPctYellow << "mid" << kAnsiReset << " " << kPctRed << "hot" << kAnsiReset
            << "\n";

        // ---- htop-style meter view -------------------------------------
        // One meter per core, laid out in as many columns as the terminal
        // fits, numbered linearly per chip. Denser than the spatial NoC grid,
        // which matters at 512 cores (8 chips x 64).
        if (meter_view) {
            const int cols = std::max(1, (term_cols() - 2) / kMeterOuterW);
            for (size_t ci = 0; ci < maps.size(); ++ci) {
                const auto& m = maps[ci];
                const auto* h = m.header;
                const auto* cores = m.cores;
                const uint32_t n = h->num_cores;

                // Chip-level averages for ALL three units are always shown, even when
                // the per-core cells display only one -- the per-chip picture is
                // cheap and is nearly always wanted as context.
                uint64_t sf = 0, ss = 0, sd = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    sf += cores[i].compute_busy_p1000;
                    ss += cores[i].sfpu_busy_p1000;
                    sd += cores[i].dispatch_busy_p1000;
                }
                const uint32_t af = n ? static_cast<uint32_t>(sf / n / 10) : 0;
                const uint32_t as = n ? static_cast<uint32_t>(ss / n / 10) : 0;
                const uint32_t ad = n ? static_cast<uint32_t>(sd / n / 10) : 0;
                // STALENESS, in the default view too.
                //
                // The other chip-title path prints (STALE); this one did not, and this is
                // the view people actually look at. A collector that hard-exits (the
                // shutdown watchdog, when a thread is stuck in a UMD poll) does not get to
                // unlink its SHM files, so they survive holding their last values -- and
                // the grid rendered them as live, indefinitely, with a plausible clock and
                // plausible percentages. Every "is this frozen or is the machine idle?"
                // question in this session came from that ambiguity.
                const bool chip_stale =
                    (monotonic_us() - h->last_update_us) > static_cast<uint64_t>(kStaleThresholdMs) * 1000;
                out << "\n" << kAnsiBold << "chip " << ci << kAnsiReset << "  " << n << " cores  @ ";
                // 0 means the ARC clock has NEVER been read successfully on this chip (the
                // collector keeps the last good value), so say that rather than print a
                // literal 0 MHz, which reads as "this chip is stopped".
                if (h->aiclk_mhz > 0) {
                    out << h->aiclk_mhz << " MHz";
                } else {
                    out << "clk n/a";
                }
                out << "   " << kMetricFpuCol << "F " << af << "%" << kAnsiReset << "  " << kMetricSfpuCol << "S " << as
                    << "%" << kAnsiReset << "  " << kMetricDispCol << "D " << ad << "%" << kAnsiReset << render_dram(h);
                if (chip_stale) {
                    const double age = static_cast<double>(monotonic_us() - h->last_update_us) / 1e6;
                    out << kAnsiBold << "   (STALE " << std::fixed << std::setprecision(0) << age
                        << "s -- no collector writing this chip)" << kAnsiReset;
                }
                out << "\n";

                const int rows = (static_cast<int>(n) + cols - 1) / cols;
                for (int r = 0; r < rows; ++r) {
                    out << " ";
                    for (int c = 0; c < cols; ++c) {
                        // Column-major so consecutive indices read down a
                        // column, matching htop's layout.
                        const int idx = c * rows + r;
                        if (idx >= static_cast<int>(n)) {
                            continue;
                        }
                        if (metric == Metric::All) {
                            out << render_meter_all(
                                       static_cast<uint32_t>(idx),
                                       cores[idx].compute_busy_p1000,
                                       cores[idx].sfpu_busy_p1000,
                                       cores[idx].dispatch_busy_p1000)
                                << " ";
                        } else {
                            out << render_meter(static_cast<uint32_t>(idx), metric_p1000(cores[idx], metric)) << " ";
                        }
                    }
                    out << "\n";
                }
            }
            out.flush();
            std::cout << out.str() << std::flush;
            continue;
        }

        // Global accumulator for the bottom program table and per-chip aggregates.
        struct KernelBucket {
            uint32_t cores = 0;
            uint64_t sum_f_p1000 = 0;
            uint64_t sum_s_p1000 = 0;
            uint64_t sum_d_p1000 = 0;
            uint32_t chip_bits = 0;  // bit i set => also present on chip i
        };
        std::unordered_map<uint32_t, KernelBucket> global_kernels;

        // Resize the per-chip timeline ring buffer to match the current
        // chip count. Adding/removing chips at runtime is rare but
        // possible (workload close+reopen) and handled cleanly.
        if (chip_timeline.size() != maps.size()) {
            chip_timeline.resize(maps.size());
        }

        // Layout: chip grids on the left, programs panel on the right. We
        // build them into separate buffers and zip them line-by-line at the
        // end so they sit side-by-side instead of stacked vertically.
        const std::streampos chips_begin = out.tellp();

        for (size_t c = 0; c < maps.size(); ++c) {
            const auto* h = maps[c].header;
            const bool stale = (monotonic_us() - h->last_update_us) > static_cast<uint64_t>(kStaleThresholdMs) * 1000;
            const auto* cores = maps[c].cores;
            const uint32_t n = h->num_cores;

            // Per-chip aggregates.
            uint64_t sum_f = 0, sum_s = 0, sum_d = 0;
            uint8_t min_x = 255, max_x = 0, min_y = 255, max_y = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const auto& v = cores[i];
                sum_f += v.compute_busy_p1000;
                sum_s += v.sfpu_busy_p1000;
                sum_d += v.dispatch_busy_p1000;
                if (v.noc_x < min_x) {
                    min_x = v.noc_x;
                }
                if (v.noc_x > max_x) {
                    max_x = v.noc_x;
                }
                if (v.noc_y < min_y) {
                    min_y = v.noc_y;
                }
                if (v.noc_y > max_y) {
                    max_y = v.noc_y;
                }
                if ((v.dispatch_busy_p1000 / 10u) > 0 && v.last_kernel_id != 0) {
                    // Aggregate by DECODED program id (bits 30:10). The lower
                    // 10 bits of host_assigned_id encode device/sub-device/
                    // dispatch-slot information that varies within a single
                    // chip — keying on the raw u32 fragments same-program
                    // cores into multiple rows. Use the prog_id portion only
                    // so all cores running runtime_id=N collapse to one entry.
                    const uint32_t prog_id_key = (v.last_kernel_id >> 10) & 0x1FFFFFu;
                    auto& b = global_kernels[prog_id_key];
                    b.cores += 1;
                    b.sum_f_p1000 += v.compute_busy_p1000;
                    b.sum_s_p1000 += v.sfpu_busy_p1000;
                    b.sum_d_p1000 += v.dispatch_busy_p1000;
                    b.chip_bits |= (1u << c);
                }
            }

            // Per-frame per-chip program set for the TIMELINE pane below.
            std::set<uint32_t> frame_set;
            for (uint32_t i = 0; i < n; ++i) {
                const auto& v = cores[i];
                if ((v.dispatch_busy_p1000 / 10u) > 0 && v.last_kernel_id != 0) {
                    frame_set.insert((v.last_kernel_id >> 10) & 0x1FFFFFu);
                }
            }
            chip_timeline[c].push_back(std::move(frame_set));
            while (chip_timeline[c].size() > kTimelineWidth) {
                chip_timeline[c].pop_front();
            }
            const uint32_t f_avg = n == 0 ? 0 : static_cast<uint32_t>(sum_f / (n * 10u));
            const uint32_t s_avg = n == 0 ? 0 : static_cast<uint32_t>(sum_s / (n * 10u));
            const uint32_t d_avg = n == 0 ? 0 : static_cast<uint32_t>(sum_d / (n * 10u));

            // Chip title line.
            out << "\nchip " << c << "   " << arch_label(h->arch_id);
            if (n > 0) {
                out << " (" << (cores[0].is_remote ? "remote" : "mmio") << ")";
            }
            if (h->aiclk_mhz > 0) {
                out << " @ " << h->aiclk_mhz << " MHz";
            }
            if (stale) {
                out << "  (STALE)";
            }
            out << "   F=" << std::setw(2) << f_avg << "%  S=" << std::setw(2) << s_avg << "%  D=" << std::setw(2)
                << d_avg << "%" << render_dram(h);
            if (h->aiclk_mhz > 0 && n > 0) {
                const double peak_greq = static_cast<double>(n) * static_cast<double>(h->aiclk_mhz) / 1000.0;
                const double f_frac = static_cast<double>(sum_f) / (static_cast<double>(n) * 1000.0);
                const double achieved_greq = peak_greq * f_frac;
                // TFLOPs deliberately NOT shown. The only derivable figure is
                // LoFi-equivalent (cores x AICLK x 4096 x busy%), which
                // over-reports by exactly the math fidelity: measured 2026-08-27
                // on WH T3K as 1.04x / 2.00x / 3.94x for pinned LoFi / HiFi2 /
                // HiFi4. Correcting it needs a per-core fidelity divisor, and
                // the WH TDMA_UNPACK fidelity counters do not report it (see
                // collector/main.cpp). A single --fidelity override would still
                // be wrong for a mixed-fidelity workload, which is the normal
                // case for a real model. Greq/s below is cores x AICLK x busy%,
                // i.e. FPU-busy-cycles per second -- fidelity-independent.
                out << "   FPU " << std::fixed << std::setprecision(1) << achieved_greq << "/" << peak_greq
                    << " Greq/s";
            }
            // Dominant-program label removed: variable-length text on the
            // chip title row caused the right-side PROGRAMS panel to shift
            // horizontally as programs changed. The PROGRAMS table on the
            // right covers the same information without the layout jitter.
            out << "\n";

            if (n == 0 || min_x > max_x || min_y > max_y) {
                out << "  (no cores)\n";
                continue;
            }

            // Build a (y, x) → PerCoreView lookup so we can render spatially.
            // WH noc coords top out around 10 on each axis; this map is tiny.
            std::unordered_map<uint32_t, const ttnvtop::PerCoreView*> at;
            at.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                const auto& v = cores[i];
                const uint32_t key = (static_cast<uint32_t>(v.noc_y) << 8) | v.noc_x;
                at[key] = &v;
            }

            // Render each core as a 12-wide × 6-tall bordered box. Adjacent
            // boxes in the same y-row share one vertical border (so the run
            // of boxes reads as a table). Consecutive y-rows share the
            // horizontal border between them.
            //
            // Box content (10 visible chars inside borders):
            //    row 0: '<x>,<y> #NNN'   (noc coord + last 3 digits of prog id)
            //    row 1: 'F|||| 78%'      (bar + pct)
            //    row 2: 'S       0%'     (bar + pct)
            //    row 3: 'D||||| 99%'     (bar + pct)

            // Column header row: noc_x centered over each box.
            out << "        ";  // y-label column pad
            for (uint8_t x = min_x; x <= max_x; ++x) {
                std::ostringstream xs;
                xs << "x=" << static_cast<int>(x);
                std::string s = xs.str();
                // Center within kBoxOuterW chars.
                int pad_l = (kBoxOuterW - static_cast<int>(s.size())) / 2;
                int pad_r = kBoxOuterW - pad_l - static_cast<int>(s.size());
                if (pad_l < 0) {
                    pad_l = 0;
                }
                if (pad_r < 0) {
                    pad_r = 0;
                }
                out << std::string(pad_l, ' ') << s << std::string(pad_r, ' ');
            }
            out << "\n";

            // Horizontal border (shared between y-rows; reused for top and bottom).
            auto hline = [&]() {
                out << "        ";
                for (uint8_t x = min_x; x <= max_x; ++x) {
                    (void)x;
                    out << "+" << std::string(kBoxInnerW, '-');
                }
                out << "+\n";
            };

            // Renders one content line across all x-columns for a given y.
            // `line_idx` ∈ [0..3]: which of the four content lines.
            auto content_line = [&](uint8_t y, int line_idx) {
                // y-label on the left, only on the first content line of each y-row.
                // Leading pad must match the hline/header (8 chars) exactly or the
                // top line of each y-row drifts right by one cell.
                if (line_idx == 0) {
                    out << "  y=" << std::setw(2) << static_cast<int>(y) << "  ";  // 4+2+2 = 8
                } else {
                    out << "        ";  // 8
                }
                for (uint8_t x = min_x; x <= max_x; ++x) {
                    const uint32_t key = (static_cast<uint32_t>(y) << 8) | x;
                    auto it = at.find(key);
                    out << "|";
                    if (it == at.end()) {
                        out << std::string(kBoxInnerW, ' ');
                        continue;
                    }
                    const auto* v = it->second;
                    if (line_idx == 0) {
                        // Coord + #prog (last 3 digits).
                        std::ostringstream hdr;
                        hdr << static_cast<int>(v->noc_x) << "," << static_cast<int>(v->noc_y);
                        std::string coord = hdr.str();
                        std::string prog_str;
                        if (v->last_kernel_id != 0 && v->dispatch_busy_p1000 > 0) {
                            const uint32_t prog_id = (v->last_kernel_id >> 10) & 0x1FFFFFu;
                            std::ostringstream ps;
                            ps << "#" << (prog_id % 1000);
                            prog_str = ps.str();
                        }
                        // Lay out inside kBoxInnerW: "<coord><spaces><prog>" padded.
                        int used = static_cast<int>(coord.size()) + 1 + static_cast<int>(prog_str.size());
                        int pad = kBoxInnerW - used;
                        if (pad < 0) {
                            pad = 0;
                        }
                        out << coord << " ";
                        if (!prog_str.empty()) {
                            const uint32_t prog_id = (v->last_kernel_id >> 10) & 0x1FFFFFu;
                            out << prog_color(prog_id) << prog_str << kAnsiReset;
                        }
                        out << std::string(pad, ' ');
                    } else {
                        uint32_t pct = 0;
                        char label = '?';
                        switch (line_idx) {
                            case 1:
                                pct = v->compute_busy_p1000 / 10u;
                                label = 'F';
                                break;
                            case 2:
                                pct = v->sfpu_busy_p1000 / 10u;
                                label = 'S';
                                break;
                            case 3:
                                pct = v->dispatch_busy_p1000 / 10u;
                                label = 'D';
                                break;
                        }
                        if (pct > 100) {
                            pct = 100;
                        }
                        // Layout: label + bar (kBarWidth) + space + pct (4 wide, right-aligned).
                        std::ostringstream pcts;
                        pcts << std::setw(3) << pct << "%";
                        std::string bar;
                        append_bar(bar, pct, kBarWidth);
                        out << label << pct_color(pct) << bar << kAnsiReset << " " << pcts.str();
                        // 1 (label) + kBarWidth + 1 (sp) + 4 (pct) = 6 + kBarWidth
                        int used = 1 + kBarWidth + 1 + 4;
                        int pad = kBoxInnerW - used;
                        if (pad < 0) {
                            pad = 0;
                        }
                        out << std::string(pad, ' ');
                    }
                }
                out << "|\n";
            };

            hline();
            for (uint8_t y = min_y; y <= max_y; ++y) {
                for (int li = 0; li < 4; ++li) {
                    content_line(y, li);
                }
                hline();
            }
        }

        // End-of-chips / start-of-programs marker for the side-by-side layout.
        const std::streampos progs_begin = out.tellp();

        // ── Program table ────────────────────────────────────────────────────
        out << "\nPROGRAMS  (bits 30:10 of host_assigned_id; counted over cores with D>0)\n";
        if (global_kernels.empty()) {
            out << "  no programs running\n";
        } else {
            std::vector<std::pair<uint32_t, KernelBucket>> rows(global_kernels.begin(), global_kernels.end());
            std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
                // Primary: cores desc. Tiebreak: avg F desc.
                if (a.second.cores != b.second.cores) {
                    return a.second.cores > b.second.cores;
                }
                return a.second.sum_f_p1000 > b.second.sum_f_p1000;
            });
            // Show the full pipe-joined kernel chain. Registry caps names at
            // kRegistryNameMax-1 = 95 chars, so this is the longest possible.
            constexpr size_t kNameMaxDisp = ttnvtop::kRegistryNameMax - 1;
            constexpr size_t kNameColW = kNameMaxDisp + 2;  // +2 trailing space
            // Phase 2.1.c: precompute total cycles_in_window across all rows
            // so per-row TIME% can be expressed as a fraction. If every row
            // reports 0, we render "--" for the whole column — the LLK hook
            // that populates cycles_in_window hasn't landed yet.
            uint64_t total_cycles_in_window = 0;
            for (const auto& row : rows) {
                total_cycles_in_window += lookup_cycles(row.first);
            }
            const bool time_pct_available = (total_cycles_in_window > 0);
            // Header columns must match the per-row layout below exactly:
            //   "  " <id8> "  " <name kNameColW> setw5(chip) "  " setw5(cores)
            //   "  " <time5> "   " setw3(F)"%" "  " setw3(S)"%" "  " setw3(D)"%"
            // TIME% data cell is 5 chars total: either setw(4)+"%" (e.g. " 100%",
            // "  87%") or the literal "   --" when cycles_in_window is 0 across
            // all rows. Header label "TIME%" is exactly 5 chars.
            {
                std::string h;
                h += "  ";  // leading
                h += "ID";
                h += std::string(8 - 2, ' ');  // pad ID to 8
                h += "  ";                     // gap before NAME
                h += "NAME";
                h += std::string(kNameColW - 4, ' ');  // pad NAME to kNameColW
                h += " CHIP";                          // right-aligned in setw(5)
                h += "  ";
                h += "CORES";  // exact fit setw(5)
                h += "  ";     // 2-space gap before TIME%
                h += "TIME%";  // exact fit 5 chars (matches data setw(4)+"%" or "   --")
                h += "   ";    // 3-space gap before F%
                h += "  F%";   // right-aligned in setw(3)+"%"
                h += "  ";
                h += "  S%";
                h += "  ";
                h += "  D%";
                out << h << "\n";
            }
            // With the side-by-side layout the chip grid dictates total
            // height (often 50+ lines per chip), so we have plenty of room
            // for the program list. Cap is a sanity bound, not a layout
            // limit; bump it well past typical workload program counts.
            const size_t limit = std::min<size_t>(rows.size(), 100);
            for (size_t i = 0; i < limit; ++i) {
                // global_kernels is now keyed by the decoded prog_id directly
                // (see comment in the aggregation loop above), so no shift here.
                const uint32_t prog_id = rows[i].first;
                const auto& b = rows[i].second;
                const uint32_t fpct = static_cast<uint32_t>(b.sum_f_p1000 / (b.cores * 10u));
                const uint32_t spct = static_cast<uint32_t>(b.sum_s_p1000 / (b.cores * 10u));
                const uint32_t dpct = static_cast<uint32_t>(b.sum_d_p1000 / (b.cores * 10u));
                std::string chip_label;
                for (size_t c = 0; c < maps.size(); ++c) {
                    if (b.chip_bits & (1u << c)) {
                        if (!chip_label.empty()) {
                            chip_label.push_back(',');
                        }
                        chip_label.push_back(static_cast<char>('0' + c));
                    }
                }
                // Format the id cell so color escapes don't break column alignment.
                std::ostringstream id_cell;
                id_cell << "#" << prog_id;
                std::string idstr = id_cell.str();
                std::string id_pad = idstr.size() < 8 ? std::string(8 - idstr.size(), ' ') : std::string();
                // Name cell: truncate at kNameMaxDisp, pad to kNameColW. Color
                // escapes don't consume visible width, so we compute padding
                // from the truncated plain name.
                const std::string* name_ptr = lookup_name(prog_id);
                std::string name_plain;
                bool name_known = false;
                if (name_ptr && !name_ptr->empty()) {
                    name_plain = name_ptr->substr(0, kNameMaxDisp);
                    name_known = true;
                } else {
                    name_plain = "unnamed";
                }
                std::string name_pad(kNameColW - name_plain.size(), ' ');
                out << "  " << prog_color(prog_id) << idstr << kAnsiReset << id_pad << "  ";
                if (name_known) {
                    out << prog_color(prog_id) << name_plain << kAnsiReset;
                } else {
                    out << kAnsiDim << name_plain << kAnsiReset;
                }
                out << name_pad << std::setw(5) << chip_label << "  " << std::setw(5) << b.cores;
                // Phase 2.1.c TIME% column: cycles_in_window / total. When the
                // total across all rows is 0 (LLK hook not yet live), show "--"
                // dim for every row so users still see the column scaffold.
                out << "  ";  // gap before TIME%
                if (time_pct_available) {
                    const uint64_t cyc = lookup_cycles(rows[i].first);
                    // Round to nearest integer percent. Cap at 100.
                    uint32_t tpct =
                        static_cast<uint32_t>((cyc * 100u + total_cycles_in_window / 2) / total_cycles_in_window);
                    if (tpct > 100) {
                        tpct = 100;
                    }
                    out << pct_color(tpct) << std::setw(4) << tpct << "%" << kAnsiReset;
                } else {
                    out << kAnsiDim << "   --" << kAnsiReset;
                }
                out << "   " << pct_color(fpct) << std::setw(3) << fpct << "%" << kAnsiReset << "  " << pct_color(spct)
                    << std::setw(3) << spct << "%" << kAnsiReset << "  " << pct_color(dpct) << std::setw(3) << dpct
                    << "%" << kAnsiReset << "\n";
            }
            if (rows.size() > limit) {
                out << "  (" << (rows.size() - limit) << " more)\n";
            }
        }

        // ── Side-by-side merge ──────────────────────────────────────────────
        // Take everything we've accumulated so far and split it into:
        //   header_text  = banner + signal-legend (above chips_begin)
        //   chips_text   = per-chip grids and aggregates
        //   progs_text   = the PROGRAMS table
        // Then re-emit as: header + [chip line | program line] zipped.
        {
            const std::string full = out.str();
            const std::string header_text = full.substr(0, static_cast<size_t>(chips_begin));
            const std::string chips_text = full.substr(
                static_cast<size_t>(chips_begin), static_cast<size_t>(progs_begin) - static_cast<size_t>(chips_begin));
            const std::string progs_text = full.substr(static_cast<size_t>(progs_begin));

            auto split_lines = [](const std::string& s) {
                std::vector<std::string> v;
                size_t start = 0;
                while (start <= s.size()) {
                    size_t nl = s.find('\n', start);
                    if (nl == std::string::npos) {
                        if (start < s.size()) {
                            v.push_back(s.substr(start));
                        }
                        break;
                    }
                    v.push_back(s.substr(start, nl - start));
                    start = nl + 1;
                }
                return v;
            };

            const auto chip_lines = split_lines(chips_text);
            const auto prog_lines = split_lines(progs_text);

            // Chip grid uses ANSI color codes for the per-program #NNN label,
            // so byte length != visible width. Strip CSI sequences (ESC '['
            // … 'm') when measuring so padding aligns visually.
            auto visible_width = [](const std::string& s) -> size_t {
                size_t n = 0;
                for (size_t i = 0; i < s.size();) {
                    if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '[') {
                        size_t j = i + 2;
                        while (j < s.size() && s[j] != 'm') {
                            ++j;
                        }
                        i = (j < s.size()) ? j + 1 : s.size();
                    } else {
                        ++n;
                        ++i;
                    }
                }
                return n;
            };

            size_t chip_panel_w = 0;
            std::vector<size_t> chip_visible(chip_lines.size(), 0);
            for (size_t i = 0; i < chip_lines.size(); ++i) {
                chip_visible[i] = visible_width(chip_lines[i]);
                if (chip_visible[i] > chip_panel_w) {
                    chip_panel_w = chip_visible[i];
                }
            }

            std::ostringstream merged;
            merged << header_text;
            const size_t rows_total = std::max(chip_lines.size(), prog_lines.size());
            for (size_t i = 0; i < rows_total; ++i) {
                if (i < chip_lines.size()) {
                    merged << chip_lines[i];
                    if (chip_visible[i] < chip_panel_w) {
                        merged << std::string(chip_panel_w - chip_visible[i], ' ');
                    }
                } else {
                    merged << std::string(chip_panel_w, ' ');
                }
                if (i < prog_lines.size() && !prog_lines[i].empty()) {
                    merged << "  " << prog_lines[i];
                }
                merged << "\n";
            }

            // ── TIMELINE pane: per-chip program Gantt over the last 8 s ──
            // Each row = one program. Each column = one render frame
            // (~100 ms). Cell is filled when the program was running on at
            // least one core of this chip at that frame, blank otherwise.
            // Rows sorted by total presence (most-active first), capped at
            // 12 to keep the pane terminal-friendly.
            for (size_t ci = 0; ci < chip_timeline.size(); ++ci) {
                const auto& tl_frames = chip_timeline[ci];
                if (tl_frames.empty()) {
                    continue;
                }
                std::unordered_map<uint32_t, uint32_t> presence;
                for (const auto& fset : tl_frames) {
                    for (uint32_t p : fset) {
                        ++presence[p];
                    }
                }
                if (presence.empty()) {
                    continue;
                }
                std::vector<std::pair<uint32_t, uint32_t>> tl_rows(presence.begin(), presence.end());
                std::sort(
                    tl_rows.begin(), tl_rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                constexpr size_t kTLRowCap = 12;
                if (tl_rows.size() > kTLRowCap) {
                    tl_rows.resize(kTLRowCap);
                }

                merged << "\nTIMELINE chip " << ci << "  (last " << (tl_frames.size() * (1000 / kRenderHz)) / 1000
                       << "s, " << (1000 / kRenderHz) << "ms cells, older ──> now)\n";
                for (auto& [pid, count] : tl_rows) {
                    const std::string* nm = nullptr;
                    auto nit = name_cache.find(pid);
                    if (nit != name_cache.end()) {
                        nm = &nit->second;
                    }
                    if (!nm) {
                        for (uint32_t dev = 0; dev < 8; ++dev) {
                            auto it = name_cache.find((pid << 10) | dev);
                            if (it != name_cache.end()) {
                                nm = &it->second;
                                break;
                            }
                        }
                    }
                    std::string name_disp = nm ? *nm : std::string("?");
                    constexpr size_t kTLNameW = 36;
                    if (name_disp.size() > kTLNameW) {
                        name_disp.resize(kTLNameW);
                    }
                    if (name_disp.size() < kTLNameW) {
                        name_disp.append(kTLNameW - name_disp.size(), ' ');
                    }

                    std::ostringstream id_cell;
                    id_cell << "#" << pid;
                    std::string idstr = id_cell.str();
                    if (idstr.size() < 6) {
                        idstr.append(6 - idstr.size(), ' ');
                    }

                    merged << "  " << prog_color(pid) << idstr << kAnsiReset << " " << name_disp << "  [";
                    const size_t pad = kTimelineWidth > tl_frames.size() ? kTimelineWidth - tl_frames.size() : 0;
                    for (size_t pi = 0; pi < pad; ++pi) {
                        merged << ' ';
                    }
                    for (const auto& fset : tl_frames) {
                        if (fset.count(pid)) {
                            merged << prog_color(pid) << "#" << kAnsiReset;
                        } else {
                            merged << kAnsiDim << "." << kAnsiReset;
                        }
                    }
                    merged << "]\n";
                }
            }

            // Stage 1 (history pane): every program seen since the viewer
            // started, sorted most-recent first. The live PROGRAMS table
            // above only reflects programs currently dispatched on cores —
            // sub-period kernels run between frames and never appear there.
            // The registry has 100% coverage; this view exposes it.
            if (!history_cache.empty()) {
                std::vector<std::pair<uint32_t, const HistoryEntry*>> entries;
                entries.reserve(history_cache.size());
                for (const auto& kv : history_cache) {
                    entries.emplace_back(kv.first, &kv.second);
                }
                std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                    if (a.second->last_seen_ms != b.second->last_seen_ms) {
                        return a.second->last_seen_ms > b.second->last_seen_ms;
                    }
                    return a.first < b.first;
                });
                merged << "\nHISTORY (" << entries.size() << " unique programs since viewer start)\n";
                merged << "      ID  NAME" << std::string(static_cast<size_t>(ttnvtop::kRegistryNameMax) - 4, ' ')
                       << "      FIRST       LAST     DISP   CYCLES_TOTAL\n";
                auto fmt_ms = [](uint64_t ms) {
                    std::ostringstream o;
                    const uint64_t s = ms / 1000;
                    const uint64_t f = ms % 1000;
                    const uint64_t mm = s / 60;
                    const uint64_t ss = s % 60;
                    o << std::setw(2) << std::setfill('0') << mm << ":" << std::setw(2) << std::setfill('0') << ss
                      << "." << std::setw(3) << std::setfill('0') << f;
                    return o.str();
                };
                const size_t kHistoryRowCap = 30;
                const size_t shown = std::min(entries.size(), kHistoryRowCap);
                for (size_t i = 0; i < shown; ++i) {
                    const uint32_t rid = entries[i].first;
                    const auto& h = *entries[i].second;
                    std::string name_disp = h.name;
                    if (name_disp.size() >= ttnvtop::kRegistryNameMax) {
                        name_disp.resize(ttnvtop::kRegistryNameMax - 1);
                    }
                    if (name_disp.size() < ttnvtop::kRegistryNameMax) {
                        name_disp.append(ttnvtop::kRegistryNameMax - name_disp.size(), ' ');
                    }
                    merged << "  " << std::setw(6) << std::setfill(' ') << rid << "  " << name_disp << "  "
                           << fmt_ms(h.first_seen_ms) << "  " << fmt_ms(h.last_seen_ms) << "  " << std::setw(7)
                           << std::setfill(' ') << h.dispatch_count << "  " << std::setw(13) << std::setfill(' ')
                           << h.peak_cycles_total << "\n";
                }
                if (entries.size() > kHistoryRowCap) {
                    merged << "  (" << (entries.size() - kHistoryRowCap) << " more — see "
                           << "tt_program_registry.bin for the full list)\n";
                }
            }

            merged << "\n[Ctrl-C to exit]\n";
            std::cout << merged.str();
        }
        std::cout.flush();
    }

    for (auto& m : maps) {
        unmap_shm(m);
    }
    unmap_registry(registry);
    std::cout << "\nttnvtop: exiting.\n";
    return 0;
}
