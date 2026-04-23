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
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../collector/shm_publisher.hpp"
#include "../common/shm_schema.hpp"

namespace {

constexpr int kRenderHz = 4;
// Cell format: "(xx,yy) C[........] xxx% D[........] xxx% "
// visible widths:   8   +    17     +      17       = 42.
constexpr int kCorePanelWidth = 42;
constexpr int kBarWidth = 8;
constexpr int kStaleThresholdMs = 2000;

// ANSI color escapes. Kept inline — no curses dep.
constexpr const char* kAnsiReset = "\x1b[0m";
constexpr const char* kAnsiDim = "\x1b[2m";
constexpr const char* kAnsiGreen = "\x1b[38;5;34m";    // 1..33%
constexpr const char* kAnsiYellow = "\x1b[38;5;220m";  // 34..66%
constexpr const char* kAnsiRed = "\x1b[38;5;196m";     // 67..100%
constexpr const char* kAnsiGray = "\x1b[38;5;240m";    //  0%
constexpr const char* kAnsiBold = "\x1b[1m";

const char* color_for_pct(uint32_t pct) {
    if (pct == 0) {
        return kAnsiGray;
    }
    if (pct < 34) {
        return kAnsiGreen;
    }
    if (pct < 67) {
        return kAnsiYellow;
    }
    return kAnsiRed;
}

struct MappedShm {
    int fd = -1;
    void* map = nullptr;
    size_t map_size = 0;
    const ttnvtop::UtilShmHeader* header = nullptr;
    const ttnvtop::PerCoreView* cores = nullptr;
    std::string path;
};

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

std::string make_bar(uint32_t pct, int width) {
    uint32_t filled = (pct * static_cast<uint32_t>(width)) / 100;
    if (filled > static_cast<uint32_t>(width)) {
        filled = static_cast<uint32_t>(width);
    }
    std::string s;
    s.reserve(static_cast<size_t>(width) + 2);
    s.push_back('[');
    for (int i = 0; i < width; ++i) {
        s.push_back(i < static_cast<int>(filled) ? '#' : ' ');
    }
    s.push_back(']');
    return s;
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
    (void)argc;
    (void)argv;
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::vector<MappedShm> maps;

    // Refresh the set of SHM files: add any new ones, drop any that vanished or
    // whose collector PID is no longer alive.
    auto refresh_maps = [&]() {
        auto entries = ttnvtop::list_shm_files();
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.asic_id < b.asic_id; });

        // Drop maps whose path no longer exists.
        maps.erase(
            std::remove_if(
                maps.begin(),
                maps.end(),
                [&](MappedShm& m) {
                    bool still_there = false;
                    for (const auto& e : entries) {
                        if (e.path == m.path) {
                            still_there = true;
                            break;
                        }
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

    const auto render_period = std::chrono::milliseconds(1000 / kRenderHz);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(render_period);
        refresh_maps();

        if (maps.empty()) {
            std::cout << "\x1b[H\x1b[2J" << kAnsiBold << "ttnvtop" << kAnsiReset
                      << "  |  waiting for ttnvtop-collector ...\n\n"
                      << kAnsiDim << "  No /dev/shm/tt_device_*_util files published yet.\n"
                      << "  Start the collector in another terminal:\n"
                      << "    ttnvtop-collector\n"
                      << kAnsiReset << "\n[Ctrl-C to exit]\n";
            std::cout.flush();
            continue;
        }

        std::ostringstream out;
        out << "\x1b[H\x1b[2J";
        size_t total_cores = 0;
        for (const auto& m : maps) {
            total_cores += m.header->num_cores;
        }
        out << "ttnvtop  |  chips=" << maps.size() << "  cores=" << total_cores << "  refresh=" << kRenderHz
            << "Hz  (SHM viewer)\n";
        out << "\n";
        out << "Signals shown per Tensix (EWMA ~1s):\n";
        out << "  " << kAnsiBold << "C" << kAnsiReset << " = FPU peak-request density%    "
            << "delta(FPU_OUT_H) / delta(WALL_CLOCK).\n";
        out << "     Fraction of AICLK cycles the FPU issued a request. Tensix is\n";
        out << "     single-issue, so C=100% means the FPU issued every cycle (true\n";
        out << "     peak). C does NOT count SIMD-lane occupancy — a partial-width op\n";
        out << "     shows the same C as a full-width one. Not peak-FLOPS%.\n";
        out << "  " << kAnsiBold << "D" << kAnsiReset << " = dispatch occupancy%          "
            << "fraction of last ~1s a kernel was\n";
        out << "     dispatched to this core (go_msg.signal == RUN_MSG_GO). A kernel\n";
        out << "     stalled on NOC/CB still reads 100%.\n";
        out << "  Interpretation: high D + low C = data movement / stalls; both high =\n";
        out << "  real compute; both low = core genuinely idle.\n";
        out << "  Color: " << kAnsiGray << "idle" << kAnsiReset << " / " << kAnsiGreen << "low" << kAnsiReset << " / "
            << kAnsiYellow << "mid" << kAnsiReset << " / " << kAnsiRed << "high" << kAnsiReset << ".\n";
        out << "\n";

        // Header row: chip title per column.
        for (size_t c = 0; c < maps.size(); ++c) {
            const auto* h = maps[c].header;
            const bool stale = (monotonic_us() - h->last_update_us) > static_cast<uint64_t>(kStaleThresholdMs) * 1000;
            std::ostringstream t;
            t << "chip asic 0x" << std::hex << h->asic_id << std::dec << " " << arch_label(h->arch_id);
            if (h->num_cores > 0) {
                t << " [" << (maps[c].cores[0].is_remote ? "remote" : " mmio ") << "]";
            }
            if (h->aiclk_mhz > 0) {
                t << " @ " << h->aiclk_mhz << " MHz";
            }
            if (stale) {
                t << " (STALE)";
            }
            std::string s = t.str();
            if (static_cast<int>(s.size()) < kCorePanelWidth) {
                s.append(kCorePanelWidth - s.size(), ' ');
            } else if (static_cast<int>(s.size()) > kCorePanelWidth) {
                s.resize(kCorePanelWidth);  // truncate rather than break alignment
            }
            out << s;
            if (c + 1 < maps.size()) {
                out << " | ";
            }
        }
        out << "\n";
        for (size_t c = 0; c < maps.size(); ++c) {
            out << std::string(kCorePanelWidth, '-');
            if (c + 1 < maps.size()) {
                out << "-+-";
            }
        }
        out << "\n";

        size_t max_rows = 0;
        for (const auto& m : maps) {
            max_rows = std::max(max_rows, static_cast<size_t>(m.header->num_cores));
        }
        // Per-chip running totals of both signals for the footer averages.
        std::vector<uint64_t> compute_sum(maps.size(), 0);
        std::vector<uint64_t> dispatch_sum(maps.size(), 0);
        // Per-core row cell visible layout:
        //   "(xx,yy) C[########] xx% D[########] xx% "
        //    8       + 13        + 13        + 1 trailing = 35 chars; pad to kCorePanelWidth.
        for (size_t row = 0; row < max_rows; ++row) {
            for (size_t c = 0; c < maps.size(); ++c) {
                const uint32_t src = maps[c].header->signal_sources;
                const bool show_compute = (src & ttnvtop::SIGNAL_SRC_COMPUTE) != 0;
                const bool show_dispatch = (src & ttnvtop::SIGNAL_SRC_DISPATCH) != 0;
                if (row < maps[c].header->num_cores) {
                    const auto& v = maps[c].cores[row];
                    const uint32_t cpct = v.compute_busy_p1000 / 10u;
                    const uint32_t dpct = v.dispatch_busy_p1000 / 10u;
                    compute_sum[c] += v.compute_busy_p1000;
                    dispatch_sum[c] += v.dispatch_busy_p1000;
                    out << "(" << std::setw(2) << static_cast<int>(v.noc_x) << "," << std::setw(2)
                        << static_cast<int>(v.noc_y) << ") ";
                    if (show_compute) {
                        out << "C" << color_for_pct(cpct) << make_bar(cpct, kBarWidth) << kAnsiReset << " "
                            << std::setw(3) << cpct << "% ";
                    } else {
                        out << std::string(1 + 2 + kBarWidth + 1 + 3 + 2, ' ');
                    }
                    if (show_dispatch) {
                        out << "D" << color_for_pct(dpct) << make_bar(dpct, kBarWidth) << kAnsiReset << " "
                            << std::setw(3) << dpct << "% ";
                    } else {
                        out << std::string(1 + 2 + kBarWidth + 1 + 3 + 2, ' ');
                    }
                } else {
                    out << std::string(kCorePanelWidth, ' ');
                }
                if (c + 1 < maps.size()) {
                    out << " | ";
                }
            }
            out << "\n";
        }

        for (size_t c = 0; c < maps.size(); ++c) {
            out << std::string(kCorePanelWidth, '-');
            if (c + 1 < maps.size()) {
                out << "-+-";
            }
        }
        out << "\n";
        for (size_t c = 0; c < maps.size(); ++c) {
            const uint32_t n = maps[c].header->num_cores;
            const uint32_t c_avg = (n == 0) ? 0u : static_cast<uint32_t>(compute_sum[c] / (n * 10u));
            const uint32_t d_avg = (n == 0) ? 0u : static_cast<uint32_t>(dispatch_sum[c] / (n * 10u));
            std::ostringstream t;
            t << "avg C=" << std::setw(3) << c_avg << "%  D=" << std::setw(3) << d_avg << "%  (" << n << " cores)";
            std::string s = t.str();
            if (static_cast<int>(s.size()) < kCorePanelWidth) {
                s.append(kCorePanelWidth - s.size(), ' ');
            }
            out << s;
            if (c + 1 < maps.size()) {
                out << " | ";
            }
        }
        out << "\n";
        // Per Tensix matrix-engine tech report: one FPU request at LoFi
        // fidelity issues a 4096-muladd matmul tile, so 4 TFLOPs/Tensix at
        // 1 GHz peak. Each muladd is counted as one op, per tt-metal docs.
        constexpr uint32_t kLoFiMuladdsPerFpuReq = 4096;
        // Second footer row: absolute FPU-request throughput when AICLK known.
        //   peak_greq = num_cores × AICLK (1 req/cycle × cores at clock rate)
        //   achieved_greq = peak_greq × c_avg
        for (size_t c = 0; c < maps.size(); ++c) {
            const uint32_t n = maps[c].header->num_cores;
            const uint32_t aiclk = maps[c].header->aiclk_mhz;
            std::ostringstream t;
            if (aiclk > 0 && n > 0) {
                const double peak_greq = static_cast<double>(n) * static_cast<double>(aiclk) / 1000.0;  // Greq/s
                const double c_avg = static_cast<double>(compute_sum[c]) / (static_cast<double>(n) * 1000.0);
                const double achieved_greq = peak_greq * c_avg;
                t << "FPU req " << std::fixed << std::setprecision(1) << achieved_greq << " / " << peak_greq
                  << " Greq/s";
            } else {
                t << "FPU throughput: unknown (no AICLK)";
            }
            std::string s = t.str();
            if (static_cast<int>(s.size()) < kCorePanelWidth) {
                s.append(kCorePanelWidth - s.size(), ' ');
            } else if (static_cast<int>(s.size()) > kCorePanelWidth) {
                s.resize(kCorePanelWidth);
            }
            out << s;
            if (c + 1 < maps.size()) {
                out << " | ";
            }
        }
        out << "\n";
        // Third footer row: TFLOPs estimate assuming each FPU request is a
        // full-width LoFi matmul tile. Conservative upper bound for non-
        // matmul workloads; close to truth for prefill-style matmul.
        for (size_t c = 0; c < maps.size(); ++c) {
            const uint32_t n = maps[c].header->num_cores;
            const uint32_t aiclk = maps[c].header->aiclk_mhz;
            std::ostringstream t;
            if (aiclk > 0 && n > 0) {
                const double peak_greq = static_cast<double>(n) * static_cast<double>(aiclk) / 1000.0;
                const double c_avg = static_cast<double>(compute_sum[c]) / (static_cast<double>(n) * 1000.0);
                const double peak_tflops = peak_greq * kLoFiMuladdsPerFpuReq / 1000.0;
                const double achieved_tflops = peak_tflops * c_avg;
                t << "~" << std::fixed << std::setprecision(0) << achieved_tflops << " / " << std::setprecision(0)
                  << peak_tflops << " TFLOPs (LoFi matmul-eq)";
            } else {
                t << "";
            }
            std::string s = t.str();
            if (static_cast<int>(s.size()) < kCorePanelWidth) {
                s.append(kCorePanelWidth - s.size(), ' ');
            } else if (static_cast<int>(s.size()) > kCorePanelWidth) {
                s.resize(kCorePanelWidth);
            }
            out << s;
            if (c + 1 < maps.size()) {
                out << " | ";
            }
        }
        out << "\n\n[Ctrl-C to exit]\n";
        std::cout << out.str();
        std::cout.flush();
    }

    for (auto& m : maps) {
        unmap_shm(m);
    }
    std::cout << "\nttnvtop: exiting.\n";
    return 0;
}
