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
constexpr int kCorePanelWidth = 40;  // "(xx,yy)  [bar] pct%"
constexpr int kStaleThresholdMs = 2000;

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

    auto entries = ttnvtop::list_shm_files();
    if (entries.empty()) {
        std::cerr << "ttnvtop: no collector SHM files found in /dev/shm.\n"
                  << "        Start ttnvtop-collector first.\n";
        return 1;
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.asic_id < b.asic_id; });

    std::vector<MappedShm> maps;
    maps.reserve(entries.size());
    for (const auto& e : entries) {
        MappedShm m;
        if (map_shm(e.path, m)) {
            maps.push_back(std::move(m));
        } else {
            std::cerr << "ttnvtop: skipping " << e.path << " (bad header / version mismatch)\n";
        }
    }
    if (maps.empty()) {
        std::cerr << "ttnvtop: no readable collector SHM files found.\n";
        return 1;
    }

    const auto render_period = std::chrono::milliseconds(1000 / kRenderHz);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(render_period);

        std::ostringstream out;
        out << "\x1b[H\x1b[2J";
        size_t total_cores = 0;
        for (const auto& m : maps) {
            total_cores += m.header->num_cores;
        }
        out << "ttnvtop  |  chips=" << maps.size() << "  cores=" << total_cores << "  refresh=" << kRenderHz
            << "Hz  (SHM viewer)\n";
        out << "\n";
        out << "What this measures (Phase 1 — dispatch occupancy):\n";
        out << "  Per Tensix, fraction of the last ~1s in which a kernel was\n";
        out << "  dispatched to that core (go_msg.signal == RUN_MSG_GO).\n";
        out << "  100% = kernel dispatched the full window.  0% = idle.\n";
        out << "  NOT the same as compute busy%: a dispatched kernel stalled\n";
        out << "  on NOC/CB still reads 100%. Phase 2 will add true\n";
        out << "  FPU/PACK/UNPACK busy% via on-chip perf-counter sampling.\n";
        out << "\n";

        // Header row: chip title per column.
        for (size_t c = 0; c < maps.size(); ++c) {
            const auto* h = maps[c].header;
            const bool stale = (monotonic_us() - h->last_update_us) > static_cast<uint64_t>(kStaleThresholdMs) * 1000;
            std::ostringstream t;
            t << "chip asic 0x" << std::hex << h->asic_id << std::dec << "  " << arch_label(h->arch_id);
            // is_remote is per-core, not per-chip, in the schema; peek at first core.
            if (h->num_cores > 0) {
                t << "  [" << (maps[c].cores[0].is_remote ? "remote" : " mmio ") << "]";
            }
            if (stale) {
                t << " (STALE)";
            }
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
        std::vector<uint64_t> chip_sum(maps.size(), 0);
        for (size_t row = 0; row < max_rows; ++row) {
            for (size_t c = 0; c < maps.size(); ++c) {
                std::string cell;
                if (row < maps[c].header->num_cores) {
                    const auto& v = maps[c].cores[row];
                    // Phase 1: show dispatch-occupancy. When SIGNAL_SRC_COMPUTE is
                    // set (Phase 2+) we'll use compute_busy_p1000 instead.
                    const uint16_t busy_p1000 = (maps[c].header->signal_sources & ttnvtop::SIGNAL_SRC_COMPUTE)
                                                    ? v.compute_busy_p1000
                                                    : v.dispatch_busy_p1000;
                    const uint32_t pct = busy_p1000 / 10u;
                    chip_sum[c] += busy_p1000;
                    std::ostringstream line;
                    line << "(" << std::setw(2) << static_cast<int>(v.noc_x) << "," << std::setw(2)
                         << static_cast<int>(v.noc_y) << ")  " << make_bar(pct, 22) << " " << std::setw(3) << pct
                         << "%";
                    cell = line.str();
                }
                if (static_cast<int>(cell.size()) < kCorePanelWidth) {
                    cell.append(kCorePanelWidth - cell.size(), ' ');
                }
                out << cell;
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
            const uint32_t avg = (n == 0) ? 0u : static_cast<uint32_t>(chip_sum[c] / (n * 10u));
            std::ostringstream t;
            t << "avg " << std::setw(3) << avg << "%   (" << n << " cores)";
            std::string s = t.str();
            if (static_cast<int>(s.size()) < kCorePanelWidth) {
                s.append(kCorePanelWidth - s.size(), ' ');
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
