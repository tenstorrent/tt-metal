// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone tool that uses the tt-metal device API + SocDescriptor to
// enumerate every core on a single chip, prints a physical-NOC0 map, and
// verifies the observed map against a hard-coded Wormhole B0 layout.

#include <array>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace {

using tt::CoordSystem;
using tt::CoreType;
using tt::tt_metal::NOC;

struct CellInfo {
    CoreType type = CoreType::COUNT;  // COUNT == empty / unknown.
    bool harvested = false;
    std::optional<int> dram_channel;
    std::optional<int> dram_subchannel;
    std::optional<uint32_t> eth_channel;
};

enum class D2WAssignmentPolicy {
    FixedNoc0ReadNoc1Write,
    DynamicNoc,
};

struct D2WRouteScore {
    uint32_t score = std::numeric_limits<uint32_t>::max();
    uint32_t read_hops = 0;
    uint32_t write_hops = 0;
    NOC reader_noc = NOC::NOC_0;
    NOC writer_noc = NOC::NOC_1;
};

struct D2WAssignment {
    bool valid = false;
    uint32_t bank = 0;
    D2WRouteScore route;
};

struct BankBestWorkers {
    uint32_t score = std::numeric_limits<uint32_t>::max();
    std::vector<tt::umd::CoreCoord> workers;
};

struct DramCoreWorkAssignment {
    tt::umd::CoreCoord core;
    uint32_t dram_bank = 0;
    uint32_t slot = 0;
    uint32_t score = 0;
    uint32_t tile_ofs = 0;
    uint32_t num_tiles = 0;
};

constexpr char kGlyphEmpty = '.';
constexpr uint32_t kFirstUnavailableWorkerRow = 10;

bool is_worker_row_available(uint32_t physical_y) { return physical_y < kFirstUnavailableWorkerRow; }

char type_glyph(CoreType t) {
    switch (t) {
        case CoreType::TENSIX: return 'T';
        case CoreType::DRAM: return 'D';
        case CoreType::ETH:
        case CoreType::ACTIVE_ETH:
        case CoreType::IDLE_ETH: return 'E';
        case CoreType::PCIE: return 'P';
        case CoreType::ARC: return 'A';
        case CoreType::ROUTER_ONLY: return 'R';
        case CoreType::HARVESTED: return 'H';
        case CoreType::WORKER: return 'W';
        case CoreType::SECURITY: return 'S';
        case CoreType::L2CPU: return 'L';
        default: return kGlyphEmpty;
    }
}

// Expected WH B0 (N150 / unharvested tile ordering) layout on the 10x12 NOC0
// grid. Rows are y=0..11, columns are x=0..9.
const std::array<std::string, 12> kExpectedWormholeB0Grid = {
    std::string("DEEEEDEEEE"),  // y=0
    std::string("DTTTTDTTTT"),  // y=1
    std::string("RTTTTDTTTT"),  // y=2
    std::string("PTTTTDTTTT"),  // y=3
    std::string("RTTTTDTTTT"),  // y=4
    std::string("DTTTTDTTTT"),  // y=5
    std::string("DEEEEDEEEE"),  // y=6
    std::string("DTTTTDTTTT"),  // y=7
    std::string("RTTTTDTTTT"),  // y=8
    std::string("RTTTTDTTTT"),  // y=9
    std::string("ATTTTDTTTT"),  // y=10
    std::string("DTTTTDTTTT"),  // y=11
};

char bucket_glyph(CoreType t) {
    char g = type_glyph(t);
    if (g == 'H') {
        return 'T';
    }
    return g;
}

std::string core_type_name(CoreType t) { return tt::to_str(t); }

std::string arch_name(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return "WORMHOLE_B0";
        case tt::ARCH::BLACKHOLE: return "BLACKHOLE";
        case tt::ARCH::QUASAR: return "QUASAR";
        case tt::ARCH::Invalid: return "INVALID";
        default: return "UNKNOWN";
    }
}

std::string fmt_yx(uint32_t y, uint32_t x) {
    std::ostringstream s;
    s << "(" << y << "," << x << ")";
    return s.str();
}

std::string bank_label(uint32_t bank) {
    if (bank < 10) {
        return std::string(1, static_cast<char>('0' + bank));
    }
    if (bank < 36) {
        return std::string(1, static_cast<char>('A' + bank - 10));
    }

    std::ostringstream s;
    s << bank;
    return s.str();
}

std::vector<std::vector<CellInfo>> build_observed_grid(const metal_SocDescriptor& soc_desc) {
    const size_t gx = soc_desc.grid_size.x;
    const size_t gy = soc_desc.grid_size.y;
    std::vector<std::vector<CellInfo>> grid(gy, std::vector<CellInfo>(gx));

    auto paint = [&](CoreType type) {
        for (const auto& c : soc_desc.get_cores(type, CoordSystem::NOC0)) {
            if (c.x < gx && c.y < gy) {
                grid[c.y][c.x].type = type;
            }
        }
    };

    paint(CoreType::ARC);
    paint(CoreType::PCIE);
    paint(CoreType::ROUTER_ONLY);
    paint(CoreType::DRAM);
    paint(CoreType::ETH);
    paint(CoreType::TENSIX);

    for (const auto& c : soc_desc.get_harvested_cores(CoreType::TENSIX, CoordSystem::NOC0)) {
        if (c.x < gx && c.y < gy) {
            grid[c.y][c.x].type = CoreType::TENSIX;
            grid[c.y][c.x].harvested = true;
        }
    }

    const auto per_subch = soc_desc.get_dram_cores();
    for (int ch = 0; ch < soc_desc.get_num_dram_channels(); ++ch) {
        if (ch >= static_cast<int>(per_subch.size())) {
            break;
        }
        for (size_t sub = 0; sub < per_subch[ch].size(); ++sub) {
            const auto& c = per_subch[ch][sub];
            if (c.coord_system != CoordSystem::NOC0) {
                continue;
            }
            if (c.x >= gx || c.y >= gy) {
                continue;
            }
            grid[c.y][c.x].dram_channel = ch;
            grid[c.y][c.x].dram_subchannel = static_cast<int>(sub);
        }
    }

    for (uint32_t ch = 0; ch < soc_desc.get_num_eth_channels(); ++ch) {
        tt::umd::CoreCoord c = soc_desc.get_eth_core_for_channel(ch, CoordSystem::NOC0);
        if (c.x < gx && c.y < gy) {
            grid[c.y][c.x].eth_channel = ch;
        }
    }

    return grid;
}

void print_grid(const std::vector<std::vector<CellInfo>>& grid) {
    if (grid.empty()) {
        return;
    }
    const size_t gx = grid[0].size();
    const size_t gy = grid.size();

    std::cout << "\n=== Physical NOC0 core map (x=column, y=row) ===\n";
    std::cout << "Legend: T=TENSIX  D=DRAM  E=ETH  P=PCIE  A=ARC  R=ROUTER  H=HARVESTED_TENSIX  .=unused\n\n";

    std::cout << "     ";
    for (size_t x = 0; x < gx; ++x) {
        std::cout << std::setw(2) << x << ' ';
    }
    std::cout << "\n    +";
    for (size_t x = 0; x < gx; ++x) {
        std::cout << "---";
    }
    std::cout << "\n";

    for (size_t y = 0; y < gy; ++y) {
        std::cout << " " << std::setw(2) << y << " |";
        for (size_t x = 0; x < gx; ++x) {
            char g = grid[y][x].harvested ? 'H' : type_glyph(grid[y][x].type);
            std::cout << ' ' << g << ' ';
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

void print_per_core_table(const std::vector<std::vector<CellInfo>>& grid) {
    std::cout << "\n=== Per-core description (NOC0 coords) ===\n";
    std::cout << std::left << std::setw(10) << "y,x" << std::setw(10) << "type" << std::setw(12) << "harvested"
              << "extra\n";
    for (size_t y = 0; y < grid.size(); ++y) {
        for (size_t x = 0; x < grid[y].size(); ++x) {
            const auto& c = grid[y][x];
            if (c.type == CoreType::COUNT) {
                continue;
            }

            std::ostringstream extra;
            if (c.dram_channel) {
                extra << "dram_ch=" << *c.dram_channel << " subch=" << *c.dram_subchannel;
            }
            if (c.eth_channel) {
                if (extra.tellp() != std::streampos(0)) {
                    extra << "  ";
                }
                extra << "eth_ch=" << *c.eth_channel;
            }

            std::cout << std::left << std::setw(10) << fmt_yx(y, x) << std::setw(10) << core_type_name(c.type)
                      << std::setw(12) << (c.harvested ? "yes" : "no") << extra.str() << "\n";
        }
    }
    std::cout << std::flush;
}

void print_dram_worker_assignment(tt::tt_metal::IDevice* device, const metal_SocDescriptor& soc_desc, NOC noc) {
    std::cout << "\n=== Optimal DRAM-bank -> logical worker (NOC" << (noc == NOC::NOC_0 ? 0 : 1) << ") ===\n";
    const auto workers = device->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    const uint8_t noc_id = (noc == NOC::NOC_0 ? 0 : 1);

    std::cout << std::left << std::setw(6) << "bank" << std::setw(22) << "dram_endpoint_NOC0" << std::setw(22)
              << "dram_endpoint_TRANS" << std::setw(18) << "logical_worker" << std::setw(22) << "physical_worker_NOC0"
              << "\n";

    for (size_t bank = 0; bank < workers.size(); ++bank) {
        CoreCoord dram_translated = soc_desc.get_preferred_worker_core_for_dram_view(bank, noc_id);
        tt::umd::CoreCoord dram_noc0 = soc_desc.translate_coord_to(
            tt::umd::CoreCoord(dram_translated.x, dram_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
            CoordSystem::NOC0);
        const CoreCoord& w_logical = workers[bank];
        CoreCoord w_phys = soc_desc.get_physical_tensix_core_from_logical(w_logical);

        std::cout << std::left << std::setw(6) << bank << std::setw(22) << fmt_yx(dram_noc0.y, dram_noc0.x)
                  << std::setw(22) << fmt_yx(dram_translated.y, dram_translated.x) << std::setw(18)
                  << fmt_yx(w_logical.y, w_logical.x) << std::setw(22) << fmt_yx(w_phys.y, w_phys.x) << "\n";
    }
    std::cout << std::flush;
}

uint32_t torus_distance_positive(uint32_t src, uint32_t dst, uint32_t extent) { return (dst + extent - src) % extent; }

uint32_t route_hops(const tt::umd::CoreCoord& src, const tt::umd::CoreCoord& dst, uint32_t gx, uint32_t gy, NOC noc) {
    if (noc == NOC::NOC_0) {
        // NOC0 routes east (+x) first, then south (+y), wrapping around the torus.
        return torus_distance_positive(src.x, dst.x, gx) + torus_distance_positive(src.y, dst.y, gy);
    }

    // NOC1 routes north (-y) first, then west (-x), wrapping around the torus.
    return torus_distance_positive(dst.y, src.y, gy) + torus_distance_positive(dst.x, src.x, gx);
}

uint32_t worker_to_dram_hops(
    const tt::umd::CoreCoord& src, const tt::umd::CoreCoord& dst, uint32_t gx, uint32_t gy, NOC noc) {
    return route_hops(src, dst, gx, gy, noc);
}

tt::umd::CoreCoord dram_bank_endpoint_noc0(const metal_SocDescriptor& soc_desc, uint32_t bank, NOC noc) {
    const uint8_t noc_id = (noc == NOC::NOC_0 ? 0 : 1);
    CoreCoord dram_translated = soc_desc.get_preferred_worker_core_for_dram_view(bank, noc_id);
    return soc_desc.translate_coord_to(
        tt::umd::CoreCoord(dram_translated.x, dram_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
        CoordSystem::NOC0);
}

D2WRouteScore score_worker_for_bank(
    const tt::umd::CoreCoord& worker,
    const tt::umd::CoreCoord& dram_endpoint_noc0,
    const tt::umd::CoreCoord& dram_endpoint_noc1,
    uint32_t gx,
    uint32_t gy,
    D2WAssignmentPolicy policy) {
    D2WRouteScore score;
    if (policy == D2WAssignmentPolicy::FixedNoc0ReadNoc1Write) {
        score.reader_noc = NOC::NOC_0;
        score.writer_noc = NOC::NOC_1;
        score.read_hops = route_hops(dram_endpoint_noc0, worker, gx, gy, score.reader_noc);
        score.write_hops = route_hops(worker, dram_endpoint_noc1, gx, gy, score.writer_noc);
        score.score = score.read_hops + score.write_hops;
        return score;
    }

    const uint32_t read_noc0 = route_hops(dram_endpoint_noc0, worker, gx, gy, NOC::NOC_0);
    const uint32_t read_noc1 = route_hops(dram_endpoint_noc1, worker, gx, gy, NOC::NOC_1);
    if (read_noc0 <= read_noc1) {
        score.reader_noc = NOC::NOC_0;
        score.read_hops = read_noc0;
    } else {
        score.reader_noc = NOC::NOC_1;
        score.read_hops = read_noc1;
    }

    const uint32_t write_noc0 = route_hops(worker, dram_endpoint_noc0, gx, gy, NOC::NOC_0);
    const uint32_t write_noc1 = route_hops(worker, dram_endpoint_noc1, gx, gy, NOC::NOC_1);
    if (write_noc1 <= write_noc0) {
        score.writer_noc = NOC::NOC_1;
        score.write_hops = write_noc1;
    } else {
        score.writer_noc = NOC::NOC_0;
        score.write_hops = write_noc0;
    }

    score.score = score.read_hops + score.write_hops;
    return score;
}

void print_dram_hop_map_for_bank(
    const std::vector<std::vector<CellInfo>>& grid, const tt::umd::CoreCoord& dram_noc0, uint32_t bank, NOC noc) {
    if (grid.empty()) {
        return;
    }

    const uint32_t gx = static_cast<uint32_t>(grid[0].size());
    const uint32_t gy = static_cast<uint32_t>(grid.size());
    const char* noc_name = noc == NOC::NOC_0 ? "NOC0 east->south" : "NOC1 north->west";

    std::cout << std::right;
    std::cout << "\n--- Bank " << bank << " DRAM endpoint NOC0=" << fmt_yx(dram_noc0.y, dram_noc0.x) << " (" << noc_name
              << ") ---\n";
    std::cout << "     ";
    for (uint32_t x = 0; x < gx; ++x) {
        std::cout << std::setw(3) << x << ' ';
    }
    std::cout << "\n    +";
    for (uint32_t x = 0; x < gx; ++x) {
        std::cout << "----";
    }
    std::cout << "\n";

    for (uint32_t y = 0; y < gy; ++y) {
        std::cout << " " << std::setw(2) << y << " |";
        for (uint32_t x = 0; x < gx; ++x) {
            const auto& cell = grid[y][x];
            if (cell.type == CoreType::COUNT) {
                std::cout << std::setw(3) << "." << ' ';
                continue;
            }

            const uint32_t hops =
                worker_to_dram_hops(tt::umd::CoreCoord(x, y, cell.type, CoordSystem::NOC0), dram_noc0, gx, gy, noc);
            std::cout << std::setw(3) << hops << ' ';
        }
        std::cout << "\n";
    }
}

void print_wormhole_dram_hop_maps(
    tt::tt_metal::IDevice* device,
    const metal_SocDescriptor& soc_desc,
    const std::vector<std::vector<CellInfo>>& grid,
    NOC noc) {
    if (device->arch() != tt::ARCH::WORMHOLE_B0) {
        return;
    }

    const uint32_t bank_count = static_cast<uint32_t>(device->num_dram_channels());
    std::cout << "\n=== WH worker/core -> DRAM bank hop maps (NOC" << (noc == NOC::NOC_0 ? 0 : 1) << ") ===\n";
    std::cout << "Each value is route hops from that physical NOC0 cell to the bank endpoint; '.' means unused cell.\n";
    if (bank_count != 12) {
        std::cout << "NOTE: expected 12 WH DRAM banks, but device reports " << bank_count
                  << "; printing reported banks.\n";
    }

    for (uint32_t bank = 0; bank < bank_count; ++bank) {
        tt::umd::CoreCoord dram_noc0 = dram_bank_endpoint_noc0(soc_desc, bank, noc);
        print_dram_hop_map_for_bank(grid, dram_noc0, bank, noc);
    }
    std::cout << std::flush;
}

std::vector<std::vector<D2WAssignment>> build_worker_to_dram_assignment_map(
    const std::vector<std::vector<CellInfo>>& grid,
    const std::vector<tt::umd::CoreCoord>& dram_endpoints_noc0,
    const std::vector<tt::umd::CoreCoord>& dram_endpoints_noc1,
    D2WAssignmentPolicy policy,
    std::vector<BankBestWorkers>& bank_best_workers) {
    const uint32_t gx = static_cast<uint32_t>(grid[0].size());
    const uint32_t gy = static_cast<uint32_t>(grid.size());
    const uint32_t bank_count = static_cast<uint32_t>(dram_endpoints_noc0.size());
    std::vector<std::vector<D2WAssignment>> assignments(gy, std::vector<D2WAssignment>(gx));
    bank_best_workers.assign(bank_count, BankBestWorkers{});

    for (uint32_t y = 0; y < gy; ++y) {
        for (uint32_t x = 0; x < gx; ++x) {
            const auto& cell = grid[y][x];
            if (cell.type != CoreType::TENSIX || cell.harvested || !is_worker_row_available(y)) {
                continue;
            }

            const tt::umd::CoreCoord worker(x, y, CoreType::TENSIX, CoordSystem::NOC0);
            D2WAssignment best;
            for (uint32_t bank = 0; bank < bank_count; ++bank) {
                D2WRouteScore route =
                    score_worker_for_bank(worker, dram_endpoints_noc0[bank], dram_endpoints_noc1[bank], gx, gy, policy);

                if (route.score < bank_best_workers[bank].score) {
                    bank_best_workers[bank].score = route.score;
                    bank_best_workers[bank].workers.clear();
                    bank_best_workers[bank].workers.push_back(worker);
                } else if (route.score == bank_best_workers[bank].score) {
                    bank_best_workers[bank].workers.push_back(worker);
                }

                if (!best.valid || route.score < best.route.score) {
                    best.valid = true;
                    best.bank = bank;
                    best.route = route;
                }
            }
            assignments[y][x] = best;
        }
    }

    return assignments;
}

void print_bank_best_worker_summary(
    const std::vector<BankBestWorkers>& bank_best_workers, const std::vector<std::vector<D2WAssignment>>& assignments) {
    std::vector<uint32_t> assigned_counts(bank_best_workers.size(), 0);
    for (const auto& row : assignments) {
        for (const auto& assignment : row) {
            if (assignment.valid) {
                ++assigned_counts[assignment.bank];
            }
        }
    }

    std::cout << "\nBest worker cores per bank (minimum read+write hops):\n";
    std::cout << std::left << std::setw(6) << "bank" << std::setw(8) << "score" << std::setw(10) << "assigned"
              << "best_worker_NOC0\n";
    for (uint32_t bank = 0; bank < bank_best_workers.size(); ++bank) {
        std::ostringstream workers;
        const auto& best = bank_best_workers[bank];
        for (size_t i = 0; i < best.workers.size(); ++i) {
            if (i != 0) {
                workers << ' ';
            }
            workers << fmt_yx(best.workers[i].y, best.workers[i].x);
        }

        std::cout << std::left << std::setw(6) << bank_label(bank) << std::setw(8) << best.score << std::setw(10)
                  << assigned_counts[bank] << workers.str() << "\n";
    }
}

void print_assignment_grid(
    const std::vector<std::vector<CellInfo>>& grid,
    const std::vector<std::vector<D2WAssignment>>& assignments,
    const std::string& title) {
    const uint32_t gx = static_cast<uint32_t>(grid[0].size());
    const uint32_t gy = static_cast<uint32_t>(grid.size());

    std::cout << std::right;
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "Enabled Tensix cells show closest DRAM bank by minimum read+write hops; H=harvested Tensix.\n";
    std::cout << "     ";
    for (uint32_t x = 0; x < gx; ++x) {
        std::cout << std::setw(3) << x << ' ';
    }
    std::cout << "\n    +";
    for (uint32_t x = 0; x < gx; ++x) {
        std::cout << "----";
    }
    std::cout << "\n";

    for (uint32_t y = 0; y < gy; ++y) {
        std::cout << " " << std::setw(2) << y << " |";
        for (uint32_t x = 0; x < gx; ++x) {
            const auto& cell = grid[y][x];
            if (assignments[y][x].valid) {
                std::cout << std::setw(3) << bank_label(assignments[y][x].bank) << ' ';
            } else {
                const char glyph = cell.harvested ? 'H' : type_glyph(cell.type);
                std::cout << std::setw(3) << glyph << ' ';
            }
        }
        std::cout << "\n";
    }
}

void print_wormhole_optimal_dram_assignment_maps(
    tt::tt_metal::IDevice* device,
    const metal_SocDescriptor& soc_desc,
    const std::vector<std::vector<CellInfo>>& grid) {
    if (device->arch() != tt::ARCH::WORMHOLE_B0 || grid.empty()) {
        return;
    }

    const uint32_t bank_count = static_cast<uint32_t>(device->num_dram_channels());
    std::vector<tt::umd::CoreCoord> dram_endpoints_noc0;
    std::vector<tt::umd::CoreCoord> dram_endpoints_noc1;
    dram_endpoints_noc0.reserve(bank_count);
    dram_endpoints_noc1.reserve(bank_count);
    for (uint32_t bank = 0; bank < bank_count; ++bank) {
        dram_endpoints_noc0.push_back(dram_bank_endpoint_noc0(soc_desc, bank, NOC::NOC_0));
        dram_endpoints_noc1.push_back(dram_bank_endpoint_noc0(soc_desc, bank, NOC::NOC_1));
    }

    std::cout << "\n=== WH optimal worker -> DRAM assignment algorithm ===\n";
    std::cout << "For each enabled Tensix worker and each DRAM bank, score = read_hops + write_hops.\n";
    std::cout << "Fixed policy:  read_hops=DRAM->worker on NOC0, write_hops=worker->DRAM on NOC1.\n";
    std::cout << "Dynamic policy: read and write independently choose the shorter NOC0/NOC1 route.\n";

    std::vector<BankBestWorkers> bank_best_workers;
    auto fixed_assignments = build_worker_to_dram_assignment_map(
        grid, dram_endpoints_noc0, dram_endpoints_noc1, D2WAssignmentPolicy::FixedNoc0ReadNoc1Write, bank_best_workers);
    print_assignment_grid(grid, fixed_assignments, "WH optimal DRAM assignment map: fixed read=NOC0, write=NOC1");
    print_bank_best_worker_summary(bank_best_workers, fixed_assignments);

    auto dynamic_assignments = build_worker_to_dram_assignment_map(
        grid, dram_endpoints_noc0, dram_endpoints_noc1, D2WAssignmentPolicy::DynamicNoc, bank_best_workers);
    print_assignment_grid(grid, dynamic_assignments, "WH optimal DRAM assignment map: dynamic read/write NOC");
    print_bank_best_worker_summary(bank_best_workers, dynamic_assignments);
    std::cout << std::flush;
}

std::vector<DramCoreWorkAssignment> build_ga_optimal_dram_core_work_assignments(
    tt::tt_metal::IDevice* device,
    const metal_SocDescriptor& soc_desc,
    const std::vector<std::vector<CellInfo>>& grid,
    uint32_t num_requested_cores = 12,
    uint32_t num_tiles = 0,
    uint32_t population_size = 128,
    uint32_t generations = 256) {
    if (device->arch() != tt::ARCH::WORMHOLE_B0 || grid.empty()) {
        return {};
    }

    const uint32_t bank_count = static_cast<uint32_t>(device->num_dram_channels());
    if (bank_count == 0) {
        throw std::runtime_error("Device reports zero DRAM banks; cannot build GA core assignments.");
    }
    if (num_requested_cores < bank_count) {
        throw std::runtime_error("GA requested core count must be at least the number of DRAM banks.");
    }

    std::vector<tt::umd::CoreCoord> workers;
    for (uint32_t y = 0; y < grid.size(); ++y) {
        for (uint32_t x = 0; x < grid[y].size(); ++x) {
            const auto& cell = grid[y][x];
            if (cell.type == CoreType::TENSIX && !cell.harvested && is_worker_row_available(y)) {
                workers.emplace_back(x, y, CoreType::TENSIX, CoordSystem::NOC0);
            }
        }
    }
    if (num_requested_cores > workers.size()) {
        throw std::runtime_error("GA requested core count exceeds available enabled Tensix cores.");
    }

    const uint32_t gx = static_cast<uint32_t>(grid[0].size());
    const uint32_t gy = static_cast<uint32_t>(grid.size());
    std::vector<tt::umd::CoreCoord> dram_endpoints_noc0;
    std::vector<tt::umd::CoreCoord> dram_endpoints_noc1;
    dram_endpoints_noc0.reserve(bank_count);
    dram_endpoints_noc1.reserve(bank_count);
    for (uint32_t bank = 0; bank < bank_count; ++bank) {
        dram_endpoints_noc0.push_back(dram_bank_endpoint_noc0(soc_desc, bank, NOC::NOC_0));
        dram_endpoints_noc1.push_back(dram_bank_endpoint_noc0(soc_desc, bank, NOC::NOC_1));
    }

    std::vector<std::vector<uint32_t>> bank_worker_score(bank_count, std::vector<uint32_t>(workers.size()));
    for (uint32_t bank = 0; bank < bank_count; ++bank) {
        for (uint32_t worker_idx = 0; worker_idx < workers.size(); ++worker_idx) {
            bank_worker_score[bank][worker_idx] = score_worker_for_bank(
                                                      workers[worker_idx],
                                                      dram_endpoints_noc0[bank],
                                                      dram_endpoints_noc1[bank],
                                                      gx,
                                                      gy,
                                                      D2WAssignmentPolicy::DynamicNoc)
                                                      .score;
        }
    }

    std::vector<uint32_t> slots_per_bank(bank_count, num_requested_cores / bank_count);
    for (uint32_t bank = 0; bank < num_requested_cores % bank_count; ++bank) {
        ++slots_per_bank[bank];
    }

    std::vector<uint32_t> slot_bank;
    slot_bank.reserve(num_requested_cores);
    const uint32_t max_slots_per_bank = *std::max_element(slots_per_bank.begin(), slots_per_bank.end());
    for (uint32_t slot = 0; slot < max_slots_per_bank; ++slot) {
        for (uint32_t bank = 0; bank < bank_count; ++bank) {
            if (slot < slots_per_bank[bank]) {
                slot_bank.push_back(bank);
            }
        }
    }

    struct Chromosome {
        std::vector<uint32_t> worker_indices;
        uint32_t score = std::numeric_limits<uint32_t>::max();
    };

    auto score_chromosome = [&](Chromosome& chromosome) {
        uint32_t score = 0;
        for (uint32_t slot = 0; slot < chromosome.worker_indices.size(); ++slot) {
            score += bank_worker_score[slot_bank[slot]][chromosome.worker_indices[slot]];
        }
        chromosome.score = score;
    };

    std::vector<uint32_t> all_worker_indices(workers.size());
    for (uint32_t i = 0; i < all_worker_indices.size(); ++i) {
        all_worker_indices[i] = i;
    }

    std::mt19937 rng(0x5eed1234);
    auto make_random_chromosome = [&]() {
        std::vector<uint32_t> worker_indices = all_worker_indices;
        std::shuffle(worker_indices.begin(), worker_indices.end(), rng);
        worker_indices.resize(num_requested_cores);
        Chromosome chromosome{std::move(worker_indices)};
        score_chromosome(chromosome);
        return chromosome;
    };

    auto make_greedy_chromosome = [&]() {
        std::vector<uint32_t> worker_indices;
        worker_indices.reserve(num_requested_cores);
        std::vector<uint8_t> worker_used(workers.size(), 0);
        for (uint32_t slot = 0; slot < num_requested_cores; ++slot) {
            const uint32_t bank = slot_bank[slot];
            uint32_t best_worker_idx = std::numeric_limits<uint32_t>::max();
            uint32_t best_score = std::numeric_limits<uint32_t>::max();
            for (uint32_t worker_idx = 0; worker_idx < workers.size(); ++worker_idx) {
                if (worker_used[worker_idx]) {
                    continue;
                }
                if (bank_worker_score[bank][worker_idx] < best_score) {
                    best_score = bank_worker_score[bank][worker_idx];
                    best_worker_idx = worker_idx;
                }
            }
            if (best_worker_idx == std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("GA greedy seed failed to find an unused worker.");
            }
            worker_used[best_worker_idx] = 1;
            worker_indices.push_back(best_worker_idx);
        }
        Chromosome chromosome{std::move(worker_indices)};
        score_chromosome(chromosome);
        return chromosome;
    };

    population_size = std::max<uint32_t>(population_size, 4);
    generations = std::max<uint32_t>(generations, 1);

    std::vector<Chromosome> population;
    population.reserve(population_size);
    population.push_back(make_greedy_chromosome());
    while (population.size() < population_size) {
        population.push_back(make_random_chromosome());
    }

    auto tournament_select = [&]() -> const Chromosome& {
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(population.size() - 1));
        const Chromosome* best = &population[dist(rng)];
        for (uint32_t i = 1; i < 4; ++i) {
            const Chromosome& candidate = population[dist(rng)];
            if (candidate.score < best->score) {
                best = &candidate;
            }
        }
        return *best;
    };

    auto crossover = [&](const Chromosome& a, const Chromosome& b) {
        std::uniform_int_distribution<uint32_t> dist(0, num_requested_cores - 1);
        uint32_t begin = dist(rng);
        uint32_t end = dist(rng);
        if (begin > end) {
            std::swap(begin, end);
        }

        std::vector<uint32_t> child(num_requested_cores, std::numeric_limits<uint32_t>::max());
        std::vector<uint8_t> used(workers.size(), 0);
        for (uint32_t i = begin; i <= end; ++i) {
            child[i] = a.worker_indices[i];
            used[child[i]] = 1;
        }

        uint32_t out = (end + 1) % num_requested_cores;
        for (uint32_t i = 0; i < num_requested_cores; ++i) {
            uint32_t candidate = b.worker_indices[(end + 1 + i) % num_requested_cores];
            if (used[candidate]) {
                continue;
            }
            child[out] = candidate;
            used[candidate] = 1;
            out = (out + 1) % num_requested_cores;
        }

        Chromosome chromosome{std::move(child)};
        score_chromosome(chromosome);
        return chromosome;
    };

    auto mutate = [&](Chromosome& chromosome) {
        std::uniform_real_distribution<float> probability(0.0F, 1.0F);
        if (probability(rng) >= 0.15F || num_requested_cores < 2) {
            return;
        }

        std::uniform_int_distribution<uint32_t> dist(0, num_requested_cores - 1);
        uint32_t a = dist(rng);
        uint32_t b = dist(rng);
        if (a != b) {
            std::swap(chromosome.worker_indices[a], chromosome.worker_indices[b]);
            score_chromosome(chromosome);
        }
    };

    for (uint32_t generation = 0; generation < generations; ++generation) {
        std::sort(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) {
            return a.score < b.score;
        });

        std::vector<Chromosome> next_population;
        next_population.reserve(population_size);
        next_population.push_back(population[0]);
        next_population.push_back(population[1]);
        while (next_population.size() < population_size) {
            Chromosome child = crossover(tournament_select(), tournament_select());
            mutate(child);
            next_population.push_back(std::move(child));
        }
        population = std::move(next_population);
    }

    const auto& best =
        *std::min_element(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) {
            return a.score < b.score;
        });

    std::vector<uint32_t> next_slot_for_bank(bank_count, 0);
    std::vector<DramCoreWorkAssignment> assignments;
    assignments.reserve(num_requested_cores);
    for (uint32_t slot_idx = 0; slot_idx < num_requested_cores; ++slot_idx) {
        const uint32_t bank = slot_bank[slot_idx];
        const uint32_t slot = next_slot_for_bank[bank]++;
        const uint32_t count_b = (num_tiles > bank) ? (num_tiles - bank + bank_count - 1) / bank_count : 0u;
        const uint32_t base = count_b / slots_per_bank[bank];
        const uint32_t rem = count_b % slots_per_bank[bank];
        const uint32_t num_tiles_per_core = base + (slot < rem ? 1u : 0u);
        const uint32_t first_j = (slot < rem) ? slot * (base + 1) : rem * (base + 1) + (slot - rem) * base;
        const uint32_t tile_ofs = bank + first_j * bank_count;

        assignments.push_back(
            {workers[best.worker_indices[slot_idx]],
             bank,
             slot,
             bank_worker_score[bank][best.worker_indices[slot_idx]],
             tile_ofs,
             num_tiles_per_core});
    }

    return assignments;
}

void print_ga_optimal_dram_core_work_assignments(
    tt::tt_metal::IDevice* device,
    const metal_SocDescriptor& soc_desc,
    const std::vector<std::vector<CellInfo>>& grid,
    uint32_t num_requested_cores = 12,
    uint32_t num_tiles = 0) {
    auto assignments =
        build_ga_optimal_dram_core_work_assignments(device, soc_desc, grid, num_requested_cores, num_tiles);
    if (assignments.empty()) {
        return;
    }

    uint32_t total_score = 0;
    for (const auto& assignment : assignments) {
        total_score += assignment.score;
    }

    std::cout << "\n=== GA optimal DRAM core work assignments ===\n";
    std::cout << "Requested cores: " << num_requested_cores << ", num_tiles: " << num_tiles
              << ", total dynamic score: " << total_score << "\n";
    std::cout << std::left << std::setw(6) << "idx" << std::setw(8) << "bank" << std::setw(8) << "slot" << std::setw(12)
              << "score" << std::setw(16) << "core_NOC0" << std::setw(12) << "tile_ofs"
              << "num_tiles\n";
    for (uint32_t i = 0; i < assignments.size(); ++i) {
        const auto& assignment = assignments[i];
        std::cout << std::left << std::setw(6) << i << std::setw(8) << bank_label(assignment.dram_bank) << std::setw(8)
                  << assignment.slot << std::setw(12) << assignment.score << std::setw(16)
                  << fmt_yx(assignment.core.y, assignment.core.x) << std::setw(12) << assignment.tile_ofs
                  << assignment.num_tiles << "\n";
    }
    std::cout << std::flush;
}

size_t verify_against_wormhole_b0(const std::vector<std::vector<CellInfo>>& grid) {
    std::cout << "\n=== Verification against expected Wormhole B0 NOC0 map ===\n";
    if (grid.size() != kExpectedWormholeB0Grid.size() ||
        (!grid.empty() && grid[0].size() != kExpectedWormholeB0Grid[0].size())) {
        std::cout << "GRID SIZE MISMATCH: observed=" << grid.size() << "x" << (grid.empty() ? 0 : grid[0].size())
                  << " expected=12x10\n";
        return static_cast<size_t>(-1);
    }

    size_t mismatches = 0;
    for (size_t y = 0; y < grid.size(); ++y) {
        for (size_t x = 0; x < grid[y].size(); ++x) {
            const char expected = kExpectedWormholeB0Grid[y][x];
            const char observed = bucket_glyph(grid[y][x].type);
            if (expected != observed) {
                std::cout << "  mismatch at (" << y << "," << x << "): expected '" << expected << "' observed '"
                          << observed << "' (" << core_type_name(grid[y][x].type) << ")\n";
                ++mismatches;
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "OK: every physical NOC0 cell matches the expected Wormhole B0 layout.\n";
    } else {
        std::cout << "FAIL: " << mismatches << " cell(s) did not match the expected WH B0 layout.\n";
    }
    return mismatches;
}

void print_core_counts(const std::vector<std::vector<CellInfo>>& grid) {
    std::map<std::string, size_t> counts;
    size_t harvested = 0;
    for (const auto& row : grid) {
        for (const auto& c : row) {
            if (c.type == CoreType::COUNT) {
                continue;
            }
            counts[core_type_name(c.type)]++;
            if (c.harvested) {
                ++harvested;
            }
        }
    }

    std::cout << "\n=== Core counts (from SocDescriptor enumeration) ===\n";
    for (const auto& [name, n] : counts) {
        std::cout << "  " << std::left << std::setw(14) << name << n << "\n";
    }
    std::cout << "  (of which tensix tiles that are harvested on this board: " << harvested << ")\n";
    std::cout << std::flush;
}

}  // namespace

int main() {
    try {
        constexpr int device_id = 0;
        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        tt::tt_metal::IDevice* device = mesh_device->get_devices().front();

        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        const uint32_t harvest_mask = cluster.get_harvesting_mask(device->id());

        const auto grid_size = device->grid_size();
        const auto logical_grid = device->logical_grid_size();
        const auto cws_grid = device->compute_with_storage_grid_size();
        const auto dram_grid = device->dram_grid_size();

        std::cout << "=== Device / chip summary (device_id=" << device->id() << ") ===\n";
        std::cout << "  arch:                         " << arch_name(device->arch()) << "\n";
        std::cout << "  physical NOC grid size:       " << grid_size.x << " x " << grid_size.y << "\n";
        std::cout << "  logical (enabled) tensix:     " << logical_grid.x << " x " << logical_grid.y << "\n";
        std::cout << "  compute_with_storage_grid:    " << cws_grid.x << " x " << cws_grid.y << "\n";
        std::cout << "  dram grid (views x ports):    " << dram_grid.x << " x " << dram_grid.y << "\n";
        std::cout << "  num_dram_channels (banks):    " << device->num_dram_channels() << "\n";
        std::cout << "  soc num_dram_views:           " << soc_desc.get_num_dram_views() << "\n";
        std::cout << "  soc num_eth_channels:         " << soc_desc.get_num_eth_channels() << "\n";
        std::cout << "  tensix_harvesting_mask:       0x" << std::hex << harvest_mask << std::dec << "\n";
        std::cout << std::flush;

        auto grid = build_observed_grid(soc_desc);
        print_grid(grid);
        print_core_counts(grid);
        print_per_core_table(grid);

        print_dram_worker_assignment(device, soc_desc, NOC::NOC_0);
        print_dram_worker_assignment(device, soc_desc, NOC::NOC_1);
        print_wormhole_dram_hop_maps(device, soc_desc, grid, NOC::NOC_0);
        print_wormhole_dram_hop_maps(device, soc_desc, grid, NOC::NOC_1);
        print_wormhole_optimal_dram_assignment_maps(device, soc_desc, grid);
        print_ga_optimal_dram_core_work_assignments(device, soc_desc, grid);

        size_t fail_count = 0;
        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            size_t m = verify_against_wormhole_b0(grid);
            if (m != 0) {
                fail_count += (m == static_cast<size_t>(-1) ? 1 : m);
            }
        } else {
            std::cout << "\n(Verification against hard-coded map is only implemented for WORMHOLE_B0; skipping.)\n";
        }

        mesh_device.reset();
        return fail_count == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "core_map: caught exception: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "core_map: caught unknown exception\n";
        return 2;
    }
}
