// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone tool that uses the tt-metal device API + SocDescriptor to
// enumerate every core on a single chip, prints a physical-NOC0 map, and
// verifies the observed map against a hard-coded Wormhole B0 layout.

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
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

constexpr char kGlyphEmpty = '.';

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
