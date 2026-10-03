// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/tile_sync.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <map>
#include <optional>
#include <set>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/device/device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

int64_t TileClocks::offset(CoreType type, const CoreCoord& logical) const {
    const auto it = std::ranges::find_if(
        tiles, [&](const TileClock& tile) { return tile.type == type && tile.logical == logical; });
    TT_FATAL(
        it != tiles.end(), "streaming profiler: tile ({},{}) is not in its chip's tile clocks", logical.x, logical.y);
    return it->offset;
}

namespace {

// A reader's reads of one partner over one NoC, as its kernel posts them in a TileNetPartner. doubled_offset() widens
// doubled_median, which holds only the clocks' low words, by coarse.
struct Reading {
    uint32_t partner, noc;
    // On the both-NoC reader's extra reads, the index of its read of the same partner over the other NoC.
    std::optional<uint32_t> other_noc_read;
    int32_t doubled_median = 0;
    int64_t coarse = 0;
    int64_t doubled_offset() const {
        return 2 * coarse +
               static_cast<int32_t>(static_cast<uint32_t>(doubled_median) - static_cast<uint32_t>(2 * coarse));
    }
};

// A mirrored pair's doubled offsets differ by, and a both-NoC pair's add up to, four times the tiles' offset.
constexpr double kDoubledPairPerTick = 4.0;

struct Tile : CoreCoords {
    CoreType type;
    HalProgrammableCoreType core;
    uint32_t scratch = 0;
    uint64_t host_scratch = 0;
    std::vector<Reading> reads;
};

// NoC 0 runs towards higher raw coordinates, NoC 1 towards lower.
bool same_row_or_column(const CoreCoord& reader, const CoreCoord& partner) {
    return reader != partner && (reader.x == partner.x || reader.y == partner.y);
}
bool upward(const CoreCoord& reader, const CoreCoord& partner) {
    return reader.x == partner.x ? partner.y > reader.y : partner.x > reader.x;
}

std::vector<Tile> plan_tiles(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Tile> tiles;
    const auto add = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical, uint32_t scratch) {
        const uint64_t host_scratch = core == HalProgrammableCoreType::DRAM
                                          ? hal.get_dev_noc_addr(core, HalL1MemAddrType::UNRESERVED) +
                                                (scratch - hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED))
                                          : scratch;
        tiles.push_back(Tile{{locate(cluster, chip, logical, type)}, type, core, scratch, host_scratch});
    };
    const auto add_unreserved = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical) {
        add(type, core, logical, hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED));
    };
    // A compute tile's profiler ring can't be its scratch, even zeroed afterwards: the first frames depend on its
    // contents.
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    TT_FATAL(
        hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER) >=
            kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + sizeof(kernel_profiler::TileNetScratch),
        "streaming profiler: a profiler L1 region cannot hold the tile clock scratch");
    const uint32_t dispatch_scratch =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER)) +
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE;
    const CoreCoord compute = device->compute_with_storage_grid_size();
    const CoreCoord grid = soc.get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const bool is_compute = x < compute.x && y < compute.y;
            add(CoreType::WORKER, HalProgrammableCoreType::TENSIX, {x, y}, is_compute ? user_l1 : dispatch_scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        for (const CoreCoord& logical : sorted_yx(device->get_inactive_ethernet_cores())) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::IDLE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        for (const CoreCoord& logical :
             sorted_yx(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::ACTIVE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // The cluster gives a DRAM core's translated coordinate as its physical one. The NoC 0 grid position, which
        // says which Tensix row it shares, comes from the SoC descriptor in the same order.
        const std::vector<CoreCoord> logical = soc.get_metal_dram_cores(CoordSystem::LOGICAL);
        const std::vector<CoreCoord> noc0 = soc.get_metal_dram_cores(CoordSystem::NOC0);
        for (size_t i = 0; i < logical.size(); i++) {
            if (dram_view_endpoint_noc_mask(soc, logical[i]) == 0) {
                add_unreserved(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical[i]);
                tiles.back().phys = noc0[i];
            }
        }
    }
    // One idle eth tile also reads every eth tile in its row over the other NoC. Around the ring the two NoCs' hops
    // cancel, and the initiator's own latency is the same for every target.
    const auto both_noc_reader =
        static_cast<uint32_t>(std::ranges::find(tiles, HalProgrammableCoreType::IDLE_ETH, &Tile::core) - tiles.begin());
    for (uint32_t r = 0; r < tiles.size(); r++) {
        Tile& reader = tiles[r];
        for (uint32_t partner = 0; partner < tiles.size(); partner++) {
            if (same_row_or_column(reader.phys, tiles[partner].phys)) {
                reader.reads.push_back(
                    Reading{.partner = partner, .noc = upward(reader.phys, tiles[partner].phys) ? 0u : 1u});
            }
        }
        if (r == both_noc_reader) {
            for (uint32_t i = 0, direct_reads = static_cast<uint32_t>(reader.reads.size()); i < direct_reads; i++) {
                const Reading& direct = reader.reads[i];
                if (tiles[direct.partner].type == CoreType::ETH) {
                    reader.reads.push_back(
                        Reading{.partner = direct.partner, .noc = direct.noc ^ 1u, .other_noc_read = i});
                }
            }
        }
        TT_FATAL(
            reader.reads.size() <= kernel_profiler::kTileNetMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            reader.logical.x,
            reader.logical.y,
            reader.reads.size(),
            kernel_profiler::kTileNetMaxPartners);
    }
    return tiles;
}

KernelHandle create_tile_kernel(Program& program, HalProgrammableCoreType core, const CoreRangeSet& cores) {
    const char* src = "tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp";
    switch (core) {
        case HalProgrammableCoreType::TENSIX:
            return CreateKernel(
                program,
                src,
                cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::IDLE_ETH:
            return CreateKernel(
                program, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::ACTIVE_ETH:
            return CreateKernel(program, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::DRAM: return CreateKernel(program, src, cores, DramConfig{.noc = NOC::NOC_0});
        case HalProgrammableCoreType::DISPATCH:
        case HalProgrammableCoreType::COUNT: break;
    }
    TT_THROW("Unreachable");
}

// One chip's tiles, each with its tile kernel resident. If a later step throws, the kernels stay up waiting on their go
// words.
struct Network {
    IDevice* device = nullptr;
    uint32_t chip = 0;
    std::vector<Tile> tiles;
    struct Kind {
        std::set<CoreRange> cores;
        std::optional<Program> program;
        KernelHandle kernel = 0;
    };
    std::map<HalProgrammableCoreType, Kind> kinds;
};

uint64_t table_of(const Tile& tile) { return tile.host_scratch + offsetof(kernel_profiler::TileNetScratch, table); }

Network launch_network(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    Network net{.device = device, .chip = static_cast<uint32_t>(device->id()), .tiles = plan_tiles(device, ctx)};
    for (const Tile& tile : net.tiles) {
        net.kinds[tile.core].cores.insert(CoreRange(tile.logical, tile.logical));
    }
    for (auto& [core, kind] : net.kinds) {
        kind.program = CreateProgram();
        kind.kernel = create_tile_kernel(*kind.program, core, CoreRangeSet(kind.cores));
    }
    // Firmware takes its ring position from the control vector at the session's first launch (this one), so a stale
    // tail would put it thousands of words ahead.
    for (const Tile& tile : net.tiles) {
        std::vector<uint32_t> args{tile.scratch, static_cast<uint32_t>(tile.reads.size())};
        for (const Reading& read : tile.reads) {
            const CoreCoord& partner = net.tiles[read.partner].virt;
            args.push_back(kernel_profiler::word_of(kernel_profiler::TileNetRead{
                .x = static_cast<uint32_t>(partner.x), .y = static_cast<uint32_t>(partner.y), .noc = read.noc}));
        }
        const Network::Kind& kind = net.kinds[tile.core];
        SetRuntimeArgs(*kind.program, kind.kernel, tile.logical, args);
        zero_l1(cluster, net.chip, tile.virt, table_of(tile), 2 * sizeof(uint32_t));
        zero_profiler_control(cluster, net.chip, tile.virt, host_l1_addr(hal, tile.core, HalL1MemAddrType::PROFILER));
    }
    for (auto& [core, kind] : net.kinds) {
        launch_resident(device, *kind.program);
    }
    return net;
}

kernel_profiler::TileNetTable await_table(
    tt::Cluster& cluster, const Network& net, const Tile& tile, kernel_profiler::TileNetReady ready) {
    kernel_profiler::TileNetTable table{};
    const auto table_bytes = static_cast<uint32_t>(
        offsetof(kernel_profiler::TileNetTable, partner) + tile.reads.size() * sizeof(kernel_profiler::TileNetPartner));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do {
        cluster.read_core(&table, table_bytes, tt_cxy_pair(net.chip, tile.virt), table_of(tile));
        TT_FATAL(
            table.ready == ready || std::chrono::steady_clock::now() < deadline,
            "streaming profiler: device {} tile {} did not post its tile clock table",
            net.chip,
            tile.virt.str());
    } while (table.ready != ready);
    return table;
}

void start_tile_reads(tt::Cluster& cluster, const Network& net, const Tile& tile) {
    await_table(cluster, net, tile, kernel_profiler::TileNetReady::Up);
    const kernel_profiler::TileNetGo go_measure = kernel_profiler::TileNetGo::Measure;
    cluster.write_core(&go_measure, sizeof(go_measure), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
}

void collect_tile_reads(tt::Cluster& cluster, const Network& net, Tile& tile) {
    const kernel_profiler::TileNetTable table = await_table(cluster, net, tile, kernel_profiler::TileNetReady::Done);
    for (size_t i = 0; i < tile.reads.size(); i++) {
        tile.reads[i].doubled_median = table.partner[i].doubled_median;
        tile.reads[i].coarse = table.partner[i].coarse;
    }
}

void retire_network(ContextId ctx, Network& net) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    for (const Tile& tile : net.tiles) {
        const kernel_profiler::TileNetGo go_exit = kernel_profiler::TileNetGo::Exit;
        cluster.write_core(&go_exit, sizeof(go_exit), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
    }
    for (auto& [core, kind] : net.kinds) {
        detail::WaitProgramDone(net.device, *kind.program, false);
    }
    // Fast dispatch's go signal reaches every core and a host-launched kernel leaves its slot valid, so restore the
    // firmware's initial launch message.
    for (const Tile& tile : net.tiles) {
        auto msg = hal.get_dev_msgs_factory(tile.core).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(net.chip, tile.virt),
            host_l1_addr(hal, tile.core, HalL1MemAddrType::LAUNCH));
    }
}

bool is_active_eth(const Tile& tile) { return tile.core == HalProgrammableCoreType::ACTIVE_ETH; }

// plan_tiles lists the Tensix grid first, so tile 0 is its first Tensix tile, which every other tile is placed from.
constexpr uint32_t kGroundTile = 0;

// Every tile but the active eth ones, from the mirrored pairs, as ticks from the first Tensix tile. An active eth
// tile's reads leave its NIU about 6 cycles later than an idle eth tile's, which would bias a mirrored pair with one by
// a tick and a half.
std::vector<double> place_mirrored(const std::vector<Tile>& tiles, uint32_t chip) {
    const auto tile_count = static_cast<uint32_t>(tiles.size());
    std::vector<std::optional<size_t>> unknown_of(tile_count);
    std::vector<uint32_t> tile_of_unknown;
    for (uint32_t i = 0; i < tile_count; i++) {
        if (i != kGroundTile && !is_active_eth(tiles[i])) {
            unknown_of[i] = tile_of_unknown.size();
            tile_of_unknown.push_back(i);
        }
    }
    struct MirroredPair {
        std::optional<size_t> sender, receiver;
        double offset;
    };
    std::vector<MirroredPair> pairs;
    for (uint32_t reader = 0; reader < tile_count; reader++) {
        for (const Reading& there : tiles[reader].reads) {
            if (there.other_noc_read || there.noc != 0 || is_active_eth(tiles[reader]) ||
                is_active_eth(tiles[there.partner])) {
                continue;
            }
            const Reading& back = *std::ranges::find(tiles[there.partner].reads, reader, &Reading::partner);
            pairs.push_back(
                {.sender = unknown_of[there.partner],
                 .receiver = unknown_of[reader],
                 .offset = static_cast<double>(there.doubled_offset() - back.doubled_offset()) / kDoubledPairPerTick});
        }
    }
    const Potential potential = solve_potential(
        pairs, &MirroredPair::sender, &MirroredPair::receiver, &MirroredPair::offset, tile_of_unknown.size());
    if (potential.unreached) {
        const Tile& tile = tiles[tile_of_unknown[*potential.unreached]];
        TT_THROW(
            "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
            chip,
            tile.type == CoreType::WORKER ? "Tensix"
            : tile.type == CoreType::DRAM ? "DRAM"
                                          : "eth",
            tile.logical.x,
            tile.logical.y);
    }
    std::vector<double> ticks(tile_count, 0.0);
    for (size_t u = 0; u < tile_of_unknown.size(); u++) {
        ticks[tile_of_unknown[u]] = potential.x[u];
    }
    return ticks;
}

// Each active eth tile from its both-NoC reading, shifted onto the mirrored placement by the idle eth tiles that have
// both-NoC readings too.
void place_active_eth(const std::vector<Tile>& tiles, uint32_t chip, std::vector<double>& ticks) {
    std::vector<std::optional<double>> estimate(tiles.size());
    double shift_sum = 0.0;
    size_t reference_count = 0;
    for (const Tile& reader : tiles) {
        for (const Reading& read : reader.reads) {
            if (!read.other_noc_read) {
                continue;
            }
            estimate[read.partner] =
                static_cast<double>(reader.reads[*read.other_noc_read].doubled_offset() + read.doubled_offset()) /
                kDoubledPairPerTick;
            if (!is_active_eth(tiles[read.partner])) {
                shift_sum += ticks[read.partner] - *estimate[read.partner];
                reference_count++;
            }
        }
    }
    for (uint32_t t = 0; t < tiles.size(); t++) {
        if (!is_active_eth(tiles[t])) {
            continue;
        }
        TT_FATAL(
            reference_count != 0 && estimate[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            tiles[t].logical.x,
            tiles[t].logical.y);
        ticks[t] = shift_sum / static_cast<double>(reference_count) + *estimate[t];
    }
}

}  // namespace

void measure_tile_clocks(std::span<Device* const> devices, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    if (!can_capture(mc)) {
        return;
    }
    auto& cluster = mc.get_cluster();
    std::vector<std::future<Network>> launches;
    for (Device* device : devices) {
        if (service().tile_clocks(ctx, static_cast<uint32_t>(device->id())) == nullptr) {
            launches.push_back(std::async(std::launch::async, launch_network, device, ctx));
        }
    }
    std::vector<Network> nets;
    size_t most_tiles = 0;
    for (auto& launch : launches) {
        nets.push_back(launch.get());
        most_tiles = std::max(most_tiles, nets.back().tiles.size());
    }
    // Each chip lets one tile read at a time; the chips have their own NoCs, so they go in step.
    for (size_t t = 0; t < most_tiles; t++) {
        for (const Network& net : nets) {
            if (t < net.tiles.size()) {
                start_tile_reads(cluster, net, net.tiles[t]);
            }
        }
        for (Network& net : nets) {
            if (t < net.tiles.size()) {
                collect_tile_reads(cluster, net, net.tiles[t]);
            }
        }
    }
    for (Network& net : nets) {
        retire_network(ctx, net);
        std::vector<double> ticks = place_mirrored(net.tiles, net.chip);
        place_active_eth(net.tiles, net.chip, ticks);
        TileClocks clocks;
        for (size_t i = 0; i < net.tiles.size(); i++) {
            clocks.tiles.push_back(TileClock{
                .type = net.tiles[i].type, .logical = net.tiles[i].logical, .offset = round_nearest(ticks[i])});
        }
        service().set_tile_clocks(ctx, net.chip, std::move(clocks));
    }
}

}  // namespace tt::tt_metal::streaming_profiler
