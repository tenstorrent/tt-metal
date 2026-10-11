// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/tile_sync.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>

#include <enchantum/enchantum.hpp>
#include <fmt/chrono.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/indestructible.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/device/device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "impl/streaming_profiler/sync/link_sync.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

constexpr auto kTileTimeout = std::chrono::seconds(10);

// A reader's readings of one partner over one NoC, as its kernel posts them in a TileSyncPartner.
struct Reading {
    uint32_t partner, noc;
    // The both-NoC reader is the one idle eth tile that also reads its row's eth tiles over the other NoC. On those
    // extra reads, this is the index of its read of the same partner over the other NoC. It is unset on every other
    // read.
    std::optional<uint32_t> other_noc_read;
    int64_t doubled_offset = 0;
};

// A pair's measurement is in quarter ticks, four to a tick. Readings are doubled, because each is taken from the
// midpoint of two clock reads, which can fall on a half tick. In a mirrored pair, two tiles read each other over
// opposite NoCs, and their two readings differ by four times the tiles' clock offset. The both-NoC reader's two
// readings of one tile add up to four times the offset.
constexpr int64_t kPairUnitsPerTick = 4;

// Converts `pair_units`, a pair's measurement in quarter ticks, to ticks past `base_ticks`.
constexpr double ticks_past(int64_t pair_units, int64_t base_ticks) {
    return static_cast<double>(pair_units - kPairUnitsPerTick * base_ticks) / kPairUnitsPerTick;
}

// A tile's offset from the ground tile, as a whole-tick base plus the fitted ticks from it. An eth or DRAM tile's clock
// keeps counting while the Tensix clocks halt between sessions, so its offset can reach about 1e14 ticks. A
// least-squares fit in doubles over values that large loses tenths of a tick to cancellation, so only the small
// remainder goes through the fit.
struct Placement {
    int64_t base = 0;
    double from_base = 0.0;
    int64_t offset() const { return base + round_nearest(from_base); }
};

struct Tile : CoreCoords {
    CoreType core_type;
    HalProgrammableCoreType hal_type;
    uint32_t scratch = 0;
    uint64_t host_scratch = 0;
    std::vector<Reading> reads;
};

bool same_row_or_column(const CoreCoord& reader, const CoreCoord& partner) {
    return reader != partner && (reader.x == partner.x || reader.y == partner.y);
}
// Returns whether `partner` is at a higher raw coordinate than `reader`, along their shared row or column. NoC 0 runs
// towards higher raw coordinates, and NoC 1 towards lower ones.
bool upward(const CoreCoord& reader, const CoreCoord& partner) {
    return reader.x == partner.x ? partner.y > reader.y : partner.x > reader.x;
}

std::vector<Tile> plan_tiles(IDevice* device, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Tile> tiles;
    const auto add =
        [&](CoreType core_type, HalProgrammableCoreType hal_type, const CoreCoord& logical, uint32_t scratch) {
            tiles.push_back(Tile{
                {locate_core(cluster, chip, logical, core_type)},
                core_type,
                hal_type,
                scratch,
                scratch + hal.get_l1_noc_offset(hal_type)});
        };
    const auto add_unreserved = [&](CoreType core_type, HalProgrammableCoreType hal_type, const CoreCoord& logical) {
        add(core_type, hal_type, logical, hal.get_dev_addr(hal_type, HalL1MemAddrType::UNRESERVED));
    };
    // A compute tile's scratch goes in user L1 rather than in its profiler ring. The relays drain the compute tiles'
    // rings, and their first frames broke when a ring had held the scratch, even zeroed afterwards. Nothing drains a
    // dispatch tile's ring, so its scratch can sit just past its profiler control buffer.
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
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
        // Firmware keeps DRAM view endpoints' NIUs in NOC2AXI mode, where a read goes to GDDR, so they're skipped.
        for (const CoreCoord& logical : soc.get_metal_dram_cores(CoordSystem::LOGICAL)) {
            if (soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical)) == 0) {
                add_unreserved(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical);
            }
        }
    }
    // One idle eth tile also reads every eth tile in its row over the other NoC. The NoCs circle the row in opposite
    // directions, so the hop delays cancel when a tile's two readings are added, and the reader's latency is the same
    // for every tile.
    const auto both_noc_reader = static_cast<uint32_t>(
        std::ranges::find(tiles, HalProgrammableCoreType::IDLE_ETH, &Tile::hal_type) - tiles.begin());
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
                if (tiles[direct.partner].core_type == CoreType::ETH) {
                    reader.reads.push_back(
                        Reading{.partner = direct.partner, .noc = direct.noc ^ 1u, .other_noc_read = i});
                }
            }
        }
        TT_FATAL(
            reader.reads.size() <= kernel_profiler::kTileSyncMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            reader.logical.x,
            reader.logical.y,
            reader.reads.size(),
            kernel_profiler::kTileSyncMaxPartners);
    }
    return tiles;
}

KernelHandle create_tile_kernel(Program& program, HalProgrammableCoreType hal_type, const CoreRangeSet& cores) {
    const char* src = "tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp";
    switch (hal_type) {
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

// One chip's tiles, each running its tile kernel.
struct Network {
    IDevice* device = nullptr;
    uint32_t chip = 0;
    std::vector<Tile> tiles;
    struct Kind {
        std::set<CoreRange> cores;
        Program program;
        KernelHandle kernel = 0;
    };
    std::map<HalProgrammableCoreType, Kind> kinds;
};

uint64_t table_of(const Tile& tile) { return tile.host_scratch + offsetof(kernel_profiler::TileSyncScratch, table); }

Network launch_network(IDevice* device, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    Network net{.device = device, .chip = static_cast<uint32_t>(device->id()), .tiles = plan_tiles(device, context_id)};
    for (const Tile& tile : net.tiles) {
        net.kinds[tile.hal_type].cores.insert(CoreRange(tile.logical, tile.logical));
    }
    for (auto& [hal_type, kind] : net.kinds) {
        kind.kernel = create_tile_kernel(kind.program, hal_type, CoreRangeSet(kind.cores));
    }
    for (const Tile& tile : net.tiles) {
        std::vector<uint32_t> args{tile.scratch, static_cast<uint32_t>(tile.reads.size())};
        for (const Reading& read : tile.reads) {
            const CoreCoord& partner = net.tiles[read.partner].virt;
            args.push_back(kernel_profiler::word_of(kernel_profiler::TileSyncRead{
                .x = static_cast<uint32_t>(partner.x), .y = static_cast<uint32_t>(partner.y), .noc = read.noc}));
        }
        const Network::Kind& kind = net.kinds[tile.hal_type];
        SetRuntimeArgs(kind.program, kind.kernel, tile.logical, args);
        zero_l1(cluster, net.chip, tile.virt, table_of(tile), offsetof(kernel_profiler::TileSyncTable, partner));
        // The firmware reads its profiler ring position from the control vector only at the first launch after it
        // loads, which is this one, so a tail left by an earlier process would start the ring thousands of words ahead.
        zero_profiler_control(
            cluster, net.chip, tile.virt, hal.get_dev_noc_addr(tile.hal_type, HalL1MemAddrType::PROFILER));
    }
    for (auto& [hal_type, kind] : net.kinds) {
        launch_resident(device, kind.program);
    }
    return net;
}

kernel_profiler::TileSyncTable await_table(
    tt::Cluster& cluster, const Network& net, const Tile& tile, kernel_profiler::TileSyncReady ready) {
    kernel_profiler::TileSyncTable table{};
    const auto table_bytes = static_cast<uint32_t>(
        offsetof(kernel_profiler::TileSyncTable, partner) +
        tile.reads.size() * sizeof(kernel_profiler::TileSyncPartner));
    const auto deadline = std::chrono::steady_clock::now() + kTileTimeout;
    do {
        TT_FATAL(
            std::chrono::steady_clock::now() < deadline,
            "streaming profiler: chip {} tile {} did not reach {} within {}",
            net.chip,
            tile.virt.str(),
            enchantum::to_string(ready),
            kTileTimeout);
        cluster.read_core(&table, table_bytes, tt_cxy_pair(net.chip, tile.virt), table_of(tile));
    } while (table.ready != ready);
    return table;
}

void start_tile_reads(tt::Cluster& cluster, const Network& net, const Tile& tile) {
    await_table(cluster, net, tile, kernel_profiler::TileSyncReady::Up);
    const kernel_profiler::TileSyncGo go_measure = kernel_profiler::TileSyncGo::Measure;
    cluster.write_core(&go_measure, sizeof(go_measure), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
}

void collect_tile_reads(tt::Cluster& cluster, const Network& net, Tile& tile) {
    const kernel_profiler::TileSyncTable table = await_table(cluster, net, tile, kernel_profiler::TileSyncReady::Done);
    for (size_t i = 0; i < tile.reads.size(); i++) {
        // The kernel's median uses only the clocks' low words, so its high bits come from whole_difference.
        tile.reads[i].doubled_offset =
            widen(2 * table.partner[i].whole_difference, static_cast<uint32_t>(table.partner[i].doubled_median));
    }
}

void retire_network(ContextId context_id, Network& net) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    for (const Tile& tile : net.tiles) {
        const kernel_profiler::TileSyncGo go_exit = kernel_profiler::TileSyncGo::Exit;
        cluster.write_core(&go_exit, sizeof(go_exit), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
    }
    for (auto& [hal_type, kind] : net.kinds) {
        slow_dispatch::WaitProgramDone(*net.device, kind.program);
    }
    // The tile kernel's launch message stays valid, and fast dispatch's go signal reaches every core, so the firmware's
    // initial launch message is restored to keep the go signal from rerunning the tile kernel.
    for (const Tile& tile : net.tiles) {
        auto msg = hal.get_dev_msgs_factory(tile.hal_type).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(net.chip, tile.virt),
            hal.get_dev_noc_addr(tile.hal_type, HalL1MemAddrType::LAUNCH));
    }
}

bool is_active_eth(const Tile& tile) { return tile.hal_type == HalProgrammableCoreType::ACTIVE_ETH; }

// The tile every other tile is placed relative to. plan_tiles lists the Tensix grid first, so tile 0 is the first
// Tensix tile.
constexpr uint32_t kGroundTile = 0;

// Places every tile except the active eth ones from the mirrored pairs. Active eth tiles are left out because their
// reads leave the NIU about 6 cycles later than an idle eth tile's, which would bias a mirrored pair by a tick and a
// half.
std::vector<Placement> place_mirrored(const std::vector<Tile>& tiles, uint32_t chip) {
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
        uint32_t reader, partner;
        int64_t doubled_difference;
        std::optional<size_t> partner_unknown, reader_unknown;
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
                {.reader = reader,
                 .partner = there.partner,
                 .doubled_difference = there.doubled_offset - back.doubled_offset,
                 .partner_unknown = unknown_of[there.partner],
                 .reader_unknown = unknown_of[reader]});
        }
    }
    std::vector<std::optional<int64_t>> base(tile_count);
    base[kGroundTile] = 0;
    for (bool grew = true; grew;) {
        grew = false;
        for (const MirroredPair& pair : pairs) {
            const int64_t whole = pair.doubled_difference / kPairUnitsPerTick;
            if (base[pair.reader] && !base[pair.partner]) {
                base[pair.partner] = *base[pair.reader] + whole;
                grew = true;
            } else if (base[pair.partner] && !base[pair.reader]) {
                base[pair.reader] = *base[pair.partner] - whole;
                grew = true;
            }
        }
    }
    for (const uint32_t t : tile_of_unknown) {
        TT_FATAL(
            base[t],
            "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
            chip,
            tt::to_str(tiles[t].core_type),
            tiles[t].logical.x,
            tiles[t].logical.y);
    }
    const auto offset = [&](const MirroredPair& pair) {
        return ticks_past(pair.doubled_difference, *base[pair.partner] - *base[pair.reader]);
    };
    const std::vector<double> from_base = solve_potential(
        pairs, &MirroredPair::partner_unknown, &MirroredPair::reader_unknown, offset, tile_of_unknown.size());
    std::vector<Placement> placed(tile_count);
    for (size_t u = 0; u < tile_of_unknown.size(); u++) {
        placed[tile_of_unknown[u]] = Placement{.base = *base[tile_of_unknown[u]], .from_base = from_base[u]};
    }
    return placed;
}

// Places each active eth tile from the sum of the both-NoC reader's two readings of it, shifted onto the mirrored
// placement by the mean difference between the two placements of the idle eth tiles it also read both ways.
void place_active_eth(const std::vector<Tile>& tiles, uint32_t chip, std::vector<Placement>& placed) {
    std::vector<std::optional<int64_t>> doubled_sum(tiles.size());
    double shift_sum = 0.0;
    size_t reference_count = 0;
    for (const Tile& reader : tiles) {
        for (const Reading& read : reader.reads) {
            if (!read.other_noc_read) {
                continue;
            }
            const int64_t sum = reader.reads[*read.other_noc_read].doubled_offset + read.doubled_offset;
            doubled_sum[read.partner] = sum;
            if (!is_active_eth(tiles[read.partner])) {
                const Placement& reference = placed[read.partner];
                shift_sum += reference.from_base - ticks_past(sum, reference.base);
                reference_count++;
            }
        }
    }
    for (uint32_t t = 0; t < tiles.size(); t++) {
        if (!is_active_eth(tiles[t])) {
            continue;
        }
        TT_FATAL(
            reference_count != 0 && doubled_sum[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            tiles[t].logical.x,
            tiles[t].logical.y);
        const int64_t base = *doubled_sum[t] / kPairUnitsPerTick;
        placed[t] = Placement{
            .base = base,
            .from_base = ticks_past(*doubled_sum[t], base) + shift_sum / static_cast<double>(reference_count)};
    }
}

struct TileClockStore {
    std::mutex mu;
    std::map<std::pair<ContextId, uint32_t>, TileClocks> clocks;
};

TileClockStore& tile_clock_store() {
    static ttsl::Indestructible<TileClockStore> store;
    return store.get();
}

// Measures every tile's wall clock against its chip's first Tensix tile, on each chip not measured yet, and keeps the
// offsets for tile_clocks(). Every tile's wall clock runs on the same AICLK, so each stays a fixed whole number of
// ticks from every other while the chip is up.
void measure_tile_clocks(std::span<Device* const> devices, ContextId context_id) {
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    std::vector<std::future<Network>> launches;
    for (Device* device : devices) {
        if (tile_clocks(context_id, static_cast<uint32_t>(device->id())) == nullptr) {
            launches.push_back(std::async(std::launch::async, launch_network, device, context_id));
        }
    }
    std::vector<Network> nets;
    size_t most_tiles = 0;
    for (auto& launch : launches) {
        nets.push_back(launch.get());
        most_tiles = std::max(most_tiles, nets.back().tiles.size());
    }
    // Only one tile per chip reads at a time, but chips have their own NoCs, so they all measure in parallel.
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
        retire_network(context_id, net);
        std::vector<Placement> placed = place_mirrored(net.tiles, net.chip);
        place_active_eth(net.tiles, net.chip, placed);
        TileClocks clocks;
        for (size_t i = 0; i < net.tiles.size(); i++) {
            clocks.emplace(std::pair(net.tiles[i].core_type, net.tiles[i].logical), placed[i].offset());
        }
        TileClockStore& store = tile_clock_store();
        std::lock_guard<std::mutex> lock(store.mu);
        store.clocks.emplace(std::pair{context_id, net.chip}, std::move(clocks));
    }
}

}  // namespace

const TileClocks* tile_clocks(ContextId context_id, uint32_t chip) {
    TileClockStore& store = tile_clock_store();
    std::lock_guard<std::mutex> lock(store.mu);
    const auto it = store.clocks.find({context_id, chip});
    return it == store.clocks.end() ? nullptr : &it->second;
}

void prepare_clock_sync(std::span<Device* const> devices, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    if (!can_capture(mc)) {
        return;
    }
    measure_tile_clocks(devices, context_id);
    for (Device* device : devices) {
        for (const CoreCoord& logical : device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false)) {
            link_sync::zero_port(
                mc.hal(), mc.get_cluster(), device->id(), device->ethernet_core_from_logical_core(logical));
        }
    }
}

}  // namespace tt::tt_metal::streaming_profiler
