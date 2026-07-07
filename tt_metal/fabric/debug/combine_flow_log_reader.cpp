// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [debug] Host-side drain of the combine flow-control traces ([rxlog]/[txlog]) that fabric routers flush to
// DRAM during a combine window. See dump_combine_flow_logs() in combine_flow_log.hpp for the contract.
//
// Design (host readback, not on-device DEVICE_PRINT):
//   - The kernel appends delta-encoded records to an L1 log region and, on overflow (and again at the STOP
//     marker for the tail), bulk-writes them to a per-router DRAM buffer at a fixed per-bank byte offset in a
//     per-core DRAM bank (bank = ordinal-among-active-eth-channels % num_dram_banks). The DRAM buffer holds
//     ONLY the packed record array -- no framing. The framing (magic, per-window baseline gaps, total record
//     count) lives in the small L1 log header.
//   - So per router we (1) read the L1 header to detect a valid trace and recover the reconstruction baseline,
//     then (2) bulk-read the packed records back from DRAM, (3) reconstruct the cumulative columns exactly as
//     the on-device dump would, and (4) write one file per (device, eth core), named/tagged like the DPRINT
//     files so the two sit side by side.
//
// This keeps the large traces (the ones that spill to DRAM in the first place) off the DEVICE_PRINT path.

#include <tt-metalium/experimental/fabric/combine_flow_log.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>      // detail::ReadFromDeviceL1 / detail::ReadFromDeviceDRAMChannel
#include <tt-metalium/buffer_types.hpp>  // BufferType
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <umd/device/types/core_coordinates.hpp>  // CoordSystem, tt::CoreType

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "erisc_datamover_builder.hpp"  // FabricEriscDatamoverConfig (log buffer addresses)

namespace tt::tt_fabric {

namespace {

// ============================================================================
// Host mirrors of the on-device log layouts.
//
// KEEP IN SYNC with fabric_erisc_router.cpp: ReceiverLog / ReceiverLogRecord / SenderLog / SenderLogRecord and
// their magics. Only the fields the host reconstruction needs are named; the rest are covered by the header
// size (records follow the header in L1, but we read records from DRAM, so only the header offsets matter).
// Both host and device are little-endian, so a plain memcpy of the raw bytes reproduces the fields.
// ============================================================================

constexpr uint32_t RECEIVER_LOG_MAGIC = 0xC0FFEE02;
constexpr uint32_t SENDER_LOG_MAGIC = 0xC0FFEE03;

// Mirror of ReceiverLog up to and including the DRAM flush state (records[] follow, read from DRAM).
struct RxLogHeader {
    uint32_t magic;
    uint32_t count;    // records still resident in L1 (already duplicated into DRAM by the STOP tail flush)
    uint32_t dropped;  // records lost because both L1 and the DRAM buffer filled
    uint32_t spare;
    uint32_t last_ts, last_iter, last_ready, last_ack, last_wr_sent, last_wr_flush, last_completion;  // working state
    uint32_t base_ack_gap, base_wr_sent_gap, base_wr_flush_gap;  // reconstruction baseline (window-open backlog)
    uint32_t dram_write_offset;  // total bytes of packed records in DRAM (== count_in_dram * sizeof record)
    uint32_t dram_records;       // total records in DRAM
};
static_assert(sizeof(RxLogHeader) == 16 * sizeof(uint32_t), "RxLogHeader must mirror ReceiverLog's header");

struct RxRecord {  // ReceiverLogRecord, 8 bytes
    uint16_t ts_delta;
    uint8_t iter_delta;
    uint8_t ready;
    uint8_t ack_delta;
    uint8_t wr_sent_delta;
    uint8_t wr_flush_delta;
    uint8_t completion_delta;
};
static_assert(sizeof(RxRecord) == 8, "RxRecord must mirror ReceiverLogRecord");

// Mirror of SenderLog up to and including the DRAM flush state.
struct TxLogHeader {
    uint32_t magic;
    uint32_t count;
    uint32_t dropped;
    uint32_t spare;
    uint32_t sent[2], acked[2], cmpl[2];
    uint32_t dn_credits;
    uint32_t last_ts, last_iter, last_dn_credits;
    uint32_t last_sent[2], last_acked[2], last_cmpl[2];
    uint8_t occ[2], last_occ[2], reason[2], conn[2];  // 8 bytes == 2 words
    uint32_t base_sent_gap[2], base_acked_gap[2];     // reconstruction baseline
    uint32_t dram_write_offset;
    uint32_t dram_records;
};
static_assert(sizeof(TxLogHeader) == 28 * sizeof(uint32_t), "TxLogHeader must mirror SenderLog's header");

struct TxRecord {  // SenderLogRecord, 14 bytes
    uint16_t ts_delta;
    uint8_t iter_delta;
    uint8_t dn_credits;
    uint8_t ch_flags;  // low nibble = ch0 [conn:bit3 | reason:bits0-2]; high nibble = ch1 (same layout)
    uint8_t ch0_local_occ, ch0_sent_delta, ch0_acked_delta, ch0_cmpl_delta;
    uint8_t ch1_local_occ, ch1_sent_delta, ch1_acked_delta, ch1_cmpl_delta;
};
static_assert(sizeof(TxRecord) == 14, "TxRecord must mirror SenderLogRecord");

// Read `size` bytes from a router's DRAM log buffer. `bank_id` is the kernel's LOG_DRAM_BANK_ID; for DRAM it
// indexes the DRAM channel directly (matches the interleaved-read path in tt_metal.cpp). We add the allocator's
// DRAM bank offset to mirror get_noc_addr_from_bank_id() used by the kernel write (0 on architectures with one
// bank per channel, e.g. Blackhole -- harmless there, correct if it ever becomes nonzero).
std::vector<uint8_t> read_dram_records(tt::tt_metal::IDevice* device, uint32_t bank_id, uint64_t base, uint32_t size) {
    std::vector<uint8_t> bytes(size);
    if (size == 0) {
        return bytes;
    }
    const int32_t bank_offset = device->allocator()->get_bank_offset(tt::tt_metal::BufferType::DRAM, bank_id);
    const uint32_t addr = static_cast<uint32_t>(base) + static_cast<uint32_t>(bank_offset);
    tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
        device, static_cast<int>(bank_id), addr, std::span<uint8_t>(bytes.data(), bytes.size()));
    return bytes;
}

// Read `size` bytes of L1 from a logical ethernet core.
bool read_eth_l1(
    tt::tt_metal::IDevice* device,
    const tt::tt_metal::CoreCoord& eth_logical_core,
    uint32_t addr,
    void* dst,
    uint32_t size) {
    return tt::tt_metal::detail::ReadFromDeviceL1(
        device, eth_logical_core, addr, std::span<uint8_t>(static_cast<uint8_t*>(dst), size), tt::CoreType::ETH);
}

std::string file_header(
    int device_id,
    int physical_chip,
    const tt::tt_metal::CoreCoord& eth_logical_core,
    uint32_t eth_chan,
    uint32_t bank,
    const char* tag,
    uint32_t dram_records,
    uint32_t dropped) {
    return fmt::format(
        "# {} combine flow-control trace\n"
        "# device_id={} physical_chip={} eth_core=(x={},y={}) eth_channel={} dram_bank={}\n"
        "# records={} dropped={}\n",
        tag,
        device_id,
        physical_chip,
        eth_logical_core.x,
        eth_logical_core.y,
        eth_chan,
        bank,
        dram_records,
        dropped);
}

}  // namespace

void dump_combine_flow_logs(const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::string& out_dir) {
    const std::filesystem::path dir =
        out_dir.empty() ? std::filesystem::path("generated/combine_flow_log") : std::filesystem::path(out_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto& metal_ctx = tt::tt_metal::MetalContext::instance();
    const auto& control_plane = metal_ctx.get_control_plane();
    const auto& cluster = metal_ctx.get_cluster();

    // The L1 log-region addresses are compile-time carves identical across all routers of a config, so one
    // lookup serves every core. Bail cleanly if fabric never built its routers (nothing to drain).
    const auto& fabric_ctx = control_plane.get_fabric_context();
    if (!fabric_ctx.has_builder_context()) {
        fmt::print(stderr, "[combine-log] fabric builder context unavailable; no traces to drain\n");
        return;
    }
    const auto& edm_cfg = fabric_ctx.get_builder_context().get_fabric_router_config();
    const uint32_t rx_l1_addr = static_cast<uint32_t>(edm_cfg.receiver_log_buffer_address);
    const uint32_t tx_l1_addr = static_cast<uint32_t>(edm_cfg.sender_log_buffer_address);

    uint32_t files_written = 0;

    for (tt::tt_metal::IDevice* device : mesh_device.get_devices()) {
        const int physical_chip = device->id();
        const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip);
        const auto& soc_desc = cluster.get_soc_desc(physical_chip);
        const uint32_t num_banks = static_cast<uint32_t>(soc_desc.get_num_dram_channels());

        // Iterating the ordered set reproduces the builder's per-core `ordinal` (and thus LOG_DRAM_BANK_ID)
        // exactly -- same container, same order, same modulo. Detect DRAM-bank collisions (more active eth
        // channels than DRAM banks): colliding cores clobber each other's DRAM buffer, so their attribution is
        // unrecoverable and we warn rather than emit corrupt data.
        std::map<uint32_t, tt::tt_metal::CoreCoord> rx_bank_owner;  // bank -> first serviced eth core we saw on it
        std::map<uint32_t, tt::tt_metal::CoreCoord> tx_bank_owner;

        uint32_t ordinal = 0;
        for (const auto& [eth_chan, _direction] : control_plane.get_active_fabric_eth_channels(node)) {
            const uint32_t bank = num_banks > 0 ? (ordinal % num_banks) : 0;
            ++ordinal;

            const auto umd_eth = soc_desc.get_eth_core_for_channel(eth_chan, tt::CoordSystem::LOGICAL);
            const tt::tt_metal::CoreCoord eth_logical_core{umd_eth.x, umd_eth.y};

            // ---- Receiver trace ([rxlog]) ----
            {
                RxLogHeader h{};
                if (read_eth_l1(device, eth_logical_core, rx_l1_addr, &h, sizeof(h)) && h.magic == RECEIVER_LOG_MAGIC &&
                    h.dram_write_offset > 0) {
                    auto owner = rx_bank_owner.find(bank);
                    if (owner != rx_bank_owner.end()) {
                        fmt::print(
                            stderr,
                            "[combine-log] WARNING device {} rx: eth cores ({},{}) and ({},{}) share DRAM bank {} "
                            "-- their DRAM traces collided; skipping ({},{})\n",
                            device->id(),
                            owner->second.x,
                            owner->second.y,
                            eth_logical_core.x,
                            eth_logical_core.y,
                            bank,
                            eth_logical_core.x,
                            eth_logical_core.y);
                    } else {
                        rx_bank_owner.emplace(bank, eth_logical_core);
                        const auto bytes =
                            read_dram_records(device, bank, COMBINE_RECEIVER_LOG_DRAM_BASE, h.dram_write_offset);
                        const uint32_t n = h.dram_write_offset / sizeof(RxRecord);

                        const auto path =
                            dir /
                            fmt::format(
                                "device_{}_core_{}-{}_rxlog.txt", device->id(), eth_logical_core.y, eth_logical_core.x);
                        std::ofstream f(path);
                        f << file_header(
                            device->id(),
                            physical_chip,
                            eth_logical_core,
                            eth_chan,
                            bank,
                            "[rxlog]",
                            h.dram_records,
                            h.dropped);

                        // Reconstruct exactly as dump_receiver_log(): accumulators seeded with the base gaps so
                        // every counter is 0-based on completion; occupied = rdy + (ack - cmpl). free = slots -
                        // occupied is left to the reader (the per-channel slot count is a kernel-side constant).
                        uint32_t iter_acc = 0, cmpl_acc = 0;
                        uint32_t ack_acc = h.base_ack_gap, wsent_acc = h.base_wr_sent_gap,
                                 wflush_acc = h.base_wr_flush_gap;
                        for (uint32_t r = 0; r < n; ++r) {
                            RxRecord rec{};
                            std::memcpy(&rec, bytes.data() + r * sizeof(RxRecord), sizeof(RxRecord));
                            iter_acc += rec.iter_delta;
                            ack_acc += rec.ack_delta;
                            wsent_acc += rec.wr_sent_delta;
                            wflush_acc += rec.wr_flush_delta;
                            cmpl_acc += rec.completion_delta;
                            const uint32_t occupied = rec.ready + (ack_acc - cmpl_acc);
                            f << fmt::format(
                                "[rxlog] iter={} dt={} rdy={} ack={} wsent={} wflush={} cmpl={} occ={}\n",
                                iter_acc,
                                rec.ts_delta,
                                rec.ready,
                                ack_acc,
                                wsent_acc,
                                wflush_acc,
                                cmpl_acc,
                                occupied);
                        }
                        ++files_written;
                    }
                }
            }

            // ---- Sender trace ([txlog]) ----
            {
                TxLogHeader h{};
                if (read_eth_l1(device, eth_logical_core, tx_l1_addr, &h, sizeof(h)) && h.magic == SENDER_LOG_MAGIC &&
                    h.dram_write_offset > 0) {
                    auto owner = tx_bank_owner.find(bank);
                    if (owner != tx_bank_owner.end()) {
                        fmt::print(
                            stderr,
                            "[combine-log] WARNING device {} tx: eth cores ({},{}) and ({},{}) share DRAM bank {} "
                            "-- their DRAM traces collided; skipping ({},{})\n",
                            device->id(),
                            owner->second.x,
                            owner->second.y,
                            eth_logical_core.x,
                            eth_logical_core.y,
                            bank,
                            eth_logical_core.x,
                            eth_logical_core.y);
                    } else {
                        tx_bank_owner.emplace(bank, eth_logical_core);
                        const auto bytes =
                            read_dram_records(device, bank, COMBINE_SENDER_LOG_DRAM_BASE, h.dram_write_offset);
                        const uint32_t n = h.dram_write_offset / sizeof(TxRecord);

                        const auto path =
                            dir /
                            fmt::format(
                                "device_{}_core_{}-{}_txlog.txt", device->id(), eth_logical_core.y, eth_logical_core.x);
                        std::ofstream f(path);
                        f << file_header(
                            device->id(),
                            physical_chip,
                            eth_logical_core,
                            eth_chan,
                            bank,
                            "[txlog]",
                            h.dram_records,
                            h.dropped);

                        // Reconstruct exactly as dump_sender_log(): two output lines per record (ch0, ch1),
                        // per-channel counters seeded with base gaps. ch_flags packs reason/conn per channel.
                        uint32_t iter_acc = 0;
                        uint32_t sent0 = h.base_sent_gap[0], ack0 = h.base_acked_gap[0], cmpl0 = 0;
                        uint32_t sent1 = h.base_sent_gap[1], ack1 = h.base_acked_gap[1], cmpl1 = 0;
                        for (uint32_t r = 0; r < n; ++r) {
                            TxRecord rec{};
                            std::memcpy(&rec, bytes.data() + r * sizeof(TxRecord), sizeof(TxRecord));
                            iter_acc += rec.iter_delta;
                            sent0 += rec.ch0_sent_delta;
                            ack0 += rec.ch0_acked_delta;
                            cmpl0 += rec.ch0_cmpl_delta;
                            sent1 += rec.ch1_sent_delta;
                            ack1 += rec.ch1_acked_delta;
                            cmpl1 += rec.ch1_cmpl_delta;
                            const uint32_t fl = rec.ch_flags;
                            const uint32_t dncr = rec.dn_credits;
                            f << fmt::format(
                                "[txlog] iter={} dt={} ch=0 occ={} sent={} ack={} cmpl={} dncr={} r={} cn={}\n",
                                iter_acc,
                                rec.ts_delta,
                                rec.ch0_local_occ,
                                sent0,
                                ack0,
                                cmpl0,
                                dncr,
                                fl & 0x7,
                                (fl >> 3) & 0x1);
                            f << fmt::format(
                                "[txlog] iter={} dt=0 ch=1 occ={} sent={} ack={} cmpl={} dncr={} r={} cn={}\n",
                                iter_acc,
                                rec.ch1_local_occ,
                                sent1,
                                ack1,
                                cmpl1,
                                dncr,
                                (fl >> 4) & 0x7,
                                (fl >> 7) & 0x1);
                        }
                        ++files_written;
                    }
                }
            }
        }
    }

    fmt::print(stderr, "[combine-log] wrote {} trace file(s) to {}\n", files_written, dir.string());
}

}  // namespace tt::tt_fabric
