// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [debug] Host-side drain of the detailed fabric flow-control traces ([rxlog]/[txlog]) that fabric routers flush
// to DRAM during a logging window. See dump_detailed_fabric_logs() in detailed_fabric_log.hpp for the contract.
//
// Design (host readback, not on-device DEVICE_PRINT):
//   - The kernel appends delta-encoded records to an L1 log region and, on overflow (and again at the STOP
//     marker for the tail), bulk-writes them to a per-router DRAM buffer in a per-core DRAM bank (bank =
//     ordinal-among-active-eth-channels % num_dram_banks). At the STOP marker it also flushes a small header
//     (magic, per-window baseline gaps, record count) to the FRONT of that buffer. So each DRAM buffer is laid
//     out as [header region | packed record array] -- everything the reader needs is in DRAM; nothing is read
//     from L1.
//   - So per router we (1) read the header from the front of the DRAM buffer to detect a valid trace and recover
//     the reconstruction baseline, then (2) bulk-read the packed records back from DRAM (just past the header
//     region), (3) reconstruct the cumulative columns exactly as the on-device dump would, and (4) write one
//     file per (device, eth core), named/tagged like the DPRINT files so the two sit side by side.
//
// This keeps the large traces (the ones that spill to DRAM in the first place) off the DEVICE_PRINT path.

#include <tt-metalium/experimental/fabric/detailed_fabric_log.hpp>
#include <tt-metalium/experimental/fabric/detailed_fabric_logs.hpp>

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
#include <tt-metalium/tt_metal.hpp>      // detail::ReadFromDeviceDRAMChannel
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
// The on-device log layouts (the DRAM-flushed *LogHeader framing + the packed *LogRecord arrays and their
// magics) come from the shared detailed_fabric_logs.hpp, so this reader and the fabric router share ONE
// definition -- there is no hand-maintained host mirror to drift. Both host and device are little-endian, so the
// raw DRAM bytes map straight onto the structs. Each per-router DRAM buffer is [header region | packed records].
// ============================================================================

// Read `size` bytes from a router's DRAM log buffer at per-bank byte offset `base`. `bank_id` is the kernel's
// LOG_DRAM_BANK_ID; for DRAM it indexes the DRAM channel directly (matches the interleaved-read path in
// tt_metal.cpp). We add the allocator's DRAM bank offset to mirror get_noc_addr_from_bank_id() used by the
// kernel write (0 on architectures with one bank per channel, e.g. Blackhole -- harmless there, correct if it
// ever becomes nonzero).
std::vector<uint8_t> read_dram(tt::tt_metal::IDevice* device, uint32_t bank_id, uint64_t base, uint32_t size) {
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

// Read a POD log header (ReceiverLogHeader / SenderLogHeader) from the front of a router's DRAM buffer.
template <typename HeaderT>
HeaderT read_dram_header(tt::tt_metal::IDevice* device, uint32_t bank_id, uint64_t buffer_base) {
    HeaderT h{};
    const auto bytes = read_dram(device, bank_id, buffer_base, sizeof(HeaderT));
    std::memcpy(&h, bytes.data(), sizeof(HeaderT));
    return h;
}

std::string file_header(
    int device_id,
    int physical_chip,
    const tt::tt_metal::CoreCoord& eth_logical_core,
    uint32_t eth_chan,
    uint32_t bank,
    const char* tag,
    uint32_t window_id,
    uint32_t dram_records,
    uint32_t dropped) {
    return fmt::format(
        "# {} fabric flow-control trace\n"
        "# device_id={} physical_chip={} eth_core=(x={},y={}) eth_channel={} dram_bank={}\n"
        "# window_id={} records={} dropped={}\n",
        tag,
        device_id,
        physical_chip,
        eth_logical_core.x,
        eth_logical_core.y,
        eth_chan,
        bank,
        window_id,
        dram_records,
        dropped);
}

// [debug] Decode the per-channel sender block-reason code (mirror of SenderSendReason in
// detailed_fabric_logs.hpp) into a readable label for the dumped [txlog] lines.
const char* sender_reason_str(uint32_t reason) {
    switch (reason) {
        case 0: return "IDLE";     // no send, no obvious block (nothing pending / turn skipped)
        case 1: return "SENT";     // transmitted a packet this pass
        case 2: return "STARVED";  // downstream credit + txq free, but no unsent packet from the producer
        case 3: return "RXFULL";   // unsent packet + txq free, but no downstream receiver credit
        case 4: return "TXQBUSY";  // unsent packet + downstream credit, but the eth txq is busy
        default: return "UNKNOWN";
    }
}

// [debug] conn (ch_flags bit3/bit7) = channel_connection_established: whether an upstream worker (the tensix
// producer) currently holds an OPEN fabric connection to this VC0 sender channel this pass. "UP" = a worker is
// connected; "DN" = none attached (e.g. before the first open / after teardown, or a window edge). Reading it
// alongside reason=STARVED disambiguates "connected worker not producing fast enough" (conn=UP) from "no
// producer attached this pass" (conn=DN).
const char* sender_conn_str(uint32_t conn_bit) { return conn_bit ? "UP" : "DN"; }

}  // namespace

void dump_detailed_fabric_logs(const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::string& out_dir) {
    const std::filesystem::path dir =
        out_dir.empty() ? std::filesystem::path("generated/fabric_detailed_logs") : std::filesystem::path(out_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto& metal_ctx = tt::tt_metal::MetalContext::instance();
    const auto& control_plane = metal_ctx.get_control_plane();
    const auto& cluster = metal_ctx.get_cluster();

    // Bail cleanly if fabric never built its routers (nothing to drain). The DRAM buffer geometry is fixed and
    // shared by every router, so no per-router L1 lookup is needed anymore.
    const auto& fabric_ctx = control_plane.get_fabric_context();
    if (!fabric_ctx.has_builder_context()) {
        fmt::print(stderr, "[fabric-log] fabric builder context unavailable; no traces to drain\n");
        return;
    }

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
                const auto h =
                    read_dram_header<ReceiverLogHeader>(device, bank, DETAILED_FABRIC_RECEIVER_LOG_DRAM_BASE);
                if (h.magic == RECEIVER_LOG_MAGIC && h.dram_write_offset > 0) {
                    auto owner = rx_bank_owner.find(bank);
                    if (owner != rx_bank_owner.end()) {
                        fmt::print(
                            stderr,
                            "[fabric-log] WARNING device {} rx: eth cores ({},{}) and ({},{}) share DRAM bank {} "
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
                        const auto bytes = read_dram(
                            device,
                            bank,
                            DETAILED_FABRIC_RECEIVER_LOG_DRAM_BASE + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                            h.dram_write_offset);
                        const uint32_t n = h.dram_write_offset / sizeof(ReceiverLogRecord);

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
                            h.window_id,
                            h.dram_records,
                            h.dropped);

                        // Reconstruct the cumulative columns from the delta-encoded records: accumulators seeded
                        // with the base gaps so every counter is 0-based on completion; occupied = rdy + (ack -
                        // cmpl). free = slots - occupied is left to the reader (the per-channel slot count is a
                        // kernel-side constant).
                        uint32_t iter_acc = 0, cmpl_acc = 0;
                        uint32_t ack_acc = h.base_ack_gap, wsent_acc = h.base_wr_sent_gap,
                                 wflush_acc = h.base_wr_flush_gap;
                        for (uint32_t r = 0; r < n; ++r) {
                            ReceiverLogRecord rec{};
                            std::memcpy(&rec, bytes.data() + r * sizeof(ReceiverLogRecord), sizeof(ReceiverLogRecord));
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
                const auto h = read_dram_header<SenderLogHeader>(device, bank, DETAILED_FABRIC_SENDER_LOG_DRAM_BASE);
                if (h.magic == SENDER_LOG_MAGIC && h.dram_write_offset > 0) {
                    auto owner = tx_bank_owner.find(bank);
                    if (owner != tx_bank_owner.end()) {
                        fmt::print(
                            stderr,
                            "[fabric-log] WARNING device {} tx: eth cores ({},{}) and ({},{}) share DRAM bank {} "
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
                        const auto bytes = read_dram(
                            device,
                            bank,
                            DETAILED_FABRIC_SENDER_LOG_DRAM_BASE + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                            h.dram_write_offset);
                        const uint32_t n = h.dram_write_offset / sizeof(SenderLogRecord);

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
                            h.window_id,
                            h.dram_records,
                            h.dropped);

                        // Reconstruct the two output lines per record (ch0, ch1): per-channel counters seeded
                        // with the base gaps. ch_flags packs reason/conn per channel.
                        uint32_t iter_acc = 0;
                        uint32_t sent0 = h.base_sent_gap[0], ack0 = h.base_acked_gap[0], cmpl0 = 0;
                        uint32_t sent1 = h.base_sent_gap[1], ack1 = h.base_acked_gap[1], cmpl1 = 0;
                        for (uint32_t r = 0; r < n; ++r) {
                            SenderLogRecord rec{};
                            std::memcpy(&rec, bytes.data() + r * sizeof(SenderLogRecord), sizeof(SenderLogRecord));
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
                                "[txlog] iter={} dt={} ch=0 occ={} sent={} ack={} cmpl={} dncr={} reason={} "
                                "conn={}\n",
                                iter_acc,
                                rec.ts_delta,
                                rec.ch0_local_occ,
                                sent0,
                                ack0,
                                cmpl0,
                                dncr,
                                sender_reason_str(fl & 0x7),
                                sender_conn_str((fl >> 3) & 0x1));
                            f << fmt::format(
                                "[txlog] iter={} dt=0 ch=1 occ={} sent={} ack={} cmpl={} dncr={} reason={} "
                                "conn={}\n",
                                iter_acc,
                                rec.ch1_local_occ,
                                sent1,
                                ack1,
                                cmpl1,
                                dncr,
                                sender_reason_str((fl >> 4) & 0x7),
                                sender_conn_str((fl >> 7) & 0x1));
                        }
                        ++files_written;
                    }
                }
            }
        }
    }

    fmt::print(stderr, "[fabric-log] wrote {} trace file(s) to {}\n", files_written, dir.string());
}

}  // namespace tt::tt_fabric
