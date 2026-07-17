// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [debug] Host-side drain of the detailed fabric flow-control traces ([rxlog]/[txlog]) that fabric routers flush
// to DRAM during a logging window. See dump_detailed_fabric_logs() in detailed_fabric_log.hpp for the contract.
//
// Design (host readback, not on-device DEVICE_PRINT):
//   - The kernel appends 8-byte log WORDS to an L1 region and, on overflow (and again at the STOP marker for the
//     tail), bulk-writes them to a per-router DRAM buffer in a per-core DRAM bank (bank = ordinal-among-active-
//     eth-channels % num_dram_banks). At the STOP marker it also flushes a small header (magic, per-channel
//     baseline gaps, word count) to the FRONT of that buffer. So each DRAM buffer is [header region | packed word
//     array] -- everything the reader needs is in DRAM; nothing is read from L1.
//   - The word stream is self-describing (see detailed_fabric_logs.hpp): a COMMON word (byte0 = count >= 1) opens
//     an iteration sample carrying the shared iter/ts deltas, then `count` CHANNEL words (byte0 = 0) each carry one
//     channel's delta-encoded state. So per router we (1) read the header to detect a valid trace + recover the
//     reconstruction baselines, then (2) bulk-read the words back, (3) walk them reconstructing the CUMULATIVE
//     columns AND absolute iteration/time (not deltas), and (4) write one file per (device, eth core), tagged like
//     the DPRINT files so the two sit side by side.

#include <tt-metalium/experimental/fabric/detailed_fabric_log.hpp>
#include <tt-metalium/experimental/fabric/detailed_fabric_logs.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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
// The on-device log layouts (the DRAM-flushed *LogHeader framing, the 8 B word codec, and the nibble-packed
// channel id) come from the shared detailed_fabric_logs.hpp, so this reader and the fabric router share ONE
// definition -- there is no hand-maintained host mirror to drift. Both host and device are little-endian, so raw
// DRAM bytes map straight onto the structs / uint64_t words. Each per-router DRAM buffer is [header region | words].
// ============================================================================

// Read `size` bytes from a router's DRAM log buffer at per-bank byte offset `base`. `bank_id` is the kernel's
// LOG_DRAM_BANK_ID; for DRAM it indexes the DRAM channel directly (matches the interleaved-read path in
// tt_metal.cpp). We add the allocator's DRAM bank offset to mirror get_noc_addr_from_bank_id() used by the kernel
// write (0 on architectures with one bank per channel, e.g. Blackhole -- harmless there, correct if it ever
// becomes nonzero).
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

// Read one 8-byte log word from a packed byte buffer at word index `i` (little-endian -> uint64_t).
uint64_t word_at(const std::vector<uint8_t>& bytes, uint32_t i) {
    uint64_t w = 0;
    std::memcpy(&w, bytes.data() + static_cast<size_t>(i) * sizeof(uint64_t), sizeof(uint64_t));
    return w;
}

std::string file_header(
    int device_id,
    int physical_chip,
    const tt::tt_metal::CoreCoord& eth_logical_core,
    uint32_t eth_chan,
    uint32_t bank,
    const char* tag,
    uint32_t window_id,
    uint32_t dram_words,
    uint32_t dropped) {
    return fmt::format(
        "# {} fabric flow-control trace\n"
        "# device_id={} physical_chip={} eth_core=(x={},y={}) eth_channel={} dram_bank={}\n"
        "# window_id={} words={} dropped={}\n",
        tag,
        device_id,
        physical_chip,
        eth_logical_core.x,
        eth_logical_core.y,
        eth_chan,
        bank,
        window_id,
        dram_words,
        dropped);
}

// [debug] Decode the per-channel sender block-reason code (SenderSendReason) into a readable label.
const char* sender_reason_str(uint32_t reason) {
    switch (reason) {
        case SENDER_REASON_IDLE: return "IDLE";        // no send, no obvious block (nothing pending / turn skipped)
        case SENDER_REASON_SENT: return "SENT";        // transmitted a packet this pass
        case SENDER_REASON_STARVED: return "STARVED";  // downstream credit + txq free, but no unsent packet
        case SENDER_REASON_RXFULL: return "RXFULL";    // unsent packet + txq free, but no downstream receiver credit
        case SENDER_REASON_TXQBUSY: return "TXQBUSY";  // unsent packet + downstream credit, but the eth txq is busy
        default: return "UNKNOWN";
    }
}

// [debug] conn (flags bit3) = channel_connection_established: whether an upstream worker (the tensix producer)
// currently holds an OPEN fabric connection to this sender channel this pass. "UP" = connected; "DN" = none.
const char* sender_conn_str(bool conn) { return conn ? "UP" : "DN"; }

// Per-channel reconstruction accumulators, keyed by the nibble-packed channel id. Seeded from the header's
// [vc][local] baseline gaps the first time a channel id is seen, so every counter is 0-based on completion.
struct ReceiverAccum {
    uint32_t ack = 0;
    uint32_t wr_sent = 0;
    uint32_t wr_flush = 0;
    uint32_t cmpl = 0;
    bool seeded = false;
};
struct SenderAccum {
    uint32_t sent = 0;
    uint32_t acked = 0;
    uint32_t cmpl = 0;
    bool seeded = false;
};

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
    // shared by every router, so no per-router L1 lookup is needed.
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
                        const uint32_t n_words = h.dram_write_offset / sizeof(uint64_t);

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
                            h.dram_words,
                            h.dropped);

                        // Walk the word stream. A common word advances the shared timeline (absolute iter + t);
                        // each following channel word accumulates that channel's counters (seeded from the header
                        // base gaps) and emits one line tagged with the parent common word's reconstructed
                        // iter/t. occupied = rdy + (ack - cmpl); free = slots - occupied is left to the reader.
                        uint32_t iter_acc = 0, ts_acc = 0;
                        std::map<uint8_t, ReceiverAccum> accum;
                        for (uint32_t i = 0; i < n_words; ++i) {
                            const uint64_t w = word_at(bytes, i);
                            if (!log_word_is_channel(w)) {
                                iter_acc += common_log_word_iter_delta(w);
                                ts_acc += common_log_word_ts_delta(w);
                                continue;
                            }
                            const uint8_t cid = channel_log_word_channel_id(w);
                            const uint32_t vc = channel_id_vc(cid);
                            const uint32_t local = channel_id_local(cid);
                            auto& a = accum[cid];
                            if (!a.seeded) {
                                a.ack = h.base_ack_gap[vc][local];
                                a.wr_sent = h.base_wr_sent_gap[vc][local];
                                a.wr_flush = h.base_wr_flush_gap[vc][local];
                                a.cmpl = 0;
                                a.seeded = true;
                            }
                            const uint32_t ready = receiver_channel_log_word_ready(w);
                            a.ack += receiver_channel_log_word_ack_delta(w);
                            a.wr_sent += receiver_channel_log_word_wr_sent_delta(w);
                            a.wr_flush += receiver_channel_log_word_wr_flush_delta(w);
                            a.cmpl += receiver_channel_log_word_completion_delta(w);
                            const uint32_t occupied = ready + (a.ack - a.cmpl);
                            f << fmt::format(
                                "[rxlog] iter={} t={} vc={} ch={} rdy={} ack={} wsent={} wflush={} cmpl={} occ={}\n",
                                iter_acc,
                                ts_acc,
                                vc,
                                local,
                                ready,
                                a.ack,
                                a.wr_sent,
                                a.wr_flush,
                                a.cmpl,
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
                        const uint32_t n_words = h.dram_write_offset / sizeof(uint64_t);

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
                            h.dram_words,
                            h.dropped);

                        // Same walk as the receiver: common word advances the timeline; each sender channel word
                        // accumulates its counters (seeded from the header base gaps) and emits one line. dn_credits
                        // is shared across channels of the same VC (same vc nibble). ch_flags packs reason/conn.
                        uint32_t iter_acc = 0, ts_acc = 0;
                        std::map<uint8_t, SenderAccum> accum;
                        for (uint32_t i = 0; i < n_words; ++i) {
                            const uint64_t w = word_at(bytes, i);
                            if (!log_word_is_channel(w)) {
                                iter_acc += common_log_word_iter_delta(w);
                                ts_acc += common_log_word_ts_delta(w);
                                continue;
                            }
                            const uint8_t cid = channel_log_word_channel_id(w);
                            const uint32_t vc = channel_id_vc(cid);
                            const uint32_t local = channel_id_local(cid);
                            auto& a = accum[cid];
                            if (!a.seeded) {
                                a.sent = h.base_sent_gap[vc][local];
                                a.acked = h.base_acked_gap[vc][local];
                                a.cmpl = 0;
                                a.seeded = true;
                            }
                            a.sent += sender_channel_log_word_sent_delta(w);
                            a.acked += sender_channel_log_word_acked_delta(w);
                            a.cmpl += sender_channel_log_word_cmpl_delta(w);
                            const uint8_t flags = sender_channel_log_word_flags(w);
                            f << fmt::format(
                                "[txlog] iter={} t={} vc={} ch={} occ={} sent={} ack={} cmpl={} dncr={} reason={} "
                                "conn={}\n",
                                iter_acc,
                                ts_acc,
                                vc,
                                local,
                                sender_channel_log_word_local_occ(w),
                                a.sent,
                                a.acked,
                                a.cmpl,
                                sender_channel_log_word_dn_credits(w),
                                sender_reason_str(sender_flags_reason(flags)),
                                sender_conn_str(sender_flags_conn(flags)));
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
