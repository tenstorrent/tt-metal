// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_fabric {

// [debug] DRAM log-buffer geometry for the detailed fabric flow-control traces (single source of truth, shared
// by the EDM builder that emits these as compile-time args and the host reader that drains them). Each serviced
// router bulk-writes its trace to a per-router DRAM buffer via NOC writes to ONE DRAM bank: the sender trace at
// DETAILED_FABRIC_SENDER_LOG_DRAM_BASE and the receiver trace at DETAILED_FABRIC_RECEIVER_LOG_DRAM_BASE (one
// buffer higher), both fixed per-bank byte offsets shared by every eth core. Each buffer is laid out as
// [header region | packed record array] (see detailed_fabric_logs.hpp). What varies per core is the BANK
// (ordinal among the node's active fabric eth channels, modulo the DRAM bank count). See the long rationale in
// erisc_datamover_builder.cpp. Offsets stay < 4 GiB so they fit uint32_t.
inline constexpr uint64_t DETAILED_FABRIC_LOG_DRAM_BUFFER_SIZE = 4ull << 20;     // 4 MiB per buffer (sender, receiver)
inline constexpr uint64_t DETAILED_FABRIC_SENDER_LOG_DRAM_BASE = 0xC0000000ull;  // 3 GiB (per-bank byte offset)
inline constexpr uint64_t DETAILED_FABRIC_RECEIVER_LOG_DRAM_BASE =
    DETAILED_FABRIC_SENDER_LOG_DRAM_BASE + DETAILED_FABRIC_LOG_DRAM_BUFFER_SIZE;  // 3 GiB + 4 MiB
constexpr std::size_t RECEIVER_LOG_BUFFER_SIZE = 4096;                            // 1024 uint32 words
constexpr std::size_t SENDER_LOG_BUFFER_SIZE = 4096;                              // 1024 uint32 words

// [debug] Drain the receiver ([rxlog]) and sender ([txlog]) detailed flow-control traces that the fabric routers
// flushed to DRAM during the last logging window, and write one text file per (device, eth core) so every file
// clearly identifies the device and ethernet core it belongs to -- mirroring the naming of the DPRINT files.
//
// For each serviced router this reads the small log header + the packed delta-encoded record array back from
// DRAM (nothing is read from L1), reconstructs the cumulative flow-control columns exactly as the on-device
// STOP-marker dump would, and writes them out.
//
// Call this from the host AFTER the op that opened/closed the logging window has finished (all router writes to
// DRAM are complete). The router flushes header+tail to DRAM at the STOP marker, so this recovers the complete
// trace for every serviced router.
//
// out_dir: destination directory (created if needed). Empty => "generated/fabric_detailed_logs" under the CWD.
void dump_detailed_fabric_logs(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::string& out_dir = "");

}  // namespace tt::tt_fabric
