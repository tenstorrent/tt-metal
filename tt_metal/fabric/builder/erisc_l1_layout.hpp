// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <array>
#include <ostream>
#include <string>
#include <iomanip>
#include <sstream>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tt_align.hpp>
#include <enchantum/enchantum.hpp>

#include "mesh_channel_spec.hpp"
#include "fabric_builder_config.hpp"

namespace tt::tt_fabric {

/**
 * Names all L1 memory blocks.
 * Grouped by category for clarity. Order matches allocation order.
 */
enum class L1Block : uint8_t {
    // ═══════════════════════════════════════════════════════════
    // Telemetry & Profiling (optional, allocated early)
    // ═══════════════════════════════════════════════════════════
    TELEMETRY_BUFFER,       // Performance telemetry (BH always allocates)
    CODE_PROFILING_BUFFER,  // Code profiling timers (optional)

    // ═══════════════════════════════════════════════════════════
    // Global Control & Synchronization
    // ═══════════════════════════════════════════════════════════
    HANDSHAKE,             // Ethernet handshake
    REMOTE_COUNTER_BASES,  // Multi-TXQ counter regions (optional, 4 counter blocks)
    EDM_CHANNEL_ACK,       // Channel acknowledgment region
    TERMINATION_SIGNAL,    // Teardown signal address
    EDM_LOCAL_SYNC,        // Local sync semaphore
    EDM_STATUS,            // Router status word

    // ═══════════════════════════════════════════════════════════
    // Per-Channel Control Regions
    // ═══════════════════════════════════════════════════════════
    SENDER_CHANNEL_CONTROL,       // All sender channel control fields (per-channel stride)
    RECEIVER_DOWNSTREAM_CONTROL,  // All downstream flow control/teardown sems (per-downstream stride)

    // ═══════════════════════════════════════════════════════════
    // Tensix/UDM Mode
    // ═══════════════════════════════════════════════════════════
    TENSIX_RELAY_BUFFER_INDEX,  // Buffer index for local tensix relay
    EDM_LOCAL_TENSIX_SYNC,      // Tensix sync semaphore

    // ═══════════════════════════════════════════════════════════
    // Architecture-Specific
    // ═══════════════════════════════════════════════════════════
    NOTIFY_WORKER_SRC_ADDR,  // BH only: src addr for inline writes with spoof

    // ═══════════════════════════════════════════════════════════
    // Channel Buffers (must be last - consumes remaining space)
    // ═══════════════════════════════════════════════════════════
    CHANNEL_BUFFERS,  // Aligned buffer region for channel data

    COUNT  // Sentinel for array sizing
};

/**
 * Per-sender-channel addresses (all in local L1).
 */
struct SenderChannelAddresses {
    size_t buffer_index;
    size_t conn_info;
    size_t flow_control_sem;
    size_t terminate_conn;
    size_t connection_sem;
    size_t buffer_index_sem;
};

/**
 * Per-receiver-downstream addresses (all in local L1).
 */
struct ReceiverDownstreamAddresses {
    size_t flow_control_sem;
    size_t teardown_sem;
};

/**
 * Computes and stores L1 memory layout for ERISC fabric router.
 *
 * Stores spec by reference (not copy) - caller must ensure spec outlives layout.
 * Uses spec.get_total_*() for sizing - the per-VC breakdown is semantic info
 * that L1Layout doesn't need.
 */
class EriscL1Layout {
public:
    // L1 layout constants
    static constexpr size_t FIELD_SIZE = 16;             // Standard field size for L1 addresses
    static constexpr size_t BUFFER_ALIGNMENT = 32;       // Channel buffer alignment requirement
    static constexpr size_t ETH_CHANNEL_SYNC_SIZE = 16;  // Ethernet channel synchronization size

    EriscL1Layout(
        size_t base_address,
        size_t max_address,
        const MeshChannelSpec& spec,
        bool enable_telemetry,
        bool enable_code_profiling,
        bool enable_multi_txq,
        bool is_blackhole);

    // ═══════════════════════════════════════════════════════════
    // L1Block Helper Methods (static)
    // ═══════════════════════════════════════════════════════════

    /**
     * Convert L1Block enum to array index.
     */
    static constexpr size_t idx(L1Block block) { return static_cast<size_t>(block); }

    /**
     * Computes the actual size of an L1 block.
     * Not constexpr since spec is runtime-constructed via factories.
     */
    static size_t get_block_size(L1Block block, const MeshChannelSpec& spec);

    /**
     * Returns alignment requirement for a block.
     */
    static constexpr size_t get_block_alignment(L1Block block);

    /**
     * Returns stride for per-channel blocks.
     * Returns 0 for non-strided blocks.
     */
    static constexpr size_t get_block_stride(L1Block block);

    // ═══════════════════════════════════════════════════════════
    // Block Access
    // ═══════════════════════════════════════════════════════════

    const MemoryRegion& get(L1Block block) const { return regions_[idx(block)]; }

    // Per-channel addresses
    SenderChannelAddresses get_sender_channel_addresses(size_t ch) const;
    ReceiverDownstreamAddresses get_receiver_downstream_addresses(size_t downstream_idx) const;

    // NOTE: No emit_ct_args() method - L1 addresses are scattered throughout CT args,
    // not consecutive. Use individual accessors: get(L1Block::*).start_address

    // Debug dump
    void dump(std::ostream& os, bool show_gaps = false) const;
    std::string to_string(bool show_gaps = false) const;
    void log_layout(bool show_gaps = false) const;

private:
    void compute_layout(
        size_t base_address,
        size_t max_address,
        const MeshChannelSpec& spec,
        bool enable_telemetry,
        bool enable_code_profiling,
        bool enable_multi_txq,
        bool is_blackhole);

    // Dump helpers
    struct DumpStats {
        size_t total_used = 0;
        size_t total_gaps = 0;
        size_t prev_end = 0;
    };
    void dump_header(std::ostream& os, bool show_gaps) const;
    void dump_block_row(std::ostream& os, L1Block block, bool show_gaps, DumpStats& stats) const;
    void dump_summary(std::ostream& os, bool show_gaps, const DumpStats& stats) const;

    size_t base_address_;
    size_t max_address_;

    // Cached from spec during construction
    size_t num_sender_channels_;
    size_t num_downstream_edms_;
    size_t sender_control_stride_;
    size_t receiver_downstream_stride_;

    std::array<MemoryRegion, static_cast<size_t>(L1Block::COUNT)> regions_{};

    // Pre-computed per-channel addresses (cached after compute_layout)
    std::vector<SenderChannelAddresses> sender_channel_addresses_;
    std::vector<ReceiverDownstreamAddresses> receiver_downstream_addresses_;
};

}  // namespace tt::tt_fabric
