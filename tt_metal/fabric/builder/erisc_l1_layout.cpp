// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "erisc_l1_layout.hpp"

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric {

// ═══════════════════════════════════════════════════════════
// EriscL1Layout Static Helper Methods
// ═══════════════════════════════════════════════════════════

size_t EriscL1Layout::get_block_size(L1Block block, const MeshChannelSpec& spec) {
    constexpr size_t FIELD_SIZE = FabricEriscDatamoverConfig::field_size;                        // 16
    constexpr size_t ETH_CHANNEL_SYNC_SIZE = FabricEriscDatamoverConfig::eth_channel_sync_size;  // 16
    constexpr size_t CONN_INFO_SIZE = sizeof(EDMChannelWorkerLocationInfo);

    switch (block) {
        // Telemetry/Profiling
        case L1Block::TELEMETRY_BUFFER: return 32;  // TODO: Reference actual telemetry constant if exists

        case L1Block::CODE_PROFILING_BUFFER:
            return get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);

        // Global Control
        case L1Block::HANDSHAKE: return ETH_CHANNEL_SYNC_SIZE;

        case L1Block::REMOTE_COUNTER_BASES: {
            // 4 counter regions (ack/completion for sender/receiver)
            // Each region: align(sizeof(uint32_t) * num_sender_channels, FIELD_SIZE)
            size_t num_senders = spec.get_total_sender_channels();
            size_t per_counter_region = tt::align(sizeof(uint32_t) * num_senders, FIELD_SIZE);
            return 4 * per_counter_region;
        }

        case L1Block::EDM_CHANNEL_ACK: return 4 * ETH_CHANNEL_SYNC_SIZE;  // Padding for handshake compatibility

        case L1Block::TERMINATION_SIGNAL:
        case L1Block::EDM_LOCAL_SYNC:
        case L1Block::EDM_STATUS:
        case L1Block::TENSIX_RELAY_BUFFER_INDEX:
        case L1Block::EDM_LOCAL_TENSIX_SYNC:
        case L1Block::NOTIFY_WORKER_SRC_ADDR: return FIELD_SIZE;

        // Per-Channel Control
        case L1Block::SENDER_CHANNEL_CONTROL: {
            // Per sender: buffer_index + conn_info + flow_ctrl + term + conn_sem + buf_idx_sem
            size_t stride = FIELD_SIZE + CONN_INFO_SIZE + 4 * FIELD_SIZE;
            return stride * spec.get_total_sender_channels();
        }

        case L1Block::RECEIVER_DOWNSTREAM_CONTROL: {
            // Per downstream: padding + flow_ctrl + teardown
            size_t stride = 3 * FIELD_SIZE;
            return stride * spec.get_total_downstream_edms();
        }

        // Remaining space
        case L1Block::CHANNEL_BUFFERS: return 0;  // Size = remaining space, computed in layout

        default: TT_THROW("Unknown L1Block: {}", static_cast<int>(block));
    }
}

constexpr size_t EriscL1Layout::get_block_alignment(L1Block block) {
    constexpr size_t FIELD_SIZE = FabricEriscDatamoverConfig::field_size;
    constexpr size_t BUFFER_ALIGNMENT = FabricEriscDatamoverConfig::buffer_alignment;

    switch (block) {
        case L1Block::CHANNEL_BUFFERS: return BUFFER_ALIGNMENT;  // 32
        default: return FIELD_SIZE;                              // 16
    }
}

constexpr size_t EriscL1Layout::get_block_stride(L1Block block) {
    constexpr size_t FIELD_SIZE = FabricEriscDatamoverConfig::field_size;
    constexpr size_t CONN_INFO_SIZE = sizeof(EDMChannelWorkerLocationInfo);

    switch (block) {
        case L1Block::SENDER_CHANNEL_CONTROL: return FIELD_SIZE + CONN_INFO_SIZE + 4 * FIELD_SIZE;
        case L1Block::RECEIVER_DOWNSTREAM_CONTROL: return 3 * FIELD_SIZE;
        default: return 0;
    }
}

// ═══════════════════════════════════════════════════════════
// EriscL1Layout Implementation
// ═══════════════════════════════════════════════════════════

EriscL1Layout::EriscL1Layout(
    size_t base_address,
    size_t max_address,
    const MeshChannelSpec& spec,
    bool enable_telemetry,
    bool enable_code_profiling,
    bool enable_multi_txq,
    bool is_blackhole) :
    spec_(spec) {
    compute_layout(base_address, max_address, enable_telemetry, enable_code_profiling, enable_multi_txq, is_blackhole);
}

void EriscL1Layout::compute_layout(
    size_t base_address,
    size_t max_address,
    bool enable_telemetry,
    bool enable_code_profiling,
    bool enable_multi_txq,
    bool is_blackhole) {
    base_address_ = base_address;
    max_address_ = max_address;

    // Cache from spec for address computations
    num_sender_channels_ = spec_.get_total_sender_channels();
    num_downstream_edms_ = spec_.get_total_downstream_edms();
    sender_control_stride_ = get_block_stride(L1Block::SENDER_CHANNEL_CONTROL);
    receiver_downstream_stride_ = get_block_stride(L1Block::RECEIVER_DOWNSTREAM_CONTROL);

    size_t next_addr = base_address;

    // ═══════════════════════════════════════════════════════════
    // Allocation helpers
    // ═══════════════════════════════════════════════════════════

    auto allocate = [&](L1Block block) {
        next_addr = tt::align(next_addr, get_block_alignment(block));
        size_t size = get_block_size(block, spec_);
        regions_[idx(block)] = MemoryRegion(next_addr, size);
        next_addr += size;
    };

    auto skip = [&](L1Block block) { regions_[idx(block)] = MemoryRegion(0, 0); };

    auto allocate_remaining = [&](L1Block block) {
        size_t start = tt::align(next_addr, get_block_alignment(block));
        TT_FATAL(start < max_address, "L1 layout overflow: 0x{:x} >= 0x{:x}", start, max_address);
        regions_[idx(block)] = MemoryRegion(start, max_address - start);
    };

    // ═══════════════════════════════════════════════════════════
    // Allocation order (conditions inline for clarity)
    // ═══════════════════════════════════════════════════════════

    // 1. Telemetry (BH workaround: always allocate space on Blackhole)
    (enable_telemetry || is_blackhole) ? allocate(L1Block::TELEMETRY_BUFFER) : skip(L1Block::TELEMETRY_BUFFER);

    // 2. Code profiling
    enable_code_profiling ? allocate(L1Block::CODE_PROFILING_BUFFER) : skip(L1Block::CODE_PROFILING_BUFFER);

    // 3. Handshake (always)
    allocate(L1Block::HANDSHAKE);

    // 4. Remote counter bases (multi-TXQ mode only)
    enable_multi_txq ? allocate(L1Block::REMOTE_COUNTER_BASES) : skip(L1Block::REMOTE_COUNTER_BASES);

    // 5-8. Fixed control blocks
    allocate(L1Block::EDM_CHANNEL_ACK);
    allocate(L1Block::TERMINATION_SIGNAL);
    allocate(L1Block::EDM_LOCAL_SYNC);
    allocate(L1Block::EDM_STATUS);

    // 9-10. Per-channel control
    allocate(L1Block::SENDER_CHANNEL_CONTROL);
    allocate(L1Block::RECEIVER_DOWNSTREAM_CONTROL);

    // 11-12. Tensix extension
    allocate(L1Block::TENSIX_RELAY_BUFFER_INDEX);
    allocate(L1Block::EDM_LOCAL_TENSIX_SYNC);

    // 13. Blackhole-specific
    is_blackhole ? allocate(L1Block::NOTIFY_WORKER_SRC_ADDR) : skip(L1Block::NOTIFY_WORKER_SRC_ADDR);

    // 14. Channel buffers (remaining space)
    allocate_remaining(L1Block::CHANNEL_BUFFERS);
}

// ═══════════════════════════════════════════════════════════
// Per-Channel Address Helpers
// ═══════════════════════════════════════════════════════════

SenderChannelAddresses EriscL1Layout::get_sender_channel(size_t ch) const {
    TT_ASSERT(ch < num_sender_channels_, "Sender channel {} out of bounds (max {})", ch, num_sender_channels_);

    constexpr size_t FIELD_SIZE = FabricEriscDatamoverConfig::field_size;
    constexpr size_t CONN_INFO_SIZE = sizeof(EDMChannelWorkerLocationInfo);

    size_t base = regions_[idx(L1Block::SENDER_CHANNEL_CONTROL)].start_address;
    size_t offset = ch * sender_control_stride_;

    return SenderChannelAddresses{
        .buffer_index = base + offset,
        .conn_info = base + offset + FIELD_SIZE,
        .flow_control_sem = base + offset + FIELD_SIZE + CONN_INFO_SIZE,
        .terminate_conn = base + offset + FIELD_SIZE + CONN_INFO_SIZE + FIELD_SIZE,
        .connection_sem = base + offset + FIELD_SIZE + CONN_INFO_SIZE + 2 * FIELD_SIZE,
        .buffer_index_sem = base + offset + FIELD_SIZE + CONN_INFO_SIZE + 3 * FIELD_SIZE,
    };
}

ReceiverDownstreamAddresses EriscL1Layout::get_receiver_downstream(size_t downstream_idx) const {
    TT_ASSERT(
        downstream_idx < num_downstream_edms_,
        "Downstream EDM {} out of bounds (max {})",
        downstream_idx,
        num_downstream_edms_);

    constexpr size_t FIELD_SIZE = FabricEriscDatamoverConfig::field_size;

    size_t base = regions_[idx(L1Block::RECEIVER_DOWNSTREAM_CONTROL)].start_address;
    size_t offset = downstream_idx * receiver_downstream_stride_;

    return ReceiverDownstreamAddresses{
        .flow_control_sem = base + offset + FIELD_SIZE,  // Skip padding field
        .teardown_sem = base + offset + 2 * FIELD_SIZE,
    };
}

// ═══════════════════════════════════════════════════════════
// Debug Dump Implementation
// ═══════════════════════════════════════════════════════════

void EriscL1Layout::dump(std::ostream& os, bool show_gaps) const {
    os << "EriscL1Layout {\n";
    os << "  Base: 0x" << std::hex << base_address_ << std::dec << "\n";
    os << "  Max:  0x" << std::hex << max_address_ << std::dec << "\n";
    os << "  Sender channels: " << num_sender_channels_ << "\n";
    os << "  Downstream EDMs: " << num_downstream_edms_ << "\n\n";

    dump_header(os, show_gaps);

    DumpStats stats{.prev_end = base_address_};
    for (size_t i = 0; i < static_cast<size_t>(L1Block::COUNT); ++i) {
        dump_block_row(os, static_cast<L1Block>(i), show_gaps, stats);
    }

    dump_summary(os, show_gaps, stats);
    os << "}\n";
}

void EriscL1Layout::dump_header(std::ostream& os, bool show_gaps) const {
    os << "  " << std::setw(32) << std::left << "Block" << std::setw(12) << std::right << "Start" << std::setw(12)
       << "End" << std::setw(8) << "Size";
    if (show_gaps) {
        os << std::setw(8) << "Gap";
    }
    os << "\n";
    os << "  " << std::string(show_gaps ? 72 : 64, '-') << "\n";
}

void EriscL1Layout::dump_block_row(std::ostream& os, L1Block block, bool show_gaps, DumpStats& stats) const {
    const auto& region = regions_[idx(block)];
    const auto name = enchantum::to_string(block);

    // Skipped block
    if (region.size == 0) {
        os << "  " << std::setw(32) << std::left << name << std::setw(12) << std::right << "-" << std::setw(12) << "-"
           << std::setw(8) << "(skip)";
        if (show_gaps) {
            os << std::setw(8) << "-";
        }
        os << "\n";
        return;
    }

    // Calculate gap
    size_t gap = (region.start_address > stats.prev_end) ? region.start_address - stats.prev_end : 0;

    // Print row
    os << "  " << std::setw(32) << std::left << name << std::hex << std::setw(12) << std::right << region.start_address
       << std::setw(12) << (region.start_address + region.size) << std::dec << std::setw(8) << region.size;

    if (show_gaps) {
        if (gap > 0) {
            os << std::setw(8) << gap;
        } else {
            os << std::setw(8) << "-";
        }
    }
    os << "\n";

    // Update stats
    if (block != L1Block::CHANNEL_BUFFERS) {
        stats.total_used += region.size;
    }
    stats.total_gaps += gap;
    stats.prev_end = region.start_address + region.size;
}

void EriscL1Layout::dump_summary(std::ostream& os, bool show_gaps, const DumpStats& stats) const {
    os << "  " << std::string(show_gaps ? 72 : 64, '-') << "\n";
    os << "  Control regions: " << stats.total_used << " B\n";

    if (show_gaps && stats.total_gaps > 0) {
        os << "  Alignment gaps:  " << stats.total_gaps << " B\n";
    }

    const auto& buffers = regions_[idx(L1Block::CHANNEL_BUFFERS)];
    if (buffers.size > 0) {
        os << "  Buffer region:   " << buffers.size << " B\n";
        os << "  Total L1:        " << (stats.total_used + buffers.size + stats.total_gaps) << " B\n";
    }
}

std::string EriscL1Layout::to_string(bool show_gaps) const {
    std::ostringstream oss;
    dump(oss, show_gaps);
    return oss.str();
}

void EriscL1Layout::log_layout([[maybe_unused]] bool show_gaps) const {
    log_debug(tt::LogFabric, "L1 Layout:\n{}", to_string(show_gaps));
}

}  // namespace tt::tt_fabric
