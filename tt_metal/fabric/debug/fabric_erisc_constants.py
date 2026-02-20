# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Constants for ERISC Fabric Debugging

This module defines architecture-specific constants, register addresses,
and configuration data for debugging the fabric EDM (Ethernet Data Movement)
system on Tenstorrent devices.

The constants are organized into several categories:
- ERISC register addresses for core control and status
- Stream register indices and masking constants
- Fabric stream group definitions for flow control debugging
- Architecture mapping and detection support
"""

# ==============================================================================
# ERISC REGISTER ADDRESSES
# ==============================================================================

# Common ERISC register addresses used by the fabric EDM for debugging
ERISC_REGISTERS = {
    "ETH_RISC_RESET": 0xFFB121B0,  # Reset control register
    "ETH_RISC_WALL_CLOCK_0": 0xFFB121F0,  # Wall clock register 0 (low 32 bits)
    "ETH_RISC_WALL_CLOCK_1": 0xFFB121F4,  # Wall clock register 1 (high 32 bits)
    "ETH_RISC_REGS_START": 0x00015A10,  # Base of ERISC register space
}

# Default addresses to read when none are specified - useful for basic core health checks
DEFAULT_ADDRESSES = [
    ERISC_REGISTERS["ETH_RISC_RESET"],  # Check if core is out of reset
    ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_0"],  # Monitor core activity via wall clock
    ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_1"],  # High bits of wall clock
]

# ==============================================================================
# STREAM REGISTER CONSTANTS
# ==============================================================================

# NOC overlay stream register base addresses and layout
NOC_OVERLAY_START_ADDR = 0xFFB40000  # Base address of NOC overlay register space
NOC_STREAM_REG_SPACE_SIZE = 0x1000  # Size of register space per stream (4KB)

# Stream register value masking for fabric flow control
# On some architectures, raw register values need masking to extract meaningful data
# MEM_WORD_ADDR_WIDTH = 17, REMOTE_DEST_WORDS_FREE_WIDTH = MEM_WORD_ADDR_WIDTH
STREAM_REGISTER_MASK_BITS = 17  # Bits to preserve
STREAM_REGISTER_MASK = (1 << STREAM_REGISTER_MASK_BITS) - 1  # 0x1FFFF (131,071)

# Architecture-specific stream register indices within each stream's register space
# These values come from tt_metal/hw/inc/{arch}/noc/noc_overlay_parameters.h
STREAM_REGISTER_INDICES = {
    "wormhole": {
        "BUF_SPACE_AVAILABLE": 64,  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX
        "BUF_SIZE": 4,  # STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX
        "BUF_SPACE_UPDATE": 34,  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX
    },
    "blackhole": {
        "BUF_SPACE_AVAILABLE": 297,  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX
        "BUF_SIZE": 10,  # STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX
        "BUF_SPACE_UPDATE": 270,  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX
    },
}

# ==============================================================================
# FABRIC STREAM REGISTER GROUPS
# ==============================================================================

# Fabric stream ID assignments from EDM implementation
# Stream IDs and channel assignments come from:
# - tt_metal/fabric/erisc_datamover_builder.hpp (StreamRegAssignments)
# - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp
# - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
# These streams are used for fabric flow control and backpressure management

# IMPORTANT: Interpretation of BUF_SPACE_AVAILABLE values:
# - Streams 14-29 (buffer free slots): HIGH values = good (buffers have space)
# - Streams 0-13 (ack/completion): LOW/ZERO values = good (remote side consuming immediately)
#   For ack/completion streams, the register shows remote destination buffer space.
#   Zero means remote side is processing acks/completions as they arrive (expected when idle).

FABRIC_STREAM_GROUPS = {
    # ========== Buffer Free Slots (Flow Control) ==========
    "sender_free_slots": {
        "stream_ids": [22, 23, 24, 25, 26, 27, 28, 29],
        "labels": [
            "sender_ch0",
            "sender_ch1",
            "sender_ch2",
            "sender_ch3",
            "sender_ch4",
            "sender_ch5",
            "sender_ch6",
            "sender_ch7",
        ],
        "title": "SENDER CHANNEL FREE SLOTS",
        "description": "Fabric sender channel buffer space available (free slots) for flow control",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    "receiver_free_slots": {
        "stream_ids": [14, 15, 16, 17, 18, 19, 20, 21],
        "labels": [
            "recv_vc0_edge1",
            "recv_vc0_edge2",
            "recv_vc0_edge3",
            "recv_vc0_edge4_z",
            "recv_vc1_edge1",
            "recv_vc1_edge2",
            "recv_vc1_edge3",
            "recv_vc1_edge4_z",
        ],
        "title": "RECEIVER CHANNEL FREE SLOTS",
        "description": "Fabric receiver channel buffer space available (free slots) for flow control",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    "all_fabric_free_slots": {
        "stream_ids": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "labels": [
            "recv_vc0_edge1",  # Stream 14: VC0 downstream edge 1
            "recv_vc0_edge2",  # Stream 15: VC0 downstream edge 2
            "recv_vc0_edge3",  # Stream 16: VC0 downstream edge 3
            "recv_vc0_edge4_z",  # Stream 17: VC0 downstream edge 4 (Z)
            "recv_vc1_edge1",  # Stream 18: VC1 downstream edge 1
            "recv_vc1_edge2",  # Stream 19: VC1 downstream edge 2
            "recv_vc1_edge3",  # Stream 20: VC1 downstream edge 3
            "recv_vc1_edge4_z",  # Stream 21: VC1 downstream edge 4 (Z)
            "sender_ch0",  # Stream 22: Sender channel 0
            "sender_ch1",  # Stream 23: Sender channel 1
            "sender_ch2",  # Stream 24: Sender channel 2
            "sender_ch3",  # Stream 25: Sender channel 3
            "sender_ch4",  # Stream 26: Sender channel 4
            "sender_ch5",  # Stream 27: Sender channel 5
            "sender_ch6",  # Stream 28: Sender channel 6
            "sender_ch7",  # Stream 29: Sender channel 7 (Z)
        ],
        "title": "ALL FABRIC STREAM FREE SLOTS",
        "description": "Complete view of all fabric EDM sender/receiver buffer space for flow control debugging",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    # ========== Packet Acknowledgment Streams (Remote Buffer Status) ==========
    "sender_acks": {
        "stream_ids": [2, 3, 4, 5],
        "labels": ["sender_ch0_ack", "sender_ch1_ack", "sender_ch2_ack", "sender_ch3_ack"],
        "title": "SENDER CHANNEL PACKET ACKNOWLEDGMENT STREAMS",
        "description": "Remote buffer space for ack streams (to_sender_X_pkts_acked). LOW/ZERO = remote consuming acks",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    # ========== Packet Completion Streams (Remote Buffer Status) ==========
    "sender_completions": {
        "stream_ids": [6, 7, 8, 9, 10, 11, 12, 13],
        "labels": [
            "sender_ch0_comp",
            "sender_ch1_comp",
            "sender_ch2_comp",
            "sender_ch3_comp",
            "sender_ch4_comp",
            "sender_ch5_comp",
            "sender_ch6_comp",
            "sender_ch7_comp",
        ],
        "title": "SENDER CHANNEL PACKET COMPLETION STREAMS",
        "description": "Remote buffer space for completion streams (to_sender_X_pkts_completed). LOW/ZERO = remote consuming completions",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    # ========== Packet Sent Streams (Remote Buffer Status) ==========
    "receiver_pkts_sent": {
        "stream_ids": [0, 1],
        "labels": ["recv_ch0_sent", "recv_ch1_sent"],
        "title": "RECEIVER CHANNEL PACKET SENT STREAMS",
        "description": "Remote buffer space for packet sent streams (to_receiver_X_pkts_sent). LOW/ZERO = remote consuming packets",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    # ========== Combined Acks and Completions ==========
    "all_acks_and_completions": {
        "stream_ids": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "labels": [
            "sender_ch0_ack",  # Stream 2
            "sender_ch1_ack",  # Stream 3
            "sender_ch2_ack",  # Stream 4
            "sender_ch3_ack",  # Stream 5
            "sender_ch0_comp",  # Stream 6
            "sender_ch1_comp",  # Stream 7
            "sender_ch2_comp",  # Stream 8
            "sender_ch3_comp",  # Stream 9
            "sender_ch4_comp",  # Stream 10
            "sender_ch5_comp",  # Stream 11
            "sender_ch6_comp",  # Stream 12
            "sender_ch7_comp",  # Stream 13
        ],
        "title": "ALL SENDER ACK AND COMPLETION STREAMS",
        "description": "Remote buffer space for ack/completion streams. LOW/ZERO = remote side processing normally",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
}

# Default configuration for fabric stream debugging
DEFAULT_FABRIC_STREAM_GROUP = "all_fabric_free_slots"  # Most comprehensive view
DEFAULT_FABRIC_REGISTER = "BUF_SPACE_AVAILABLE"  # Most useful for flow control

# NOTE: Future extensions can add groups for different register types:
# - Buffer sizes: {..., "register_type": "BUF_SIZE", ...}
# - Update counters: {..., "register_type": "BUF_SPACE_UPDATE", ...}
# The dumper script infrastructure supports this with minimal changes.

# ==============================================================================
# ARCHITECTURE DETECTION AND MAPPING
# ==============================================================================

# Mapping from device architecture strings to normalized constants
# Used for selecting the correct stream register indices and behavior
ARCHITECTURE_MAPPING = {
    "wormhole_b0": "wormhole",  # Wormhole B0 silicon
    "wormhole": "wormhole",  # Generic Wormhole
    "blackhole": "blackhole",  # Blackhole architecture
}
