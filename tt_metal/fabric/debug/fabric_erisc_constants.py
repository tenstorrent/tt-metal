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
# - tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp
# - tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
# These streams are used for fabric flow control and backpressure management

FABRIC_STREAM_GROUPS = {
    "sender_free_slots": {
        "stream_ids": [17, 18, 19, 20, 21],
        "labels": ["sender_ch0", "sender_ch1", "sender_ch2", "sender_ch3", "sender_ch4_vc1"],
        "title": "SENDER CHANNEL FREE SLOTS",
        "description": "Fabric sender channel buffer space available (free slots) for flow control",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    "receiver_free_slots": {
        "stream_ids": [12, 13, 14, 15, 16],
        "labels": ["recv_east", "recv_west", "recv_north", "recv_south", "recv_downstream_vc1"],
        "title": "RECEIVER CHANNEL FREE SLOTS",
        "description": "Fabric receiver channel buffer space available (free slots) for flow control",
        "register_type": "BUF_SPACE_AVAILABLE",
    },
    "all_fabric_free_slots": {
        "stream_ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "labels": [
            "recv_east",  # Stream 12: East receiver channel
            "recv_west",  # Stream 13: West receiver channel
            "recv_north",  # Stream 14: North receiver channel
            "recv_south",  # Stream 15: South receiver channel
            "recv_downstream_vc1",  # Stream 16: Downstream VC1 receiver
            "sender_ch0",  # Stream 17: Sender channel 0
            "sender_ch1",  # Stream 18: Sender channel 1
            "sender_ch2",  # Stream 19: Sender channel 2
            "sender_ch3",  # Stream 20: Sender channel 3
            "sender_ch4_vc1",  # Stream 21: Sender channel 4 VC1
        ],
        "title": "ALL FABRIC STREAM FREE SLOTS",
        "description": "Complete view of all fabric EDM sender/receiver buffer space for flow control debugging",
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
