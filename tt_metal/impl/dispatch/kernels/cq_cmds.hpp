// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetcher/Dispatcher CMD interfaces
//  - CMD ID enums: identify the command to execute
//  - CMD structures: contain parameters for each command
//  - FLAGs: densely packed bits to configure commands

#pragma once

constexpr uint32_t CQ_PREFETCH_CMD_BARE_MIN_SIZE = 32; // for NOC PCIe alignemnt
constexpr uint32_t CQ_DISPATCH_CMD_SIZE = 16;          // for L1 alignment

// Prefetcher CMD ID enums
enum CQPrefetchCmdId : uint8_t {
    CQ_PREFETCH_CMD_ILLEGAL = 0,              // common error value
    CQ_PREFETCH_CMD_RELAY_PAGED = 1,          // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE = 2,         // relay (inline) data from CmdDatQ to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH = 3, // same as above, but doesn't flush the page to dispatcher
    CQ_PREFETCH_CMD_STALL = 4,                // drain pipe through dispatcher
    CQ_PREFETCH_CMD_DEBUG = 5,                // log waypoint data to watcher, checksum
    CQ_PREFETCH_CMD_TERMINATE = 6,            // quit
};

// Dispatcher CMD ID enums
enum CQDispatchCmdId : uint8_t {
    CQ_DISPATCH_CMD_ILLEGAL = 0,            // common error value
    CQ_DISPATCH_CMD_WRITE = 1,              // write data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_PAGED = 2,        // write banked/paged data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_PACKED = 3,       // write to multiple noc addresses with packed data
    CQ_DISPATCH_CMD_WAIT = 4,               // wait until workers are done
    CQ_DISPATCH_CMD_GO = 5,                 // send go message
    CQ_DISPATCH_CMD_SINK = 6,               // act as a data sink (for testing)
    CQ_DISPATCH_CMD_DEBUG = 7,              // log waypoint data to watcher, checksum
    CQ_DISPATCH_CMD_TERMINATE = 8,          // quit
};

//////////////////////////////////////////////////////////////////////////////

// Shared commands
struct CQGenericDebugCmd {
    uint8_t pad;
    uint16_t key;                          // prefetcher/dispatcher all write to watcher
    uint32_t checksum;                     // checksum of payload
    uint32_t size;                         // size of payload
    uint32_t stride;                       // stride to next Cmd (may be within the payload)
} __attribute__((packed));

//////////////////////////////////////////////////////////////////////////////

// Prefetcher CMD structures
struct CQPrefetchBaseCmd {
    enum CQPrefetchCmdId cmd_id;
} __attribute__((packed));

struct CQPrefetchRelayPagedCmd {
    uint8_t pad1;
    uint8_t is_dram;          // one flag, false=l1
    uint8_t start_page;
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));;

struct CQPrefetchRelayInlineCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t length;
    uint32_t stride;          // explicit stride saves a few insns on device
} __attribute__((packed));;

struct CQPrefetchCmd {
    CQPrefetchBaseCmd base;
    union {
        CQPrefetchRelayPagedCmd relay_paged;
        CQPrefetchRelayInlineCmd relay_inline;
        CQGenericDebugCmd debug;
    } __attribute__((packed));
};

//////////////////////////////////////////////////////////////////////////////

// Dispatcher CMD structures
struct CQDispatchBaseCmd {
    enum CQDispatchCmdId cmd_id;
} __attribute__((packed));

struct CQDispatchWriteCmd {
    uint8_t is_multicast;
    uint16_t pad1;
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWritePagedCmd {
    uint8_t pad1;
    uint8_t is_dram;          // one flag, false=l1
    uint8_t start_page;
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));

struct CQDispatchWritePackedCmd {
    uint8_t is_multicast;
    uint16_t pad1;
    uint32_t count;           // number of sub-cmds
    uint32_t size;            // size of each packet, stride is padded to L1 alignment
    uint32_t addr;            // common memory address across all packed SubCmds
} __attribute__((packed));

struct CQDispatchWritePackedSubCmd {
    uint32_t noc_xy_addr;     // unique XY address for each SubCmd
} __attribute__((packed));

struct CQDispatchWaitCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t addr;
    uint32_t count;
};

struct CQDispatchCmd {
    CQDispatchBaseCmd base;

    union {
        CQDispatchWriteCmd write;
        CQDispatchWritePagedCmd write_paged;
        CQDispatchWritePackedCmd write_packed;
        CQDispatchWaitCmd wait;
        CQGenericDebugCmd debug;
    } __attribute__((packed));
};


static_assert(sizeof(CQPrefetchBaseCmd) == sizeof(uint8_t)); // if this fails, padding above needs to be adjusted
static_assert(sizeof(CQDispatchBaseCmd) == sizeof(uint8_t)); // if this fails, padding above needs to be adjusted
static_assert((sizeof(CQPrefetchCmd) & 0xf) == 0);
static_assert((sizeof(CQDispatchCmd) & 0xf) == 0);
