// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetcher/Dispatcher CMD interfaces
//  - CMD ID enums: identify the command to execute
//  - CMD structures: contain parameters for each command
//  - FLAGs: densely packed bits to configure commands

#pragma once

// Prefetcher CMD ID enums
enum CQPrefetchCmdId : uint8_t {
    CQ_PREFETCH_CMD_ILLEGAL = 0,              // common error value
    CQ_PREFETCH_CMD_RELAY_DRAM_PAGED = 1,     // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE = 2,         // relay (inline) data from CmdDatQ to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH = 3, // same as above, but doesn't flush the page to dispatcher
    CQ_PREFETCH_CMD_WRAP = 4,                 // go to top of host pull buffer
    CQ_PREFETCH_CMD_STALL = 5,                // drain pipe through dispatcher
    CQ_PREFETCH_CMD_DEBUG = 6,                // log waypoint data to watcher
    CQ_PREFETCH_CMD_TERMINATE = 7,            // quit
};

// Dispatcher CMD ID enums
enum CQDispatchCmdId : uint8_t {
    CQ_DISPATCH_CMD_ILLEGAL = 0,            // common error value
    CQ_DISPATCH_CMD_WRITE = 1,              // write data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_PAGED = 2,        // write banked/paged data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WAIT = 3,               // wait until workers are done
    CQ_DISPATCH_CMD_GO = 4,                 // send go message
    CQ_DISPATCH_CMD_SINK = 5,               // act as a data sink (for testing)
    CQ_DISPATCH_CMD_DEBUG = 6,              // log waypoint data to watcher
    CQ_DISPATCH_CMD_TERMINATE = 7,          // quit
};

//////////////////////////////////////////////////////////////////////////////

// Shared commands
struct CQGenericDebugCmd {
    uint16_t key;                          // prefetcher/dispatcher all write to watcher
    uint32_t checksum;                     // checksum of payload
    uint32_t size;                         // size of payload
    uint32_t stride;                       // stride to next command
} __attribute__((packed));

//////////////////////////////////////////////////////////////////////////////

// Prefetcher CMD structures
struct CQPrefetchBaseCmd {
    enum CQPrefetchCmdId cmd_id;
    uint8_t flags;
} __attribute__((packed));

struct CQPrefetchRelayDramPagedCmd {
    uint8_t pad;
    uint8_t start_page_id;                    // 0..nDRAMs-1, first bank
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));;

struct CQPrefetchRelayInlineCmd {
    uint16_t pad;
    uint32_t length;
    uint32_t stride;
} __attribute__((packed));;

struct CQPrefetchCmd {
    CQPrefetchBaseCmd base;
    union {
        CQPrefetchRelayDramPagedCmd relay_dram_paged;
        CQPrefetchRelayInlineCmd relay_inline;
        CQGenericDebugCmd debug;
    } __attribute__((packed));
};

//////////////////////////////////////////////////////////////////////////////

// Dispatcher CMD structures
struct CQDispatchBaseCmd {
    enum CQDispatchCmdId cmd_id;
    uint8_t flags;
} __attribute__((packed));

struct CQDispatchWriteCmd {
    uint16_t pad;
    uint32_t dst_noc_addr;
    uint32_t dst_addr;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWritePagedCmd {
    uint16_t pad;
    uint32_t dst_base_addr;
    uint32_t dst_page_size;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWaitCmd {
    uint16_t pad;
    uint32_t count;
};

struct CQDispatchCmd {
    CQDispatchBaseCmd base;

    union {
        CQDispatchWriteCmd write;
        CQDispatchWritePagedCmd write_paged;
        CQDispatchWaitCmd wait;
        CQGenericDebugCmd debug;
    } __attribute__((packed));
};


// Dispatcher CMD flags
constexpr uint32_t CQ_DISPATCH_CMD_FLAG_MULTICAST     = 0x1;
constexpr uint32_t CQ_DISPATCH_CMD_FLAG_GO_BARRIER    = 0x2;


static_assert(sizeof(CQPrefetchBaseCmd) == sizeof(uint16_t)); // if this fails, padding above needs to be adjusted
static_assert(sizeof(CQDispatchBaseCmd) == sizeof(uint16_t)); // if this fails, padding above needs to be adjusted
static_assert((sizeof(CQPrefetchCmd) & 0xf) == 0);
static_assert((sizeof(CQDispatchCmd) & 0xf) == 0);
