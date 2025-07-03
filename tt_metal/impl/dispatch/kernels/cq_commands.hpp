// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file is shared by host and device CQ dispatch

// Prefetcher/Dispatcher CMD interfaces
//  - CMD ID enums: identify the command to execute
//  - CMD structures: contain parameters for each command

#pragma once

#include <cstdint>

constexpr uint32_t CQ_DISPATCH_CMD_SIZE = 16;  // for L1 alignment

// Prefetcher CMD ID enums
enum CQPrefetchCmdId : uint8_t {
    CQ_PREFETCH_CMD_ILLEGAL = 0,               // common error value
    CQ_PREFETCH_CMD_RELAY_LINEAR = 1,          // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_PAGED = 2,           // relay banked/paged data from src_noc to dispatcher
    CQ_PREFETCH_CMD_RELAY_PAGED_PACKED = 3,    // relay banked/paged data from multiple srcs to dispacher
    CQ_PREFETCH_CMD_RELAY_INLINE = 4,          // relay (inline) data from CmdDatQ to dispatcher
    CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH = 5,  // same as above, but doesn't flush the page to dispatcher
    CQ_PREFETCH_CMD_EXEC_BUF = 6,              // execute commands from a buffer
    CQ_PREFETCH_CMD_EXEC_BUF_END = 7,  // finish executing commands from a buffer (return), payload like relay_inline
    CQ_PREFETCH_CMD_STALL = 8,         // drain pipe through dispatcher
    CQ_PREFETCH_CMD_DEBUG = 9,         // log waypoint data to watcher, checksum
    CQ_PREFETCH_CMD_TERMINATE = 10,    // quit
    CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER = 11,    // Copy paged data to the ringbuffer
    CQ_PREFETCH_CMD_SET_RINGBUFFER_OFFSET = 12,  // Set an offset in the ringbuffer for later reads.
    CQ_PREFETCH_CMD_RELAY_RINGBUFFER = 13,       // Relay data from the ringbuffer to the dispatcher
    CQ_PREFETCH_CMD_MAX_COUNT,                   // for checking legal IDs
};

// Dispatcher CMD ID enums
enum CQDispatchCmdId : uint8_t {
    CQ_DISPATCH_CMD_ILLEGAL = 0,              // common error value
    CQ_DISPATCH_CMD_WRITE_LINEAR = 1,         // write data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_LINEAR_H = 2,       // write data from dispatcher to dst_noc on dispatch_h chip
    CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST = 3,  // like write, dedicated to writing to host
    CQ_DISPATCH_CMD_WRITE_PAGED = 4,          // write banked/paged data from dispatcher to dst_noc
    CQ_DISPATCH_CMD_WRITE_PACKED = 5,         // write to multiple noc addresses with packed data
    CQ_DISPATCH_CMD_WRITE_PACKED_LARGE = 6,   // write to multiple noc/dst addresses and varying lengnths w/ packed data
    CQ_DISPATCH_CMD_WAIT = 7,                 // wait until workers are done
    CQ_DISPATCH_CMD_SINK = 8,                 // act as a data sink (for testing)
    CQ_DISPATCH_CMD_DEBUG = 9,                // log waypoint data to watcher, checksum
    CQ_DISPATCH_CMD_DELAY = 10,               // insert delay (for testing)
    CQ_DISPATCH_CMD_EXEC_BUF_END = 11,        // dispatch_d notify prefetch_h that exec_buf has completed
    CQ_DISPATCH_CMD_SET_WRITE_OFFSET = 12,  // set the offset to add to all non-host destination addresses (relocation)
    CQ_DISPATCH_CMD_TERMINATE = 13,         // quit
    CQ_DISPATCH_CMD_SEND_GO_SIGNAL = 14,
    CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL = 15,
    CQ_DISPATCH_SET_NUM_WORKER_SEMS = 16,
    CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA = 17,
    CQ_DISPATCH_CMD_MAX_COUNT,  // for checking legal IDs
};

enum GoSignalMcastSettings : uint8_t {
    SEND_MCAST = 1,
    SEND_UNICAST = 2,
};

enum DispatcherSelect : uint8_t {
    DISPATCH_MASTER = 0,
    DISPATCH_SUBORDINATE = 1,
};

//////////////////////////////////////////////////////////////////////////////

// Shared commands
struct CQGenericDebugCmd {
    uint8_t pad;
    uint16_t key;       // prefetcher/dispatcher all write to watcher
    uint32_t size;      // size of payload
    uint32_t stride;    // stride to next Cmd (may be within the payload)
} __attribute__((packed));

//////////////////////////////////////////////////////////////////////////////

// Prefetcher CMD structures
struct CQPrefetchBaseCmd {
    enum CQPrefetchCmdId cmd_id;
} __attribute__((packed));

struct CQPrefetchRelayLinearCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint32_t length;
} __attribute__((packed));
;

constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK = 0xff;
constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT = 15;
constexpr uint32_t CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK = 0x7fff;

struct CQPrefetchRelayPagedCmd {
    uint8_t start_page;
    uint16_t is_dram_and_length_adjust;  // is_dram flag and bytes subtracted from size (multiple of 32)
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));

struct CQPrefetchRelayPagedPackedCmd {
    uint8_t pad1;
    uint16_t count;
    uint32_t total_length;  // aggregate length of all sub-read-cmds
    uint32_t stride;        // stride to start of next cmd
} __attribute__((packed));

struct CQPrefetchRelayPagedPackedSubCmd {
    uint16_t start_page;  // 0..nbanks-1
    uint16_t log_page_size;
    uint32_t base_addr;
    uint32_t length;  // multiple of DRAM alignment, <= half scratch_db_size
} __attribute__((packed));

// Current implementation limit is based on size of the l1_cache which stores the sub_cmds
constexpr uint32_t CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS = 35;

struct CQPrefetchRelayInlineCmd {
    uint8_t dispatcher_type;
    uint16_t pad;
    uint32_t length;
    uint32_t stride;  // explicit stride saves a few insns on device
} __attribute__((packed));

struct CQPrefetchExecBufCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t base_addr;
    uint32_t log_page_size;
    uint32_t pages;
} __attribute__((packed));

// Reset the write pointer to the start of the ring buffer. If this isn't set,
// the ringbuffer will never wrap around.
constexpr uint32_t CQ_PREFETCH_PAGED_TO_RING_BUFFER_FLAG_RESET_TO_START = 1;
// Will always flush writes before reading.
struct CQPrefetchPagedToRingbufferCmd {
    uint8_t flags;
    uint8_t log2_page_size;
    uint8_t start_page;
    uint32_t wp_offset_update;  // set final increment ringbuffer write pointer
    uint32_t base_addr;  // Base address of the interleaved buffer to read from.
    uint32_t length;     // multiple of DRAM alignment
} __attribute__((packed));

struct CQPrefetchSetRingbufferOffsetCmd {
    uint32_t offset;
    uint16_t pad1;
    uint8_t pad2;
    uint8_t update_wp;  // if set, the ringbuffer write pointer will be updated to the offset
} __attribute__((packed));

// Current implementation limit is based on size of the l1_cache which stores the sub_cmds
constexpr uint32_t CQ_PREFETCH_CMD_RELAY_RINGBUFFER_MAX_SUB_CMDS = 52;
struct CQPrefetchRelayRingbufferCmd {
    uint8_t pad1;
    uint16_t count;
    uint32_t stride;  // Size of the full command, including subcmds
} __attribute__((packed));

struct CQPrefetchRelayRingbufferSubCmd {
    uint32_t start;  // The ringbuffer offset will be added to this.
    uint32_t length;
} __attribute__((packed));

struct CQPrefetchCmd {
    CQPrefetchBaseCmd base;
    union {
        CQPrefetchRelayLinearCmd relay_linear;
        CQPrefetchRelayPagedCmd relay_paged;
        CQPrefetchRelayPagedPackedCmd relay_paged_packed;
        CQPrefetchRelayInlineCmd relay_inline;
        CQPrefetchExecBufCmd exec_buf;
        CQGenericDebugCmd debug;
        CQPrefetchPagedToRingbufferCmd paged_to_ringbuffer;
        CQPrefetchSetRingbufferOffsetCmd set_ringbuffer_offset;
        CQPrefetchRelayRingbufferCmd relay_ringbuffer;
    } __attribute__((packed));
};

//////////////////////////////////////////////////////////////////////////////

// Dispatcher CMD structures
struct CQDispatchBaseCmd {
    enum CQDispatchCmdId cmd_id;
} __attribute__((packed));

struct CQDispatchWriteCmd {
    uint8_t num_mcast_dests;  // 0 = unicast, 1+ = multicast
    uint8_t write_offset_index;
    uint8_t pad1;
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint32_t length;
} __attribute__((packed));

struct CQDispatchWriteHostCmd {
    uint8_t is_event;  // one flag, false=read buffer
    uint16_t pad1;
    uint32_t pad2;
    uint32_t pad3;
    uint32_t length;
} __attribute__((packed));

constexpr uint16_t CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX = 0xFFFF;

struct CQDispatchWritePagedCmd {
    uint8_t is_dram;  // one flag, false=l1
    uint16_t start_page;
    uint32_t base_addr;
    uint32_t page_size;
    uint32_t pages;
} __attribute__((packed));

constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE = 0x00;
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST = 0x01;
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE = 0x02;

constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT = 4;
enum CQDispatchCmdPackedWriteType {
    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_MASK = 0xf << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT,
    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_RTA = 0x1 << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT,
    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_LAUNCH = 0x2 << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT,
    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_SEMS = 0x3 << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT,
    CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_EVENT = 0x4 << CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT,
};

struct CQDispatchWritePackedCmd {
    uint8_t flags;   // see above
    uint16_t count;  // number of sub-cmds (max 1020 unicast, 510 mcast). Max num sub-cmds =
                     // (DispatchSettings::TRANSFER_PAGE_SIZE - sizeof(CQDispatchCmd)) /
                     // sizeof(CQDispatchWritePacked*castSubCmd)
    uint16_t write_offset_index;
    uint16_t size;  // size of each packet, stride is padded to L1 alignment and less than dispatch_cb_page_size
    uint32_t addr;  // common memory address across all packed SubCmds
} __attribute__((packed));

struct CQDispatchWritePackedUnicastSubCmd {
    uint32_t noc_xy_addr;  // unique XY address for each SubCmd
} __attribute__((packed));

struct CQDispatchWritePackedMulticastSubCmd {
    uint32_t noc_xy_addr;  // unique XY address for each SubCmd
    uint32_t num_mcast_dests;
} __attribute__((packed));

constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_NONE = 0x00;
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK = 0x01;
struct CQDispatchWritePackedLargeSubCmd {
    uint32_t noc_xy_addr;
    uint32_t addr;
    uint16_t length;  // multiples of L1 cache line alignment
    uint8_t num_mcast_dests;
    uint8_t flags;
} __attribute__((packed));

constexpr inline __attribute__((always_inline)) uint32_t
get_packed_write_max_multicast_sub_cmds(uint32_t packed_write_max_unicast_sub_cmds) {
    uint32_t packed_write_max_multicast_sub_cmds = packed_write_max_unicast_sub_cmds *
                                                   sizeof(CQDispatchWritePackedUnicastSubCmd) /
                                                   sizeof(CQDispatchWritePackedMulticastSubCmd);
    return packed_write_max_multicast_sub_cmds;
}

// Current implementation limit is based on size of the l1_cache which stores the sub_cmds
constexpr uint32_t CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS = 35;

enum CQDispatchCmdPackedWriteLargeType {
    CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN = 0,
    CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_CBS_SEMS_CRTAS = 1,
    CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_PROGRAM_BINARIES = 2,
};

// More flexible/slower than WritePacked
// Removes size constraints
// Implicitly mcast
struct CQDispatchWritePackedLargeCmd {
    uint8_t type;
    uint16_t count;  // number of sub-cmds
    uint16_t alignment;
    uint16_t write_offset_index;
} __attribute__((packed));

constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_NONE = 0x00;
// Issue a write barrier
constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER = 0x01;
// Increment prefetch semaphore
constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_NOTIFY_PREFETCH = 0x02;
// Wait for a count value on memory.
constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY = 0x04;
// Wait for a count value on a stream
constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM = 0x08;
// Clear a count value on a stream.
constexpr uint32_t CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM = 0x10;

struct CQDispatchWaitCmd {
    uint8_t flags;    // see above
    uint16_t stream;  // stream to read/write
    uint32_t addr;   // address to read
    uint32_t count;  // wait while address is < count
} __attribute__((packed));

struct CQDispatchDelayCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t delay;
} __attribute__((packed));

// The maximum value allowed for offset_count in CQDispatchSetWriteOffsetCmd.
constexpr uint32_t CQ_DISPATCH_MAX_WRITE_OFFSETS = 4;

struct CQDispatchSetWriteOffsetCmd {
    uint8_t offset_count;  // Number of uint32_t offsets this command sets. Offsets are stored after the CQDispatchCmd.
    uint16_t program_host_id;  // Program Host ID for upcoming commands. Used for profiling.
} __attribute__((packed));

struct CQDispatchSetUnicastOnlyCoresCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t num_unicast_only_cores;
} __attribute__((packed));

struct CQDispatchGoSignalMcastCmd {
    uint32_t go_signal;
    uint8_t num_mcast_txns;
    uint8_t num_unicast_txns;
    uint8_t noc_data_start_index;
    uint32_t wait_count;
    uint32_t wait_stream;  // Index of the stream to wait on
} __attribute__((packed));

struct CQDispatchNotifySubordinateGoSignalCmd {
    // sends a counter update to dispatch_s when it sees this cmd
    uint8_t wait;  // if true, issue a write barrier before sending signal to dispatch_s
    uint16_t index_bitmask;
    uint32_t pad3;
} __attribute__((packed));

struct CQDispatchSetNumWorkerSemsCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t num_worker_sems;
} __attribute__((packed));

struct CQDispatchSetGoSignalNocDataCmd {
    uint8_t pad1;
    uint16_t pad2;
    uint32_t num_words;
} __attribute__((packed));

struct CQDispatchCmd {
    CQDispatchBaseCmd base;

    union {
        CQDispatchWriteCmd write_linear;
        CQDispatchWriteHostCmd write_linear_host;
        CQDispatchWritePagedCmd write_paged;
        CQDispatchWritePackedCmd write_packed;
        CQDispatchWritePackedLargeCmd write_packed_large;
        CQDispatchWaitCmd wait;
        CQGenericDebugCmd debug;
        CQDispatchDelayCmd delay;
        CQDispatchSetWriteOffsetCmd set_write_offset;
        CQDispatchGoSignalMcastCmd mcast;
        CQDispatchSetUnicastOnlyCoresCmd set_unicast_only_cores;
        CQDispatchNotifySubordinateGoSignalCmd notify_dispatch_s_go_signal;
        CQDispatchSetNumWorkerSemsCmd set_num_worker_sems;
        CQDispatchSetGoSignalNocDataCmd set_go_signal_noc_data;
    } __attribute__((packed));
};

//////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(CQPrefetchBaseCmd) == sizeof(uint8_t));  // if this fails, padding above needs to be adjusted
static_assert(sizeof(CQDispatchBaseCmd) == sizeof(uint8_t));  // if this fails, padding above needs to be adjusted
static_assert((sizeof(CQPrefetchCmd) & (CQ_DISPATCH_CMD_SIZE - 1)) == 0);
static_assert((sizeof(CQDispatchCmd) & (CQ_DISPATCH_CMD_SIZE - 1)) == 0);
