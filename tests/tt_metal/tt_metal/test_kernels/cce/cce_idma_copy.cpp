// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One iDMA copy through command buffer 0, with every poll bounded. Completion is a bounded ack poll on trid 0, then a
// bounded poll of the destination data. The command-buffer ack follows the write response; MIM-628 (done fires on
// accept) covers only the iDMA sideband done_id, which no mode here uses. The status line is written only after the
// data poll (MIM-560), through the uncached alias. In kModeIdmaCopyHostVerified the kernel loads no remote operand:
// when the destination is its own TL1 it polls the XOR of the destination against kArgExpectedXor, otherwise there is
// no data poll; the host checks the data. kModeIdmaStrided issues one command per element from address generator 0.
// The MM4x modes (kModeTridBatch, kModeAccumulate, kModeTridTable, kModeTridStream) set the trid per command and wait
// on each trid's pending count, all under one budget. kModeTridBatch and kModeAccumulate run on command buffer 0 or 1;
// the other modes on command buffer 0.

#include "cce_idma_copy_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "risc_common.h"

namespace abi = cce_idma_copy_abi;
using cce_test::arg_u64;
using cce_test::DataCheck;
using cce_test::poll_bounded;
using overlay::ADDRGEN_0;
using overlay::CmdBuf;
using overlay::CMDBUF_0;
using overlay::CMDBUF_1;

static_assert(abi::kMaxBytesInPacket == NOC_OVERLAY_MAX_BYTES_IN_PACKET);

struct Operand {
    uint64_t command_address = 0;
    uint64_t check_address = 0;
};

static bool resolve_operand(uint32_t kind, uint32_t selector, uint64_t offset, Operand& operand) {
    switch (kind) {
        case abi::kKindOwnTl1:
            operand.command_address = experimental::cce_sram_address(selector, offset);
            operand.check_address = reinterpret_cast<uint64_t>(cce_test::own_tl1_words(offset));
            return true;
        case abi::kKindSramSpa:
            operand.command_address = experimental::cce_sram_address(selector, offset);
            operand.check_address = operand.command_address;
            return true;
        case abi::kKindGddr:
            operand.command_address = experimental::cce_gddr_address(selector, offset);
            operand.check_address = operand.command_address;
            return true;
        default: return false;
    }
}

// Element i of the source, at source + i * source_stride_bytes, against element i of the destination, at destination +
// i * destination_stride_bytes; `check` numbers the destination words.
static bool data_landed(
    const Operand& source,
    const Operand& destination,
    uint32_t len_bytes,
    uint32_t num_elements,
    uint32_t source_stride_bytes,
    uint32_t destination_stride_bytes,
    DataCheck& check) {
    check = DataCheck{};
    const uint32_t element_words = len_bytes / sizeof(uint32_t);
    for (uint32_t element = 0; element < num_elements; ++element) {
        const volatile uint32_t* source_words = reinterpret_cast<const volatile uint32_t*>(
            source.check_address + static_cast<uint64_t>(element) * source_stride_bytes);
        const volatile uint32_t* destination_words = reinterpret_cast<const volatile uint32_t*>(
            destination.check_address + static_cast<uint64_t>(element) * destination_stride_bytes);
        for (uint32_t word = 0; word < element_words; ++word) {
            check.record(element * element_words + word, destination_words[word], source_words[word]);
        }
    }
    return check.mismatched_words == 0;
}

// XOR of the destination words against the host's XOR of the source; on a mismatch, `check` holds the two XORs.
static bool destination_xor_matches(
    const volatile uint32_t* destination_words, uint32_t num_words, uint32_t expected_xor, DataCheck& check) {
    uint32_t observed_xor = 0;
    for (uint32_t word = 0; word < num_words; ++word) {
        observed_xor ^= destination_words[word];
    }
    check = DataCheck{};
    check.record(0, observed_xor, expected_xor);
    return check.mismatched_words == 0;
}

// In strided mode address generator 0 supplies every command's addresses (upstream idma_1d_strided_example.cpp), so
// the command buffer gets no source or destination.
static void program_copy(
    uint32_t backend_vc,
    uint32_t len_bytes,
    const Operand& source,
    const Operand& destination,
    bool strided,
    uint32_t num_elements,
    uint32_t source_stride_bytes) {
    overlay::reset_cmdbuf<CMDBUF_0>();
    if (strided) {
        overlay::reset_addrgen<ADDRGEN_0>();
        overlay::setup_src_base_start_addrgen<ADDRGEN_0>(source.command_address);
        overlay::setup_src_inner_loop_addrgen<ADDRGEN_0>(
            source_stride_bytes, static_cast<uint64_t>(num_elements) * source_stride_bytes);
        overlay::setup_dest_base_start_addrgen<ADDRGEN_0>(destination.command_address);
        overlay::setup_dest_inner_loop_addrgen<ADDRGEN_0>(
            len_bytes, static_cast<uint64_t>(num_elements) * len_bytes + 1);
    }
    overlay::idma_setup_as_copy_cmdbuf<CMDBUF_0>(/*wrapping=*/false);
    overlay::setup_ongoing_cmdbuf<CMDBUF_0>({});
    overlay::setup_wrapping_vcs_cmdbuf<CMDBUF_0>(
        {.start = backend_vc, .end = backend_vc},
        {.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
    overlay::setup_trids_cmdbuf<CMDBUF_0>(abi::kTrid);
    overlay::setup_max_bytes_in_packet_cmdbuf<CMDBUF_0>(NOC_OVERLAY_MAX_BYTES_IN_PACKET);
    if (!strided) {
        overlay::set_src_cmdbuf<CMDBUF_0>(source.command_address);
        overlay::set_dest_cmdbuf<CMDBUF_0>(destination.command_address);
    }
    overlay::set_len_cmdbuf<CMDBUF_0>(len_bytes);
}

struct Report {
    uint32_t code = 0;
    uint32_t flags = 0;
    uint32_t vc_space_cycles = 0;
    uint32_t ack_cycles = 0;
    uint32_t data_cycles = 0;
    uint32_t elements_issued = 0;
    uint32_t last_poll = abi::kPollNone;
    bool last_poll_ready = false;
    uint32_t pending_high_water = 0;
    uint32_t probe_first = 0;
    uint32_t probe_second = 0;
    DataCheck check;
};

static void write_status(uint32_t status_offset, const Report& report) {
    volatile uint32_t* status = cce_test::own_tl1_words(status_offset);
    status[abi::kStatusCode] = report.code;
    status[abi::kStatusFlags] = report.flags;
    status[abi::kStatusMismatchedWords] = report.check.mismatched_words;
    status[abi::kStatusFirstMismatchWord] = report.check.first_mismatch_word;
    status[abi::kStatusObserved] = report.check.observed;
    status[abi::kStatusExpected] = report.check.expected;
    status[abi::kStatusVcSpaceCycles] = report.vc_space_cycles;
    status[abi::kStatusAckCycles] = report.ack_cycles;
    status[abi::kStatusDataCycles] = report.data_cycles;
    status[abi::kStatusLastPoll] = report.last_poll;
    status[abi::kStatusLastPollReady] = report.last_poll_ready ? 1 : 0;
    status[abi::kStatusElementsIssued] = report.elements_issued;
    status[abi::kStatusPendingHighWater] = report.pending_high_water;
    status[abi::kStatusProbeFirst] = report.probe_first;
    status[abi::kStatusProbeSecond] = report.probe_second;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}

// Commands of `trid` issued on command buffer CMDBUF and not yet acked.
template <CmdBuf CMDBUF>
static uint32_t pending_count(uint32_t trid) {
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, trid);
}

// Like cce_test::poll_bounded, against a deadline the caller shares across several polls.
template <typename Ready>
static bool poll_until(Ready ready, const cce_test::WaitDeadline& deadline, uint32_t& cycles_spent) {
    const uint64_t start = cce_test::read_cycle_counter();
    bool is_ready = ready();
    while (!is_ready && !deadline.expired()) {
        is_ready = ready();
    }
    cycles_spent += static_cast<uint32_t>(cce_test::read_cycle_counter() - start);
    return is_ready;
}

struct BatchArgs {
    uint32_t backend_vc = 0;
    uint32_t len_bytes = 0;
    uint32_t num_commands = 0;
    uint32_t source_stride_bytes = 0;
    uint32_t destination_stride_bytes = 0;
    uint32_t first_trid = 0;
    uint32_t trid_stride = 0;
    bool snoop = false;
    bool host_verified = false;
    uint32_t expected_xor = 0;
    bool accumulate = false;
    uint32_t accumulate_format = 0;
    uint32_t accumulate_op = 0;
    bool accumulate_disable_saturation = false;
    uint32_t expected_lo = 0;
    uint32_t expected_hi = 0;
    uint32_t poll_region_offset = 0;
    uint32_t poll_region_bytes = 0;
    bool check_at_ack = false;
};

static uint32_t command_trid(const BatchArgs& batch, uint32_t command) {
    return (batch.first_trid + command * batch.trid_stride) % abi::kNumTrids;
}

template <CmdBuf CMDBUF>
static void setup_batch(const BatchArgs& batch, uint32_t first_trid) {
    overlay::reset_cmdbuf<CMDBUF>();
    if (batch.accumulate) {
        overlay::idma_setup_as_atomic_accum_cmdbuf<CMDBUF>(/*wrapping=*/false);
        overlay::l1_atomic_instr_cmdbuf<CMDBUF>(
            batch.accumulate_format, batch.accumulate_disable_saturation, batch.accumulate_op);
    } else {
        overlay::idma_setup_as_copy_cmdbuf<CMDBUF>(/*wrapping=*/false);
    }
    overlay::setup_ongoing_cmdbuf<CMDBUF>({});
    overlay::setup_wrapping_vcs_cmdbuf<CMDBUF>(
        {.start = batch.backend_vc, .end = batch.backend_vc},
        {.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
    overlay::setup_trids_cmdbuf<CMDBUF>(first_trid);
    overlay::setup_max_bytes_in_packet_cmdbuf<CMDBUF>(NOC_OVERLAY_MAX_BYTES_IN_PACKET);
    if (batch.snoop) {
        overlay::setup_packet_tags_cmdbuf<CMDBUF>({.snoop = true});
    }
}

// Issues one command on `trid` once `backend_vc` has space. Returns false when the deadline passed first.
template <CmdBuf CMDBUF>
static bool issue_one(
    uint64_t source_address,
    uint64_t destination_address,
    uint32_t len_bytes,
    uint32_t trid,
    uint32_t backend_vc,
    const cce_test::WaitDeadline& deadline,
    Report& report) {
    overlay::setup_trids_cmdbuf<CMDBUF>(trid);
    overlay::set_src_cmdbuf<CMDBUF>(source_address);
    overlay::set_dest_cmdbuf<CMDBUF>(destination_address);
    overlay::set_len_cmdbuf<CMDBUF>(len_bytes);
    report.last_poll = abi::kPollVcSpace;
    report.last_poll_ready = poll_until(
        [backend_vc] { return overlay::idma_free_space_cmdbuf<CMDBUF>(backend_vc) != 0; },
        deadline,
        report.vc_space_cycles);
    if (!report.last_poll_ready) {
        return false;
    }
    overlay::issue_cmdbuf<CMDBUF>();
    report.elements_issued++;
    return true;
}

// Issues command `command` of the batch on `trid`.
template <CmdBuf CMDBUF>
static bool issue_command(
    const BatchArgs& batch,
    const Operand& source,
    const Operand& destination,
    uint32_t command,
    uint32_t trid,
    const cce_test::WaitDeadline& deadline,
    Report& report) {
    return issue_one<CMDBUF>(
        source.command_address + static_cast<uint64_t>(command) * batch.source_stride_bytes,
        destination.command_address + static_cast<uint64_t>(command) * batch.destination_stride_bytes,
        batch.len_bytes,
        trid,
        batch.backend_vc,
        deadline,
        report);
}

static void record_failure(Report& report, uint32_t success_code, uint32_t failure_code) {
    if (report.code == success_code) {
        report.code = failure_code;
    }
}

static void record_pending(uint32_t pending, Report& report) {
    report.pending_high_water = pending > report.pending_high_water ? pending : report.pending_high_water;
}

// Waits until every trid in `used_trids` (bit per trid) reads 0 pending, recording the rank and cycles of each in the
// trace.
template <CmdBuf CMDBUF>
static void wait_for_trids(
    uint32_t used_trids,
    uint64_t last_issue_cycles,
    const cce_test::WaitDeadline& deadline,
    uint32_t success_code,
    uint32_t* trace,
    Report& report) {
    for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
        trace[trid * abi::kTraceWordsPerTrid] =
            (used_trids & (1u << trid)) != 0 ? abi::kTraceNotAcked : abi::kTraceNotUsed;
        trace[trid * abi::kTraceWordsPerTrid + 1] = 0;
    }
    uint32_t remaining = used_trids;
    uint32_t rank = 0;
    report.last_poll = abi::kPollAck;
    report.last_poll_ready = poll_until(
        [&] {
            for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
                if ((remaining & (1u << trid)) != 0 && pending_count<CMDBUF>(trid) == 0) {
                    remaining &= ~(1u << trid);
                    trace[trid * abi::kTraceWordsPerTrid] = rank++;
                    trace[trid * abi::kTraceWordsPerTrid + 1] =
                        static_cast<uint32_t>(cce_test::read_cycle_counter() - last_issue_cycles);
                }
            }
            return remaining == 0;
        },
        deadline,
        report.ack_cycles);
    if (report.last_poll_ready) {
        report.flags |= abi::kFlagAcked;
    } else {
        record_failure(report, success_code, abi::kCodeAckTimeout);
    }
}

// XOR of the polled words and the first of them against the host's model; `check` numbers them 0 and 1. The polled
// words run from the first command's destination to the end of the last command's, or are the poll region when the
// batch has one.
static bool accumulate_result_matches(const Operand& destination, const BatchArgs& batch, DataCheck& check) {
    const bool poll_region = batch.poll_region_bytes != 0;
    const volatile uint32_t* polled_words = poll_region
                                                ? cce_test::own_tl1_words(batch.poll_region_offset)
                                                : reinterpret_cast<const volatile uint32_t*>(destination.check_address);
    const uint32_t span_words =
        (poll_region ? batch.poll_region_bytes
                     : (batch.num_commands - 1) * batch.destination_stride_bytes + batch.len_bytes) /
        sizeof(uint32_t);
    uint32_t observed_xor = 0;
    for (uint32_t word = 0; word < span_words; ++word) {
        observed_xor ^= polled_words[word];
    }
    check = DataCheck{};
    check.record(0, observed_xor, batch.expected_lo);
    check.record(1, polled_words[0], batch.expected_hi);
    return check.mismatched_words == 0;
}

// Waits for every trid in `used_trids`, then polls the data of every command unless the batch is host-verified with a
// destination other than the kernel's own TL1. An accumulate batch polls its result against the host's model. A
// host-verified batch with a poll region polls the XOR of the region instead of the destination's. With check_at_ack
// the data is first compared once, right after the trids read 0, so a done that comes before the data is not waited
// out.
template <CmdBuf CMDBUF>
static void complete_batch(
    const BatchArgs& batch,
    const Operand& source,
    const Operand& destination,
    uint32_t destination_kind,
    uint32_t used_trids,
    uint64_t last_issue_cycles,
    const cce_test::WaitDeadline& deadline,
    uint32_t success_code,
    uint32_t* trace,
    Report& report) {
    wait_for_trids<CMDBUF>(used_trids, last_issue_cycles, deadline, success_code, trace, report);
    if (batch.check_at_ack && (report.flags & abi::kFlagAcked) != 0) {
        DataCheck at_ack;
        data_landed(
            source,
            destination,
            batch.len_bytes,
            batch.num_commands,
            batch.source_stride_bytes,
            batch.destination_stride_bytes,
            at_ack);
        report.probe_first = at_ack.mismatched_words;
        if (at_ack.mismatched_words != 0) {
            record_failure(report, success_code, abi::kCodeNotLandedAtAck);
        }
    }

    if (batch.host_verified && destination_kind != abi::kKindOwnTl1) {
        return;
    }
    const bool poll_region = batch.poll_region_bytes != 0;
    const volatile uint32_t* polled_words = poll_region
                                                ? cce_test::own_tl1_words(batch.poll_region_offset)
                                                : reinterpret_cast<const volatile uint32_t*>(destination.check_address);
    const uint32_t num_words =
        (poll_region ? batch.poll_region_bytes : batch.num_commands * batch.len_bytes) / sizeof(uint32_t);
    report.last_poll = abi::kPollData;
    report.last_poll_ready = poll_until(
        [&] {
            if (batch.accumulate) {
                return accumulate_result_matches(destination, batch, report.check);
            }
            return batch.host_verified
                       ? destination_xor_matches(polled_words, num_words, batch.expected_xor, report.check)
                       : data_landed(
                             source,
                             destination,
                             batch.len_bytes,
                             batch.num_commands,
                             batch.source_stride_bytes,
                             batch.destination_stride_bytes,
                             report.check);
        },
        deadline,
        report.data_cycles);
    if (report.last_poll_ready) {
        report.flags |= abi::kFlagLanded;
    } else {
        record_failure(report, success_code, abi::kCodeDataTimeout);
    }
}

static void write_trace(uint32_t trace_offset, const uint32_t* trace) {
    if (trace_offset == 0) {
        return;
    }
    volatile uint32_t* words = cce_test::own_tl1_words(trace_offset);
    for (uint32_t word = 0; word < abi::kNumTrids * abi::kTraceWordsPerTrid; ++word) {
        words[word] = trace[word];
    }
}

// kModeTridBatch and kModeAccumulate.
template <CmdBuf CMDBUF>
static Report run_trid_issuer(
    uint32_t mode,
    const BatchArgs& batch,
    const Operand& source,
    const Operand& destination,
    uint32_t destination_kind,
    uint32_t peer_flag_offset,
    uint64_t budget_cycles,
    uint32_t* trace) {
    Report report;
    const uint32_t success_code = batch.host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    report.code = success_code;
    const cce_test::WaitDeadline deadline(budget_cycles);
    setup_batch<CMDBUF>(batch, command_trid(batch, 0));

    if (mode == abi::kModeTridBatch && peer_flag_offset != 0) {
        const volatile uint32_t* start_flag = cce_test::own_tl1_words(peer_flag_offset);
        report.last_poll = abi::kPollPeer;
        uint32_t peer_cycles = 0;
        report.last_poll_ready = poll_until([start_flag] { return *start_flag != 0; }, deadline, peer_cycles);
        if (!report.last_poll_ready) {
            record_failure(report, success_code, abi::kCodePeerTimeout);
        }
    }

    uint32_t used_trids = 0;
    for (uint32_t command = 0; command < batch.num_commands && report.code == success_code; ++command) {
        const uint32_t trid = command_trid(batch, command);
        if (!issue_command<CMDBUF>(batch, source, destination, command, trid, deadline, report)) {
            record_failure(report, success_code, abi::kCodeVcSpaceTimeout);
            break;
        }
        report.flags |= abi::kFlagVcSpace;
        used_trids |= 1u << (trid % abi::kNumTrids);
        record_pending(pending_count<CMDBUF>(trid % abi::kNumTrids), report);
    }
    const uint64_t last_issue_cycles = cce_test::read_cycle_counter();
    if (report.code != success_code) {
        report.flags &= ~abi::kFlagVcSpace;
    }

    if (used_trids != 0) {
        BatchArgs issued = batch;
        issued.num_commands = report.elements_issued;
        complete_batch<CMDBUF>(
            issued,
            source,
            destination,
            destination_kind,
            used_trids,
            last_issue_cycles,
            deadline,
            success_code,
            trace,
            report);
    }
    return report;
}

struct TableCommand {
    Operand source;
    Operand destination;
    uint32_t backend_vc = 0;
    uint32_t trid = 0;
    uint32_t len_bytes = 0;
    bool host_verified = false;
};

// Reads entry `command` of the kModeTridTable table. Returns false when an operand kind is unknown, the VC, trid or
// length is out of range, or a host-verified entry writes the kernel's own TL1: it gets no data poll, so the status
// store could overlap its landing (MIM-560).
static bool read_table_command(uint32_t table_offset, uint32_t command, TableCommand& table_command) {
    const volatile uint32_t* entry =
        cce_test::own_tl1_words(table_offset + command * abi::kTableWords * sizeof(uint32_t));
    table_command.backend_vc = entry[abi::kTableBackendVc];
    table_command.trid = entry[abi::kTableTrid];
    table_command.len_bytes = entry[abi::kTableLenBytes];
    table_command.host_verified = entry[abi::kTableHostVerified] != 0;
    const uint64_t source_offset =
        (static_cast<uint64_t>(entry[abi::kTableSrcOffsetHi]) << 32) | entry[abi::kTableSrcOffsetLo];
    const uint64_t destination_offset =
        (static_cast<uint64_t>(entry[abi::kTableDstOffsetHi]) << 32) | entry[abi::kTableDstOffsetLo];
    return resolve_operand(
               entry[abi::kTableSrcKind], entry[abi::kTableSrcSelector], source_offset, table_command.source) &&
           resolve_operand(
               entry[abi::kTableDstKind],
               entry[abi::kTableDstSelector],
               destination_offset,
               table_command.destination) &&
           table_command.backend_vc < overlay::CMDBUF_NUM_IDMA_VCS && table_command.trid < abi::kNumTrids &&
           table_command.len_bytes != 0 && table_command.len_bytes % sizeof(uint32_t) == 0 &&
           table_command.len_bytes <= abi::kMaxBytesInPacket &&
           !(table_command.host_verified && entry[abi::kTableDstKind] == abi::kKindOwnTl1);
}

// Every entry of the first `num_commands` not marked host-verified against its source. `check` numbers the words of
// every entry in table order, host-verified entries included.
static bool table_data_landed(uint32_t table_offset, uint32_t num_commands, DataCheck& check) {
    check = DataCheck{};
    uint32_t first_word = 0;
    TableCommand table_command;
    for (uint32_t command = 0; command < num_commands; ++command) {
        read_table_command(table_offset, command, table_command);
        const uint32_t command_words = table_command.len_bytes / sizeof(uint32_t);
        if (!table_command.host_verified) {
            const volatile uint32_t* source_words =
                reinterpret_cast<const volatile uint32_t*>(table_command.source.check_address);
            const volatile uint32_t* destination_words =
                reinterpret_cast<const volatile uint32_t*>(table_command.destination.check_address);
            for (uint32_t word = 0; word < command_words; ++word) {
                check.record(first_word + word, destination_words[word], source_words[word]);
            }
        }
        first_word += command_words;
    }
    return check.mismatched_words == 0;
}

// What the issue loop needs of a table entry.
struct IssueEntry {
    uint64_t source_address = 0;
    uint64_t destination_address = 0;
    uint32_t len_bytes = 0;
    uint32_t trid = 0;
    uint32_t backend_vc = 0;
};

// kModeTridTable. Every entry is checked and resolved before the first command is issued, so consecutive issues are
// separated only by command buffer writes and pending count reads. The success code is kCodeAcked when every entry is
// host-verified and kCodeLanded otherwise. kStatusProbeFirst is the largest gap between consecutive issues and
// kStatusProbeSecond the high water of the total pending count.
static Report run_trid_table(
    uint32_t table_offset, uint32_t num_commands, bool snoop, uint64_t budget_cycles, uint32_t* trace) {
    Report report;
    TableCommand table_command;
    IssueEntry issue_entries[abi::kMaxElements];
    bool every_entry_host_verified = true;
    for (uint32_t command = 0; command < num_commands; ++command) {
        if (!read_table_command(table_offset, command, table_command)) {
            report.code = abi::kCodeBadArgs;
            return report;
        }
        every_entry_host_verified = every_entry_host_verified && table_command.host_verified;
        issue_entries[command] = {
            .source_address = table_command.source.command_address,
            .destination_address = table_command.destination.command_address,
            .len_bytes = table_command.len_bytes,
            .trid = table_command.trid,
            .backend_vc = table_command.backend_vc};
    }
    const uint32_t success_code = every_entry_host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    report.code = success_code;
    const cce_test::WaitDeadline deadline(budget_cycles);
    setup_batch<CMDBUF_0>(BatchArgs{.backend_vc = issue_entries[0].backend_vc, .snoop = snoop}, issue_entries[0].trid);

    uint32_t used_trids = 0;
    uint64_t previous_issue_cycles = 0;
    for (uint32_t command = 0; command < num_commands; ++command) {
        const IssueEntry& entry = issue_entries[command];
        overlay::setup_wrapping_vcs_cmdbuf<CMDBUF_0>(
            {.start = entry.backend_vc, .end = entry.backend_vc},
            {.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
        if (!issue_one<CMDBUF_0>(
                entry.source_address,
                entry.destination_address,
                entry.len_bytes,
                entry.trid,
                entry.backend_vc,
                deadline,
                report)) {
            record_failure(report, success_code, abi::kCodeVcSpaceTimeout);
            break;
        }
        const uint64_t issue_cycles = cce_test::read_cycle_counter();
        if (command != 0 && issue_cycles - previous_issue_cycles > report.probe_first) {
            report.probe_first = static_cast<uint32_t>(issue_cycles - previous_issue_cycles);
        }
        previous_issue_cycles = issue_cycles;
        report.flags |= abi::kFlagVcSpace;
        used_trids |= 1u << entry.trid;
        record_pending(pending_count<CMDBUF_0>(entry.trid), report);
        uint32_t total_pending = 0;
        for (uint32_t trid = 0; trid < abi::kNumTrids; ++trid) {
            if ((used_trids & (1u << trid)) != 0) {
                total_pending += pending_count<CMDBUF_0>(trid);
            }
        }
        report.probe_second = total_pending > report.probe_second ? total_pending : report.probe_second;
    }
    const uint64_t last_issue_cycles = cce_test::read_cycle_counter();
    if (report.code != success_code) {
        report.flags &= ~abi::kFlagVcSpace;
    }
    if (used_trids == 0) {
        return report;
    }
    wait_for_trids<CMDBUF_0>(used_trids, last_issue_cycles, deadline, success_code, trace, report);
    if (every_entry_host_verified) {
        return report;
    }
    const uint32_t issued = report.elements_issued;
    report.last_poll = abi::kPollData;
    report.last_poll_ready =
        poll_until([&] { return table_data_landed(table_offset, issued, report.check); }, deadline, report.data_cycles);
    if (report.last_poll_ready) {
        report.flags |= abi::kFlagLanded;
    } else {
        record_failure(report, success_code, abi::kCodeDataTimeout);
    }
    return report;
}

// kModeTridStream. A poll that uses the budget mid-round keeps its own timeout code; kCodeStreamBudget means the budget
// had run out at a round boundary, with nothing in flight. The trace is the last round's.
static Report run_trid_stream(
    const BatchArgs& batch,
    const Operand& source,
    const Operand& destination,
    uint32_t destination_kind,
    uint32_t stop_offset,
    uint32_t progress_offset,
    uint64_t budget_cycles,
    uint32_t* trace) {
    Report report;
    const uint32_t success_code = batch.host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    report.code = success_code;
    const cce_test::WaitDeadline deadline(budget_cycles);
    setup_batch<CMDBUF_0>(batch, command_trid(batch, 0));
    const volatile uint32_t* stop = cce_test::own_tl1_words(stop_offset);
    volatile uint32_t* progress = cce_test::own_tl1_words(progress_offset);

    uint32_t rounds = 0;
    while (*stop != abi::kStreamStop) {
        if (deadline.expired()) {
            report.code = abi::kCodeStreamBudget;
            break;
        }
        const uint32_t issued_before = report.elements_issued;
        uint32_t used_trids = 0;
        for (uint32_t command = 0; command < batch.num_commands; ++command) {
            const uint32_t trid = command_trid(batch, command);
            if (!issue_command<CMDBUF_0>(batch, source, destination, command, trid, deadline, report)) {
                record_failure(report, success_code, abi::kCodeVcSpaceTimeout);
                report.flags &= ~abi::kFlagVcSpace;
                break;
            }
            report.flags |= abi::kFlagVcSpace;
            used_trids |= 1u << trid;
            record_pending(pending_count<CMDBUF_0>(trid), report);
        }
        const uint64_t last_issue_cycles = cce_test::read_cycle_counter();
        if (used_trids != 0) {
            BatchArgs issued = batch;
            issued.num_commands = report.elements_issued - issued_before;
            complete_batch<CMDBUF_0>(
                issued,
                source,
                destination,
                destination_kind,
                used_trids,
                last_issue_cycles,
                deadline,
                success_code,
                trace,
                report);
        }
        if (report.code != success_code) {
            break;
        }
        rounds++;
        *progress = rounds;
    }
    report.probe_first = rounds;
    return report;
}

static Report run_scope_observer(uint32_t trid, uint32_t peer_flag_offset, uint64_t budget_cycles) {
    Report report;
    report.code = abi::kCodeAcked;
    const cce_test::WaitDeadline deadline(budget_cycles);
    volatile uint32_t* issuer_flag = cce_test::own_tl1_words(peer_flag_offset);
    report.last_poll = abi::kPollPeer;
    uint32_t peer_cycles = 0;
    report.last_poll_ready = poll_until([issuer_flag] { return *issuer_flag == 1; }, deadline, peer_cycles);
    if (!report.last_poll_ready) {
        report.code = abi::kCodePeerTimeout;
        return report;
    }
    report.probe_first = pending_count<CMDBUF_0>(trid);
    *cce_test::own_tl1_words(peer_flag_offset + abi::kLineBytes) = 1;
    return report;
}

static bool line_offset_valid(uint32_t offset) { return offset != 0 && offset % abi::kLineBytes == 0; }

// The MM4x modes. The status line and trace are written after the last data poll (MIM-560).
static void run_mm4x_mode(uint32_t mode, uint32_t status_offset, uint64_t budget_cycles) {
    const uint32_t destination_kind = get_arg_val<uint32_t>(abi::kArgDstKind);
    const uint32_t peer_flag_offset = get_arg_val<uint32_t>(abi::kArgPeerFlagOffset);
    const uint32_t trace_offset = get_arg_val<uint32_t>(abi::kArgTraceOffset);
    const uint32_t table_offset = get_arg_val<uint32_t>(abi::kArgCmdTableOffset);
    const uint32_t stop_offset = get_arg_val<uint32_t>(abi::kArgStopOffset);
    const uint32_t progress_offset = get_arg_val<uint32_t>(abi::kArgProgressOffset);
    const uint32_t command_buffer = get_arg_val<uint32_t>(abi::kArgCommandBuffer);
    const uint64_t destination_offset = arg_u64(abi::kArgDstOffsetLo, abi::kArgDstOffsetHi);
    const bool accumulate = mode == abi::kModeAccumulate;
    BatchArgs batch{
        .backend_vc = get_arg_val<uint32_t>(abi::kArgBackendVc),
        .len_bytes = get_arg_val<uint32_t>(abi::kArgLenBytes),
        .num_commands = get_arg_val<uint32_t>(abi::kArgNumElements),
        .source_stride_bytes = get_arg_val<uint32_t>(abi::kArgSourceStrideBytes),
        .destination_stride_bytes = get_arg_val<uint32_t>(abi::kArgDestinationStrideBytes),
        .first_trid = get_arg_val<uint32_t>(abi::kArgTrid),
        .trid_stride = get_arg_val<uint32_t>(abi::kArgTridStride),
        .snoop = get_arg_val<uint32_t>(abi::kArgSnoop) != 0,
        .host_verified = !accumulate && get_arg_val<uint32_t>(abi::kArgHostVerified) != 0,
        .expected_xor = get_arg_val<uint32_t>(abi::kArgExpectedXor),
        .accumulate = accumulate,
        .accumulate_format = get_arg_val<uint32_t>(abi::kArgAccumulateFormat),
        .accumulate_op = get_arg_val<uint32_t>(abi::kArgAccumulateOp),
        .accumulate_disable_saturation = get_arg_val<uint32_t>(abi::kArgAccumulateDisableSaturation) != 0,
        .expected_lo = get_arg_val<uint32_t>(abi::kArgExpectedLo),
        .expected_hi = get_arg_val<uint32_t>(abi::kArgExpectedHi),
        .poll_region_offset = get_arg_val<uint32_t>(abi::kArgPollRegionOffset),
        .poll_region_bytes = get_arg_val<uint32_t>(abi::kArgPollRegionBytes),
        .check_at_ack = get_arg_val<uint32_t>(abi::kArgCheckAtAck) != 0};
    Operand source;
    Operand destination;
    const bool operands_known =
        resolve_operand(
            get_arg_val<uint32_t>(abi::kArgSrcKind),
            get_arg_val<uint32_t>(abi::kArgSrcSelector),
            arg_u64(abi::kArgSrcOffsetLo, abi::kArgSrcOffsetHi),
            source) &&
        resolve_operand(destination_kind, get_arg_val<uint32_t>(abi::kArgDstSelector), destination_offset, destination);
    const bool commands_valid = batch.num_commands != 0 && batch.num_commands <= abi::kMaxElements;
    const bool batch_valid = operands_known && batch.len_bytes != 0 && batch.len_bytes % 4 == 0 &&
                             batch.len_bytes <= abi::kMaxBytesInPacket && commands_valid &&
                             batch.source_stride_bytes % 4 == 0 && batch.destination_stride_bytes % 4 == 0;
    const bool command_buffer_valid =
        mode == abi::kModeTridBatch || accumulate ? command_buffer <= 1 : command_buffer == 0;
    const bool trid_valid = batch.first_trid < abi::kNumTrids;
    const bool poll_region_valid =
        batch.poll_region_bytes == 0 ||
        (line_offset_valid(batch.poll_region_offset) && batch.poll_region_bytes % abi::kLineBytes == 0 &&
         (accumulate || (mode == abi::kModeTridBatch && batch.host_verified && destination_kind == abi::kKindOwnTl1)));
    const bool check_at_ack_valid = !batch.check_at_ack || (mode == abi::kModeTridBatch && !batch.host_verified);
    bool mode_valid = false;
    switch (mode) {
        case abi::kModeTridBatch:
            mode_valid = batch_valid && (peer_flag_offset == 0 || line_offset_valid(peer_flag_offset));
            break;
        case abi::kModeAccumulate:
            mode_valid = batch_valid && destination_kind == abi::kKindOwnTl1 && batch.accumulate_format < 8 &&
                         batch.accumulate_op < 16 && get_arg_val<uint32_t>(abi::kArgAccumulateDisableSaturation) <= 1;
            break;
        case abi::kModeTridTable: mode_valid = commands_valid && line_offset_valid(table_offset); break;
        case abi::kModeTridStream:
            mode_valid = batch_valid && destination_kind != abi::kKindOwnTl1 && line_offset_valid(stop_offset) &&
                         line_offset_valid(progress_offset);
            break;
        case abi::kModeTridScopeObserver: mode_valid = line_offset_valid(peer_flag_offset); break;
        default: break;
    }
    const bool args_valid = budget_cycles != 0 && trace_offset % abi::kLineBytes == 0 && trid_valid &&
                            command_buffer_valid && poll_region_valid && check_at_ack_valid && mode_valid;

    uint32_t trace[abi::kNumTrids * abi::kTraceWordsPerTrid];
    for (uint32_t& word : trace) {
        word = abi::kTraceNotUsed;
    }
    Report report;
    if (!args_valid) {
        report.code = abi::kCodeBadArgs;
    } else if (mode == abi::kModeTridScopeObserver) {
        report = run_scope_observer(batch.first_trid, peer_flag_offset, budget_cycles);
    } else if (mode == abi::kModeTridTable) {
        report = run_trid_table(table_offset, batch.num_commands, batch.snoop, budget_cycles, trace);
    } else if (mode == abi::kModeTridStream) {
        report = run_trid_stream(
            batch, source, destination, destination_kind, stop_offset, progress_offset, budget_cycles, trace);
    } else if (command_buffer == 1) {
        report = run_trid_issuer<CMDBUF_1>(
            mode, batch, source, destination, destination_kind, peer_flag_offset, budget_cycles, trace);
    } else {
        report = run_trid_issuer<CMDBUF_0>(
            mode, batch, source, destination, destination_kind, peer_flag_offset, budget_cycles, trace);
    }
    if (args_valid) {
        write_trace(trace_offset, trace);
    }
    write_status(status_offset, report);
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t backend_vc = get_arg_val<uint32_t>(abi::kArgBackendVc);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint32_t len_bytes = get_arg_val<uint32_t>(abi::kArgLenBytes);
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const bool host_verified = mode == abi::kModeIdmaCopyHostVerified;
    const bool strided = mode == abi::kModeIdmaStrided;
    const uint32_t num_elements = get_arg_val<uint32_t>(abi::kArgNumElements);
    const uint32_t source_stride_bytes = get_arg_val<uint32_t>(abi::kArgSourceStrideBytes);
    if (mode >= abi::kModeTridBatch) {
        run_mm4x_mode(mode, status_offset, budget_cycles);
        return;
    }

    const uint32_t success_code = host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    uint32_t code = success_code;
    uint32_t flags = 0;
    uint32_t vc_space_cycles = 0;
    uint32_t ack_cycles = 0;
    uint32_t data_cycles = 0;
    uint32_t elements_issued = 0;
    uint32_t last_poll = abi::kPollNone;
    bool last_poll_ready = false;
    DataCheck check;

    const uint32_t destination_kind = get_arg_val<uint32_t>(abi::kArgDstKind);
    Operand source;
    Operand destination;
    const bool operands_known = resolve_operand(
                                    get_arg_val<uint32_t>(abi::kArgSrcKind),
                                    get_arg_val<uint32_t>(abi::kArgSrcSelector),
                                    arg_u64(abi::kArgSrcOffsetLo, abi::kArgSrcOffsetHi),
                                    source) &&
                                resolve_operand(
                                    destination_kind,
                                    get_arg_val<uint32_t>(abi::kArgDstSelector),
                                    arg_u64(abi::kArgDstOffsetLo, abi::kArgDstOffsetHi),
                                    destination);
    const bool elements_valid =
        strided ? num_elements != 0 && num_elements <= abi::kMaxElements && source_stride_bytes % 4 == 0
                : num_elements == 1;
    const bool args_valid =
        operands_known && budget_cycles != 0 && len_bytes != 0 && len_bytes % 4 == 0 &&
        len_bytes <= abi::kMaxBytesInPacket && elements_valid && get_arg_val<uint32_t>(abi::kArgPollRegionBytes) == 0 &&
        get_arg_val<uint32_t>(abi::kArgCheckAtAck) == 0 && (mode == abi::kModeIdmaCopy || host_verified || strided);

    if (!args_valid) {
        code = abi::kCodeBadArgs;
    } else {
        const auto record_timeout = [&code, success_code](uint32_t timeout_code) {
            if (code == success_code) {
                code = timeout_code;
            }
        };
        const auto run_poll = [&last_poll, &last_poll_ready, budget_cycles](
                                  uint32_t poll, auto ready, uint32_t& cycles_spent) {
            last_poll = poll;
            last_poll_ready = poll_bounded(ready, budget_cycles, cycles_spent);
            return last_poll_ready;
        };
        program_copy(backend_vc, len_bytes, source, destination, strided, num_elements, source_stride_bytes);

        bool vc_space_ready = true;
        for (uint32_t element = 0; element < num_elements && vc_space_ready; ++element) {
            uint32_t element_vc_space_cycles = 0;
            vc_space_ready = run_poll(
                abi::kPollVcSpace,
                [backend_vc] { return overlay::idma_free_space_cmdbuf<CMDBUF_0>(backend_vc) != 0; },
                element_vc_space_cycles);
            vc_space_cycles += element_vc_space_cycles;
            if (vc_space_ready) {
                if (strided) {
                    overlay::push_both_addrgen<ADDRGEN_0>();
                }
                overlay::issue_cmdbuf<CMDBUF_0>();
                elements_issued++;
            }
        }
        if (!vc_space_ready) {
            record_timeout(abi::kCodeVcSpaceTimeout);
        } else {
            flags |= abi::kFlagVcSpace;
            if (run_poll(abi::kPollAck, [] { return overlay::idma_acked_cmdbuf<CMDBUF_0>(abi::kTrid); }, ack_cycles)) {
                flags |= abi::kFlagAcked;
            } else {
                record_timeout(abi::kCodeAckTimeout);
            }
        }

        const uint32_t num_words = num_elements * len_bytes / sizeof(uint32_t);
        const uint32_t expected_xor = get_arg_val<uint32_t>(abi::kArgExpectedXor);
        if (!host_verified || destination_kind == abi::kKindOwnTl1) {
            if (run_poll(
                    abi::kPollData,
                    [&] {
                        return host_verified
                                   ? destination_xor_matches(
                                         reinterpret_cast<const volatile uint32_t*>(destination.check_address),
                                         num_words,
                                         expected_xor,
                                         check)
                                   : data_landed(
                                         source,
                                         destination,
                                         len_bytes,
                                         num_elements,
                                         source_stride_bytes,
                                         len_bytes,
                                         check);
                    },
                    data_cycles)) {
                flags |= abi::kFlagLanded;
            } else {
                record_timeout(abi::kCodeDataTimeout);
            }
        }
    }

    write_status(
        status_offset,
        Report{
            .code = code,
            .flags = flags,
            .vc_space_cycles = vc_space_cycles,
            .ack_cycles = ack_cycles,
            .data_cycles = data_cycles,
            .elements_issued = elements_issued,
            .last_poll = last_poll,
            .last_poll_ready = last_poll_ready,
            .check = check});
}
