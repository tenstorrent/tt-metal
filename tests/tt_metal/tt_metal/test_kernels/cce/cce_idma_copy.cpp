// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One iDMA copy through command buffer 0, with every poll bounded. Completion is a bounded ack poll on trid 0, then a
// bounded poll of the destination data (MIM-628: done fires on accept). The status line is written only after the data
// poll (MIM-560), through the uncached alias. In kModeIdmaCopyHostVerified the kernel loads no remote operand: when the
// destination is its own TL1 it polls the XOR of the destination against kArgExpectedXor, otherwise there is no data
// poll; the host checks the data.

#include "cce_idma_copy_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "risc_common.h"

namespace abi = cce_idma_copy_abi;
using cce_test::arg_u64;
using cce_test::DataCheck;
using cce_test::poll_bounded;
using overlay::CMDBUF_0;

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
            operand.command_address = experimental::cce_gddr_address(
                abi::gddr_selector_local(selector), abi::gddr_selector_target(selector), offset);
            operand.check_address = operand.command_address;
            return true;
        default: return false;
    }
}

static bool data_landed(const Operand& source, const Operand& destination, uint32_t num_words, DataCheck& check) {
    return cce_test::words_match(
        reinterpret_cast<const volatile uint32_t*>(source.check_address),
        reinterpret_cast<const volatile uint32_t*>(destination.check_address),
        num_words,
        check);
}

// XOR of the destination words against the host's XOR of the source; on a mismatch, `check` holds the two XORs.
static bool destination_xor_matches(
    const Operand& destination, uint32_t num_words, uint32_t expected_xor, DataCheck& check) {
    const volatile uint32_t* destination_words = reinterpret_cast<const volatile uint32_t*>(destination.check_address);
    uint32_t observed_xor = 0;
    for (uint32_t word = 0; word < num_words; ++word) {
        observed_xor ^= destination_words[word];
    }
    check = DataCheck{};
    check.record(0, observed_xor, expected_xor);
    return check.mismatched_words == 0;
}

static void program_copy(uint32_t backend_vc, uint32_t len_bytes, const Operand& source, const Operand& destination) {
    overlay::reset_cmdbuf<CMDBUF_0>();
    overlay::idma_setup_as_copy_cmdbuf<CMDBUF_0>(/*wrapping=*/false);
    overlay::setup_ongoing_cmdbuf<CMDBUF_0>({});
    overlay::setup_wrapping_vcs_cmdbuf<CMDBUF_0>(
        {.start = backend_vc, .end = backend_vc},
        {.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
    overlay::setup_trids_cmdbuf<CMDBUF_0>(abi::kTrid);
    overlay::setup_max_bytes_in_packet_cmdbuf<CMDBUF_0>(NOC_OVERLAY_MAX_BYTES_IN_PACKET);
    overlay::set_src_cmdbuf<CMDBUF_0>(source.command_address);
    overlay::set_dest_cmdbuf<CMDBUF_0>(destination.command_address);
    overlay::set_len_cmdbuf<CMDBUF_0>(len_bytes);
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t backend_vc = get_arg_val<uint32_t>(abi::kArgBackendVc);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint32_t len_bytes = get_arg_val<uint32_t>(abi::kArgLenBytes);
    const uint64_t budget_cycles = arg_u64(abi::kArgPollBudgetLo, abi::kArgPollBudgetHi);
    const bool host_verified = mode == abi::kModeIdmaCopyHostVerified;

    const uint32_t success_code = host_verified ? abi::kCodeAcked : abi::kCodeLanded;
    uint32_t code = success_code;
    uint32_t flags = 0;
    uint32_t vc_space_cycles = 0;
    uint32_t ack_cycles = 0;
    uint32_t data_cycles = 0;
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
    const bool args_valid = operands_known && budget_cycles != 0 && len_bytes != 0 && len_bytes % 4 == 0 &&
                            len_bytes <= abi::kMaxBytesInPacket && (mode == abi::kModeIdmaCopy || host_verified);

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
        program_copy(backend_vc, len_bytes, source, destination);

        if (!run_poll(
                abi::kPollVcSpace,
                [backend_vc] { return overlay::idma_free_space_cmdbuf<CMDBUF_0>(backend_vc) != 0; },
                vc_space_cycles)) {
            record_timeout(abi::kCodeVcSpaceTimeout);
        } else {
            flags |= abi::kFlagVcSpace;
            overlay::issue_cmdbuf<CMDBUF_0>();
            if (run_poll(abi::kPollAck, [] { return overlay::idma_acked_cmdbuf<CMDBUF_0>(abi::kTrid); }, ack_cycles)) {
                flags |= abi::kFlagAcked;
            } else {
                record_timeout(abi::kCodeAckTimeout);
            }
        }

        const uint32_t num_words = len_bytes / sizeof(uint32_t);
        const uint32_t expected_xor = get_arg_val<uint32_t>(abi::kArgExpectedXor);
        if (!host_verified || destination_kind == abi::kKindOwnTl1) {
            if (run_poll(
                    abi::kPollData,
                    [host_verified, &source, &destination, num_words, expected_xor, &check] {
                        return host_verified ? destination_xor_matches(destination, num_words, expected_xor, check)
                                             : data_landed(source, destination, num_words, check);
                    },
                    data_cycles)) {
                flags |= abi::kFlagLanded;
            } else {
                record_timeout(abi::kCodeDataTimeout);
            }
        }
    }

    volatile uint32_t* status = cce_test::own_tl1_words(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusFlags] = flags;
    status[abi::kStatusMismatchedWords] = check.mismatched_words;
    status[abi::kStatusFirstMismatchWord] = check.first_mismatch_word;
    status[abi::kStatusObserved] = check.observed;
    status[abi::kStatusExpected] = check.expected;
    status[abi::kStatusVcSpaceCycles] = vc_space_cycles;
    status[abi::kStatusAckCycles] = ack_cycles;
    status[abi::kStatusDataCycles] = data_cycles;
    status[abi::kStatusLastPoll] = last_poll;
    status[abi::kStatusLastPollReady] = last_poll_ready ? 1 : 0;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
