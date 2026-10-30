// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cce_copy_ops_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "risc_common.h"

namespace abi = cce_copy_ops_abi;

static bool resolve_endpoint(uint32_t kind, uint32_t selector, uint64_t offset, uint64_t& address) {
    switch (kind) {
        case abi::kKindOwnTl1: address = offset + static_cast<uint64_t>(MEM_L1_UNCACHED_BASE); return true;
        case abi::kKindGddr: address = experimental::cce_gddr_address(selector, offset); return true;
        case abi::kKindSramSpa: address = experimental::cce_sram_address(selector, offset); return true;
        default: return false;
    }
}

void kernel_main() {
    const uint32_t done_offset = get_arg_val<uint32_t>(abi::kArgDoneOffset);
    const uint32_t n_ops = get_arg_val<uint32_t>(abi::kArgNumOps);

    uint32_t xor_signature = 0;

    for (uint32_t op_index = 0; op_index < n_ops; op_index++) {
        const uint32_t src_kind = get_arg_val<uint32_t>(abi::op_arg_index(op_index, abi::kOpSrcKind));
        const uint32_t src_sel = get_arg_val<uint32_t>(abi::op_arg_index(op_index, abi::kOpSrcSelector));
        const uint64_t src_offset = cce_test::arg_u64(
            abi::op_arg_index(op_index, abi::kOpSrcOffsetLo), abi::op_arg_index(op_index, abi::kOpSrcOffsetHi));
        const uint32_t dst_kind = get_arg_val<uint32_t>(abi::op_arg_index(op_index, abi::kOpDstKind));
        const uint32_t dst_sel = get_arg_val<uint32_t>(abi::op_arg_index(op_index, abi::kOpDstSelector));
        const uint64_t dst_offset = cce_test::arg_u64(
            abi::op_arg_index(op_index, abi::kOpDstOffsetLo), abi::op_arg_index(op_index, abi::kOpDstOffsetHi));
        const uint32_t n_words = get_arg_val<uint32_t>(abi::op_arg_index(op_index, abi::kOpNumWords));

        uint64_t src_address = 0;
        uint64_t dst_address = 0;
        const bool src_known = resolve_endpoint(src_kind, src_sel, src_offset, src_address);
        const bool dst_known = resolve_endpoint(dst_kind, dst_sel, dst_offset, dst_address);
        // An unknown kind leaves the done block unwritten, so the host sees done == false.
        if (!src_known || !dst_known) {
            return;
        }

        cce_test::copy_words(
            reinterpret_cast<volatile uint32_t*>(dst_address),
            reinterpret_cast<const volatile uint32_t*>(src_address),
            n_words,
            xor_signature);
    }

    volatile uint32_t* done = cce_test::own_tl1_words(done_offset);
    done[abi::kDoneXorWord] = xor_signature;
    asm volatile("fence rw, rw" ::: "memory");
    done[abi::kDoneMagicWord] = abi::kDoneMagic;
}
