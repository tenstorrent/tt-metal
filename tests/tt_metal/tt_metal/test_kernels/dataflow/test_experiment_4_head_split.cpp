// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

using namespace overlay;

void kernel_main() {
    constexpr std::uint32_t in_addr = get_arg(args::in_addr);
    constexpr std::uint32_t out_addr = get_arg(args::out_addr);
    constexpr std::uint32_t num_heads = get_arg(args::num_heads);
    constexpr std::uint32_t seq_len = get_arg(args::seq_len);
    constexpr std::uint32_t head_dim = get_arg(args::head_dim);
    constexpr std::uint32_t element_bytes = get_arg(args::element_bytes);

    constexpr std::uint32_t transfer_bytes = head_dim * element_bytes;

    reset_cmdbuf_0();
    idma_setup_as_copy_cmdbuf_0(false);

    setup_vcs_cmdbuf_0(NocVcs::READ);
    setup_trids_cmdbuf_0(0);

    set_len_cmdbuf_0(transfer_bytes);

    reset_addrgen_0();

    setup_src_base_start_addrgen_0(in_addr);
    setup_src_inner_loop_addrgen_0({.stride = num_heads * transfer_bytes, .end = seq_len * num_heads * transfer_bytes});
    setup_src_outer_loop_addrgen_0({.stride = transfer_bytes, .end = num_heads * transfer_bytes});

    setup_dest_base_start_addrgen_0(out_addr);
    setup_dest_inner_loop_addrgen_0({.stride = transfer_bytes, .end = seq_len * num_heads * transfer_bytes});

    for (std::uint32_t i = 0; i < seq_len * num_heads; ++i) {
        push_both_addrgen_0();
        issue_cmdbuf_0();
    }

    while (!idma_acked_cmdbuf_0()) {
    }

    DEVICE_PRINT("head split done: {} transfers of {} B\n", seq_len * num_heads, transfer_bytes);
}
