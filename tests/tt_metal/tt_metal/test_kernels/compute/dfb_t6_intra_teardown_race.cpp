// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Intra-tensix push/pop split across the packer remapper teardown (issue #57950).
// Pack posts, waits until those posts have landed on ClientL, then returns. Unpack waits for that
// flag, spins so pack firmware has left the kernel, then pops. Without the teardown wait, pack
// clears the remapper before unpack's pops, and on A0 those acks alias onto overlay counter
// (ClientL id & 0xF).

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "ckernel_trisc_common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {
volatile std::uint32_t* scratch_word(std::uint32_t l1_address) {
    return reinterpret_cast<volatile std::uint32_t*>(l1_address + MEM_L1_UNCACHED_BASE);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t scratch_l1_address = get_arg(args::scratch_l1_address);
    constexpr std::uint32_t posts_landed = get_arg(args::posts_landed);
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr std::uint32_t client_l_tc = get_arg(args::client_l_tc);
    constexpr std::uint32_t unpack_spin = get_arg(args::unpack_spin);

    DataflowBuffer dfb(dfb::out);
    compute_kernel_hw_startup(dfb::out, dfb::out);

#ifdef UCK_CHLKC_PACK
    dfb.reserve_back(num_tiles);
    ckernel::dummy_pack(dfb::out);
    dfb.push_back(num_tiles);
    while ((ckernel::trisc::tile_counters[client_l_tc].f.posted & 0xFFFFu) != num_tiles) {
    }
    *scratch_word(scratch_l1_address) = posts_landed;
#endif

#ifdef UCK_CHLKC_UNPACK
    while (*scratch_word(scratch_l1_address) != posts_landed) {
    }
    // Pack has returned by the time this spin finishes, so its firmware teardown has run.
    for (std::uint32_t i = 0; i < unpack_spin; i++) {
        *scratch_word(scratch_l1_address) = posts_landed;
    }
    dfb.wait_front(num_tiles);
    ckernel::dummy_unpack(dfb::out);
    dfb.pop_front(num_tiles);
#endif
}
