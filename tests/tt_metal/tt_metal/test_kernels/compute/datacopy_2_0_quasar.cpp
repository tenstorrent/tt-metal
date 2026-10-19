// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 datacopy: in DFB -> copy (unpack + math) -> Dest -> pack -> out DFB, using only the id-free 2.0
// compute API. Operands come from the runtime DataflowBuffers (front/back) typed by their binding tokens
// (LLKOperandFrom). Dest sync is explicit and outside the ops: the kernel declares its chain (FPU -> PACK) once
// and each client acquires/releases Dest around its op. Every op init is re-called before its op (Quasar rule:
// the init programs the buffer descriptor at the DFB's current address). Between each DFB wait/reserve and its
// pop/push sits the op's TDMA (UNPACR / PACR), as Quasar requires.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/dest_sync.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    namespace ex = ckernel::experimental;
    using In = LLKOperandFrom<dfb::in>;
    using Out = LLKOperandFrom<dfb::out>;
    using Chain = ex::DestChain<ex::DestClient::FPU, ex::DestClient::PACK>;

    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr std::uint32_t block = get_arg(args::block_tiles);  // tiles per Dest section
    constexpr std::uint32_t num_sections = num_tiles / block;

    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);

    ckernel::compute_kernel_hw_startup(in.front<In>(), out.back<Out>());
    ex::dest_sync_init<Chain>();

    for (std::uint32_t s = 0; s < num_sections; ++s) {
        const std::uint32_t sections_left = num_sections - 1 - s;

        in.wait_front(block);
        ex::copy_init(in.front<In>());
        ex::dest_acquire<Chain, ex::DestClient::FPU>();
        ex::copy_block(in.front<In>(), 0, 0, block);
        ex::dest_release<Chain, ex::DestClient::FPU>(sections_left);
        in.pop_front(block);

        out.reserve_back(block);
        ex::pack_init(out.back<Out>());
        ex::dest_acquire<Chain, ex::DestClient::PACK>();
        ex::pack_block(out.back<Out>(), 0, block);
        ex::dest_release<Chain, ex::DestClient::PACK>(sections_left);
        out.push_back(block);
    }
}
