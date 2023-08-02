#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "fw_debug.h"
#include "cunpack_common.h"
#include "llk_param_structs.h"

#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel;
using namespace ckernel::unpacker;

void llk_zero_operand(std::uint32_t operand) {
    std::uint32_t input = get_operand_id(operand);

    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    std::uint32_t fifo_base_addr = (cb_read_interface[input].f.fifo_limit + 1) - operands[input].fifo_size;
    TT_SETDMAREG(0, LOWER_HALFWORD(fifo_base_addr), 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(fifo_base_addr), 0, HI_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

    for (std::uint32_t i = 0; i < cb_read_interface[input].fifo_size; i++) {
        TTI_STOREIND(
            1,
            0,
            p_ind::LD_16B,
            LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR),
            p_ind::INC_16B,
            p_gpr_unpack::ZERO_0,
            p_gpr_unpack::OPERAND_BASE_ADDR);
    }
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_unpack_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t *p_tile) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_read_interface[input].fifo_rd_ptr;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
    std::uint32_t byte_address = (base_address + offset_address + TILE_HEADER_SIZE)<<4;

    if constexpr (mail2math) {
       mailbox_write(ThreadId::MathThreadId, byte_address);
       semaphore_post(semaphore::UNPACK_OPERAND_SYNC);
    }

    if constexpr (mail2pack) {
       mailbox_write(ThreadId::PackThreadId, byte_address);
       semaphore_post(semaphore::UNPACK_OPERAND_SYNC);
    }

    *p_tile = byte_address;
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_unpack_release_tile(std::uint32_t operand) {
    while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) > 0);
}
