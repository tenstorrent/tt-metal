// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Functions for encoding and decoding PC buffer writes
namespace ckernel
{

inline uint get_pc_buf_cmd(uint ckernel_id, uint command_addr)
{
    // Ckernel fast launch cmd has MSB set
    return ckernel_id | ((command_addr & 0x7FFFF) << 12) | (1 << 31);
}

inline uint get_pc_buf_cmd(uint ckernel_id, uint iterations, uint number_of_extra_params)
{
    // Ckernel ID can be a max of 12 bits.
    return ckernel_id | ((iterations & 0x7FFF) << 16) | ((number_of_extra_params & 0xF) << 12);
}

inline bool is_pc_buf_fast_cmd(uint cmd)
{
    // MSB stores fast kernel mode
    return (cmd >> 31) & 0x1;
}

inline void decode_pc_buf_cmd(uint cmd, uint &ckernel_id, uint &command_addr)
{
    // Ckernel ID can be a max of 12 bits.
    ckernel_id   = cmd & 0xFFF;
    command_addr = (cmd >> 12) & 0x7FFFF;
}

inline void decode_pc_buf_cmd(uint cmd, uint &ckernel_id, uint &iterations, uint &number_of_extra_params)
{
    // Ckernel ID can be a max of 12 bits.
    ckernel_id             = cmd & 0xFFF;
    iterations             = (cmd >> 16) & 0x7FFF;
    number_of_extra_params = (cmd >> 12) & 0xF;
}

} // namespace ckernel
