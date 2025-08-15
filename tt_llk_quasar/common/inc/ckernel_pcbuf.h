// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once // Cosmetic change to test CODEOWNERS2

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
    return ckernel_id | ((iterations & 0xFFFF) << 16) | ((number_of_extra_params & 0xF) << 12);
}

inline void decode_pc_buf_cmd(uint cmd, uint &ckernel_id, uint &iterations, uint &number_of_extra_params)
{
    // Ckernel ID can be a max of 12 bits.
    ckernel_id             = cmd & 0xFFF;
    iterations             = (cmd >> 16) & 0xFFFF;
    number_of_extra_params = (cmd >> 12) & 0xF;
}

} // namespace ckernel
