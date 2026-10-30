// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Each function issues exactly one Tensix instruction.  Tool: 0 for all (some listed as MMIO, some invisible).
void ib_raw_sym(const std::uint32_t w)    { __instrn_buffer[0] = w; }                                   // invisible
void ib_offset4(const std::uint32_t w)    { ckernel::instrn_buffer[4] = w; }                            // invisible
void ib_ptr_alias(const std::uint32_t w)  { volatile std::uint32_t *p = ckernel::instrn_buffer; p[0] = w; } // MMIO write &__instrn_buffer[0]
void ib_const_addr(const std::uint32_t w) { *reinterpret_cast<volatile std::uint32_t *>(INSTRN_BUF_BASE) = w; } // MMIO write INSTRN_BUF_BASE
void ib_store_blocking(const std::uint32_t w) { store_blocking(&ckernel::instrn_buffer[0], w); }       // MMIO asm
