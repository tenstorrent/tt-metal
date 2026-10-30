// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_a() { TTI_STALLWAIT(1,2); *reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE) = 1; }
void a() { TT_STALLWAIT(1,2); *reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE) = 1; }
