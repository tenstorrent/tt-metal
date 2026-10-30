// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_rows.h"
#include "llk_unpack_untilize.h"
#include "llk_pack_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_A.h"
#include "llk_pack.h"

void k_pack_rows_init(const std::uint32_t n) { _llk_pack_rows_init_(n); }
void k_pack_rows(const std::uint32_t t, const std::uint32_t a) { _llk_pack_rows_(t, a); }
void k_pack_rows_uninit() { _llk_pack_rows_uninit_(); }
void k_untilize_init(const std::uint32_t fmt, const std::uint32_t ts) { _llk_unpack_untilize_init_(fmt, ts); }
void k_untilize_pass(const std::uint32_t base, const std::uint32_t cols) { _llk_unpack_untilize_pass_(base, cols); }
void k_untilize_uninit(const std::uint32_t fmt) { _llk_unpack_untilize_uninit_(fmt, 16); }
void k_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }
void k_pack_reduce_mask_config() { _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_ROW, PackMode::Default>(); }
void k_pack_dest_section_done() { _llk_pack_dest_section_done_<DstSync::SyncHalf, false>(); }
void k_pack_dest_init() { _llk_pack_dest_init_<DstSync::SyncHalf, false>(); }
void k_unpack_hw_configure() { _llk_unpack_hw_configure_<false>(5, 5, 5, 5, 16, 16, 4, 4); }
void k_matmul_unpack_init() { _llk_unpack_AB_matmul_init_(0, 1, 1, 1, 16, 16, 4, 4, false, false); }
void k_matmul_unpack_init_rt(const std::uint32_t ct, const std::uint32_t rt, const std::uint32_t kt, const bool pf) { _llk_unpack_AB_matmul_init_(0, ct, rt, kt, 16, 16, 4, 4, pf, pf); }
void k_matmul_unpack(const std::uint32_t a, const std::uint32_t b, const std::uint32_t ta, const std::uint32_t tb, const bool pf) { _llk_unpack_AB_matmul_(a, b, ta, tb, 128, 128, pf, pf, 2, 2, 2); }
void k_unpack_A(const std::uint32_t a) { _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(a, 5, 5); }
