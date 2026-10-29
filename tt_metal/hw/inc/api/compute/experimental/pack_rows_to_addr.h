// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generic pack_rows_to_addr: pack N row-major rows from DEST to an arbitrary
// L1 address (in 16B units). Reusable across cache-update ops (dsa_indexer_cache_update,
// kv_cache_update, etc.) for patching rows directly into untilized buffers.

#pragma once

#include "api/compute/common.h"
#if defined(TRISC_PACK)
#include "llk_pack_rows_api.h"
#endif

namespace ckernel {

// On Quasar the L1 format is the output of the last pack init (pack_init / compute_kernel_hw_startup); unlike WH/BH,
// pack_reconfig_data_format does not change it. MX output formats are not supported.
ALWI void pack_rows_to_addr_init(uint32_t num_rows) { PACK((llk_pack_rows_init(num_rows))); }

// NOTE: this raw path bypasses the CB/DFB write pointer, because it packs to an arbitrary L1 address.
// On WH/BH it also bypasses the llk_pack_rows wrapper's dst_index bound check, so callers are responsible
// for keeping idst within DEST capacity. On Quasar these packs name no DFB: when the rows patch a reserved
// CB, call dummy_pack(cb) before cb_push_back.

ALWI void pack_rows_to_addr(uint32_t idst, uint32_t l1_addr) {
#ifndef ARCH_QUASAR
    PACK((_llk_pack_rows_(idst, l1_addr - 1)));
#else
    // Quasar L1 address fields have no off-by-one convention.
    PACK((llk_pack_rows_to_addr(idst, l1_addr)));
#endif
}

ALWI void pack_rows_to_addr_uninit() {
    PACK((llk_pack_rows_uninit()));
#ifndef ARCH_QUASAR
    // pack_rows_to_addr_init (llk_pack_rows_init) overwrote ADDR_MOD_0/1 with
    // row-pack address mods; llk_pack_rows_uninit only restores the X counter.
    // Restore the Default tile-pack addrmods so the init/uninit pair is symmetric
    // and downstream ops that pack via MOP-only pack_block_contiguous_init (which
    // does not reconfigure addrmods) do not inherit the row-pack mods. This is the
    // same primitive llk_pack_init<Default> uses to set ADDR_MOD_0/1/2.
    PACK((_llk_pack_configure_addrmod_<PackMode::Default>()));
#endif
}

}  // namespace ckernel
