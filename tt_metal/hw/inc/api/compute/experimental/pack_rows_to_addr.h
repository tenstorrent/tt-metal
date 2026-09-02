// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generic pack_rows_to_addr: pack N row-major rows from DEST to an arbitrary
// L1 byte address. Reusable across cache-update ops (dsa_indexer_cache_update,
// kv_cache_update, etc.) for patching rows directly into untilized buffers.

#pragma once

#include "api/compute/common.h"

namespace ckernel {

// Blackhole + Wormhole: the row-pack LLKs (llk_pack_rows.h) exist in both of those llk_libs but not in
// tt_llk_quasar, so this is an exclusion guard rather than the ARCH_BLACKHOLE inclusion guard the other
// experimental headers here use.
#ifndef ARCH_QUASAR

ALWI void pack_rows_to_addr_init(uint32_t num_rows) { PACK((llk_pack_rows_init(num_rows))); }

// NOTE: this raw path bypasses the llk_pack_rows wrapper's dst_index bound check, because it packs to an
// arbitrary L1 address rather than a CB. Callers are responsible for keeping idst within DEST capacity.
ALWI void pack_rows_to_addr(uint32_t idst, uint32_t l1_addr) { PACK((_llk_pack_rows_(idst, l1_addr - 1))); }

ALWI void pack_rows_to_addr_uninit() {
    PACK((llk_pack_rows_uninit()));
    // pack_rows_to_addr_init (llk_pack_rows_init) overwrote ADDR_MOD_0/1 with
    // row-pack address mods; llk_pack_rows_uninit only restores the X counter.
    // Restore the Default tile-pack addrmods so the init/uninit pair is symmetric
    // and downstream ops that pack via MOP-only pack_block_contiguous_init (which
    // does not reconfigure addrmods) do not inherit the row-pack mods. This is the
    // same primitive llk_pack_init<Default> uses to set ADDR_MOD_0/1/2.
    PACK((_llk_pack_configure_addrmod_<PackMode::Default>()));
}

#endif

}  // namespace ckernel
