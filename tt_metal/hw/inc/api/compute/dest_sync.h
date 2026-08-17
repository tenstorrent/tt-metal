// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#if defined(ARCH_QUASAR)
#include "llk_dest_dvalid_api.h"
#endif

namespace ckernel {

#define DEST_UNPACK ::dest_order::client::UNPACK
#define DEST_FPU ::dest_order::client::FPU
#define DEST_SFPU ::dest_order::client::SFPU
#define DEST_PACK ::dest_order::client::PACK
#define DEST_CHAIN(...) ::dest_order::chain<__VA_ARGS__>

template <typename CHAIN>
ALWI void compute_kernel_dest_sync_init() {
#if defined(ARCH_QUASAR)
    llk_dest_dvalid_configure<CHAIN>();
#endif
}

template <typename CHAIN>
ALWI void compute_kernel_dest_sync_uninit() {
#if defined(ARCH_QUASAR)
    llk_dest_dvalid_teardown<CHAIN>();
#endif
}

template <typename CHAIN, dest_order::client FROM, dest_order::client TO>
ALWI void dest_sync_reconfig() {
#if defined(ARCH_QUASAR)
    llk_dest_dvalid_reconfig<CHAIN, FROM, TO>();
#endif
}

ALWI void dest_sync_section_done() {
#if defined(ARCH_QUASAR)
    llk_dest_dvalid_passthrough_if_skipped();
#endif
}

}  // namespace ckernel
