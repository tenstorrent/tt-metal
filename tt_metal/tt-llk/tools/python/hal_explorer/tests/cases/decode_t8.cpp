// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
#include "llk_unpack_common.h"
#include "llk_pack_untilize.h"
void hwcfg(const std::uint32_t a)
{
    _llk_unpack_hw_configure_<false>(5, 5, 5, 5, 16, 16, 4, 4, a, a);
}
void relu(const std::uint32_t a)
{
    _llk_pack_relu_config_(ckernel::ReluConfig::from_packed(a));
}
void untilize_init()
{
    _llk_pack_untilize_init_<8, 8, false, false>(5, 5, 16, 4);
}
