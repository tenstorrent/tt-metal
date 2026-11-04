// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
// TTI_REPLAY comes from the includer's ckernel.h (a TT-Metal kernel) or the deployment's headers.
namespace ckernel::sfpu::bf16::hardmish {
// An SFPU call owns replay slots 0-15: matmul and reduce record theirs from replay_buf_offset = 16 at init
// and replay them across the call. A row that would record past slot 15 is issued, or records only a head that fits
// (record_split).
#ifndef TTPOLY_SFPU_REPLAY_SLOTS
#define TTPOLY_SFPU_REPLAY_SLOTS 16
#endif
}  // namespace ckernel::sfpu::bf16::hardmish
