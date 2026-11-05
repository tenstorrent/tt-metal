// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
// TTI_REPLAY comes from the includer's ckernel.h (a TT-Metal kernel) or the deployment's headers.
namespace ckernel::sfpu::bf16::logsigmoid {
// An SFPU call owns replay slots 0-15: matmul and reduce record theirs from replay_buf_offset = 16 at init
// and replay them across the call. A row that would record past slot 15 is issued, or records only a head that fits
// (record_split).
}  // namespace ckernel::sfpu::bf16::logsigmoid
