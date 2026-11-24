// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7a swiglu phase (megakernel FFN half, tt/mk_ffn.py): m = SiLU(min(gate, limit)) * clamp(up, -limit, limit)
// (the dense MLP's ttnn.clamped_silu_glu) on the [1,1,B<=32, 2I] gate | up output of the gate_up phase; core k owns the
// NT column tiles c0 .. c0 + NT of m (gate tile c, up tile I/32 + c). CB indices at SWI_CB0 + n.
#pragma once

#include <cstdint>

#ifndef SWI_CB0
#define SWI_CB0 0
#endif

namespace swiglu {
constexpr uint32_t CB_G = SWI_CB0 + 0;  // bf16 gate tiles (NT)
constexpr uint32_t CB_U = SWI_CB0 + 1;  // bf16 up tiles (NT)
constexpr uint32_t CB_M = SWI_CB0 + 2;  // bf16 output tiles (2)
}  // namespace swiglu
