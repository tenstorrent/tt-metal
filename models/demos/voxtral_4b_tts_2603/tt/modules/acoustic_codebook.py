# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `acoustic_codebook` (`AcousticCodebook.decode`).

Finite scalar quantization, weight-free. The reference dequantizes integer codes back to the
[-1, 1] latent range:

    decode(codes) = codes * 2 / (n_levels - 1) - 1

Written as `(codes - (n_levels - 1) / 2) * (2 / (n_levels - 1))` instead: the subtraction is then
exact on the integer codes (they arrive as bfloat16, and `codes - 10` for codes in `[0, 20]` has an
exact bfloat16 representation), so the only rounding left is the single scale multiply.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    inner = getattr(torch_module, "inner", torch_module)
    n_levels = int(inner.n_levels)
    shift = (n_levels - 1) / 2.0
    scale = 2.0 / (n_levels - 1)

    def acoustic_codebook(codes, **kwargs):
        # The codes arrive as INTEGERS (uint32, ROW_MAJOR). Subtracting the shift from an unsigned
        # integer tensor wraps instead of going negative -- it read as PCC -0.87 -- so tilize and
        # widen first. Both calls are no-ops if a caller already hands over a tilized float.
        widened = ttnn.typecast(ttnn.to_layout(codes, ttnn.TILE_LAYOUT), ttnn.bfloat16)
        return ttnn.multiply(ttnn.subtract(widened, shift), scale)

    return acoustic_codebook
