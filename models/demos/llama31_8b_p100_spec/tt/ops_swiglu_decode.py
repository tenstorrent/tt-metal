# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("swiglu_decode", "swiglu_decode_supported")


def covers(w1_out, w3_out, activation, dtype, prefetcher):
    return (
        prefetcher is None
        and activation == ttnn.UnaryOpType.SILU
        and dtype == ttnn.bfloat8_b
        and ttnn.experimental.llama31_8b_spec.swiglu_decode_supported(w1_out, w3_out)
    )


def swiglu_decode(w1_out, w3_out):
    return ttnn.experimental.llama31_8b_spec.swiglu_decode(w1_out, w3_out)
