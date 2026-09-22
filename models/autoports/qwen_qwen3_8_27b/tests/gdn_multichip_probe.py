# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapt flat model Q/K/V to the monolithic GDN API for a TP4 comparison.

This test-only shim includes head reshaping and L2 normalization in the measured
forward. Set QWEN_GDN_PHASED=0 before launching. Normal model dispatch is unchanged.
"""

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder import main

native_gdn = ttnn.transformer.chunk_gated_delta_rule


def monolithic_gdn(q, k, v, g, beta, **kwargs):
    def heads(x):
        return ttnn.reshape(x, [x.shape[0], x.shape[1], x.shape[2] // 128, 128])

    def normalize(x):
        xf = ttnn.typecast(heads(x), ttnn.float32)
        return ttnn.typecast(
            ttnn.mul(xf, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.mul(xf, xf), dim=-1, keepdim=True), 1e-6))),
            ttnn.bfloat16,
        )

    return native_gdn(normalize(q), normalize(k), heads(v), g, beta, **kwargs)


if __name__ == "__main__":
    ttnn.transformer.chunk_gated_delta_rule = monolithic_gdn
    main()
