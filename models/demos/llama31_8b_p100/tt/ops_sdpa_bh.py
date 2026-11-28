# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("sdpa_bh", "sdpa_bh_supported")


def sdpa(q, k, v, scale, program_config, compute_kernel_config, output_concat_heads=False):
    if not ttnn.experimental.llama31_8b.sdpa_bh_supported(
        q, k, v, program_config, compute_kernel_config=compute_kernel_config
    ):
        return None
    return ttnn.experimental.llama31_8b.sdpa_bh(
        q,
        k,
        v,
        program_config,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        output_concat_heads=output_concat_heads,
    )
