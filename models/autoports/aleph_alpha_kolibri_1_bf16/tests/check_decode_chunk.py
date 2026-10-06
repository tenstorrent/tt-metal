# SPDX-License-Identifier: Apache-2.0
"""Precision diagnostic for decode's online BF16 softmax accumulation."""

import os

import ttnn

from .run_decoder import main

original = ttnn.transformer.paged_scaled_dot_product_attention_decode


def controlled(*args, **kwargs):
    kwargs["program_config"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        q_chunk_size=32,
        k_chunk_size=int(os.environ["DIAG_K_CHUNK"]),
        max_cores_per_head_batch=1,
        exp_approx_mode=False,
    )
    return original(*args, **kwargs)


ttnn.transformer.paged_scaled_dot_product_attention_decode = controlled
if os.environ.get("DIAG_COVERAGE"):
    from .run_coverage import run

    run(
        4,
        synthetic=os.environ["DIAG_COVERAGE"] == "synthetic",
        output="k256_" + os.environ["DIAG_COVERAGE"] + "_4.json",
    )
else:
    main()
