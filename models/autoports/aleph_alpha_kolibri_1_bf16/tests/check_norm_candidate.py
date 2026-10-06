# SPDX-License-Identifier: Apache-2.0
"""Diagnostic candidate; production source is not patched by this runner."""

import os

import ttnn

from .diagnostic_baseline import FunctionalDecoder
from .run_coverage import run

original = FunctionalDecoder._norm


def candidate(self, x, name):
    if os.environ.get("DIAG_NORM_ALL") != "1" and name != "input_layernorm":
        return original(self, x, name)
    normalized = ttnn.rms_norm(
        x, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.multiply(normalized, self.norms[name])


if os.environ.get("DIAG_NORM_NONE") != "1":
    FunctionalDecoder._norm = candidate
if os.environ.get("DIAG_ACCURATE_EXP") == "1":
    original_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode

    def accurate_sdpa(*args, **kwargs):
        kwargs["program_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=32,
            k_chunk_size=32,
            max_cores_per_head_batch=1,
            exp_approx_mode=False,
        )
        return original_sdpa(*args, **kwargs)

    ttnn.transformer.paged_scaled_dot_product_attention_decode = accurate_sdpa
run(4, synthetic=True, output=os.environ.get("DIAG_OUTPUT", "synthetic_norm_candidate_4.json"))
