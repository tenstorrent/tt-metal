# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Prefill block-level device-perf tests on 8x4 galaxy with real pretrained weights.

- `block_8x4_layer0_dense`: full MLA + dense FFN block. Measures MLA, dense FFN, norms,
  residual adds. Gives a baseline without the MoE path.
- `block_8x4_layer3_moe`: full MLA + MoE block (gate + dispatch + routed experts + combine
  + shared expert + reduce). Subtract the dense-layer number to isolate MoE overhead.

Both run 1 iteration with `skip_reference=True` (no torch reference, bfp8/bfp4 production
dtypes) at isl_total=25*1024 (`isl_25k`). Requires a galaxy runner (32 BH chips in 8x4
topology) and DEEPSEEK_V3_HF_MODEL pointing to the pretrained checkpoint.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            1,  # TODO: set baseline after first run
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            1,  # TODO: set baseline after first run
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights",
        ),
    ],
    ids=[
        "block_8x4_layer0_dense",
        "block_8x4_layer3_moe",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_prefill_block_perf(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
