# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Prefill block-level device-perf tests with real pretrained weights.

- `block_8x4_layer0_dense`: full MLA + dense FFN block on an 8x4 galaxy mesh. Measures
  MLA, dense FFN, norms, and residual adds. Gives a baseline without the MoE path.
- `block_8x4_layer3_moe_5k` / `block_8x4_layer3_moe_25k`: full MLA + MoE block on an 8x4 galaxy
  mesh (gate + dispatch + routed experts + combine + shared expert + reduce), at
  `isl_5k` and `isl_25k` respectively. Subtract the dense-layer number to isolate MoE
  overhead. The two run at different sequence lengths, so they use distinct subdir /
  model_name / comments to keep their perf baselines separate.
- `block_2x4_layer3_moe`: full MLA + MoE block on a 2x4 2-link mesh, using the smaller
  `isl_6k4` configuration for real-weights perf coverage on that topology.

All run 1 iteration with `skip_reference=True` (no torch reference, bfp8/bfp4 production
dtypes). The 8x4 layer3 cases run at `isl_total=5*1024` (`isl_5k`) and `isl_total=25*1024`
(`isl_25k`), the 8x4 layer0 case at `isl_25k`, while the 2x4 case runs at
`isl_total=6*1024+512` (`isl_6k4`). Requires the appropriate hardware for the selected mesh
topology and `DEEPSEEK_V3_HF_MODEL` pointing to the pretrained checkpoint.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            20_680_586,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer3 and gate_device and no_ref and isl_5k'",
            33_691_097,
            "deepseek_v3_prefill_block_5k",
            "deepseek_v3_prefill_block_8x4_layer3_moe_5k",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_5k",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            130_873_080,
            "deepseek_v3_prefill_block_25k",
            "deepseek_v3_prefill_block_8x4_layer3_moe_25k",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_25k",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4'",
            107_554_551,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_2link",
        ),
    ],
    ids=[
        "block_8x4_layer0_dense",
        "block_8x4_layer3_moe_5k",
        "block_8x4_layer3_moe_25k",
        "block_2x4_layer3_moe",
    ],
)
@pytest.mark.timeout(0)
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
