# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Prefill block-level device-perf tests with real pretrained weights.

- `block_8x4_layer0_dense`: full MLA + dense FFN block on an 8x4 galaxy mesh. Measures
  MLA, dense FFN, norms, and residual adds. Gives a baseline without the MoE path.
- `block_8x4_layer3_moe`: full MLA + MoE block on an 8x4 galaxy mesh (gate + dispatch +
  routed experts + combine + shared expert + reduce). Subtract the dense-layer number to
  isolate MoE overhead.
- `block_2x4_layer3_moe`: full MLA + MoE block on a 2x4 2-link mesh, using the smaller
  `isl_6k4` configuration for real-weights perf coverage on that topology.

All three run 1 iteration with `skip_reference=True` (no torch reference, bfp8/bfp4
production dtypes). The 8x4 cases run at `isl_total=25*1024` (`isl_25k`), while the 2x4
case runs at `isl_total=6*1024+512` (`isl_6k4`). Requires the appropriate hardware for the
selected mesh topology and `DEEPSEEK_V3_HF_MODEL` pointing to the pretrained checkpoint.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py"


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        # FABRIC_1D baselines — tightened with `and not fabric2d-` to exclude the new
        # 2D parametrize ids in test_prefill_block_loop.py (substring `mesh-8x4`/`mesh-2x4`
        # would otherwise match both 1D and 2D variants).
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer0 and gate_device and no_ref and isl_25k and not fabric2d-'",
            19_185_763,  # Recalibrated 2026-06-02 (CI run 26781295939): FABRIC_1D path sped up ~7% by the
            "deepseek_v3_prefill_block",  # ccl_routing_utils migration (non-blocking atomic-inc flush, etc.).
            "deepseek_v3_prefill_block_8x4_layer0_dense",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-8x4 and layer3 and gate_device and no_ref and isl_25k and not fabric2d-'",
            73_644_850,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights",
        ),
        (
            f"pytest {_TEST_PATH} -k 'mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4'",
            61_188_063,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_2link",
        ),
        # FABRIC_2D variants. The layer3 MoE entry has been calibrated against a real run on
        # BH Galaxy (~149.6 ms measured); the other two are still uncalibrated placeholders
        # with a wide margin so the first run will pass and surface the measured number.
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-8x4 and layer0 and gate_device and no_ref and isl_25k'",
            26_070_387,  # Recalibrated 2026-06-05 on bh-glx-d07u02 (post-main-rebase).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer0_dense_fabric2d",
            1,
            1,
            0.03,
            "glx_8x4_layer0_dense_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-8x4 and layer3 and gate_device and no_ref and isl_25k'",
            86_581_430,  # Recalibrated 2026-06-05 on bh-glx-d07u02 — post-main-rebase speedup from #45189 (unified_routed_expert_ffn).
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_8x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "glx_8x4_layer3_moe_real_weights_fabric2d",
        ),
        (
            f"pytest {_TEST_PATH} -k 'fabric2d-mesh-2x4 and layer3 and gate_device and no_ref and isl_6k4'",
            72_625_506,
            "deepseek_v3_prefill_block",
            "deepseek_v3_prefill_block_2x4_layer3_moe_fabric2d",
            1,
            1,
            0.03,
            "2x4_layer3_moe_real_weights_fabric2d",
        ),
    ],
    ids=[
        "block_8x4_layer0_dense",
        "block_8x4_layer3_moe",
        "block_2x4_layer3_moe",
        "block_8x4_layer0_dense_fabric2d",
        "block_8x4_layer3_moe_fabric2d",
        "block_2x4_layer3_moe_fabric2d",
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
