# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import models.perf.device_perf_utils as perf_utils


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "module, filter, num_iterations, expected_perf",
    [
        ["segformer_attention", "", 1, 620.0],
        ["segformer_decode_head", "", 1, 245.0],
        ["segformer_dwconv", "", 1, 445.0],
        ["segformer_efficient_selfattention", "", 1, 650.0],
        ["segformer_encoder", "", 1, 190.0],
        ["segformer_layer", "", 1, 215.0],
        ["segformer_mix_ffn", "", 1, 370.0],
        ["segformer_mlp", "", 1, 8900.0],
        ["segformer_model", "", 1, 190.0],
        ["segformer_overlap_path_embeddings", "", 1, 2048.0],
        ["segformer_selfoutput", "", 1, 14500.0],
    ],
)
def test_perf_device_bare_metal(module, filter, num_iterations, expected_perf):
    batch_size = 1
    subdir = "segformer"
    margin = 0.05

    command = f"pytest tests/ttnn/integration_tests/segformer/test_{module}.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = perf_utils.run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = perf_utils.check_device_perf(
        post_processed_results, margin, expected_perf_cols, assert_on_fail=True
    )
    perf_utils.prep_device_perf_report(
        model_name=f"segformer_module_{module}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"{num_iterations}_iterations",
    )
