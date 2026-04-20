# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "batch_size, seq_len, chunk_size, num_heads, head_k_dim, head_v_dim, expected_perf",
    [
        # Must match a parametrized case in test_ttnn_validation_2.py::test_fused_chunked_delta_rule_ttnn.
        # Tuple order HERE is (batch_size, seq_len, chunk_size, num_heads, head_k_dim, head_v_dim) but the
        # pytest node id is built as seq_len-chunk_size-batch_size-... (see `command` below).
        # (1, 64, 64, 4, 128, 256, 1.0),
        (2, 64, 64, 4, 128, 256, 1.0),
        # (1, 128, 64, 4, 128, 256, 1.0),
        # (2, 128, 64, 4, 128, 256, 1.0),
        # (2, 32, 32, 2, 64, 128, 1.0),
        # (4, 64, 32, 8, 128, 256, 1.0),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_fused_chunked_delta_rule(
    batch_size,
    seq_len,
    chunk_size,
    num_heads,
    head_k_dim,
    head_v_dim,
    expected_perf,
):
    model_name = "fused_chunked_delta_rule_ttnn"
    subdir = model_name
    num_iterations = 1
    margin = 0.04

    command = (
        f"pytest models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation_2.py"
        f"::test_fused_chunked_delta_rule_ttnn"
        f"[{seq_len}-{chunk_size}-{batch_size}-{num_heads}-{head_k_dim}-{head_v_dim}]"
    )

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=True)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"seq_{seq_len}-chunk_{chunk_size}-heads_{num_heads}-kdim_{head_k_dim}-vdim_{head_v_dim}",
    )
