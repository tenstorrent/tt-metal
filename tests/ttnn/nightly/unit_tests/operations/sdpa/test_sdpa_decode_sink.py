# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from models.common.utility_functions import skip_for_blackhole

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    run_sdpa_decode_sink_impl,
)


@skip_for_blackhole("Failing on Blackhole, Issue #27193")
@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, dim",
    [
        (1, 256, 32, 4, 64),  # GPT-OSS 20B TP=2
        (64, 1024, 32, 8, 64),
        (16, 1024, 32, 1, 64),
        (32, 256, 32, 1, 128),
        (32, 128, 8, 1, 128),
        (32, 128, 8, 8, 128),
        # (1, 256, 64, 8, 64),  # GPT-OSS 20B TP=1 (FIXME: Fails)
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
    ],
)
def test_sdpa_decode(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    dim,
    q_dtype,
    dtype,
    function_level_defaults,
    reset_seeds,
):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("Only bfloat16 is supported for multi-head queries")

    run_sdpa_decode_sink_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        dim,
        q_dtype,
        dtype,
    )
