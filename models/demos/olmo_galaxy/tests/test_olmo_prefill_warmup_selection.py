# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.olmo_galaxy.tt.generator import get_olmo_long_isl_warmup_seqlens, should_warmup_prefill_traces


@pytest.mark.parametrize(
    ("requested_prefill_len", "expected_warmups"),
    [
        (4096, []),
        (8160, []),
        (8192, []),
        (16384, []),
        (32768, [8192, 16384]),
        (65536, [8192, 16384, 32768]),
    ],
)
def test_olmo_long_isl_warmup_is_scoped_to_requested_prefill_len(requested_prefill_len, expected_warmups):
    assert get_olmo_long_isl_warmup_seqlens(requested_prefill_len) == expected_warmups


@pytest.mark.parametrize(
    ("already_warmed", "enable_trace", "expected"),
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_prefill_trace_warmup_only_runs_for_traced_prefill(already_warmed, enable_trace, expected):
    assert should_warmup_prefill_traces(already_warmed, enable_trace) is expected
