# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bucket selection and the served max_seq_len for the configured context and mesh (CPU only)."""

import pytest

from models.demos.pplx_decider_v1_27b.tt.bucket_trace import BUCKETS, bucket_sizes_for, max_seq_len_for


@pytest.fixture(autouse=True)
def _default_buckets(monkeypatch):
    monkeypatch.delenv("PPLX_BUCKETS", raising=False)


@pytest.mark.parametrize(
    "max_seq_len, capacity, expected",
    [
        (8192, 10240, BUCKETS),  # the default PplxDecider: every bucket
        (2048, 4096, (256, 512, 768, 1024, 1536, 2048)),
        (1024, 2048, (256, 512, 768, 1024)),
        (1000, 2048, (256, 512, 768, 1024)),  # rounds up to the next bucket
        (8192, 2048, (256, 512, 768, 1024, 1536, 2048)),  # capped by the KV capacity
    ],
)
def test_bucket_sizes_for(max_seq_len, capacity, expected):
    assert bucket_sizes_for(max_seq_len, capacity) == expected


@pytest.mark.parametrize(
    "max_seq_len, two_d, expected",
    [
        (8192, False, 8192),  # 1xN: the chunk trace serves prompts past the largest bucket
        (8192, True, 4096),  # 2D: no chunked prefill, so the largest bucket is the limit
        (2048, True, 2048),
    ],
)
def test_max_seq_len_for(max_seq_len, two_d, expected):
    assert max_seq_len_for(max_seq_len, two_d) == expected


def test_max_seq_len_for_follows_pplx_buckets(monkeypatch):
    monkeypatch.setenv("PPLX_BUCKETS", "256,512")
    assert max_seq_len_for(8192, True) == 512
