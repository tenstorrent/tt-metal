# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMoRuntimeOptions (host only): defaults = production, from_env mapping, invalid combinations."""

import pytest

import ttnn
from models.demos.mimo_v2_d_p.tt.moe_ag import flat_rows
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions


def test_defaults_and_from_env():
    assert MiMoRuntimeOptions.from_env({}) == MiMoRuntimeOptions()
    o = MiMoRuntimeOptions.from_env({"MIMO_EXPERT_DTYPE": "bf8", "MIMO_TTNN_CACHE": "0", "MIMO_FLAT_EXPERT": "0"})
    assert o.expert_dtype == ttnn.bfloat8_b and not o.ttnn_cache and o.routed_expert == "unified" and not o.use_moe_ag
    assert MiMoRuntimeOptions.from_env({"TT_METAL_SDPA_RING_TWO_LEVEL": "1"}).sdpa_two_level


def test_py_expert_is_dispatch_only(expect_error):
    with expect_error(ValueError, "moe_ag=False"):
        MiMoRuntimeOptions(routed_expert="py")
    o = MiMoRuntimeOptions.from_env({"MIMO_FLAT_EXPERT": "py"})  # old command lines: the dispatch path
    assert o.routed_expert == "py" and not o.moe_ag and not o.use_moe_ag
    with expect_error(ValueError, "moe_ag=False"):
        MiMoRuntimeOptions.from_env({"MIMO_FLAT_EXPERT": "py", "MIMO_MOE_AG": "1"})


@pytest.mark.parametrize(
    "tokens,k,epc,expected",
    [
        (1280, 8, 64, 10240 + 32 * 63),  # 2x2, 640 tokens / chip
        (4096, 8, 64, 32768 + 32 * 63),  # 2x2, 2048
        (5120, 8, 8, 40960 + 32 * 7),  # Galaxy 8x4, 640
        (16384, 8, 8, 131072 + 32 * 7),  # Galaxy 8x4, 2048
    ],
)
def test_flat_rows_worst_case(tokens, k, epc, expected):
    assert flat_rows(tokens, k, epc) == expected


def test_flat_rows_bound_is_tight_and_safe():
    """Brute force over small cases: the worst 32-padded layout of P pairs over <= experts_per_chip experts fits."""
    import itertools

    for epc in (1, 2, 3):
        for tokens in range(1, 40):
            k = 1
            P = tokens * min(k, epc)
            worst = max(
                sum(-(-c // 32) * 32 for c in split)
                for n in range(1, epc + 1)
                for cuts in itertools.combinations(range(1, P), n - 1)
                for split in [[b - a for a, b in zip((0,) + cuts, cuts + (P,))]]
            )
            assert flat_rows(tokens, k, epc) >= worst, (tokens, epc, worst)
