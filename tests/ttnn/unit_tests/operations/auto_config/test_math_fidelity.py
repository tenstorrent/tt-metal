# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for math fidelity module (CPU-only, no hardware required)."""

import pytest


from ttnn.operations.auto_config.math_fidelity import (
    MathFidelity,
    CYCLES_PER_TILE,
    MAX_CYCLES_PER_TILE,
    DTYPE_FIDELITY_CONSTRAINTS,
    valid_fidelities,
    default_fidelity,
    fidelity_cycle_cost,
    fidelity_to_ttnn_string,
    GPT_ATTENTION_SHAPES,
)


class TestMathFidelityEnum:
    """Tests for the MathFidelity enum values and ordering."""

    def test_enum_values(self):
        assert MathFidelity.LoFi.value == 0
        assert MathFidelity.HiFi2.value == 1
        assert MathFidelity.HiFi3.value == 2
        assert MathFidelity.HiFi4.value == 3

    def test_enum_ordering(self):
        assert MathFidelity.LoFi < MathFidelity.HiFi2
        assert MathFidelity.HiFi2 < MathFidelity.HiFi3
        assert MathFidelity.HiFi3 < MathFidelity.HiFi4

    def test_enum_count(self):
        assert len(MathFidelity) == 4


class TestCyclesPerTile:
    """Tests for the cycle cost table."""

    def test_known_values(self):
        assert CYCLES_PER_TILE[MathFidelity.LoFi] == 16
        assert CYCLES_PER_TILE[MathFidelity.HiFi2] == 32
        assert CYCLES_PER_TILE[MathFidelity.HiFi3] == 48
        assert CYCLES_PER_TILE[MathFidelity.HiFi4] == 64

    def test_all_fidelities_have_cost(self):
        for fid in MathFidelity:
            assert fid in CYCLES_PER_TILE

    def test_monotonically_increasing(self):
        costs = [CYCLES_PER_TILE[f] for f in MathFidelity]
        assert costs == sorted(costs)
        assert len(set(costs)) == len(costs)  # all distinct

    def test_max_constant(self):
        assert MAX_CYCLES_PER_TILE == 64
        assert MAX_CYCLES_PER_TILE == max(CYCLES_PER_TILE.values())


class TestValidFidelities:
    """Tests for dtype-to-fidelity constraint mapping."""

    def test_bfp8_bfp8_needs_hifi2(self):
        valid = valid_fidelities("BFLOAT8_B", "BFLOAT8_B")
        assert MathFidelity.HiFi2 in valid
        assert MathFidelity.LoFi not in valid

    def test_bf16_bf16_needs_hifi4_only(self):
        valid = valid_fidelities("BFLOAT16", "BFLOAT16")
        assert valid == [MathFidelity.HiFi4]

    def test_bf16_bfp4_prefers_hifi3_over_hifi2(self):
        """PR #39628: bfp4 SrcB has few mantissa bits, HiFi3 reads SrcB LSBs."""
        valid = valid_fidelities("BFLOAT16", "BFLOAT4_B")
        assert MathFidelity.HiFi3 in valid
        assert MathFidelity.HiFi2 not in valid

    def test_bf16_bfp8_allows_hifi2_and_above(self):
        valid = valid_fidelities("BFLOAT16", "BFLOAT8_B")
        assert MathFidelity.HiFi2 in valid
        assert MathFidelity.HiFi3 in valid
        assert MathFidelity.HiFi4 in valid
        assert MathFidelity.LoFi not in valid

    def test_bfp4_bfp4_allows_lofi(self):
        valid = valid_fidelities("BFLOAT4_B", "BFLOAT4_B")
        assert MathFidelity.LoFi in valid

    def test_unknown_dtype_returns_defaults(self):
        valid = valid_fidelities("fp32", "fp32")
        assert len(valid) > 0

    def test_case_insensitive_normalization(self):
        v1 = valid_fidelities("bfloat16", "bfloat16")
        v2 = valid_fidelities("BFLOAT16", "BFLOAT16")
        assert v1 == v2

    def test_datatype_prefix_stripped(self):
        v1 = valid_fidelities("DataType.BFLOAT16", "DataType.BFLOAT16")
        v2 = valid_fidelities("BFLOAT16", "BFLOAT16")
        assert v1 == v2

    def test_all_constraint_entries_nonempty(self):
        for key, fids in DTYPE_FIDELITY_CONSTRAINTS.items():
            assert len(fids) > 0, f"Empty fidelity list for {key}"

    def test_hifi4_always_valid(self):
        """HiFi4 should always be valid — it's the most accurate."""
        for (a, b) in DTYPE_FIDELITY_CONSTRAINTS:
            valid = valid_fidelities(a, b)
            assert MathFidelity.HiFi4 in valid, f"HiFi4 not valid for {a}×{b}"


class TestDefaultFidelity:
    """Tests for the default fidelity selection."""

    def test_default_is_first_valid(self):
        for (a, b), fids in DTYPE_FIDELITY_CONSTRAINTS.items():
            default = default_fidelity(a, b)
            assert default == fids[0]

    def test_bf16_bfp4_default_is_hifi3(self):
        assert default_fidelity("BFLOAT16", "BFLOAT4_B") == MathFidelity.HiFi3

    def test_bf16_bf16_default_is_hifi4(self):
        assert default_fidelity("BFLOAT16", "BFLOAT16") == MathFidelity.HiFi4


class TestFidelityCycleCost:
    """Tests for the cycle cost lookup."""

    @pytest.mark.parametrize("fid,expected", [
        (MathFidelity.LoFi, 16),
        (MathFidelity.HiFi2, 32),
        (MathFidelity.HiFi3, 48),
        (MathFidelity.HiFi4, 64),
    ])
    def test_parametrized(self, fid, expected):
        assert fidelity_cycle_cost(fid) == expected


class TestFidelityToString:
    """Tests for ttnn string conversion."""

    def test_all_fidelities(self):
        assert fidelity_to_ttnn_string(MathFidelity.LoFi) == "MathFidelity.LoFi"
        assert fidelity_to_ttnn_string(MathFidelity.HiFi2) == "MathFidelity.HiFi2"
        assert fidelity_to_ttnn_string(MathFidelity.HiFi3) == "MathFidelity.HiFi3"
        assert fidelity_to_ttnn_string(MathFidelity.HiFi4) == "MathFidelity.HiFi4"


class TestGPTAttentionShapes:
    """Tests for the GPT training shape table."""

    def test_shapes_exist(self):
        assert len(GPT_ATTENTION_SHAPES) >= 15

    def test_shapes_are_tuples(self):
        for shape in GPT_ATTENTION_SHAPES:
            assert len(shape) == 4
            m, k, n, desc = shape
            assert isinstance(m, int) and m > 0
            assert isinstance(k, int) and k > 0
            assert isinstance(n, int) and n > 0
            assert isinstance(desc, str)

    def test_no_duplicate_shapes(self):
        mkn = [(m, k, n) for m, k, n, _ in GPT_ATTENTION_SHAPES]
        assert len(set(mkn)) == len(mkn)
