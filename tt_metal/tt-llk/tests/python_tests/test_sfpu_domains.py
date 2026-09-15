# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the sfpu_domains gates that decide what a sweep may inject.

No kernel, no device: these are pure-Python assertions about metadata. They are here
because specials_safe() is a *measured* matrix — 250 hardware variants reduced to a
handful of rules (see the section comment in sfpu_domains) — and until now nothing
executed it. Both production callers short-circuit on SPECIALS_READY_OPS, which is empty,
so the rules and the enum-normalisation trap underneath them could be rewritten without a
single test changing outcome. The measurement is expensive to redo and cheap to pin, so it
is pinned here.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.sfpu_domains import specials_safe, specials_safe_formats

# The formats the measurement covered: the 5x5 matrix driven over the isinf / isposinf /
# isneginf / isnan / isfinite predicates on Wormhole n150. Integer and Fp8/MX *output*
# formats are deliberately absent — the measurement never covered them and the gate makes
# no claim there, so pinning a verdict for them would invent one.
_MEASURED_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
]

# The whole accepted set, written out. Seven cells of fifty, and each one is a rule:
#
#   * A Float32 input carries specials at either dest_acc, into any non-block output —
#     Float16 included, but only with the 32-bit dest (breaker 1, output side).
#   * A Float16_b input carries them only at dest_acc=No; the bf16 -> fp32 dest unpack
#     loses -inf and NaN (breaker 2), and cannot reach a Float16 output at all.
#   * A Float16 input never carries them, at any dest_acc, into any output (breaker 1).
#   * A block-float input cannot carry them in the first place, and a block-float output
#     cannot express the result.
_ACCEPTED = frozenset(
    {
        (DataFormat.Float32, DataFormat.Float32, False),
        (DataFormat.Float32, DataFormat.Float16_b, False),
        (DataFormat.Float32, DataFormat.Float32, True),
        (DataFormat.Float32, DataFormat.Float16, True),
        (DataFormat.Float32, DataFormat.Float16_b, True),
        (DataFormat.Float16_b, DataFormat.Float32, False),
        (DataFormat.Float16_b, DataFormat.Float16_b, False),
    }
)

_MEASURED_MATRIX = [
    (inp, out, dest_acc)
    for inp in _MEASURED_FORMATS
    for out in _MEASURED_FORMATS
    for dest_acc in (False, True)
]


@pytest.mark.parametrize(
    "input_format,output_format,dest_acc",
    _MEASURED_MATRIX,
    ids=[
        f"{inp.name}-{out.name}-dest_acc_{'Yes' if d else 'No'}"
        for inp, out, d in _MEASURED_MATRIX
    ],
)
def test_specials_safe_matches_measured_matrix(input_format, output_format, dest_acc):
    """Every cell of the measured matrix, against the checked-in verdict."""
    expected = (input_format, output_format, dest_acc) in _ACCEPTED
    assert specials_safe(input_format, output_format, dest_acc) is expected, (
        f"specials_safe({input_format.name}, {output_format.name}, dest_acc={dest_acc}) "
        f"changed: expected {expected}. Either a rule moved or the measurement did — if "
        f"the latter, re-measure with the predicate sweep described in sfpu_domains and "
        f"update _ACCEPTED with the new verdict."
    )


def test_specials_safe_accepts_nothing_outside_the_measured_matrix():
    """The gate defaults to deny, so an unlisted input format is False rather than a
    wall of failures with one root cause."""
    for unlisted in (
        DataFormat.Tf32,
        DataFormat.Int32,
        DataFormat.UInt32,
        DataFormat.Fp8_e4m3,
        DataFormat.MxFp8P,
        DataFormat.Bfp2_b,
    ):
        for dest_acc in (False, True):
            assert not specials_safe(unlisted, DataFormat.Float32, dest_acc)


def test_specials_safe_rejects_mx_outputs():
    """MX outputs are excluded statically, not by measurement: an inf/NaN inside a block
    whose shared exponent is finite is not a value the format can express."""
    for mx_output in (DataFormat.MxFp8P, DataFormat.MxFp8R, DataFormat.MxFp4):
        assert not specials_safe(DataFormat.Float32, mx_output, True)


@pytest.mark.parametrize(
    "dest_acc,flag",
    [
        (True, True),
        (False, False),
        (DestAccumulation.Yes, True),
        (DestAccumulation.No, False),
    ],
    ids=["bool_True", "bool_False", "DestAccumulation_Yes", "DestAccumulation_No"],
)
def test_dest_acc_normalisation(dest_acc, flag):
    """A DestAccumulation member must read as its value, not as its truthiness.

    Both members are truthy objects, so a `bool(dest_acc)` implementation would evaluate
    DestAccumulation.No as the 32-bit-dest case and silently flip whole rows of the
    matrix. Float16_b -> Float16_b is the probe because its verdict inverts with the flag:
    accepted at dest_acc=No, rejected at dest_acc=Yes (breaker 2). So a member that
    normalised to the wrong bool would fail here rather than pass by coincidence.
    """
    assert specials_safe(DataFormat.Float16_b, DataFormat.Float16_b, dest_acc) is (
        not flag
    )


@pytest.mark.parametrize(
    "bad", [1, 0, "Yes", None, 1.0], ids=["int_1", "int_0", "str", "None", "float"]
)
def test_dest_acc_rejects_non_flags(bad):
    """Anything that is neither a bool nor a DestAccumulation raises rather than being
    guessed at. 1 and 0 included: they would work by accident and hide a caller bug."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe(DataFormat.Float32, DataFormat.Float32, bad)


def test_specials_safe_formats_filters_to_the_accepted_rows():
    formats = [
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.No)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float16_b, DataFormat.Float16_b),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.Yes)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float32, DataFormat.Float16),
    ]


def test_specials_safe_formats_validates_dest_acc_on_an_empty_list():
    """Normalisation happens once up front, so a bad flag raises even with nothing to
    filter — otherwise the error surfaces only for callers that happen to pass formats.
    """
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe_formats([], "Yes")
