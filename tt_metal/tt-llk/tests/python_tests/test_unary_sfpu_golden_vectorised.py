# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The whole-tile unary SFPU golden must agree bit-exactly with the per-element one.

``UnarySFPUGolden`` keeps two ways to evaluate an op: the per-element method registered
in ``self.ops`` (the oracle, and the only place each op's semantics are written out) and
the whole-tile entry in ``_VECTOR_TORCH_FNS`` / ``_VECTOR_SPECIAL_OPS`` that ``__call__``
prefers when one exists. This file is the contract between them.

Bit-exactness, not tolerance, is the bar. The golden is what every SFPU test asserts
against, so a vectorised op that is merely *close* to the scalar one silently moves the
pass/fail boundary of the whole sweep. torch is entitled to pick a different code path
for a 1-element tensor than for a 1024-element one -- and for some ops it does -- which
is exactly why membership of the table has to be measured rather than assumed. For the
same reason the comparison is repeated at every window size the sweep dispatches (see
``WINDOWS``) rather than at one tile.

An op with no table entry is not a failure; it falls back to the per-element loop.
``test_no_unlisted_op_is_silently_bit_exact_capable`` reports the fallback set so the
coverage stays visible instead of quietly shrinking.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden
from helpers.llk_params import DestAccumulation, MathOperation, format_dict

# One tile: the base population size, and the smallest window __call__ can produce.
TILE = 1024

# The window sizes __call__ actually hands to a vector op. It evaluates the whole
# ``TILE_SIZE * iterations`` slice at once, and the sweep's input shapes are 64x64 and
# 128x256, so in production that slice is 4,096 or 32,768 elements -- not one tile.
# torch selects its kernel on numel, which is the very reason several ops had to be kept
# off the table, so agreement has to be *measured* at each dispatched size rather than
# inferred from the smallest one.
#
# The larger windows are the base population tiled. That keeps the scalar oracle at
# 1,024 evaluations without weakening the check: every op on the table is a pure
# elementwise map, so the per-element answer for a repeated input is the repeated
# answer, and what the larger window changes is the shape torch dispatches on.
#
# Measured on torch 2.9: no op on the table is shape-sensitive across these three
# windows -- the 0-d-vs-N-d split that kept 18 ops off it is a boundary at N == 1, not a
# threshold further up. The check is here so that stays a measured fact; it costs ~1.3 s.
WINDOWS = [1024, 4096, 32768]

# Both formats matter: _torch_unary's inf->NaN substitution fires only for an
# A-exponent (Float16) format, so an op is only proven equivalent if it agrees under
# both branches of that rule.
# Float32 is in the list because the Dest-dtype family (_VECTOR_DST_TORCH_FNS)
# evaluates *in* dst_format, so each of the three Dest dtypes is a different
# computation and has to be compared separately.
FORMATS = [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32]


def _population(name):
    """Inputs chosen to reach each op's interesting region *and* its specials.

    A single uniform population would leave most of these ops evaluated only in their
    smooth interior, where scalar and vectorised torch trivially agree; the divergences
    worth catching live at poles, branch cuts and non-finite inputs.
    """
    torch.manual_seed(0)
    if name == "unit_interval":
        # (-1, 1): the domain of asin/acos/atanh/erfinv.
        return torch.rand(TILE) * 1.98 - 0.99
    if name == "positive":
        # (0, 100]: log/sqrt/rsqrt/lgamma/digamma/reciprocal.
        return torch.rand(TILE) * 100 + 1e-3
    if name == "signed_wide":
        # Straddles zero with a wide range: sign-sensitive and saturating ops.
        return torch.randn(TILE) * 20
    if name == "above_one":
        # [1, 50]: acosh's domain.
        return torch.rand(TILE) * 49 + 1.0
    if name == "specials":
        # Every special the pipeline can carry, padded with ordinary values so the
        # comparison is not dominated by NaN-vs-NaN cases.
        base = torch.rand(TILE) * 4 - 2
        specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            torch.inf,
            -torch.inf,
            torch.nan,
            -torch.nan,
            1e-45,
            -1e-45,
            3.4e38,
            -3.4e38,
            65504.0,
            -65504.0,
        ]
        for i, v in enumerate(specials):
            base[i] = v
        return base
    raise AssertionError(f"unknown population {name}")


POPULATIONS = ["unit_interval", "positive", "signed_wide", "above_one", "specials"]


# One instance for the file, built directly rather than fetched with
# ``get_golden_generator``.
#
# The registry is what the device tests use, but it is not usable here: the harness
# replaces ``get_golden_generator`` with one that hands back a ``DummyGoldenGenerator``
# under ``--compile-producer`` -- which is how the Quasar compile lane runs the whole
# suite -- and that stub has none of the vector tables, so consulting the registry turns
# this file's *collection* into an error there rather than into a test result. Same fix,
# and same reason, as ``test_eltwise_binary_sfpu.py``'s ``_classify_edge_result`` and
# ``helpers/compressed_utils.py``'s matmul golden.
#
# Shared across the file rather than built per test, which is also what the registry did
# before, so the change is only in where the instance comes from. Safe because the only
# state the vector paths read is ``data_format`` / ``dst_format`` / ``dest_acc``, and every
# test binds all three through ``_bind_formats`` before dispatching. (``__init__``
# assembles a 114-entry dispatch dict; per-test instantiation measured ~0.5 s slower over
# the file -- small, but there is no reason to pay it.)
_GOLDEN = UnarySFPUGolden()


def _by_name(ops):
    """Sorted so the parametrize ids are stable across runs."""
    return sorted(ops, key=lambda o: o.name)


def _vector_ops():
    golden = _GOLDEN
    return _by_name(
        set(golden._VECTOR_TORCH_FNS)
        | set(golden._VECTOR_DST_TORCH_FNS)
        | set(golden._VECTOR_PY_FNS)
        | set(golden._VECTOR_SPECIAL_OPS)
    )


VECTOR_OPS = _vector_ops()

# _VECTOR_SPECIAL_OPS entries whose method routes through ``_torch_unary_vec`` or calls
# ``handle_infinite_numbers`` itself, so the NaN rule reaches them as well. The rest of
# that table (_vec_sign, _vec_cast_fp32_to_fp16a, the clamp/relu/Hardsigmoid family) has
# no such rule and neither do their scalar twins, so requiring it of them would assert a
# behaviour the golden does not have -- they are checked for format *invariance*
# instead. See test_nan_rule_holds_for_every_op_that_carries_it.
_NAN_RULE_SPECIAL_OPS = frozenset(
    {
        MathOperation.I0,
        MathOperation.I1,
        MathOperation.Square,
    }
)


def _nan_rule_partition():
    """Split the vector tables into the ops the NaN rule applies to and the rest.

    ``_VECTOR_DST_TORCH_FNS`` is in neither group: it evaluates *in* dst_format, so
    Float16 and Float16_b are two different computations there and nothing can be said
    about the pair.
    """
    golden = _GOLDEN
    rule = {
        "_VECTOR_TORCH_FNS": _by_name(golden._VECTOR_TORCH_FNS),
        "_VECTOR_SPECIAL_OPS": _by_name(
            set(golden._VECTOR_SPECIAL_OPS) & _NAN_RULE_SPECIAL_OPS
        ),
    }
    invariant = _by_name(
        (set(golden._VECTOR_SPECIAL_OPS) - _NAN_RULE_SPECIAL_OPS)
        | set(golden._VECTOR_PY_FNS)
    )
    return rule, invariant


_NAN_RULE_TABLES, _FORMAT_INVARIANT_OPS = _nan_rule_partition()

# The floor this file guards against a regression that quietly empties the vector
# tables: 88 of the 114 registered ops were measured bit-exact and are on the table.
# Asserted as a floor, not an equality -- adding an op is the intended direction, and
# only a *drop* means the optimisation has silently gone away.
MIN_VECTORISED_OPS = 88


def _bind_formats(golden, data_format):
    """Bind the fields ``__call__`` sets before dispatching.

    ``_torch_unary_vec`` and ``handle_infinite_numbers`` read ``data_format`` and
    ``_torch_dst_vec`` reads ``dst_format``, so they have to be bound the same way on
    both sides for the comparison to be fair.
    """
    golden.data_format = data_format
    golden.dst_format = data_format
    golden.dest_acc = DestAccumulation.No


def _scalar_reference(golden, operation, values):
    """Evaluate the per-element oracle, flagging the inputs it cannot handle.

    Returns ``(results, evaluable)``, where *evaluable* is False wherever the scalar
    method raised. Only the ``math``-backed methods do this, and only for inputs outside
    the domain the op is ever swept over -- see the caller.
    """
    out, ok = [], []
    for x in values.tolist():
        try:
            out.append(float(golden.ops[operation](x)))
            ok.append(True)
        except (OverflowError, ValueError):
            out.append(torch.nan)
            ok.append(False)
    return torch.tensor(out, dtype=torch.float32), torch.tensor(ok)


def _bits_agree(a, b, nan_sign_matters=False):
    """Elementwise bitwise equality on fp32, treating any-NaN as equal to any-NaN.

    NaN *payloads* are not part of what the golden asserts: the payload never survives
    the cast to Dest. Everything else -- including the distinction between NaN and
    infinity, which is the one the pack path turns into a visible +inf/-inf disagreement
    -- is compared exactly.

    The NaN *sign* is a different matter, and *nan_sign_matters* is the caller's way of
    saying so. ``__call__`` canonicalises the sign of every NaN except for the three ops
    in ``_NAN_SIGN_TRANSPARENT_OPS`` (Neg, Abs, Identity) -- precisely the ops whose NaN
    sign is meaningful, and all three are on the vector path. For them a sign
    disagreement becomes a visible +inf vs -inf once ``convert_nan_to_inf`` runs, so
    folding it into "both NaN" would hide the one property ``cast_to_dest_dtype`` is
    there to preserve.
    """
    both_nan = a.isnan() & b.isnan()
    if nan_sign_matters:
        both_nan &= torch.signbit(a) == torch.signbit(b)
    # `a == b` plus matching sign bits *is* bitwise equality for floats: the only two
    # distinct patterns that compare equal are +0.0 and -0.0. Written this way rather
    # than as an int32 bit view so it also applies to the float64 and 16-bit tensors the
    # tables produce before __call__'s cast.
    return ((a == b) & (torch.signbit(a) == torch.signbit(b))) | both_nan


def _first_difference(a, b, values, nan_sign_matters=False):
    bad = (~_bits_agree(a, b, nan_sign_matters)).nonzero().flatten()
    i = int(bad[0])
    return (
        f"{len(bad)}/{a.numel()} elements differ; first at [{i}]: "
        f"input {values[i]!r} -> vector {a[i].item()!r} vs scalar {b[i].item()!r}"
    )


def _compare(golden, operation, vector, scalar, evaluable, values, label):
    """Assert bit-exact agreement, dropping the inputs the oracle could not evaluate.

    Some scalar methods cannot evaluate every input: _cosh/_sinh go through
    math.cosh/sinh, which raise OverflowError near the fp32 maximum instead of returning
    an infinity (torch returns inf there, so the vectorised path is strictly more
    total). Those elements drop out of the comparison rather than being papered over --
    the oracle has no answer to compare against. The sweep never reaches them, because
    each op's stimuli come from its registered domain; this file is the only thing that
    feeds an op its format limits.
    """
    if not bool(evaluable.all()):
        excluded = int((~evaluable).sum())
        assert excluded < values.numel() // 2, (
            f"{operation.name}: the scalar oracle could not evaluate {excluded} of "
            f"{values.numel()} inputs, too many for the comparison to mean anything"
        )
        vector = vector[evaluable]
        scalar = scalar[evaluable]

    nan_sign_matters = operation in golden._NAN_SIGN_TRANSPARENT_OPS
    # raise rather than `assert cond, msg`: pytest's assertion rewriting renders both
    # tensors into the failure report, which buries the one line that says what
    # actually differs.
    if not _bits_agree(vector, scalar, nan_sign_matters).all():
        raise AssertionError(
            f"{operation.name} / {label}: "
            + _first_difference(
                vector, scalar, values[evaluable].tolist(), nan_sign_matters
            )
        )


@pytest.mark.parametrize("operation", VECTOR_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("data_format", FORMATS, ids=lambda f: f.name)
def test_vector_op_matches_scalar_op(operation, population, data_format):
    """Every table entry, over every population, at every window, under both NaN rules."""
    golden = _GOLDEN
    _bind_formats(golden, data_format)
    base = _population(population).to(torch.float32)

    vector_op = golden._vector_op(operation)
    assert vector_op is not None, f"{operation.name} is in the table but has no impl"

    scalar, evaluable = _scalar_reference(golden, operation, base)

    for window in WINDOWS:
        repeats = window // TILE
        values = base.repeat(repeats)
        _compare(
            golden,
            operation,
            vector_op(values).to(torch.float32),
            scalar.repeat(repeats),
            evaluable.repeat(repeats),
            values,
            f"{population} / {data_format.name} / window {window}",
        )


@pytest.mark.parametrize("operation", VECTOR_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("data_format", FORMATS, ids=lambda f: f.name)
def test_vector_op_matches_scalar_op_in_the_tile_dtype(operation, data_format):
    """The same agreement with the window arriving in the tile's own dtype.

    ``__call__`` does not hand ``vector_op`` an fp32 tensor: ``result`` follows
    *input_format* through ``tilize_block``, so for a 16-bit format the window is
    bfloat16 or float16 while the scalar path sees the Python doubles that
    ``result.tolist()`` widens the same elements to. Every other check here feeds fp32,
    the one dtype where that narrowing cannot lose anything -- a NaN sign included, which
    is exactly what ``cast_to_dest_dtype`` exists to preserve.

    One cell rather than the full matrix: the populations are already covered in fp32,
    and what is being pinned here is the dtype the window arrives in, so the specials
    are the population that matters.
    """
    golden = _GOLDEN
    _bind_formats(golden, data_format)
    values = _population("specials").to(format_dict[data_format])

    vector_op = golden._vector_op(operation)
    scalar, evaluable = _scalar_reference(golden, operation, values)
    _compare(
        golden,
        operation,
        vector_op(values).to(torch.float32),
        scalar,
        evaluable,
        values,
        f"specials in {format_dict[data_format]} / {data_format.name}",
    )


@pytest.mark.parametrize("operation", VECTOR_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("window", WINDOWS)
def test_vector_op_preserves_shape_and_dtype(operation, window):
    """The whole-tile result must be a same-length float tensor.

    ``__call__`` writes it straight into a slice of ``result``, so a shape or dtype
    surprise corrupts the tile rather than raising. Asserted on the *pre-cast* tensor:
    the ``.to(torch.float32)`` every other test applies would launder a bool- or
    int-returning table entry (several entries end in ``.to(d.dtype)`` for exactly that
    reason) into a passing float.
    """
    golden = _GOLDEN
    _bind_formats(golden, DataFormat.Float16_b)
    values = _population("signed_wide").to(torch.float32).repeat(window // TILE)
    raw = golden._vector_op(operation)(values)
    assert raw.shape == values.shape
    assert raw.dtype.is_floating_point, f"{operation.name} returned {raw.dtype}"


def test_every_table_entry_is_a_registered_op():
    """A table entry for an unregistered op would never be reached.

    ``_vector_op`` is consulted with whatever ``__call__`` was handed, so an entry whose
    MathOperation is not in ``self.ops`` is dead weight that reads as coverage.
    """
    golden = _GOLDEN
    unreachable = sorted(op.name for op in VECTOR_OPS if op not in golden.ops)
    assert not unreachable, f"vector table entries with no registered op: {unreachable}"


def test_no_unlisted_op_is_silently_bit_exact_capable():
    """Report which ops still take the per-element path, and hold the table's floor.

    The report is the point: ops legitimately stay scalar, and this is the only place
    the fallback set is written down where a reader will see it.

    The assertion is on the size of the *vector* table, not the fallback set. Asserting
    the fallback set instead is what makes this test vacuous: it is a subset of
    ``golden.ops`` by construction, so any bound against ``len(golden.ops)`` holds even
    with every vector table emptied -- which is the regression the test exists for.
    """
    golden = _GOLDEN
    scalar_only = sorted(op.name for op in golden.ops if op not in set(VECTOR_OPS))
    print(f"\n{len(VECTOR_OPS)} vectorised, {len(scalar_only)} still per-element:")
    print("  " + ", ".join(scalar_only))
    assert len(VECTOR_OPS) >= MIN_VECTORISED_OPS, (
        f"only {len(VECTOR_OPS)} ops are on the vector tables, down from the measured "
        f"{MIN_VECTORISED_OPS}; the sweep has silently gone back to the per-element "
        f"golden for {MIN_VECTORISED_OPS - len(VECTOR_OPS)} op(s)"
    )


def _eval_under(golden, operation, values, data_format):
    """Evaluate *operation* with *data_format* bound, in the table's own precision.

    Deliberately **not** cast to fp32. The NaN rule is about the infinities the op
    itself produces, and an fp32 cast manufactures others: ``_vec_square`` works in
    float64, where 3.4e38 squares to a finite 1.16e77 and never reaches
    ``handle_infinite_numbers`` -- it only becomes an infinity in the later cast, under
    both formats alike. Comparing after the cast would read that as the rule having been
    dropped.
    """
    _bind_formats(golden, data_format)
    return golden._vector_op(operation)(values)


@pytest.mark.parametrize("table", sorted(_NAN_RULE_TABLES))
def test_nan_rule_holds_for_every_op_that_carries_it(table):
    """``+/-inf`` under exponent-B must be NaN under an A-exponent format. Per op.

    ``_torch_unary_vec``'s inf->NaN substitution is the one behaviour that differs
    between the ``FORMATS`` entries, and the Float16 half of this file's matrix rests on
    it. Asserted as the rule itself, for every op that carries it, rather than by proxy:
    "some op produced an infinity" is satisfied by ``Abs`` -- a ``_VECTOR_PY_FNS`` entry
    with no such rule -- and stays green with the substitution deleted outright.

    Both directions are needed. Wherever exponent-B kept an infinity the A-exponent
    format must hold a NaN, *and* nothing may differ between the two formats anywhere
    else; either half alone passes on an implementation that has stopped applying the
    rule and started doing something else with the format.

    Parametrised per table so each one has to exercise the rule on its own. Rolling them
    together lets ``_vec_square`` -- which calls ``handle_infinite_numbers`` directly and
    so keeps the rule even if ``_torch_unary_vec`` drops it -- stand in for the whole
    ``_VECTOR_TORCH_FNS`` table.
    """
    golden = _GOLDEN
    values = _population("specials").to(torch.float32)

    exercised = []
    for operation in _NAN_RULE_TABLES[table]:
        exponent_b = _eval_under(golden, operation, values, DataFormat.Float16_b)
        exponent_a = _eval_under(golden, operation, values, DataFormat.Float16)

        infinite = exponent_b.isinf()
        if bool(infinite.any()):
            exercised.append(operation.name)
            kept = infinite & ~exponent_a.isnan()
            assert not bool(kept.any()), (
                f"{operation.name}: {int(kept.sum())} of {int(infinite.sum())} "
                "infinities under Float16_b were not substituted with NaN under "
                "Float16, so the inf->NaN rule is no longer applied"
            )

        # NaN sign is compared here: the substitution's output is what the pack path
        # later turns into a visible +inf/-inf, so a sign that varies with the format is
        # a real disagreement.
        differs = ~_bits_agree(exponent_a, exponent_b, nan_sign_matters=True)
        elsewhere = differs & ~infinite
        assert not bool(elsewhere.any()), (
            f"{operation.name}: Float16_b and Float16 differ in "
            f"{int(elsewhere.sum())} elements where Float16_b held no infinity, so "
            "something other than the inf->NaN rule varies with the format"
        )

    assert exercised, (
        f"no op in {table} produced an infinity under Float16_b on the specials "
        "population, so the inf->NaN substitution is untested from this table and the "
        "Float16 half of this file's matrix proves nothing about it"
    )


@pytest.mark.parametrize("operation", _FORMAT_INVARIANT_OPS, ids=lambda o: o.name)
def test_ops_without_the_nan_rule_are_format_invariant(operation):
    """The complement of the rule, pinned so a new op cannot land on the wrong side.

    ``_VECTOR_PY_FNS`` and the non-``_torch_unary_vec`` half of ``_VECTOR_SPECIAL_OPS``
    carry no format-dependent behaviour -- their scalar twins do not either. Asserting
    that keeps ``_NAN_RULE_SPECIAL_OPS`` honest: an op added to the wrong table fails
    here (invariance broken) or in the sibling test (rule never exercised), rather than
    silently narrowing what the rule is checked against.
    """
    golden = _GOLDEN
    values = _population("specials").to(torch.float32)
    exponent_b = _eval_under(golden, operation, values, DataFormat.Float16_b)
    exponent_a = _eval_under(golden, operation, values, DataFormat.Float16)
    differs = ~_bits_agree(exponent_a, exponent_b, nan_sign_matters=True)
    assert not bool(differs.any()), (
        f"{operation.name} changes with the data format in {int(differs.sum())} "
        "elements but is not listed as carrying the inf->NaN rule; if it does carry "
        "it, add it to _NAN_RULE_SPECIAL_OPS"
    )
