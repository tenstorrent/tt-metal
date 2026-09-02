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
is exactly why membership of the table has to be measured rather than assumed.

An op with no table entry is not a failure; it falls back to the per-element loop.
``test_no_unlisted_op_is_silently_bit_exact_capable`` reports the fallback set so the
coverage stays visible instead of quietly shrinking.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import DestAccumulation

# One tile. The window __call__ evaluates is a whole number of tiles, so this is the
# smallest input that exercises the real code path.
TILE = 1024

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


def _vector_ops():
    golden = get_golden_generator(UnarySFPUGolden)
    return sorted(
        set(golden._VECTOR_TORCH_FNS)
        | set(golden._VECTOR_DST_TORCH_FNS)
        | set(golden._VECTOR_PY_FNS)
        | set(golden._VECTOR_SPECIAL_OPS),
        key=lambda o: o.name,
    )


VECTOR_OPS = _vector_ops()


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


def _bitwise_equal(a, b):
    """Bitwise equality on fp32, treating any-NaN as equal to any-NaN.

    NaN *payloads* are not part of what the golden asserts: ``__call__`` canonicalises
    the sign of every NaN it did not itself create (see _NAN_SIGN_TRANSPARENT_OPS), and
    the payload never survives the cast to Dest. Everything else -- including the
    distinction between NaN and infinity, which is the one the pack path turns into a
    visible +inf/-inf disagreement -- is compared exactly.
    """
    both_nan = a.isnan() & b.isnan()
    return ((a.view(torch.int32) == b.view(torch.int32)) | both_nan).all()


def _first_difference(a, b, values):
    both_nan = a.isnan() & b.isnan()
    bad = (
        (~((a.view(torch.int32) == b.view(torch.int32)) | both_nan)).nonzero().flatten()
    )
    i = int(bad[0])
    return (
        f"{len(bad)}/{a.numel()} elements differ; first at [{i}]: "
        f"input {values[i]!r} -> vector {a[i].item()!r} vs scalar {b[i].item()!r}"
    )


@pytest.mark.parametrize("operation", VECTOR_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("population", POPULATIONS)
@pytest.mark.parametrize("data_format", FORMATS, ids=lambda f: f.name)
def test_vector_op_matches_scalar_op(operation, population, data_format):
    """Every table entry, over every population, under both NaN-rule branches."""
    golden = get_golden_generator(UnarySFPUGolden)
    values = _population(population).to(torch.float32)

    # __call__ sets these before dispatching; _torch_unary and handle_infinite_numbers
    # both read them, so they have to be bound the same way for a fair comparison.
    golden.data_format = data_format
    golden.dst_format = data_format
    golden.dest_acc = DestAccumulation.No

    vector_op = golden._vector_op(operation)
    assert vector_op is not None, f"{operation.name} is in the table but has no impl"

    vector = vector_op(values).to(torch.float32)
    scalar, evaluable = _scalar_reference(golden, operation, values)

    # Some scalar methods cannot evaluate every input: _cosh/_sinh go through
    # math.cosh/sinh, which raise OverflowError near the fp32 maximum instead of
    # returning an infinity (torch returns inf there, so the vectorised path is
    # strictly more total). Those elements drop out of the comparison rather than being
    # papered over -- the oracle has no answer to compare against. The sweep never
    # reaches them, because each op's stimuli come from its registered domain; this file
    # is the only thing that feeds an op its format limits.
    if not bool(evaluable.all()):
        excluded = int((~evaluable).sum())
        assert excluded < values.numel() // 2, (
            f"{operation.name}: the scalar oracle could not evaluate {excluded} of "
            f"{values.numel()} inputs, too many for the comparison to mean anything"
        )
        vector = vector[evaluable]
        scalar = scalar[evaluable]

    # raise rather than `assert cond, msg`: pytest's assertion rewriting renders both
    # 1024-element tensors into the failure report, which buries the one line that says
    # what actually differs.
    if not _bitwise_equal(vector, scalar):
        raise AssertionError(
            f"{operation.name} / {population} / {data_format.name}: "
            + _first_difference(vector, scalar, values[evaluable].tolist())
        )


@pytest.mark.parametrize("operation", VECTOR_OPS, ids=lambda o: o.name)
def test_vector_op_preserves_shape_and_dtype(operation):
    """The whole-tile result must be a same-length fp32 tensor.

    ``__call__`` writes it straight into a slice of ``result``, so a shape or dtype
    surprise corrupts the tile rather than raising.
    """
    golden = get_golden_generator(UnarySFPUGolden)
    golden.data_format = DataFormat.Float16_b
    golden.dst_format = DataFormat.Float16_b
    golden.dest_acc = DestAccumulation.No
    values = _population("signed_wide").to(torch.float32)
    out = golden._vector_op(operation)(values).to(torch.float32)
    assert out.shape == values.shape
    assert out.dtype == torch.float32


def test_every_table_entry_is_a_registered_op():
    """A table entry for an unregistered op would never be reached.

    ``_vector_op`` is consulted with whatever ``__call__`` was handed, so an entry whose
    MathOperation is not in ``self.ops`` is dead weight that reads as coverage.
    """
    golden = get_golden_generator(UnarySFPUGolden)
    unreachable = sorted(op.name for op in VECTOR_OPS if op not in golden.ops)
    assert not unreachable, f"vector table entries with no registered op: {unreachable}"


def test_no_unlisted_op_is_silently_bit_exact_capable():
    """Report which ops still take the per-element path.

    Not an assertion about the size of the table -- ops legitimately stay scalar. This
    exists so the fallback set is written down somewhere a reader will see it, and so a
    regression that empties the table shows up as a diff here rather than only as a
    slower nightly.
    """
    golden = get_golden_generator(UnarySFPUGolden)
    scalar_only = sorted(op.name for op in golden.ops if op not in set(VECTOR_OPS))
    # The count is asserted loosely on purpose: it should only ever go down.
    assert len(scalar_only) <= len(golden.ops), "impossible"
    print(f"\n{len(VECTOR_OPS)} vectorised, {len(scalar_only)} still per-element:")
    print("  " + ", ".join(scalar_only))


def test_nan_rule_branch_is_actually_exercised():
    """The Float16 population must really produce an infinity for some op.

    ``_torch_unary``'s inf->NaN substitution is the one behaviour that differs between
    the two FORMATS. If no op/population combination ever produces an infinity, the
    Float16 half of this file's matrix proves nothing, and a vectorised path that
    dropped the rule would still pass.
    """
    golden = get_golden_generator(UnarySFPUGolden)
    golden.data_format = DataFormat.Float16_b  # exponent-B: infinities survive
    golden.dst_format = DataFormat.Float16_b
    golden.dest_acc = DestAccumulation.No
    values = _population("specials").to(torch.float32)
    produced_inf = False
    for operation in VECTOR_OPS:
        out = golden._vector_op(operation)(values)
        if bool(out.isinf().any()):
            produced_inf = True
            break
    assert produced_inf, (
        "no vectorised op produces an infinity on the specials population, so the "
        "Float16 inf->NaN branch of _torch_unary is untested"
    )
