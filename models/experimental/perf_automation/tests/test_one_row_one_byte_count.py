# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A DECODE row that quoted two different byte counts, three lines apart.

    memory ← binds │ 4.59 ms    │ ... │ 13.27 ms      floor  -> 4.59ms x 512GB/s = 2.350 GB
                   │ 512.0 GB/s │ ... │ 360.5 GB/s    bw     -> 360.5 x 13.27ms  = 4.784 GB
                   │ 217.9 tok/s/u    │ 75.3          rate   -> 1000/4.59, agrees with the floor

2.04x apart, in one row, and only the recurring stage was affected -- encode and prefill were
self-consistent. The cause was a special case that had quietly outlived its reason:

    _mb = bw_gbps if (rf["tokens"] == 1 and bw_gbps) else rf["bytes"] / ms

with the comment "the caller already computed this one from the same bytes; recomputing it here
differs in the last digit". TRUE while a stage's read set WAS the model-level one. The moment stages
got their own measured bytes it stopped being a rounding nicety and became a second source -- and
nothing failed, because the premise lived in a comment rather than in a test.

THE SHAPE OF THE WHOLE CLASS. Eleven commits have now been about this one quantity -- the census
total, the anchor's key, the checkpoint inference, the pinning, the per-stage split -- and each fixed
the consumer that happened to look wrong. None began by asking who else read it. So each fix left the
others on a different source, and the next symptom appeared somewhere new.

This file exists so the twelfth attempt fails here instead of shipping.
"""
import ast
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from cc_optimize.summary import _measured_bw_gbps  # noqa: E402


def test_the_row_divides_by_its_own_bytes():
    """THE INVARIANT. Whatever bytes a row's floor is built from, its bandwidth uses the same ones."""
    for gb, ms in ((2.350, 13.27), (0.512, 1.0), (11.9, 43.0)):
        rf = {"bytes": int(gb * 1e9), "tokens": 1}
        got = _measured_bw_gbps(rf, ms)
        assert abs(got - (gb / (ms / 1000.0))) < 1e-6, (gb, ms, got)


def test_the_recurring_stage_is_not_special():
    """tokens == 1 was the branch that substituted a model-level number. A stage's identity must not
    change which bytes its own row divides by."""
    a = _measured_bw_gbps({"bytes": int(2.35e9), "tokens": 1}, 13.27)
    b = _measured_bw_gbps({"bytes": int(2.35e9), "tokens": 128}, 13.27)
    assert a == b, "the recurring stage still takes a different path"


def test_the_decode_row_from_the_live_report_is_now_coherent():
    """Run 20's actual numbers: a 4.59 ms floor against 13.27 ms measured is 34.6% of the ceiling, so
    the achieved bandwidth must be 34.6% of peak -- not the 70% that the model-level figure printed."""
    rf = {"bytes": int(round(4.59e-3 * 512e9)), "tokens": 1}
    bw = _measured_bw_gbps(rf, 13.27)
    assert abs(bw - 177.1) < 1.0, bw
    assert abs((100 * 4.59 / 13.27) - (100 * bw / 512.0)) < 0.5, "floor %% and bandwidth %% disagree"


def test_zero_and_missing_inputs_refuse_rather_than_divide():
    for rf, ms in (({}, 10.0), ({"bytes": 0}, 10.0), ({"bytes": 100}, 0), ({"bytes": 100}, None), (None, 10.0)):
        assert _measured_bw_gbps(rf, ms) is None, (rf, ms)


def test_it_is_the_only_place_measured_bandwidth_is_computed():
    """The expression stood at two call sites, identical, each with its own copy of the special case.
    Two copies is how one of them keeps a premise the other has dropped."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "_measured_bw_gbps":
            continue
        body = ast.unparse(node)
        # dividing a byte count by a millisecond figure and scaling to GB/s, anywhere else
        if "/ 1000.0) / 1e+09" in body.replace(" ", "") or '["bytes"] / (' in body:
            offenders.append(node.name)
    assert not offenders, "measured bandwidth is computed outside its owner: %s" % offenders


def test_no_renderer_reaches_for_the_model_level_bytes_to_price_a_stage():
    """`active_bytes` is the MODEL's read set. A per-stage row using it is the bug this file is about.

    The DRAM-capacity panel legitimately uses it -- that question really is model-level ("how much of
    the device does this model occupy"), not "what does one unit read"."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_stage_roofs":
            continue
        body = ast.unparse(node)
        # the model-level figure may seed the recurring stage's FALLBACK, never a rendered rate
        assert "active_bytes / (" not in body.replace(" ", ""), "a stage rate divided by model-level bytes"


def test_the_owner_is_registered():
    """So a new consumer is a test failure rather than a discovery six commits later."""
    src = (_PA / "tests" / "test_single_source_of_truth.py").read_text()
    assert '"the per-stage read set"' in src
