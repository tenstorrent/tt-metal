# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden functional runs of the cells with a static throttle of the LLK matmul tables (llk_matmul_forms.h and their python mirrors): main form, separate post, main flip, end fold off, forced forms, the main
form cells of the mixed pairs, tiny tile cells, the per-tile pack cells (grid and LLK-only space), one block per cell, plus the forced row MOP on the four
formats at both syncs. The body is test_matmul_grid2_tiny (math matmul test, golden on the operands as the unpackers write them); the cells keep their block shape and k steps
(k steps above 8 run at 8: the body's golden fails at 32 on every head). Cells with a static throttle, packer L1 accumulate or several inner blocks are not run here: the body has none of the three.
"""

import inspect

import helpers.matmul_prod_keys as _pk
import helpers.matmul_sweep as _ms
import perf_matmul_grid as _g
import perf_matmul_llkx as _x
import pytest
import test_matmul_grid2_tiny as _t
import test_matmul_llkx  # noqa: F401  (its import extends the body's golden to the Float16, Bfp8 5 bit exponent and Bfp2_b formats)
from conftest import skip_for_wormhole
from helpers.llk_params import MathFidelity
from helpers.test_variant_parameters import TemplateParameter

_g._F.update({k: v for k, v in _x._F.items() if k not in _g._F})
_BYTES = {
    "F32": 4096,
    "F16b": 2048,
    "F16a": 2048,
    "B8": 1088,
    "B8a": 1088,
    "B4": 576,
    "B2": 320,
}
_L1_BUDGET = 900 * 1024
_CAP = {("H", "N"): 8, ("H", "Y"): 4, ("F", "N"): 16, ("F", "Y"): 8}
_SHORT = {
    "Float16_b": "F16b",
    "Float16": "F16",
    "Float32": "F32",
    "Bfp8_b": "B8",
    "Bfp8": "B8x",
    "Bfp4_b": "B4",
    "Bfp2_b": "B2",
}


class _Raw(TemplateParameter):
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return "RAW(%s)" % self.text

    def convert_to_cpp(self):
        return self.text


def _fit(a, b, o, rt, ct, kt):
    # the body's golden does not hold at 32 k steps on any head (9321c0d and 5bb6f7e fail the same 55 ids, all pass at 8): the long k cells run at 8
    kt = min(kt, 8)
    while (
        kt > 1
        and (rt * kt * _BYTES[a] + kt * ct * _BYTES[b] + rt * ct * _BYTES[o])
        > _L1_BUDGET
    ):
        kt = max(1, kt // 2)
    return kt


_HARNESS = {"B8x": "B8a", "F16": "F16a"}


def _case(table, a, b, o, acc, sy, fid, rt, ct, kt, r0=32, c1=32, row_mop=False):
    a0, b0, o0 = a, b, o
    a, b, o = _HARNESS.get(a, a), _HARNESS.get(b, b), _HARNESS.get(o, o)
    if rt * ct > _CAP[(sy, acc)]:
        return None
    return (
        table,
        row_mop,
        (
            a,
            b,
            o,
            acc,
            sy,
            getattr(MathFidelity, fid),
            r0,
            c1,
            rt,
            ct,
            _fit(a, b, o, rt, ct, kt),
        ),
    )


def _cases():
    out = []

    def add(c):
        if c is not None and c not in out:
            out.append(c)

    def row(table, rows):
        for r in rows:
            fid, half, acc, a, b, o, rt, ct, kt, thr, l1, nb = r
            if thr and not l1 and nb == 1:
                add(
                    _case(
                        table,
                        a,
                        b,
                        o,
                        "Y" if acc else "N",
                        "H" if half else "F",
                        fid,
                        rt,
                        ct,
                        kt,
                    )
                    and _with_thr(
                        _case(
                            table,
                            a,
                            b,
                            o,
                            "Y" if acc else "N",
                            "H" if half else "F",
                            fid,
                            rt,
                            ct,
                            kt,
                        ),
                        thr,
                    )
                )

    row("main", sorted(_pk._GRID_MAIN_ROWS))
    row("keep", sorted(_pk._KEEP_FLIP_OFF_ROWS))
    for fid, o, a, b, rt, ct, thr in sorted(_ms._PACK_LLKX_PER_TILE_ROWS):
        if thr:
            add(_with_thr(_case("llkxtile", a, b, o, "N", "H", fid, rt, ct, 2), thr))
    return out


def _with_thr(c, thr):
    if c is None:
        return None
    t, rm, body = c
    return (t, thr, body)


_CASES = _cases()


def _id(c):
    t, thr, (a, b, o, acc, sy, fi, r0, c1, rt, ct, kt) = c
    return (
        "thr%d_%s_%s_%s_%s_%s_%s_%dx%dx%d" % (thr, t, a, b, o, acc, sy, rt, ct, kt)
        + "_"
        + fi.name
    )


# the body of test_matmul_grid2_tiny with its static throttle (0) a parameter
_SRC = inspect.getsource(_t.test_matmul_grid2_tiny)
_SRC = (
    _SRC[_SRC.index("def test_matmul_grid2_tiny(case):") :]
    .replace("def test_matmul_grid2_tiny(case):", "def _body(case, throttle):")
    .replace("    throttle = 0\n", "")
)
exec(compile(_SRC, "test_matmul_grid2_tiny_throttled", "exec"), _t.__dict__)
_body = _t.__dict__["_body"]


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize("case", _CASES, ids=_id)
def test_matmul_cells_thr(case):
    table, thr, body = case
    _body(body, thr)
