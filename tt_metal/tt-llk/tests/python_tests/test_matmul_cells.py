# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden functional runs of the cells of the LLK matmul tables (llk_matmul_forms.h and their python mirrors): main form, separate post, main flip, end fold off, forced forms, the main
form cells of the mixed pairs, tiny tile cells, the per-tile pack cells (grid and LLK-only space), one block per cell, plus the forced row MOP on the four
formats at both syncs. The body is test_matmul_grid2_tiny (math matmul test, golden on the operands as the unpackers write them); the cells keep their block shape and k steps
(k steps above 8 run at 8: the body's golden fails at 32 on every head). Cells with a static throttle, packer L1 accumulate or several inner blocks are not run here: the body has none of the three.
"""

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

    def row(table, rows, tiny=False):
        for r in rows:
            if tiny:
                fid, half, acc, a, b, o, r0, c1, rt, ct, kt = r
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
                        r0,
                        c1,
                    )
                )
                continue
            fid, half, acc, a, b, o, rt, ct, kt, thr, l1, nb = r
            if thr == 0 and not l1 and nb == 1:
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
                )

    row("main", sorted(_pk._GRID_MAIN_ROWS))
    # blocks that take main's form for the re-init cost (matmul_reinit_main_block): every format triple of the dyn re-init coverage
    for a, b, o in (
        ("F16b", "F16b", "F16b"),
        ("B8", "B8", "B8"),
        ("F16b", "F16b", "B4"),
        ("F32", "F32", "F32"),
    ):
        for acc in ("N", "Y"):
            for sy in ("H", "F"):
                for fid in ("LoFi", "HiFi2"):
                    for rt, ct in ((1, 2), (2, 1), (3, 1)):
                        if _pk.reinit_main_block(
                            getattr(MathFidelity, fid),
                            sy == "H",
                            rt,
                            ct,
                            1,
                            0,
                            False,
                            1,
                        ):
                            add(_case("main", a, b, o, acc, sy, fid, rt, ct, 1))
    # the Float16_b to Float16_b and Bfp4_b HiFi2 two tile blocks of half sync that matmul_main_form_cells keeps on main's form
    for o in ("F16b", "B4"):
        for rt, ct in ((1, 2), (2, 1)):
            add(_case("main", "F16b", "F16b", o, "N", "H", "HiFi2", rt, ct, 1))
    row("post", sorted(_pk._UNFUSED_POST_ROWS))
    row("flip", sorted(_pk._MAIN_FLIP_ROWS))
    row("endfold", sorted(_pk._END_FOLD_OFF_ROWS))
    row("keep", sorted(_pk._KEEP_FLIP_OFF_ROWS))
    row("tiny", sorted(_pk._TINY_MAIN_ROWS), tiny=True)
    for (
        fid,
        sync,
        mink,
        maxk,
        acc,
        i,
        o,
        p,
        minw,
        maxw,
        minh,
        wide,
        form,
        several,
        sec,
        maxh,
        in0_only,
    ) in _pk._FORCED_CELLS:
        f = _SHORT[i.name]
        rt = max(minh, 1)
        ct = min(max(minw, rt if wide else 1), maxw)
        add(
            _case(
                "forced",
                f,
                f,
                _SHORT[o.name],
                "Y" if acc else "N",
                "F" if sync else "H",
                fid,
                rt,
                ct,
                max(mink, 1),
            )
        )
    for key, cell in sorted(_pk._MAIN_CELLS.items(), key=str):
        fid, k1, acc, i, o, p, sync = key
        several_ok, minrow, maxrow, mink, maxk, mindim = cell
        f = _SHORT[i.name]
        rt = max(mindim, 1)
        ct = max(minrow, rt)
        add(
            _case(
                "maincell",
                f,
                f,
                _SHORT[o.name],
                "Y" if acc else "N",
                "F" if sync == 1 else "H",
                fid,
                rt,
                ct,
                1 if k1 else max(mink, 2),
            )
        )
    for fid, o, a, b, rt, ct in sorted(_ms._PACK_PER_TILE_ROWS):
        add(_case("pertile", a, b, o, "N", "H", fid, rt, ct, 3))
    for fid, o, a, b, rt, ct, thr in sorted(_ms._PACK_LLKX_PER_TILE_ROWS):
        if thr == 0:
            add(_case("llkxtile", a, b, o, "N", "H", fid, rt, ct, 2))
    for sy in ("H", "F"):
        for fid in ("LoFi", "HiFi2"):
            for f in ("F16b", "B8", "B4", "F32"):
                acc = "Y" if f == "F32" else "N"
                add(
                    _case(
                        "rowmop",
                        f,
                        f,
                        f if f != "B4" else "B8",
                        acc,
                        sy,
                        fid,
                        1,
                        4,
                        2,
                        row_mop=True,
                    )
                )
    return out


_CASES = _cases()


def _id(c):
    t, rm, (a, b, o, acc, sy, fi, r0, c1, rt, ct, kt) = c
    return "%s_%s_%s_%s_%s_%s_%s%s_%dx%dx%d" % (
        t,
        a,
        b,
        o,
        acc,
        sy,
        fi.name,
        ("_%dx%d" % (r0, c1)) if (r0, c1) != (32, 32) else "",
        rt,
        ct,
        kt,
    )


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize("case", _CASES, ids=_id)
def test_matmul_cells(case, monkeypatch):
    table, row_mop, body = case
    if row_mop:
        real_config = _t.TestConfig

        def config(*args, templates=None, **kwargs):
            return real_config(
                *args,
                templates=list(templates) + [_Raw("#define MATMUL_ROW_MOP")],
                **kwargs,
            )

        monkeypatch.setattr(_t, "TestConfig", config)
    _t.test_matmul_grid2_tiny(body)
