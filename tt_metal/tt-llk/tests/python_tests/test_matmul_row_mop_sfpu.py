# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden of a matmul form that writes address mods beyond main's set followed by an SFPU op in one kernel: the form must leave the SFPU invariants (ADDR_MOD_7 = {0,0,0},
ADDR_MOD_6 = dest += 2, set by hw_configure) as it found them. The body is test_matmul_grid2_tiny (math matmul test) with the form forced and a negation of every tile of
the block after the matmul, loading through ADDR_MOD_7 and stepping DEST through ADDR_MOD_6, at both syncs and every fidelity; the golden is the negated matmul golden.
"""

import pytest
import test_matmul_grid2_tiny as _t
from conftest import skip_for_wormhole
from helpers.llk_params import MathFidelity
from helpers.test_variant_parameters import TemplateParameter

pytestmark = pytest.mark.llk_sweep


class _Raw(TemplateParameter):
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return "RAW(%s)" % self.text

    def convert_to_cpp(self):
        return self.text


# the forms that write address mods beyond main's set (0, 1, 2, 4, 5): the opt-in row MOP (3, 5, 6 in the call), the hoisted rows forms (3, 5), the row end fold (3, 6 in the call), the default form of the cell
_FORMS = {
    "rowmop": "#define MATMUL_ROW_MOP",
    "rowsR": "#define MATMUL_FORCE_FORM_CHAR 'R'",
    "rowsr": "#define MATMUL_FORCE_FORM_CHAR 'r'",
    "tileh": "#define MATMUL_FORCE_FORM_CHAR 'h'",
    "rowfold": "#define MATMUL_ROW_END_FOLD",
    "default": "",
}


def _cases():
    return [
        (
            form,
            (
                "F16b",
                "F16b",
                "F16b",
                "N",
                sy,
                getattr(MathFidelity, fi),
                32,
                32,
                rt,
                ct,
                2,
            ),
        )
        for form in _FORMS
        for sy in ("H", "F")
        for fi in ("LoFi", "HiFi2", "HiFi4")
        for rt, ct in ((1, 4), (2, 2), (4, 1))
    ]


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    _cases(),
    ids=lambda c: "sfpu_after_%s_%s_%s_%dx%dx%d"
    % (c[0], c[1][4], c[1][5].name, c[1][8], c[1][9], c[1][10]),
)
def test_matmul_row_mop_sfpu(case, monkeypatch):
    form, case = case
    real_config = _t.TestConfig
    real_golden = _t.get_golden_generator

    def config(*args, templates=None, **kwargs):
        return real_config(
            *args,
            templates=list(templates)
            + [_Raw(_FORMS[form] + "\n#define MATMUL_SFPU_AFTER")],
            **kwargs,
        )

    def golden(cls):
        gen = real_golden(cls)
        if cls is not _t.MatmulGolden:
            return gen
        return lambda *a, **k: -gen(*a, **k)

    monkeypatch.setattr(_t, "TestConfig", config)
    monkeypatch.setattr(_t, "get_golden_generator", golden)
    _t.test_matmul_grid2_tiny(case)
