# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden check of candidate forms (the probe arms of the speed sweeps: forced form, ROW_MOP, STREAM_Z, TTSYNC, unfused post, keep flip, pack form) on the cells that read slower than
main, before any of them is measured for speed: the same body as test_matmul_cells_thr with the arm applied to the templates.
"""


import helpers.test_variant_parameters as tvp
import pytest
import test_matmul_cells_thr as _c
import test_matmul_grid2_tiny as _t
from conftest import skip_for_wormhole
from helpers.test_variant_parameters import TemplateParameter

_SPEC = [""]


class _RawDef(TemplateParameter):
    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return "RAW(" + self.d + ")"

    def convert_to_cpp(self):
        return "#define " + self.d.replace("=", " ")

    def __eq__(self, o):
        return isinstance(o, _RawDef) and o.d == self.d

    def __hash__(self):
        return hash(self.d)


def _mm_arm(templates):
    """probe arms: MM_ARM = comma list of -NAME (drop MATMUL_NAME), +NAME (add MATMUL_NAME), PACK=n, FK:unfused|keep|nokeep|nounfused, RAW:DEFINE"""
    import os

    spec = _SPEC[0]
    if not spec:
        return templates
    out = list(templates)
    for tok in [t for t in spec.split(",") if t]:
        if tok[0] == "-":
            cls = getattr(tvp, "MATMUL_" + tok[1:])
            out = [t for t in out if not isinstance(t, cls)]
        elif tok[0] == "+":
            cls = getattr(tvp, "MATMUL_" + tok[1:])
            out = [t for t in out if not isinstance(t, cls)] + [cls()]
        elif tok.startswith("PACK="):
            out = [
                (
                    tvp.MATMUL_PACK_BLOCK(int(tok[5:]))
                    if isinstance(t, tvp.MATMUL_PACK_BLOCK)
                    else t
                )
                for t in out
            ]
        elif tok.startswith("FK:"):
            for t in out:
                if isinstance(t, tvp.MATMUL_FORM_KEYS):
                    v = tok[3:]
                    if v == "unfused":
                        t.unfused_post = True
                    elif v == "nounfused":
                        t.unfused_post = False
                    elif v == "keep":
                        t.keep_flip = True
                    elif v == "nokeep":
                        t.keep_flip = False
                    elif v.startswith("force="):
                        t.force_form = v[6:]
        elif tok.startswith("RD="):
            out = [t for t in out if not isinstance(t, tvp.MATMUL_REUSE_DIR)] + [
                tvp.MATMUL_REUSE_DIR(int(tok[3:]))
            ]
        elif tok.startswith("RAW:"):
            out.append(_RawDef(tok[4:]))
    if os.environ.get("MM_DUMP"):
        print("MM_DUMP", [repr(t) for t in out], flush=True)
    return out


class _RawDef(
    __import__("helpers.test_variant_parameters", fromlist=["x"]).TemplateParameter
):
    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return "RAW(" + self.d + ")"

    def convert_to_cpp(self):
        return "#define " + self.d.replace("=", " ")

    def __eq__(self, o):
        return isinstance(o, _RawDef) and o.d == self.d

    def __hash__(self):
        return hash(self.d)


# (name, case with its throttle) of the slower configs and the arms every one gets
_CELLS = {
    # the 32-bit DEST accumulate of the cell fails the body's golden for a Bfp8 5 bit exponent output on main's LLK as well (acc_case.md): its 16-bit DEST sibling stands in for the forms
    "a_f16_b8_b8_1x4x3_acc_no": (
        0,
        ("F16a", "B8a", "B8a", "N", "H", _c.MathFidelity.LoFi, 32, 32, 1, 4, 3),
    ),
    "b_f16b_b4_b8_thr2_2x4x8": (
        2,
        ("F16b", "B4", "B8", "N", "H", _c.MathFidelity.LoFi, 32, 32, 2, 4, 8),
    ),
}
_ARMS = [
    "-",
    "FK:force=R",
    "FK:force=r",
    "FK:force=h",
    "FK:force=d",
    "+ROW_MOP",
    "+STREAM_Z",
    "RAW:MATMUL_UNPACK_TTSYNC",
    "FK:unfused",
    "FK:nokeep",
    "FK:keep",
    "PACK=0",
    "PACK=2",
    "FK:force=R,RAW:MATMUL_UNPACK_TTSYNC",
    "FK:force=R,FK:nokeep",
    "FK:force=R,FK:unfused",
    "FK:force=R,+STREAM_Z",
    "FK:force=h,FK:nokeep",
    "FK:force=h,RAW:MATMUL_UNPACK_TTSYNC",
    "FK:force=h,FK:unfused",
    "FK:force=R,PACK=0",
    "FK:force=h,PACK=0",
]
_CASES = [(n, arm) for n in _CELLS for arm in _ARMS]


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda c: "arm_%s_%s"
    % (
        c[0],
        c[1].replace("=", "").replace(",", "+").replace(":", "").replace("-", "base"),
    ),
)
def test_arm_golden(case, monkeypatch):
    name, arm = case
    thr, body = _CELLS[name]
    real_config = _t.TestConfig
    _SPEC[0] = "" if arm == "-" else arm

    def config(*args, templates=None, **kwargs):
        return real_config(*args, templates=_mm_arm(list(templates)), **kwargs)

    monkeypatch.setattr(_t, "TestConfig", config)
    _c._body(body, thr)
