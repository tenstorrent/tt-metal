# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-call A3 norm-mode selection of ``TtMHCFast`` (host only, no device).

Measured on the 8x4 (2026-10-26, 45 layers, 4 users/row, 128-token prompt, one run each): ``GLM53_MHC_NORM=sharded``
decode 79.75 ms / TTFT 297 ms, ``single`` 83.06 / 266: the sharded norm is worth -3.3 ms per decode step but a
128-token prompt on 8 rows is 16-row slabs = one tile, which ``TtMHCSplit.select`` routes to the fast path, so its 4
norm ops (typecast, reshard, sharded rms_norm, reshard) ran inside the traced prefill (+31 ms of TTFT). The fix:
``TtMHCFast.forward(norm_mode=..., prefill=...)`` -- an explicit per-call mode, else ``single`` inside a prefill scope
(``GLMMesh.prefill_scope`` / ``mesh.in_prefill``, passed by ``decoder_layer._hc_call_impl``), else the configured
``GLM53_MHC_NORM`` (default ``sharded``). ``GLM53_MHC_PREFILL_NORM=same`` keeps the configured mode everywhere (A/B).

    pytest --noconftest -p no:cacheprovider models/demos/glm53_flash/tests/test_mhc_norm_select_host.py -q
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.tt import mhc as mhc_mod  # noqa: E402
from models.demos.glm53_flash.tt.mesh import GLMMesh  # noqa: E402
from models.demos.glm53_flash.tt.mhc import NORM_MODES, TtMHCFast, effective_norm_mode  # noqa: E402

DECODER_SRC = Path(mhc_mod.__file__).with_name("decoder_layer.py")


@pytest.mark.parametrize("configured", NORM_MODES)
def test_effective_norm_mode_table(configured, monkeypatch):
    monkeypatch.setattr(mhc_mod, "MHC_PREFILL_NORM", "single")
    # decode (no prefill scope): the configured mode
    assert effective_norm_mode(configured) == configured
    assert effective_norm_mode(configured, prefill=False) == configured
    # a prefill-scope call: the one-core norm, whatever the configured mode
    assert effective_norm_mode(configured, prefill=True) == "single"
    # an explicit per-call override wins over both
    for override in NORM_MODES:
        assert effective_norm_mode(configured, prefill=True, override=override) == override
        assert effective_norm_mode(configured, prefill=False, override=override) == override
    # the A/B escape: the configured mode inside prefill scopes too
    monkeypatch.setattr(mhc_mod, "MHC_PREFILL_NORM", "same")
    assert effective_norm_mode(configured, prefill=True) == configured
    assert effective_norm_mode(configured, prefill=True, override="single") == "single"


def _rejects(fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except AssertionError:
        return True
    return False


def test_effective_norm_mode_rejects_unknown():
    assert _rejects(effective_norm_mode, "sharded", override="multicore")
    assert _rejects(effective_norm_mode, "multicore")


def test_default_knobs():
    """The defaults the coordinator asked for: GLM53_MHC_NORM=sharded (decode), prefill-scope calls single."""
    assert mhc_mod.MHC_NORM == "sharded"
    assert mhc_mod.MHC_PREFILL_NORM == "single"
    assert effective_norm_mode(mhc_mod.MHC_NORM) == "sharded"
    assert effective_norm_mode(mhc_mod.MHC_NORM, prefill=True) == "single"


def test_prefill_scope_drives_the_selection(monkeypatch):
    """``mesh.in_prefill`` (what decoder_layer passes) is True exactly inside ``prefill_scope``, nested or not."""
    monkeypatch.setattr(mhc_mod, "MHC_PREFILL_NORM", "single")
    mesh = GLMMesh(device=None, rows=8, cols=4)
    assert effective_norm_mode("sharded", prefill=mesh.in_prefill) == "sharded"
    with mesh.prefill_scope():
        assert effective_norm_mode("sharded", prefill=mesh.in_prefill) == "single"
        with mesh.prefill_scope():  # a layer prefill inside a model prefill
            assert effective_norm_mode("sharded", prefill=mesh.in_prefill) == "single"
        assert effective_norm_mode("sharded", prefill=mesh.in_prefill) == "single"
    assert effective_norm_mode("sharded", prefill=mesh.in_prefill) == "sharded"


def _bare_fast(norm_mode: str) -> TtMHCFast:
    """A ``TtMHCFast`` without its device uploads: only the attributes ``forward`` reads on the way to ``_pre``."""
    hc = object.__new__(TtMHCFast)
    hc.norm_mode = norm_mode
    hc.post_kernel = 2
    hc.pre_block_on = False
    return hc


@pytest.mark.parametrize("configured", ["sharded", "sharded_cast"])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("override", [None, "single", "sharded"])
def test_forward_threads_the_mode_to_pre(configured, prefill, override, monkeypatch):
    """``forward`` hands ``_pre`` the effective mode (which ``_collapse`` -> ``_norm_decode`` consume)."""
    monkeypatch.setattr(mhc_mod, "MHC_PREFILL_NORM", "single")
    monkeypatch.setattr(mhc_mod.ttnn, "deallocate", lambda t, *a, **k: None)
    hc = _bare_fast(configured)
    seen = {}

    def fake_pre(x, norm=None, norm_mode=None):
        seen["mode"] = norm_mode
        seen["norm"] = norm
        return "h", "post", "comb", "x32", None, False

    def fake_post(y, x32, x_sm, post, comb, collapse=False):
        seen["post_args"] = (y, x32, x_sm, post, comb, collapse)
        return "out"

    monkeypatch.setattr(hc, "_pre", fake_pre)
    monkeypatch.setattr(hc, "_post", fake_post)
    norm = ("w", 1e-5)
    out = hc.forward("x", lambda t: t, sublayer_dtype=ttnn.bfloat16, norm=norm, norm_mode=override, prefill=prefill)
    assert out == "out"
    assert seen["norm"] == norm
    assert seen["mode"] == effective_norm_mode(configured, prefill=prefill, override=override)
    assert seen["post_args"] == ("h", "x32", None, "post", "comb", False)
    # the decode default on the model's path: prefill=False, no override -> the configured (sharded) mode
    if not prefill and override is None:
        assert seen["mode"] == configured
    if prefill and override is None:
        assert seen["mode"] == "single"


def test_norm_decode_honours_the_per_call_mode(monkeypatch):
    """``_norm_decode(h, norm, mode)``: ``single`` -> typecast + interleaved rms_norm (2 ops, no reshard);
    ``sharded`` -> typecast, interleaved_to_sharded, sharded rms_norm, sharded_to_interleaved; the per-call mode
    overrides the instance's."""
    calls = []
    fake = mhc_mod.ttnn

    def rec(name):
        def f(*a, **k):
            calls.append(name)
            return f"{name}_out"

        return f

    for name in ("typecast", "rms_norm", "interleaved_to_sharded", "sharded_to_interleaved"):
        monkeypatch.setattr(fake, name, rec(name))
    monkeypatch.setattr(fake, "deallocate", lambda t, *a, **k: None)
    hc = _bare_fast("sharded")
    hc.ckc = None
    hc.h_mem = None
    hc.pc_norm = None
    hc.C = 4096

    class _H:  # the shape checks of the sharded branch
        padded_shape = [1, 1, 32, 4096]

    calls.clear()
    assert hc._norm_decode(_H(), ("w", 1e-5), "single") == "rms_norm_out"
    assert calls == ["typecast", "rms_norm"]
    calls.clear()
    assert hc._norm_decode(_H(), ("w", 1e-5), "sharded") == "sharded_to_interleaved_out"
    assert calls == ["typecast", "interleaved_to_sharded", "rms_norm", "sharded_to_interleaved"]
    calls.clear()
    assert hc._norm_decode(_H(), ("w", 1e-5), "sharded_cast") == "sharded_to_interleaved_out"
    assert calls == ["interleaved_to_sharded", "rms_norm", "sharded_to_interleaved"]
    calls.clear()
    assert hc._norm_decode(_H(), ("w", 1e-5)) == "sharded_to_interleaved_out"  # None -> the instance's mode
    assert calls[0] == "typecast" and len(calls) == 4
    assert _rejects(hc._norm_decode, _H(), ("w", 1e-5), "multicore")


def test_forward_signature_and_decoder_touch_point():
    sig = inspect.signature(TtMHCFast.forward)
    assert sig.parameters["norm_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["prefill"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["norm_mode"].default is None and sig.parameters["prefill"].default is False
    src = DECODER_SRC.read_text()
    impl = src[src.index("def _hc_call_impl") :]
    impl = impl[: impl.index("def ", 10)]
    # the fuses_norm branch (TtMHCFast) passes the mesh's prefill flag; the composite wrap branch does not take it
    assert re.search(r"prefill=self\.mesh\.in_prefill", impl), impl
    assert impl.index("fuses_norm") < impl.index("prefill=self.mesh.in_prefill")
    assert "with self.mesh.prefill_scope():" in src  # decoder prefill_forward enters the scope
