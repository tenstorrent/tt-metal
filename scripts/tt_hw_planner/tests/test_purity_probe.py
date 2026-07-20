# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
def test_pure_call_returns_normally(monkeypatch):
    import types

    fake_ttnn = types.SimpleNamespace(to_torch=lambda *a, **k: None, from_torch=lambda *a, **k: None)
    monkeypatch.setitem(__import__("sys").modules, "ttnn", fake_ttnn)

    from scripts.tt_hw_planner._cli_helpers.purity_probe import assert_pure_on_device

    def pure_fn(x):
        return x * 2

    assert assert_pure_on_device(pure_fn, 21) == 42


def test_to_torch_call_raises_purity_violation(monkeypatch):
    import types

    fake_ttnn = types.SimpleNamespace(to_torch=lambda *a, **k: None, from_torch=lambda *a, **k: None)
    import sys

    sys.modules["ttnn"] = fake_ttnn

    from scripts.tt_hw_planner._cli_helpers.purity_probe import PurityViolation, assert_pure_on_device

    def dirty_fn(x):
        import ttnn

        ttnn.to_torch(x)
        return x

    err = None
    try:
        assert_pure_on_device(dirty_fn, 1)
    except PurityViolation as e:
        err = e
    assert err is not None, "expected PurityViolation to be raised"
    assert "ttnn.to_torch" in str(err)


def test_from_torch_also_flagged(monkeypatch):
    import sys
    import types

    fake_ttnn = types.SimpleNamespace(to_torch=lambda *a, **k: None, from_torch=lambda *a, **k: None)
    sys.modules["ttnn"] = fake_ttnn

    from scripts.tt_hw_planner._cli_helpers.purity_probe import PurityViolation, assert_pure_on_device

    def dirty_fn():
        import ttnn

        ttnn.from_torch(None)

    err = None
    try:
        assert_pure_on_device(dirty_fn)
    except PurityViolation as e:
        err = e
    assert err is not None, "expected PurityViolation to be raised"
    assert "from_torch" in str(err)


def test_bringup_gate_now_flags_free_form_torch_compute():
    """After the cli.py alignment, _stub_uses_torch_wrapper should return
    True for a stub that has torch.softmax hidden in a helper called from
    __call__ — even though it doesn't call self._torch_module."""
    from pathlib import Path

    from scripts.tt_hw_planner.cli import _stub_uses_torch_wrapper

    dirty_src = """
import torch
import ttnn

class M:
    def __call__(self, x_ttnn):
        return self._self_attn(x_ttnn)

    def _self_attn(self, x_ttnn):
        qkv = ttnn.linear(x_ttnn, self.w)
        qkv_t = ttnn.to_torch(qkv)
        return torch.softmax(qkv_t, dim=-1)
"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(dirty_src)
        p = Path(f.name)
    try:
        assert _stub_uses_torch_wrapper(p), (
            "bring-up gate must catch torch.softmax + ttnn.to_torch in a helper "
            "reached from __call__ — regression: it previously only caught "
            "day-0 template shims"
        )
    finally:
        p.unlink(missing_ok=True)


def test_bringup_gate_still_accepts_pure_ttnn_stub():
    from pathlib import Path

    from scripts.tt_hw_planner.cli import _stub_uses_torch_wrapper

    pure_src = """
import ttnn

class M:
    def __call__(self, x_ttnn):
        return self._layer(x_ttnn)

    def _layer(self, x_ttnn):
        y = ttnn.linear(x_ttnn, self.w)
        return ttnn.softmax(y, dim=-1)
"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(pure_src)
        p = Path(f.name)
    try:
        assert not _stub_uses_torch_wrapper(p), "pure ttnn stub should not be flagged as wrapper"
    finally:
        p.unlink(missing_ok=True)
