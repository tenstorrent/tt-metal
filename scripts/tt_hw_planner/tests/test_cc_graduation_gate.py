# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The cc bring-up engine must apply the SAME graduation criterion as the fsm loop, UNCONDITIONALLY
and with NO env flags: a component graduates iff its PCC test PASSED and the stub is a native ttnn
forward (NOT a torch-delegating wrapper). A trivial PCC pass by delegating to the torch reference
(output == golden) is not evidence of a native port (the seamless-m4t / XTTS permissive-run bug), so
it never graduates."""
import importlib

import pytest

TORCH_WRAPPER = "class C:\n    def __call__(self, *a, **k):\n        return self._torch_module(*a, **k)\n"

NATIVE = "import ttnn\nclass C:\n    def forward(self, x):\n        return ttnn.matmul(x, x)\n"


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


def _write_stub(tmp, comp, body):
    p = tmp / "_stubs" / f"{comp}.py"
    p.write_text(body)
    return p


def test_block_reason_torch_wrapper_is_unconditional(bmcp):
    m, tmp = bmcp
    stub = _write_stub(tmp, "block", TORCH_WRAPPER)
    assert m._graduation_block_reason(stub) is not None


def test_block_reason_native_allowed(bmcp):
    m, tmp = bmcp
    stub = _write_stub(tmp, "block", NATIVE)
    assert m._graduation_block_reason(stub) is None


def test_record_result_refuses_torch_wrapper_without_flag(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "block", TORCH_WRAPPER)
    r = m.record_result("block", ok=True, pcc=1.0)
    assert r["graduated"] is False
    assert r.get("reason")
    assert not (tmp / "_stubs" / "block.py.last_good_native").is_file()


def test_record_result_graduates_native(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "block", NATIVE)
    r = m.record_result("block", ok=True, pcc=0.999)
    assert r["graduated"] is True
    assert (tmp / "_stubs" / "block.py.last_good_native").is_file()


def test_no_env_flag_changes_the_criterion(bmcp, monkeypatch):
    # The graduation criterion is flag-free: setting the (now-removed) enforcement env vars must not
    # change the verdict for either a wrapper or a native stub.
    m, tmp = bmcp
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "1")
    monkeypatch.setenv("E2E_ALLOW_HOST_DECODE", "1")
    assert m._graduation_block_reason(_write_stub(tmp, "wrap", TORCH_WRAPPER)) is not None
    assert m._graduation_block_reason(_write_stub(tmp, "nat", NATIVE)) is None


OP_SYNTH_STUB = (
    "class C:\n"
    "    def __call__(self, *args, **kwargs):\n"
    "        args = tuple(_coerce_to_torch(a) for a in args)\n"
    "        kwargs = {k: _coerce_to_torch(v) for k, v in kwargs.items()}\n"
    "        return self._op(*args, **kwargs)\n"
)

SELF_TORCH_MODULE_WRAPPER = (
    "class C:\n" "    def __call__(self, *a, **k):\n" "        return self._torch_module(*a, **k)\n"
)

TORCH_NO_GRAD_WRAPPER = (
    "import torch\n"
    "class C:\n"
    "    def forward(self, x):\n"
    "        with torch.no_grad():\n"
    "            return self._ref(x)\n"
)


def _place_snapshot(tmp, comp: str) -> None:
    (tmp / "_stubs" / f"{comp}.py.last_good_native").write_text("")


def test_is_graduated_rejects_stale_snapshot_over_torch_wrapper(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", TORCH_WRAPPER)
    _place_snapshot(tmp, "comp")
    assert m._is_graduated("comp") is False


def test_is_graduated_accepts_snapshot_over_native_stub(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", NATIVE)
    _place_snapshot(tmp, "comp")
    assert m._is_graduated("comp") is True


def test_is_graduated_false_without_snapshot(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", NATIVE)
    assert m._is_graduated("comp") is False


def test_is_graduated_rejects_op_synth_coerce_to_torch(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", OP_SYNTH_STUB)
    _place_snapshot(tmp, "comp")
    assert m._is_graduated("comp") is False


def test_is_graduated_rejects_self_torch_module_wrapper(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", SELF_TORCH_MODULE_WRAPPER)
    _place_snapshot(tmp, "comp")
    assert m._is_graduated("comp") is False


def test_is_graduated_rejects_torch_no_grad_wrapper(bmcp):
    m, tmp = bmcp
    _write_stub(tmp, "comp", TORCH_NO_GRAD_WRAPPER)
    _place_snapshot(tmp, "comp")
    assert m._is_graduated("comp") is False
