# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A component may only graduate if it is BOUND to a module of the model, and only
on a real pass.

FLUX.2-klein's transformer was reported "6/6 on device (100%)" for six components
whose `submodule_path` was null. They were generic roles copied from another
model's template after discovery failed, so no module of FLUX corresponded to any
of them and their PCC tests had nothing from the model to compare against. The
native-stub rule did not catch it: the reused ttnn code was genuinely native, and
genuinely passed — against the wrong model.

A skip is likewise not a pass: it means the test could not evaluate the component.
"""
import importlib
import json

import pytest

NATIVE = "import ttnn\nclass C:\n    def forward(self, x):\n        return ttnn.matmul(x, x)\n"
TORCH_WRAPPER = "class C:\n    def __call__(self, *a, **k):\n        return self._torch_module(*a, **k)\n"


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


def _stub(tmp, comp, body=NATIVE):
    p = tmp / "_stubs" / f"{comp}.py"
    p.write_text(body)
    return p


def _status(tmp, components):
    (tmp / "bringup_status.json").write_text(json.dumps({"components": components}))


# ─── binding ─────────────────────────────────────────────────────


def test_unbound_component_cannot_graduate(bmcp):
    """submodule_path null -> no module of this model to compare against."""
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "ADAPT", "submodule_path": None}])
    reason = m._graduation_block_reason(_stub(tmp, comp), comp)
    assert reason is not None
    assert "not bound" in reason


def test_blank_submodule_path_counts_as_unbound(bmcp):
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "ADAPT", "submodule_path": "   "}])
    assert m._graduation_block_reason(_stub(tmp, comp), comp) is not None


def test_component_absent_from_status_cannot_graduate(bmcp):
    """A component the plan never declared has no binding either."""
    m, tmp = bmcp
    _status(tmp, [{"name": "other", "submodule_path": "blocks.0"}])
    assert m._graduation_block_reason(_stub(tmp, "ghost"), "ghost") is not None


def test_bound_native_component_graduates(bmcp):
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "NEW", "submodule_path": "blocks.0.attn"}])
    assert m._graduation_block_reason(_stub(tmp, comp), comp) is None


# ─── a skip is not a pass ────────────────────────────────────────


def test_skipped_run_cannot_graduate(bmcp):
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "NEW", "submodule_path": "blocks.0"}])
    (tmp / "state.json").write_text(json.dumps({"harness_skip_reason": {comp: "submodule not callable"}}))
    reason = m._graduation_block_reason(_stub(tmp, comp), comp)
    assert reason is not None
    assert "SKIPPED" in reason


def test_clearing_the_skip_restores_eligibility(bmcp):
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "NEW", "submodule_path": "blocks.0"}])
    (tmp / "state.json").write_text(json.dumps({"harness_skip_reason": {}}))
    assert m._graduation_block_reason(_stub(tmp, comp), comp) is None


# ─── the existing rule still applies, and back-compat ────────────


def test_torch_wrapper_still_blocks_even_when_bound(bmcp):
    m, tmp = bmcp
    comp = "some_component"
    _status(tmp, [{"name": comp, "status": "NEW", "submodule_path": "blocks.0"}])
    reason = m._graduation_block_reason(_stub(tmp, comp, TORCH_WRAPPER), comp)
    assert reason is not None
    assert "torch" in reason.lower()


def test_call_without_component_keeps_old_behaviour(bmcp):
    """Existing callers that pass only the stub must be unaffected."""
    m, tmp = bmcp
    assert m._graduation_block_reason(_stub(tmp, "c", NATIVE)) is None
    assert m._graduation_block_reason(_stub(tmp, "c", TORCH_WRAPPER)) is not None


def test_missing_status_file_does_not_crash(bmcp):
    m, tmp = bmcp
    assert m._graduation_block_reason(_stub(tmp, "c"), "c") is not None
