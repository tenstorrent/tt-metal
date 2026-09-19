# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A generated perf test must take its mesh shape from resolve_mesh_shape, whichever way it gets a device.

--devices/--mesh are planned by the tool (commands/optimize.py:_derive_topology_env) and exported as
TT_PERF_MESH_ROWS/COLS; perf_adapter.resolve_mesh_shape is the only thing that reads them back. If a
generated test instead keeps the demo's MESH_DEVICE board table, the export lands nowhere and the flags
silently do nothing.

That is exactly what happened on llama3_1_8b_p150 (2026-07-26): the builder was told to reuse the demo's
fixtures+parametrize verbatim, the mesh rule only covered the SELF-OPEN case, so the copied table became
the only source of the shape and --devices single could not reach the device open. These tests pin the
rule as unconditional.
"""
from __future__ import annotations

from models.experimental.perf_automation.agent import perf_test_gen as G


def _skeletons():
    return [s for s in (G._SKELETON_REF, G._SELF_TRACED_SKELETON_REF, G._SKELETON_COMPONENT) if s]


def test_the_pipeline_skeleton_resolves_the_mesh_shape():
    """Structural, not prose: the builder copies the skeleton far more reliably than it applies a rule."""
    assert "resolve_mesh_shape" in G._SKELETON_REF
    assert "TT_PERF_MESH_ROWS" in G._SKELETON_REF


def test_the_self_traced_skeleton_resolves_the_mesh_shape():
    assert "resolve_mesh_shape" in G._SELF_TRACED_SKELETON_REF


def test_no_skeleton_hardcodes_a_mesh_shape():
    """A literal shape cannot honour --devices/--mesh."""
    for s in _skeletons():
        for literal in ("MeshShape(1, 1)", "MeshShape(1,1)", "MeshShape(8, 4)"):
            assert literal not in s, "skeleton hardcodes %s" % literal


def test_the_rule_is_not_conditioned_on_self_opening(tmp_path):
    """THE REGRESSION THIS FILE EXISTS FOR. The rule used to read 'when you self-open a mesh, ...', so a
    fixture+parametrize test -- the common tt-metal shape -- fell outside it."""
    prompt = G._perf_prompt_for(tmp_path) if hasattr(G, "_perf_prompt_for") else None
    src = prompt or "\n".join(_skeletons())
    assert "when you self-open a mesh, derive" not in src


def test_resolve_mesh_shape_honours_the_export_and_falls_back_otherwise(monkeypatch):
    """The behaviour the skeleton depends on: export wins, absence keeps the source's own shape."""
    from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

    monkeypatch.delenv("TT_PERF_MESH_ROWS", raising=False)
    monkeypatch.delenv("TT_PERF_MESH_COLS", raising=False)
    assert resolve_mesh_shape(default_rows=1, default_cols=2) == (1, 2)

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "1")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "1")
    assert resolve_mesh_shape(default_rows=1, default_cols=2) == (1, 1)

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "2")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "4")
    assert resolve_mesh_shape(default_rows=1, default_cols=1) == (2, 4)


def test_skeletons_gate_fabric_on_a_multichip_mesh():
    """Fabric must not be enabled for a 1x1 run.

    A demo targeting multi-chip shapes hardcodes fabric_config=True. Carried verbatim into a 1x1 perf
    run, device-open still trains ethernet across every VISIBLE chip -- on a p300c (4 chips) Device 3
    times out with "Fabric Router Sync: Timeout ... ethernet handshake failed" before the model is
    built. Measured 2026-07-26: that failure is why TT_MESH_GRAPH_DESC_PATH appeared to be required;
    with fabric gated, the same run passes with no descriptor and no MESH_DEVICE.
    """
    for s in (G._SKELETON_REF, G._SELF_TRACED_SKELETON_REF):
        assert "_MESH_SHAPE[0] * _MESH_SHAPE[1] > 1" in s, "skeleton does not gate fabric on mesh size"
        i_gate = s.index("_MESH_SHAPE[0] * _MESH_SHAPE[1] > 1")
        i_def = s.index("_MESH_SHAPE = resolve_mesh_shape")
        assert i_def < i_gate, "mesh shape must be resolved BEFORE device params use it"


def test_no_skeleton_enables_fabric_unconditionally():
    for s in _skeletons():
        assert '_DEV_PARAMS = {"fabric_config": True' not in s
        assert '"fabric_config": True, "num_command_queues"' not in s
