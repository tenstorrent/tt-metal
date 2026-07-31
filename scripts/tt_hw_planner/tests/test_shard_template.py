# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Piece D: the additive sharded-test emitter. It must render valid Python that uses the mesh_device
fixture and reuses the single-device test's helpers by path — and must refuse to emit when the
single-device test isn't present (a component has to be single-device graduated first)."""
from scripts.tt_hw_planner.bringup_loop import emit_shard_test


def test_returns_none_without_sibling(tmp_path):
    assert emit_shard_test(tmp_path, "self_attn") is None


def _make_sibling(tmp_path, safe="self_attn"):
    d = tmp_path / "tests" / "pcc"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"test_{safe}.py").write_text("# single-device test placeholder\n")
    return d


def test_emits_valid_python_using_mesh_fixture(tmp_path):
    _make_sibling(tmp_path)
    p = emit_shard_test(tmp_path, "self_attn", tp_default=2)
    assert p is not None and p.name == "test_self_attn_sharded.py"
    src = p.read_text()
    compile(src, str(p), "exec")
    assert 'parametrize("mesh_device", [_MESH], indirect=True)' in src
    assert "_MESH = (_DP, _TP) if _DP > 1 else _TP" in src
    assert "def test_self_attn_sharded(mesh_device):" in src
    assert "test_self_attn.py" in src
    assert "comp_pcc" in src
    assert "_sd._build_ttnn_port(mesh_device, torch_module)" in src
    assert 'os.environ.get("TT_HW_PLANNER_SHARD_TP", "2")' in src
    assert 'os.environ.get("TT_HW_PLANNER_SHARD_DP", "1")' in src
    assert "FabricConfig.FABRIC_1D" in src


def test_idempotent_without_overwrite(tmp_path):
    _make_sibling(tmp_path)
    p1 = emit_shard_test(tmp_path, "self_attn")
    p1.write_text(p1.read_text() + "# edited\n")
    p2 = emit_shard_test(tmp_path, "self_attn")
    assert p2 == p1
    assert "# edited" in p2.read_text()


def test_overwrite_regenerates(tmp_path):
    _make_sibling(tmp_path)
    p = emit_shard_test(tmp_path, "self_attn")
    p.write_text("# stale\n")
    emit_shard_test(tmp_path, "self_attn", overwrite=True)
    assert "# stale" not in p.read_text()
    assert "test_self_attn_sharded" in p.read_text()
