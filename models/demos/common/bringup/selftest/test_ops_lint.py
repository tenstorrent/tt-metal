# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F113: bring-up ops before stock TTNN (INDEX.md "Consider instead of", per-op README.md)."""

from pathlib import Path

from models.demos.common.bringup.testing import ops_lint as L

INDEX = """# Derived TTNN ops

- `ttnn.bringup.fabric_all_gather` (fabric_all_gather_ttnn): all-gather on the fabric.
  Consider instead of: ttnn.experimental.high_bw_all_gather, ttnn.all_gather. -> fabric_all_gather_ttnn/README.md
- `ttnn.bringup.fabric_reduce_scatter` (fabric_reduce_scatter_ttnn): reduce-scatter on the fabric.
  Consider instead of: ttnn.reduce_scatter; ttnn.all_gather + ttnn.add for exact sums. -> fabric_reduce_scatter_ttnn/README.md
- `ttnn.bringup.indexer_score_dsa` (indexer_score): scorer.
  Consider instead of: ttnn.experimental.indexer_score_dsa / ring_indexer_score_dsa. -> indexer_score/README.md
- `ttnn.bringup.mhc_post` (mhc_post_ttnn): fused mix.
  Consider instead of: per-stream ttnn.mul + ttnn.addcmul chains. -> mhc_post_ttnn/README.md
"""
README = "\n\n".join(L.SECTIONS) + "\n"


def test_parse_index_keeps_standalone_stock_ops_only():
    idx = L.parse_index(INDEX)
    assert idx["fabric_all_gather_ttnn"]["consider"] == ["ttnn.experimental.high_bw_all_gather", "ttnn.all_gather"]
    assert idx["fabric_reduce_scatter_ttnn"]["consider"] == ["ttnn.reduce_scatter"]  # not the composition
    assert idx["indexer_score"]["consider"] == [
        "ttnn.experimental.indexer_score_dsa",
        "ttnn.experimental.ring_indexer_score_dsa",
    ]
    assert idx["mhc_post_ttnn"] == {"op": "ttnn.bringup.mhc_post", "consider": []}


def _ops_root(tmp: Path, readmes: dict[str, str]) -> Path:
    root = tmp / "ttnn" / "ttnn" / "bringup"
    for f in ("fabric_all_gather_ttnn", "fabric_reduce_scatter_ttnn", "indexer_score", "mhc_post_ttnn", "new_fork"):
        (root / f).mkdir(parents=True)
        if f in readmes:
            (root / f / "README.md").write_text(readmes[f])
    (root / "__pycache__").mkdir()
    (root / "INDEX.md").write_text(INDEX)
    return root


def test_doc_problems(tmp_path):
    full = {f: README for f in ("fabric_all_gather_ttnn", "fabric_reduce_scatter_ttnn", "indexer_score")}
    root = _ops_root(tmp_path, {**full, "mhc_post_ttnn": "## What it does\n"})
    bad = L.doc_problems(root)
    assert any("mhc_post_ttnn/README.md: missing sections" in b for b in bad)
    assert any("new_fork: no README.md" in b for b in bad) and any("new_fork: no INDEX.md entry" in b for b in bad)
    assert not any("fabric_all_gather_ttnn" in b or "__pycache__" in b for b in bad)
    assert L.doc_problems(root, ["fabric_all_gather_ttnn"]) == []


def test_stock_call_needs_a_breadcrumbs_reason(tmp_path):
    _ops_root(tmp_path, {})
    tt = tmp_path / "models" / "demos" / "m" / "tt"
    tt.mkdir(parents=True)
    (tt / "ccl.py").write_text(
        "y = ttnn.all_gather(x, dim=3)\n"
        "z = ttnn.bringup.fabric_all_gather(x)\n"
        "w = ttnn.all_gather_async(x)\n"  # another op, not listed
        "# ttnn.reduce_scatter(x) in a comment\n"
        "s = ttnn.add(a, b)\n"  # only part of a composition
    )
    crumbs = tmp_path / "models" / "demos" / "m" / "bringup" / "BREADCRUMBS.md"
    crumbs.parent.mkdir(parents=True)
    files = ["models/demos/m/tt/ccl.py"]
    bad = L.stock_problems(tmp_path, "models/demos/m", files, crumbs, base=None)
    assert len(bad) == 1 and "ccl.py:1: calls ttnn.all_gather" in bad[0]
    assert "ttnn.bringup.fabric_all_gather" in bad[0]
    crumbs.write_text("- stock ttnn.all_gather instead of ttnn.bringup.fabric_all_gather: 1-device axis, no fabric\n")
    assert L.stock_problems(tmp_path, "models/demos/m", files, crumbs, base=None) == []


def test_fork_op_writes_every_readme_section():
    src = (Path(L.__file__).resolve().parents[5] / L.FORKS / "fork_op.py").read_text()
    assert all(s in src for s in L.SECTIONS) and "Consider instead of:" in src
