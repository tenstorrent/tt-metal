# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Containment tests for common/safe_paths.py.

Host-only (no device), so these run anywhere and guard the path-traversal fixes.
"""

import os
from pathlib import Path

import pytest

from models.experimental.vibevoice.common.safe_paths import safe_join, safe_output_path


@pytest.mark.timeout(60)
def test_safe_join_allows_paths_inside_base(tmp_path):
    assert safe_join(tmp_path, "model.safetensors") == tmp_path / "model.safetensors"
    assert safe_join(tmp_path, "sub", "shard-00001.safetensors") == tmp_path / "sub" / "shard-00001.safetensors"
    # A no-op join lands on the base itself, which is inside it.
    assert safe_join(tmp_path, ".") == tmp_path


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "sub/../../outside.txt",
        "../../etc/passwd",
        "/etc/passwd",  # absolute part would silently discard the base in os.path.join
    ],
)
def test_safe_join_rejects_escapes(tmp_path, escape, expect_error):
    with expect_error(ValueError, "outside base directory"):
        safe_join(tmp_path, escape)


@pytest.mark.timeout(60)
def test_safe_join_rejects_sibling_prefix(tmp_path, expect_error):
    """``/base`` must not accept ``/basefoo`` — the check is on a separator boundary."""
    base = tmp_path / "base"
    base.mkdir()
    with expect_error(ValueError, "outside base directory"):
        safe_join(base, "../basefoo/x.txt")


@pytest.mark.timeout(60)
def test_safe_join_does_not_resolve_symlinks(tmp_path):
    """HF snapshots symlink shards into a sibling blobs/ tree; that must stay legal."""
    snapshot = tmp_path / "snapshots" / "abc"
    blobs = tmp_path / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "deadbeef"
    blob.write_bytes(b"weights")
    link = snapshot / "model.safetensors"
    link.symlink_to(blob)

    joined = safe_join(snapshot, "model.safetensors")
    assert joined == snapshot / "model.safetensors"
    assert joined.read_bytes() == b"weights"


@pytest.mark.timeout(60)
def test_safe_output_path_absolutizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = safe_output_path("traj.csv")
    assert out.is_absolute()
    assert out == Path(os.path.abspath(tmp_path / "traj.csv"))


@pytest.mark.timeout(60)
def test_safe_output_path_enforces_suffix(tmp_path, expect_error):
    assert safe_output_path(tmp_path / "traj.csv", suffix=".csv").name == "traj.csv"
    with expect_error(ValueError, r"expected a \.csv file"):
        safe_output_path(tmp_path / "traj.txt", suffix=".csv")


@pytest.mark.timeout(60)
def test_load_weights_rejects_traversal_shard_name(tmp_path, expect_error):
    """A crafted ``weight_map`` in the checkpoint index must not escape the checkpoint dir."""
    import json

    from models.experimental.vibevoice.tt.load_weights import load_vibevoice_state_dict

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"some.weight": "../../../../etc/passwd"}})
    )
    with expect_error(ValueError, "outside base directory"):
        load_vibevoice_state_dict(str(tmp_path))
