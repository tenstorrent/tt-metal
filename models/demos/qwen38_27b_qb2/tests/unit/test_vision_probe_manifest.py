# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.vision_reference_probe import main, verify_manifest


def test_manifest_rejects_changes_and_escaping_paths(tmp_path, expect_error):
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "probe.py"
    tracked.write_text("pass\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"probe.py": hashlib.sha256(tracked.read_bytes()).hexdigest()}}))
    verify_manifest(source, manifest)
    tracked.write_text("changed\n")
    with expect_error(ValueError, "mismatch"):
        verify_manifest(source, manifest)
    manifest.write_text(json.dumps({"files": {"../outside": "sha"}}))
    with expect_error(ValueError, "escapes"):
        verify_manifest(source, manifest)
    manifest.write_text(json.dumps({"files": {}}))
    with expect_error(ValueError, "must contain"):
        verify_manifest(source, manifest)


def test_verify_only_never_imports_native_or_opens_devices(tmp_path, monkeypatch, capsys):
    source = Path(__file__).resolve().parents[5]
    name = str(Path(__file__).resolve().relative_to(source))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}))
    monkeypatch.setitem(sys.modules, "ttnn", None)
    main(["--checkpoint", str(tmp_path), "--manifest", str(manifest), "--verify-only"])
    assert json.loads(capsys.readouterr().out) == {"source_files_verified": 1, "device_opened": False}
