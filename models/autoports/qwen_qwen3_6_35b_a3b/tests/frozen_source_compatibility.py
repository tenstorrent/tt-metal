# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify exact historical sources or explicitly reviewed formatting migrations.

This never substitutes a historical digest for a live digest. Runtime acceptance
uses pinned byte hashes only, not an AST normalizer or a general formatting rule.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "doc/fused_decoder/endurance/source_format_compatibility/manifest.json"
MANIFEST_SHA256 = "fe5a3db52da5678a8d8041cd5078ff89f55838256bd9ec3c93fae220e87a085e"


def raw_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scoped_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source compatibility artifact escapes the model directory")
    return path


def _verify_source(path, historical_sha256, *, root, manifest_path, manifest_sha256):
    path, root = Path(path).resolve(), Path(root).resolve()
    actual = raw_sha256(path)
    if actual == historical_sha256:
        return dict(path=str(path), actual_sha256=actual, historical_sha256=historical_sha256, compatibility="exact")
    if not path.is_relative_to(root):
        raise ValueError("Changed external/native sources are never compatibility-accepted")
    manifest_bytes = Path(manifest_path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("Source compatibility manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown source compatibility schema")
    relative = str(path.relative_to(root))
    matches = [
        entry
        for entry in manifest["entries"]
        if entry["path"] == relative
        and entry["historical_sha256"] == historical_sha256
        and entry["current_sha256"] == actual
    ]
    if len(matches) != 1:
        raise ValueError(f"Unapproved source bytes: {relative} ({actual}); historical {historical_sha256}")
    entry = matches[0]
    archived = _scoped_path(root, entry["archive"])
    if raw_sha256(archived) != historical_sha256:
        raise ValueError("Historical source snapshot hash mismatch")
    proof_path = _scoped_path(root, entry["proof"])
    if raw_sha256(proof_path) != entry["proof_sha256"]:
        raise ValueError("Source compatibility proof hash mismatch")
    proof = json.loads(proof_path.read_text())
    if (
        proof.get("status") != "reviewed_inverse_delta_pass"
        or proof.get("path") != relative
        or proof.get("historical_sha256") != historical_sha256
        or proof.get("current_sha256") != actual
        or proof.get("arithmetic_ast_unchanged") is not True
        or proof.get("kernel_literals_unchanged") is not True
    ):
        raise ValueError("Source compatibility proof does not bind this exact pair")
    return dict(
        path=str(path),
        actual_sha256=actual,
        historical_sha256=historical_sha256,
        compatibility="reviewed_format_and_verification_migration",
        historical_source=str(archived),
        proof=str(proof_path),
        proof_sha256=entry["proof_sha256"],
        manifest_sha256=manifest_sha256,
    )


def verify_source(path, historical_sha256):
    """Accept only exact historical bytes or a pinned, reviewed old/new pair."""
    return _verify_source(
        path,
        historical_sha256,
        root=ROOT,
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA256,
    )
