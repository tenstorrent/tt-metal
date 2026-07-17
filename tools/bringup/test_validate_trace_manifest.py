#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/bringup/validate_trace_manifest.

Covers the fail-fast guarantees the manifest validator provides:
  §1  A valid manifest passes; --print-resolved reports paths
  §2  Missing / wrong-typed required keys and metadata mismatches fail
  §3  Unsupported kind and missing Conv2d params fail
  §4  Non-4D / non-positive shapes fail
  §5  Missing artifacts fail with the resolved path
  §6  Tensor shape consistency (injected loader + torch-gated real tensors)
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# The validator is a standalone script (not an installed package); ensure it is
# importable under pytest's importlib import mode regardless of rootdir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_trace_manifest import main, validate_manifest


# ── Fixtures / builders ────────────────────────────────────────────────────


CONV_PARAMS = {
    "in_channels": 3,
    "out_channels": 4,
    "kernel_size": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1],
    "dilation": [1, 1],
    "groups": 1,
}


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _valid_record(tmp_path, idx=0, name="conv", kind="Conv2d", with_weight=True):
    """Build a valid record and create its (placeholder) artifact files."""
    tensors = tmp_path / "tensors"
    in_rel = f"tensors/{idx:05d}_{name}_in.pt"
    out_rel = f"tensors/{idx:05d}_{name}_out.pt"
    _touch(tmp_path / in_rel)
    _touch(tmp_path / out_rel)
    rec = {
        "idx": idx,
        "name": name,
        "kind": kind,
        "params": dict(CONV_PARAMS) if kind == "Conv2d" else {},
        "in_shape": [1, 3, 8, 8],
        "out_shape": [1, 4, 8, 8],
        "in_path": in_rel,
        "out_path": out_rel,
        "w_path": None,
        "b_path": None,
    }
    if with_weight and kind == "Conv2d":
        w_rel = f"tensors/{idx:05d}_{name}_w.pt"
        _touch(tmp_path / w_rel)
        rec["w_path"] = w_rel
    return rec


def _write_manifest(tmp_path, records, input_shape=None, num_records=None, name="manifest.json"):
    if input_shape is None:
        input_shape = [1, 3, 8, 8]
    manifest = {
        "input_shape": input_shape,
        "num_records": len(records) if num_records is None else num_records,
        "records": records,
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ── §1 Valid manifest ───────────────────────────────────────────────────────


def test_valid_manifest_passes(tmp_path):
    path = _write_manifest(tmp_path, [_valid_record(tmp_path)])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 0


def test_print_resolved_reports_paths(tmp_path, capsys):
    records = [_valid_record(tmp_path, idx=0), _valid_record(tmp_path, idx=1, name="relu", kind="ReLU")]
    path = _write_manifest(tmp_path, records)
    main(["--manifest", str(path), "--no-shape-check", "--print-resolved", "2"])
    out = capsys.readouterr().out
    assert "Resolved artifact paths (first 2 records)" in out
    assert "[0] conv (Conv2d)" in out
    assert "[1] relu (ReLU)" in out
    assert "OK" in out


# ── §2 Required keys / types / metadata ──────────────────────────────────────


@pytest.mark.parametrize(
    "key", sorted(["idx", "name", "kind", "params", "in_shape", "out_shape", "in_path", "out_path"])
)
def test_missing_required_key_fails(tmp_path, capsys, key):
    rec = _valid_record(tmp_path)
    del rec[key]
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert key in capsys.readouterr().out


def test_wrong_type_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["in_shape"] = "1,3,8,8"  # should be a list
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "in_shape" in capsys.readouterr().out


def test_idx_mismatch_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path, idx=5)  # position will be 0
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "idx" in capsys.readouterr().out


def test_num_records_mismatch_fails(tmp_path, capsys):
    path = _write_manifest(tmp_path, [_valid_record(tmp_path)], num_records=99)
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "num_records" in capsys.readouterr().out


def test_bad_input_shape_fails(tmp_path, capsys):
    path = _write_manifest(tmp_path, [_valid_record(tmp_path)], input_shape=[1, 3, 8])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "input_shape" in capsys.readouterr().out


def test_manifest_not_found_fails(tmp_path, capsys):
    missing = tmp_path / "nope.json"
    assert main(["--manifest", str(missing), "--no-shape-check"]) == 1
    assert "file not found" in capsys.readouterr().out


def test_malformed_json_fails(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{ not valid json ")
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_manifest_argument_required(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "--manifest" in capsys.readouterr().err


# ── §3 kind / Conv2d params ───────────────────────────────────────────────────


def test_unsupported_kind_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path, kind="ReLU")
    rec["kind"] = "MysteryOp"
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "MysteryOp" in capsys.readouterr().out


def test_conv2d_missing_param_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    del rec["params"]["groups"]
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "groups" in capsys.readouterr().out


# ── §4 Shape well-formedness ──────────────────────────────────────────────────


def test_non_4d_shape_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["in_shape"] = [1, 3, 8]
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "in_shape" in capsys.readouterr().out


def test_non_positive_shape_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["out_shape"] = [1, 4, 0, 8]
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    assert "out_shape" in capsys.readouterr().out


# ── §5 Artifact existence ─────────────────────────────────────────────────────


def test_missing_artifact_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["w_path"] = "tensors/does_not_exist_w.pt"  # referenced but never created
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    out = capsys.readouterr().out
    assert "w_path" in out and "not found" in out


# ── §6 Shape consistency ──────────────────────────────────────────────────────


def test_shape_consistency_match_via_loader(tmp_path):
    rec = _valid_record(tmp_path, with_weight=False)
    path = _write_manifest(tmp_path, [rec])

    def loader(p):
        s = str(p)
        if s.endswith("_in.pt"):
            return (1, 3, 8, 8)
        if s.endswith("_out.pt"):
            return (1, 4, 8, 8)
        return (1,)

    report = validate_manifest(path, shape_loader=loader)
    assert report.ok
    assert report.shape_checks == 2


def test_shape_consistency_mismatch_via_loader(tmp_path):
    rec = _valid_record(tmp_path, with_weight=False)
    path = _write_manifest(tmp_path, [rec])

    def loader(p):
        return (1, 3, 16, 16)  # wrong for both in and out

    report = validate_manifest(path, shape_loader=loader)
    assert not report.ok
    assert any("does not match manifest" in e for e in report.errors)


def test_shape_consistency_real_torch(tmp_path):
    torch = pytest.importorskip("torch")
    tensors = tmp_path / "tensors"
    tensors.mkdir(parents=True)
    torch.save(torch.randn(1, 3, 8, 8), tensors / "00000_conv_in.pt")
    torch.save(torch.randn(1, 4, 8, 8), tensors / "00000_conv_out.pt")
    rec = {
        "idx": 0,
        "name": "conv",
        "kind": "ReLU",
        "params": {},
        "in_shape": [1, 3, 8, 8],
        "out_shape": [1, 4, 8, 8],
        "in_path": "tensors/00000_conv_in.pt",
        "out_path": "tensors/00000_conv_out.pt",
        "w_path": None,
        "b_path": None,
    }
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path)]) == 0

    # Corrupt the input tensor shape -> mismatch is detected.
    torch.save(torch.randn(1, 3, 16, 16), tensors / "00000_conv_in.pt")
    assert main(["--manifest", str(path)]) == 1


# ── §7 Conv2d shape/param consistency ─────────────────────────────────────────


def test_conv2d_consistent_record_passes(tmp_path):
    # in_shape=[1,3,8,8], out_shape=[1,4,8,8], k=3/s=1/p=1/d=1 -> output H/W == 8.
    path = _write_manifest(tmp_path, [_valid_record(tmp_path)])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 0


def test_conv2d_in_channels_mismatch_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["params"]["in_channels"] = 5  # in_shape channels is 3
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    out = capsys.readouterr().out
    assert "in_channels" in out and "in_shape channels" in out


def test_conv2d_out_channels_mismatch_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["params"]["out_channels"] = 9  # out_shape channels is 4
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    out = capsys.readouterr().out
    assert "out_channels" in out and "out_shape channels" in out


def test_conv2d_output_spatial_mismatch_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["params"]["padding"] = [0, 0]  # with k=3/s=1 -> output 6x6, not 8x8
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    out = capsys.readouterr().out
    assert "inconsistent" in out and "expected 6" in out


def test_conv2d_groups_not_dividing_channels_fails(tmp_path, capsys):
    rec = _valid_record(tmp_path)
    rec["params"]["groups"] = 2  # in_channels=3 not divisible by 2
    path = _write_manifest(tmp_path, [rec])
    assert main(["--manifest", str(path), "--no-shape-check"]) == 1
    out = capsys.readouterr().out
    assert "groups" in out and "divisible" in out


def test_shape_check_without_torch_errors(tmp_path, monkeypatch, capsys):
    # Simulate torch being unavailable when a shape check is requested.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = _write_manifest(tmp_path, [_valid_record(tmp_path, with_weight=False)])
    assert main(["--manifest", str(path)]) == 1
    assert "torch is required" in capsys.readouterr().out
