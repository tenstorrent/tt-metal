# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests of tt/package_boot.py (the package's first-boot resident weight cache). No device; a fake converter.

The kill tests SIGKILL a child process at every stage of a rebuild (no finally / atexit runs) and check that no
directory a later boot would accept is left behind, and that the next boot rebuilds a valid cache.
"""

import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import models.demos.gemma4_31b_qb2.tt.package_boot as pb

ROOT = Path(__file__).resolve().parents[4]  # the tt-metal tree these modules come from
PACKAGE = "resident_m3_2"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"

# A fake converter, shared verbatim with the child processes of the kill tests.
FAKE_CONVERTER = r"""
from pathlib import Path

FILES = ["L00/qkv_s_c0.tensorbin", "L00/down_c3.tensorbin", "L03/qkv_f_c1.tensorbin", "L59/gu_c2.tensorbin",
         "head/head_c0.tensorbin", "embed_scaled.pt", "vectors.pt"]


def payload(rel):
    return (rel.encode() + b"\n") * (64 + len(rel))


def fake_convert(cache, calls=None, die_at=None, die=None, **kwargs):
    if calls is not None:
        calls.append((str(cache), kwargs))
    for i, rel in enumerate(FILES):
        p = Path(cache) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload(rel))
        if i == die_at:
            die()
"""
_ns = {}
exec(FAKE_CONVERTER, _ns)
FILES, payload = _ns["FILES"], _ns["payload"]


def converter(calls):
    return lambda cache, **kw: _ns["fake_convert"](cache, calls=calls, **kw)


def refuse(cache, **kw):
    raise AssertionError("a complete cache must be reused, not converted again")


@pytest.fixture
def env(tmp_path, monkeypatch):
    hf = tmp_path / "hf" / "hub" / "models--google--gemma-4-31B-it" / "snapshots" / REVISION
    hf.mkdir(parents=True)
    (hf / "model.safetensors.index.json").write_text("{}")
    cache = tmp_path / "tensor-cache" / "gemma4_resident_raw_bf4"
    monkeypatch.setenv("HF_MODEL", str(hf))
    monkeypatch.setenv("GRES_CACHE", str(cache))
    monkeypatch.setenv("GEMMA4_RESIDENT_ENGINE_PACKAGE", PACKAGE)
    monkeypatch.delenv("GEMMA4_RESIDENT_PRECISION", raising=False)
    monkeypatch.delenv("GEMMA4_RESIDENT_CACHE_VERIFY", raising=False)
    return {"hf": hf, "cache": cache, "partial": cache.with_name(cache.name + ".partial"),
            "key": pb.cache_key(PACKAGE, hf, "bf4")}


def build(env):
    calls = []
    pb.ensure_resident_cache(convert=converter(calls))
    return calls


def marker(env):
    return json.loads((env["cache"] / pb.MARKER).read_text())


def test_first_boot_converts_into_partial_and_marks_the_key_and_manifest(env):
    calls = build(env)
    assert calls == [(str(env["partial"]), pb.CONVERT_ARGS)]
    assert not env["partial"].exists()
    record = marker(env)
    assert record["key"] == env["key"]
    assert [rel for rel, _, _ in record["files"]] == sorted(FILES)
    assert all(size == len(payload(rel)) for rel, size, _ in record["files"])
    assert pb.check_cache(env["cache"], env["key"], "sha256") is None


def test_key_names_weights_engine_converter_and_precision(env):
    key = env["key"]
    assert (key["weights_repo"], key["weights_revision"]) == ("google/gemma-4-31B-it", REVISION)
    assert key["engine_package"] == PACKAGE and key["precision"] == "bf4"
    assert key["entry_dtypes"]["qkv_s"] == "bf4" and key["entry_dtypes"]["head"] == "bf8"
    assert len(key["converter_sha256"]) == len(key["engine_sha256"]) == 64
    assert key["convert_args"] == {"fold": False, "head": True, "embed": True, "reuse": None}


def test_a_complete_cache_is_reused_without_converting(env):
    build(env)
    before = (env["cache"] / pb.MARKER).stat().st_mtime_ns
    pb.ensure_resident_cache(convert=refuse)
    pb.ensure_resident_cache(convert=refuse, verify="sha256")
    assert (env["cache"] / pb.MARKER).stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "field, value",
    [("weights_revision", "0" * 40), ("weights_repo", "google/other"), ("engine_package", "resident_m3_3"),
     ("converter_sha256", "f" * 64), ("engine_sha256", "e" * 64), ("precision", "bf8"),
     ("entry_dtypes", {"qkv_s": "bf8"}), ("format", 0), ("convert_args", {"fold": True})],
)
def test_a_marker_with_a_mismatched_key_rebuilds(env, field, value):
    build(env)
    record = marker(env)
    record["key"][field] = value
    (env["cache"] / pb.MARKER).write_text(json.dumps(record))
    assert field in pb.check_cache(env["cache"], env["key"])
    assert len(build(env)) == 1
    assert marker(env)["key"] == env["key"]


def _missing(cache):
    (cache / FILES[0]).unlink()


def _truncated(cache):
    p = cache / FILES[1]
    p.write_bytes(p.read_bytes()[:-1])


def _extra(cache):
    (cache / "L00" / "stray_c0.tensorbin").write_bytes(b"x")


def _symlinked(cache):
    p = cache / FILES[2]
    copy = cache.parent / "elsewhere.tensorbin"
    copy.write_bytes(p.read_bytes())
    p.unlink()
    p.symlink_to(copy)


def _unreadable_marker(cache):
    (cache / pb.MARKER).write_text("{not json")


def _no_marker(cache):
    (cache / pb.MARKER).unlink()


@pytest.mark.parametrize("damage", [_missing, _truncated, _extra, _symlinked, _unreadable_marker, _no_marker])
def test_a_cache_that_differs_from_its_manifest_rebuilds(env, damage):
    build(env)
    damage(env["cache"])
    assert pb.check_cache(env["cache"], env["key"]) is not None
    assert len(build(env)) == 1
    assert pb.check_cache(env["cache"], env["key"], "sha256") is None


def test_sha256_verification_catches_a_same_size_corruption(env):
    build(env)
    p = env["cache"] / FILES[3]
    data = bytearray(p.read_bytes())
    data[5] ^= 1
    p.write_bytes(bytes(data))
    assert pb.check_cache(env["cache"], env["key"], "sizes") is None  # the default boot check reads sizes only
    assert "sha256" in pb.check_cache(env["cache"], env["key"], "sha256")
    assert len(_rebuild_with(env, verify="sha256")) == 1
    assert pb.check_cache(env["cache"], env["key"], "sha256") is None


def _rebuild_with(env, **kw):
    calls = []
    pb.ensure_resident_cache(convert=converter(calls), **kw)
    return calls


def test_a_stale_partial_is_discarded(env):
    env["partial"].mkdir(parents=True)
    (env["partial"] / "L00").mkdir()
    (env["partial"] / "L00" / "half_written.tensorbin").write_bytes(b"\0" * 7)
    build(env)
    assert not env["partial"].exists()
    assert "L00/half_written.tensorbin" not in {rel for rel, _, _ in marker(env)["files"]}


def test_a_converter_error_leaves_only_partial_and_the_next_boot_rebuilds(env):
    def broken(cache, **kw):
        _ns["fake_convert"](cache, **kw)
        raise RuntimeError("checkpoint read failed")

    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        pb.ensure_resident_cache(convert=broken)
    assert not env["cache"].exists() and env["partial"].exists()
    assert len(build(env)) == 1 and not env["partial"].exists()


def test_a_symlinked_cache_is_unlinked_not_emptied(env, tmp_path):
    target = tmp_path / "validated"
    target.mkdir()
    (target / "keep.tensorbin").write_bytes(b"k")
    env["cache"].parent.mkdir(parents=True)
    env["cache"].symlink_to(target)
    build(env)
    assert (target / "keep.tensorbin").read_bytes() == b"k"
    assert not env["cache"].is_symlink() and pb.check_cache(env["cache"], env["key"]) is None


def test_a_declared_precision_the_converter_does_not_write_is_refused(env):
    with pytest.raises(ValueError, match="Declared precision bf8"):
        pb.ensure_resident_cache(precision="bf8", convert=refuse)
    with pytest.raises(ValueError, match="GEMMA4_RESIDENT_PRECISION"):
        pb.ensure_resident_cache(precision="fp8", convert=refuse)
    assert not env["cache"].exists()


@pytest.mark.parametrize("var", ["GRES_CACHE", "GEMMA4_RESIDENT_ENGINE_PACKAGE", "HF_MODEL"])
def test_missing_configuration_is_refused(env, monkeypatch, var):
    monkeypatch.delenv(var)
    with pytest.raises(ValueError, match=var):
        pb.ensure_resident_cache(convert=refuse)


def test_weights_identity_reads_a_hub_snapshot_path(env):
    assert pb.weights_identity(env["hf"]) == ("google/gemma-4-31B-it", REVISION)
    assert pb.weights_identity("/data/ckpt")[0] == "local"


def test_the_real_converter_takes_the_arguments_passed_to_it():
    import importlib

    convert = importlib.import_module(f"models.demos.gemma4_31b_qb2.{PACKAGE}.weights").convert
    params = inspect.signature(convert).parameters
    assert set(pb.CONVERT_ARGS) <= set(params)
    assert "layers" not in pb.CONVERT_ARGS and params["layers"].default == 60  # all 60 layers


def test_the_packaged_class_ensures_the_cache_before_building_the_model(monkeypatch):
    order = []
    monkeypatch.setattr(pb, "ensure_resident_cache", lambda: order.append("cache"))

    def parent(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        order.append(("model", cls, max_batch_size, kwargs))
        return "model"

    monkeypatch.setattr(pb.Gemma4ResidentSpecForCausalLM, "initialize_vllm_model", classmethod(parent))
    out = pb.Gemma4ResidentSpecPackaged.initialize_vllm_model("cfg", "mesh", 1, max_seq_len=262144)
    assert out == "model"
    assert order == ["cache", ("model", pb.Gemma4ResidentSpecPackaged, 1, {"max_seq_len": 262144})]


def test_the_package_bundle_names_the_packaged_class():
    meta = json.loads((ROOT / "models/demos/gemma4_31b_qb2/vllm_models/resident_spec/vllm_metadata.json").read_text())
    module, name = meta["main_class"].split(":")
    import importlib

    assert getattr(importlib.import_module(module), name) is pb.Gemma4ResidentSpecPackaged
    assert meta["arch"] == "Gemma4QB2ResidentSpecForCausalLM"


# ---------------------------------------------------------------- kill tests (SIGKILL, so no cleanup code runs)

CHILD = FAKE_CONVERTER + r"""
import os, signal, sys
from pathlib import Path

import models.demos.gemma4_31b_qb2.tt.package_boot as pb

point = sys.argv[1]


def die(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)


def half_rmtree(path, *args, **kwargs):
    # Half of a rejected cache's files removed, then killed. Never the marker: discard() must have removed it first.
    files = sorted(p for p in Path(path).rglob("*") if p.is_file() and p.name != pb.MARKER)
    for p in files[: len(files) // 2]:
        p.unlink()
    die()


if point == "rmtree":
    pb.shutil.rmtree = half_rmtree
elif point == "sync":  # converted, nothing flushed or marked yet
    pb.os.sync = die
elif point == "manifest":
    pb.file_manifest = die
elif point == "marker":  # the marker's tmp file written, not yet renamed to COMPLETE.json
    pb.os.replace = die
elif point == "rename":  # .partial complete and marked, not yet renamed into place
    pb.os.rename = die
die_at = {"convert0": 0, "convert3": 3}.get(point)  # convertN: killed right after writing file N
pb.ensure_resident_cache(convert=lambda cache, **kw: fake_convert(cache, die_at=die_at, die=die, **kw))
"""


def run_child(point):
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, point],
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == -signal.SIGKILL, (point, proc.returncode, proc.stderr[-2000:])


@pytest.mark.parametrize("point", ["convert0", "convert3", "sync", "manifest", "marker", "rename"])
def test_a_kill_during_a_first_boot_leaves_only_partial_and_the_next_boot_rebuilds(env, point):
    run_child(point)
    assert not env["cache"].exists()
    assert env["partial"].exists()
    assert pb.check_cache(env["cache"], env["key"]) is not None
    assert len(build(env)) == 1
    assert not env["partial"].exists() and pb.check_cache(env["cache"], env["key"], "sha256") is None


@pytest.mark.parametrize("point", ["rmtree", "convert3", "rename"])
def test_a_kill_while_replacing_a_rejected_cache_leaves_nothing_accepted(env, point):
    build(env)
    record = marker(env)
    record["key"]["weights_revision"] = "0" * 40  # an old cache, for other weights
    (env["cache"] / pb.MARKER).write_text(json.dumps(record))
    run_child(point)
    assert pb.check_cache(env["cache"], env["key"]) is not None
    if point == "rmtree":  # discard() removed the marker before the tree: what is left is not a cache
        assert env["cache"].exists() and not (env["cache"] / pb.MARKER).exists()
    else:
        assert not env["cache"].exists() and env["partial"].exists()
    assert len(build(env)) == 1
    assert not env["partial"].exists() and pb.check_cache(env["cache"], env["key"], "sha256") is None
