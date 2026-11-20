# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F260: tests read goldens from a node-local mirror of the shared copy (NFS), synced at run start; a stale or missing
mirror falls back to the shared copy with a warning; frozen files stay checked on the shared copy."""

import json
import os

import pytest
import torch
import yaml
from safetensors.torch import save_file

from models.demos.common.bringup.core import freeze, stamps
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.reference import golden as GD
from models.demos.common.bringup.reference.golden import Golden


def _golden(d):
    """A small golden in the trace layout (golden.py): manifest with its content_hash, metadata, a chunk, a state."""
    (d / "chunk_00").mkdir(parents=True)
    (d / "kv_cache").mkdir()
    save_file({"out": torch.arange(64.0).reshape(8, 8)}, str(d / "chunk_00" / "layer_00.safetensors"))
    save_file({"x": torch.ones(4)}, str(d / "chunk_00" / "model.safetensors"))
    save_file({"key_cache_layer_0": torch.zeros(2, 2)}, str(d / "kv_cache" / "layer_0.safetensors"))
    (d / "metadata.json").write_text(json.dumps({"token_ids": [1, 2, 3], "seq_len": 256}))
    man = {"seq": 256, "chunk": 64, "layers": [0], "content_hash": GD.content_hash(d)}
    (d / "manifest.json").write_text(json.dumps(man))
    GD.write_manifest(d)  # as generate_golden does after the rename


@pytest.fixture
def gm(fx, tmp_path):
    """A spec whose goldens live outside art (the shared copy, e.g. NFS), with the s256 golden there."""
    shared = tmp_path / "nfs" / "fixture" / "golden_full" / "box"
    path = fx(paths={"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo"), "golden": str(shared)})
    s = Spec.load(path)
    _golden(GD.rung_dir(s, s.rung("s256")))
    GD._warned.clear()
    return s


def _dirs(s):
    d = GD.rung_dir(s, s.rung("s256"))
    return d, s.golden_mirror_root / d.name


def test_mirror_root_follows_the_shared_path_after_the_model_dir(gm, fx, tmp_path):
    assert gm.golden_mirror_root == tmp_path / "art" / "fixture" / "golden_full" / "box"
    local = Spec.load(fx(paths={**gm.data["paths"], "golden_local": "/elsewhere/g"}))
    assert local.golden_mirror_root.as_posix() == "/elsewhere/g"
    under_art = Spec.load(fx(paths={"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo")}))
    assert under_art.golden_mirror_root is None  # already node-local: nothing to mirror
    off = Spec.load(fx(paths=gm.data["paths"], goldens={"mirror": False}))
    assert off.golden_mirror_root is None


def test_the_manifest_beside_the_shared_copy_lists_its_files(gm):
    shared, _ = _dirs(gm)
    man = json.loads((shared / GD.MIRROR).read_text())
    assert man["files"] == GD.listing(shared) and "manifest.json" in man["files"] and GD.MIRROR not in man["files"]
    assert Golden(shared).verify()  # mirror.json is derived: not in the content_hash


def test_missing_mirror_reads_the_shared_copy_with_a_warning(gm, capsys):
    shared, mirror = _dirs(gm)
    assert Golden.for_rung(gm, "s256").dir == shared
    err = capsys.readouterr().err
    assert "WARNING (F260)" in err and "not synced" in err and "golden sync" in err
    Golden.for_rung(gm, "s256")
    assert "WARNING" not in capsys.readouterr().err  # one line per golden, not per read


def test_sync_makes_it_complete_and_tests_read_the_mirror(gm, capsys):
    shared, mirror = _dirs(gm)
    res = GD.sync(gm)
    assert res["s256_c64"].startswith("synced") and "s256_c64-official" not in res  # no official generated
    assert res["s512_c128"] == "not generated"
    g = Golden.for_rung(gm, "s256")
    assert g.dir == mirror and g.shared_dir == shared and "WARNING" not in capsys.readouterr().err
    assert g.verify() and json.loads((mirror / GD.MIRROR).read_text()) == json.loads((shared / GD.MIRROR).read_text())
    assert GD.sync(gm)["s256_c64"] == "in sync"  # idempotent
    assert GD.main(["status", "--spec", str(gm.path), "--rung", "s256"]) == 0
    assert str(mirror) in capsys.readouterr().out


def test_a_changed_shared_file_makes_the_mirror_stale_until_the_next_sync(gm, capsys):
    shared, mirror = _dirs(gm)
    GD.sync(gm)
    (shared / "drift-s256.json").write_text("{}")  # e.g. a drift file rewritten in the shared copy
    assert Golden.for_rung(gm, "s256").dir == shared
    assert "stale" in capsys.readouterr().err
    assert GD.sync(gm)["s256_c64"].startswith("synced")
    assert Golden.for_rung(gm, "s256").dir == mirror and (mirror / "drift-s256.json").read_text() == "{}"


def test_manifest_mismatch_is_detected(gm):
    shared, mirror = _dirs(gm)
    GD.sync(gm)
    f = mirror / "metadata.json"
    f.write_text(f.read_text() + " ")  # a mirror file edited in place
    assert "incomplete: metadata.json" in GD.mirror_problem(shared, mirror)
    GD.sync(gm)
    assert GD.mirror_problem(shared, mirror) is None
    (mirror / "chunk_00" / "model.safetensors").unlink()
    assert "missing" in GD.mirror_problem(shared, mirror)
    GD.sync(gm)
    man = json.loads((mirror / GD.MIRROR).read_text())
    man["files"]["extra.bin"] = [1, 1]
    (mirror / GD.MIRROR).write_text(json.dumps(man))
    assert "stale" in GD.mirror_problem(shared, mirror)


def test_the_official_sibling_must_be_complete_too(gm):
    shared, mirror = _dirs(gm)
    off = shared.with_name(shared.name + GD.OFFICIAL_SUFFIX)
    off.mkdir()
    (off / "final_norm.json").write_text("{}")
    GD.sync(gm)
    assert GD.read_dir(gm, gm.rung("s256")) == mirror and (mirror.with_name(off.name) / "final_norm.json").exists()
    (off / "window_kv.json").write_text("{}")
    assert GD.read_dir(gm, gm.rung("s256")) == shared


def test_files_written_only_into_the_mirror_are_reported(gm):
    _, mirror = _dirs(gm)
    GD.sync(gm)
    (mirror / "only_here.json").write_text("{}")
    assert "only in the mirror" in GD.sync(gm)["s256_c64"]
    assert GD.read_dir(gm, gm.rung("s256")) == mirror  # an extra file does not make the mirror wrong


def test_frozen_files_stay_checked_on_the_shared_copy(gm):
    """Frozen keys are the shared paths (the same on every box); syncing, and reading the mirror, changes neither
    their hashes nor the goldens' stamp key."""
    shared, mirror = _dirs(gm)
    key = stamps.goldens_key(gm)
    task = {"frozen": {"files": freeze.hash_paths(gm.repo, [str(shared / "manifest.json")])}}
    GD.sync(gm)
    assert Golden.for_rung(gm, "s256").dir == mirror
    assert freeze.verify(gm.repo, task) == [] and stamps.goldens_key(gm) == key
    (shared / GD.MIRROR).write_text("{}")  # a rewritten mirror manifest is not a new golden
    assert stamps.goldens_key(gm) == key


def test_orchestrator_syncs_a_stale_mirror(gm):
    from models.demos.common.bringup.orchestrator import Orchestrator

    shared, mirror = _dirs(gm)
    gm.ledger_dir.mkdir(parents=True, exist_ok=True)
    (gm.ledger_dir / "tasks.yaml").write_text(yaml.safe_dump({"tasks": []}))
    lines = []
    o = Orchestrator(gm, echo=lines.append)
    o.sync_goldens(always=False)
    assert GD.mirror_problem(shared, mirror) is None and "complete" in lines[-1]
    n = len(lines)
    o.sync_goldens(always=False)  # in sync: nothing to do, nothing printed
    assert len(lines) == n
    os.utime(shared / "metadata.json", ns=(1, 1))
    o.sync_goldens(always=False)
    assert GD.mirror_problem(shared, mirror) is None and len(lines) == n + 1
