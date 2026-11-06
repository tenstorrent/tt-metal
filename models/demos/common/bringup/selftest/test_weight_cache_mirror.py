# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F261: the weight cache's shared master (paths.tt_cache_shared) and its node-local mirror (spec.tt_cache()):
pull what the master has at run start, push what a run built (atomic per file, never over a newer master file,
readable by others), core/mirror.py shared with the goldens (F260)."""

import json
import os
import stat

import pytest
import yaml

from models.demos.common.bringup.core import mirror as MR
from models.demos.common.bringup.core import weight_cache as WC
from models.demos.common.bringup.core.spec import Spec

PATS = ["*.bm.bin", "*.done.json"]


@pytest.fixture
def wc(fx, tmp_path):
    """A spec whose weight cache has a master outside art (e.g. NFS)."""
    paths = {"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo"), "tt_cache_shared": str(tmp_path / "nfs")}
    return Spec.load(fx(paths=paths))


def _put(root, rel, data=b"x" * 100, mtime=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, ns=(mtime, mtime))
    return p


def _layer(root, n, size=100, mtime=None):
    for s in (".flat.bf4.lk.w_gu.bm.bin", ".flat.bf4.lk.w_d.bm.bin", ".done.json"):
        _put(root, f"experts-r/1x1/L{n:02d}.bfp4_mse.h{n}{s}", b"y" * size, mtime)


def test_master_path_follows_the_local_mesh_and_version(wc, fx, tmp_path):
    assert wc.tt_cache() == tmp_path / "art" / "fixture" / "tt_cache" / "1x1" / "v1"
    assert wc.tt_cache_shared() == tmp_path / "nfs" / "1x1" / "v1"
    assert Spec.load(fx(paths={"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo")})).tt_cache_shared() is None
    same = {**wc.data["paths"], "tt_cache_shared": str(tmp_path / "art" / "fixture" / "tt_cache")}
    assert Spec.load(fx(paths=same)).tt_cache_shared() is None  # the master is the local cache: nothing to mirror


def test_push_creates_the_master_atomically_readable_and_only_the_cache_files(wc):
    local, shared = WC.roots(wc)
    _layer(local, 0)
    _put(local, "experts-r/1x1/L00.bfp4_mse.h0.flat.bf4.lk.w_gu.tensorbin")  # the old layout
    _put(local, "experts-r/1x1/L01.bfp4_mse.h1.flat.bf4.lk.w_gu.bm.bin.tmp77")  # being written
    _put(local, "experts-r/1x1/.L01.bm.bin.Xy12ab")  # rsync's temp name
    os.chmod(local / "experts-r/1x1/L00.bfp4_mse.h0.done.json", 0o600)
    assert WC.plan(wc, PATS)["push"] == sorted(MR.listing(local, MR.matcher(PATS)))
    res = WC.push(wc, patterns=PATS, echo=lambda m: None)
    assert res.startswith("pushed 3 files"), res
    got = MR.listing(shared, MR.matcher(["*"]))
    assert got == MR.listing(local, MR.matcher(PATS))  # same sizes and mtimes (rsync -a), nothing else
    for rel in got:
        assert os.stat(shared / rel).st_mode & stat.S_IROTH, rel  # readable by other users
    assert os.stat(shared / "experts-r/1x1").st_mode & stat.S_IXOTH
    man = json.loads((shared / MR.MANIFEST).read_text())
    assert man["files"] == got and json.loads((local / MR.MANIFEST).read_text()) == man
    assert not list(shared.rglob(".*")) and not list(shared.rglob("*.tmp*"))
    assert WC.push(wc, patterns=PATS).startswith("the master has every local file")  # (+ a note: pytest tmp is 0700)


def test_pull_fills_the_local_cache_and_never_takes_a_file_twice(wc):
    local, shared = WC.roots(wc)
    _layer(shared, 0)
    _layer(shared, 1)
    _layer(local, 1, mtime=10**18)  # the same files (same size) built on this box at another time: kept as they are
    p = WC.plan(wc, PATS)
    assert len(p["pull"]) == 3 and p["bytes_pull"] == 300 and not p["push"]
    assert WC.pull(wc, patterns=PATS, echo=lambda m: None).startswith("pulled 3 files")
    assert MR.listing(local, MR.matcher(["L00*"])) == MR.listing(shared, MR.matcher(["L00*"]))
    assert os.stat(local / "experts-r/1x1/L01.bfp4_mse.h1.done.json").st_mtime_ns == 10**18
    assert WC.pull(wc, patterns=PATS) == "local has every master file"
    assert json.loads((local / MR.MANIFEST).read_text())["files"] == MR.listing(shared, MR.matcher(PATS))


def test_a_newer_master_file_is_never_overwritten_and_a_newer_local_one_never_pulled_over(wc):
    local, shared = WC.roots(wc)
    rel = "experts-r/1x1/L02.bfp4_mse.h2.done.json"
    _put(local, rel, b"old", mtime=1_000_000_000_000_000_000)
    _put(shared, rel, b"newer master", mtime=1_700_000_000_000_000_000)
    p = WC.plan(wc, PATS)
    assert p["push"] == [] and p["kept_newer_master"] == [rel] and p["pull"] == [rel]
    assert "newer than the local ones kept" in WC.push(wc, patterns=PATS)
    assert (shared / rel).read_bytes() == b"newer master"
    _put(local, rel, b"newest local!", mtime=1_800_000_000_000_000_000)
    p = WC.plan(wc, PATS)
    assert p["pull"] == [] and p["kept_newer_local"] == [rel] and p["push"] == [rel]
    WC.pull(wc, patterns=PATS)
    assert (local / rel).read_bytes() == b"newest local!"


def test_pull_refuses_without_the_room(wc, monkeypatch):
    local, shared = WC.roots(wc)
    _layer(shared, 0)
    local.mkdir(parents=True)
    monkeypatch.setattr(WC.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    assert WC.pull(wc, patterns=PATS).startswith("FAILED: 3 files")
    assert not MR.listing(local, MR.matcher(PATS))


def test_one_push_at_a_time_and_a_master_we_cannot_write(wc, tmp_path):
    local, shared = WC.roots(wc)
    _layer(local, 0)
    lk = WC._Lock(local)
    assert lk.ok and WC.pushing(wc)
    assert WC.push(wc, patterns=PATS) == "another push is running on this box"
    assert WC.spawn_push(wc, tmp_path / "push.log") is None
    lk.release()
    assert not WC.pushing(wc)
    if os.getuid() == 0:
        pytest.skip("root writes anywhere")
    (tmp_path / "nfs").mkdir()
    os.chmod(tmp_path / "nfs", 0o555)
    try:
        assert WC.push(wc, patterns=PATS).startswith("master not writable")
    finally:
        os.chmod(tmp_path / "nfs", 0o755)


def test_no_master_is_a_no_op(fx, tmp_path):
    s = Spec.load(fx(paths={"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo")}))
    assert WC.pull(s).startswith("no master") and WC.push(s).startswith("no master")
    assert "no master" in WC.status(s)[0]


def test_status_and_cli(wc, capsys):
    local, shared = WC.roots(wc)
    _layer(local, 0)
    _layer(shared, 1)
    assert WC.main(["status", "--spec", str(wc.path), "--glob", "*.bm.bin", "--glob", "*.done.json"]) == 0
    out = capsys.readouterr().out
    assert "to pull: 3 files" in out and "to push: 3 files" in out and "mirror.json: missing or stale" in out
    assert WC.main(["push", "--spec", str(wc.path), "--glob", "*.bm.bin", "--glob", "*.done.json"]) == 0
    assert "pushed 3 files" in capsys.readouterr().out
    assert WC.main(["pull", "--spec", str(wc.path), "--glob", "*.bm.bin", "--glob", "*.done.json"]) == 0
    assert WC.plan(wc, PATS)["pull"] == [] and WC.plan(wc, PATS)["push"] == []


def test_orchestrator_pulls_at_start_and_pushes_a_new_build_in_the_background(wc, monkeypatch):
    from models.demos.common.bringup.orchestrator import Orchestrator

    local, shared = WC.roots(wc)
    _layer(shared, 0)
    wc.ledger_dir.mkdir(parents=True, exist_ok=True)
    (wc.ledger_dir / "tasks.yaml").write_text(yaml.safe_dump({"tasks": []}))
    spawned = []
    monkeypatch.setattr(
        WC, "spawn_push", lambda spec, log: spawned.append(log) or (4242 if WC.plan(spec)["push"] else None)
    )
    lines = []
    o = Orchestrator(wc, echo=lines.append)
    o.sync_weight_cache()
    assert "pulled 3 files" in lines[-1] and WC.plan(wc)["pull"] == [] and spawned and "4242" not in "".join(lines)
    _layer(local, 1)  # a task built layer 1
    o.sync_weight_cache(pull=False)
    assert "in the background (pid 4242)" in lines[-1]
    monkeypatch.setattr(WC, "pull", lambda *a, **k: 1 / 0)
    o.sync_weight_cache()  # never stops the run
    assert "sync failed (ZeroDivisionError" in lines[-1]
