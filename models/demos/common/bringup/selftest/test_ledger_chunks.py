# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F265 (owner, FIX.27): the ledger files that grow with every task are chunk files below the repo's 500 KB limit;
nothing is compacted, rounded or dropped. DeepSeek-V4.1 FIX.26's passed gate could not commit: state_metrics.json
had reached 515,706 bytes. CPU only."""

import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from models.demos.common.bringup.core import chunked
from models.demos.common.bringup.core.chunked import (
    CHUNK_BYTES,
    REPO_LIMIT_BYTES,
    ChunkedJson,
    key_hash,
    log_chunks,
    read_log,
    roll_log,
)
from models.demos.common.bringup.core.ledger import Ledger

REPO = Path(__file__).resolve().parents[5]


def metrics_like(n_tasks: int, per_task: int, seed: int = 0) -> dict:
    """A state_metrics-shaped object: {task: {metric: value}} with 6-digit floats, ints and strings."""
    rnd = random.Random(seed)
    out = {}
    for t in range(n_tasks):
        m = {}
        for i in range(per_task):
            m[f"err_layer_L{i:02d}"] = float(f"{rnd.random():.6g}")
            m[f"err_layer_L{i:02d}_limit"] = float(f"{rnd.random():.6g}")
        m["chips"], m["note"] = 32, f"task {t}"
        out[f"P.{t}"] = m
    return out


def sizes(store: ChunkedJson) -> list[int]:
    return [p.stat().st_size for p in store.chunk_files()]


# ---------------------------------------------------------------- the store
def test_round_trip_of_a_file_past_the_repo_limit(tmp_path):
    data = metrics_like(200, 40)
    store = ChunkedJson(tmp_path / "state_metrics.json")
    assert len(store.dump(data).encode()) > REPO_LIMIT_BYTES  # the single file would be refused
    files = store.write(data)
    assert len(files) >= 2 and [p.name for p in files] == [f"{i:04d}.json" for i in range(1, len(files) + 1)]
    assert store.read() == data
    assert max(sizes(store)) <= CHUNK_BYTES
    assert not store.legacy.exists()
    # the same data written again changes no file
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    store.write(store.read())
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in store.chunk_files()} == before


def test_every_chunk_stays_within_the_bound(tmp_path):
    store = ChunkedJson(tmp_path / "x.json", bound=4096)
    data = metrics_like(60, 6, seed=1)
    store.write(data)
    assert store.read() == data
    assert all(s <= 4096 for s in sizes(store)) and len(store.chunk_files()) > 5
    # every key in exactly one chunk
    keys = [k for _, obj in store.chunks() for k in obj]
    assert sorted(keys) == sorted(data) and len(keys) == len(set(keys))


def test_appending_across_a_chunk_boundary_rewrites_only_the_newest_chunk(tmp_path):
    store = ChunkedJson(tmp_path / "x.json", bound=4096)
    data = {}
    seen_new_chunk = False
    for t, (k, v) in enumerate(metrics_like(40, 6, seed=2).items()):
        old = {p.name: p.read_bytes() for p in store.chunk_files()}
        newest = max(old, default=None)
        data[k] = v
        store.write(data)
        now = {p.name: p.read_bytes() for p in store.chunk_files()}
        assert store.read() == data
        for name, b in old.items():  # an older chunk is never touched by an append
            if name != newest:
                assert now[name] == b, (t, name)
        if len(now) > len(old) and old:
            seen_new_chunk = True
            assert now[newest] == old[newest]  # the full chunk stays as it was; the key opened the next one
            assert list(json.loads(now[max(now)])) == [k]
        assert all(s <= 4096 for s in sizes(store))
    assert seen_new_chunk


def test_a_grown_record_moves_its_chunks_tail_and_a_huge_record_gets_its_own_chunk(tmp_path):
    store = ChunkedJson(tmp_path / "x.json", bound=4096)
    data = metrics_like(30, 6, seed=3)
    store.write(data)
    first = store.chunks()[0][1]
    k0 = sorted(first)[0]
    data[k0] = {**data[k0], **{f"extra_{i}": float(i) for i in range(60)}}  # a re-run records more metrics
    store.write(data)
    assert store.read() == data
    assert all(s <= 4096 for p, s in zip(store.chunk_files(), sizes(store)) if len(json.loads(p.read_text())) > 1)
    huge = {f"m{i}": float(i) for i in range(1000)}  # one record over the bound: never split, alone in its chunk
    data["HUGE"] = huge
    store.write(data)
    assert store.read() == data
    alone = [obj for _, obj in store.chunks() if "HUGE" in obj]
    assert alone == [{"HUGE": huge}]


def test_removed_keys_leave_no_empty_chunk(tmp_path):
    store = ChunkedJson(tmp_path / "x.json", bound=2048)
    data = metrics_like(20, 4, seed=4)
    store.write(data)
    first = list(store.chunks()[0][1])
    for k in first:
        del data[k]
    store.write(data)
    assert store.read() == data and "0001.json" not in [p.name for p in store.chunk_files()]


def test_duplicate_keys_across_chunks_are_an_error(tmp_path):
    store = ChunkedJson(tmp_path / "x.json")
    store.write({"A": 1})
    (store.dir / "0002.json").write_text(json.dumps({"A": 2}))
    with pytest.raises(chunked.ChunkError, match="key 'A' in both 0001.json and 0002.json"):
        store.read()


# ---------------------------------------------------------------- migration
def test_migrate_is_lossless_and_leaves_a_manifest(tmp_path):
    data = metrics_like(200, 40, seed=5)
    legacy = tmp_path / "state_metrics.json"
    legacy.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    raw = legacy.read_bytes()
    store = ChunkedJson(legacy)
    assert store.read() == data  # before: the old single file is read as it is
    store.migrate()
    assert not legacy.exists() and store.read() == data
    assert max(sizes(store)) <= CHUNK_BYTES
    man = json.loads(store.manifest.read_text())
    assert man["bytes"] == len(raw) and man["keys"] == {k: key_hash(v) for k, v in data.items()}
    assert store.migrate() == store.chunk_files()  # once: a second call changes nothing


def test_a_late_old_format_write_merges_over_the_chunks(tmp_path):
    """The orchestrator that runs FIX.27's gate still holds the pre-F265 ledger code: it reads no state.json after the
    migration and writes one with only what it touched. The next reader merges it in; the next write folds it into
    the chunks and deletes it."""
    led = Ledger(tmp_path / "led")
    led.write_tasks({"tasks": [{"id": t, "title": t, "gate": {"cmd": "true"}} for t in ("A.1", "B.1")]})
    led.update("A.1", status="PASS", metrics={"m": 1.0}, history_add={"t": "1", "status": "PASS"})
    led.update("B.1", status="RUNNING", agent_runs=[{"attempt": 1}], history_add={"t": "2", "status": "RUNNING"})
    # what the old code writes after the migration: only B.1, its lists only from what it appended
    led.state_path.write_text(json.dumps({"B.1": {"status": "PASS", "history": [{"t": "3", "status": "PASS"}]}}))
    led.metrics_path.write_text(json.dumps({"B.1": {"m": 2.0}}))
    led.side_path.write_text(json.dumps({"B.1": {"agent_runs": [{"attempt": 2}]}}))
    st = led.state()
    assert st["A.1"]["metrics"] == {"m": 1.0} and st["A.1"]["status"] == "PASS"
    assert st["B.1"]["status"] == "PASS" and st["B.1"]["metrics"] == {"m": 2.0}
    assert [h["t"] for h in st["B.1"]["history"]] == ["2", "3"]
    assert st["B.1"]["agent_runs"] == [{"attempt": 1}, {"attempt": 2}]
    led.update("B.1", reason=[])
    assert not led.state_path.exists() and not led.metrics_path.exists() and not led.side_path.exists()
    st2 = led.state()
    assert st2["B.1"]["history"] == st["B.1"]["history"] and st2["B.1"]["agent_runs"] == st["B.1"]["agent_runs"]


# ---------------------------------------------------------------- readers and writers through the ledger
def test_the_ledger_reads_and_writes_every_chunk(tmp_path):
    led = Ledger(tmp_path / "led")
    for st in led.stores:
        st.bound = 2048
    tids = [f"T.{i}" for i in range(40)]
    led.write_tasks({"tasks": [{"id": t, "title": t, "gate": {"cmd": "true"}} for t in tids]})
    for i, t in enumerate(tids):
        led.update(
            t,
            status="PASS",
            metrics={f"err_{j}": 0.001 * (i + j) for j in range(8)},
            agent_runs=[{"attempt": 1, "role": "fix", "note": "x" * 50}],
            history_add={"t": f"{i}", "status": "PASS"},
        )
    for st in led.stores:
        assert len(st.chunk_files()) > 1 and all(s <= 2048 for s in sizes(st)), st.dir
    state = led.state()
    assert sorted(k for k in state) == sorted(tids)
    for i, t in enumerate(tids):
        assert state[t]["metrics"] == {f"err_{j}": float(f"{0.001 * (i + j):.6g}") for j in range(8)}
        assert state[t]["agent_runs"][0]["attempt"] == 1 and state[t]["history"][-1]["t"] == f"{i}"
    # the old record of a task not re-run is never trimmed again, even when its gate's globs change
    led.write_tasks(
        {"tasks": [{"id": t, "title": t, "gate": {"cmd": "true", "metrics": {"pcc_*": ">= 0.9"}}} for t in tids]}
    )
    led.update("T.0", status="PASS")
    assert led.state()["T.1"]["metrics"] == state["T.1"]["metrics"]
    rec = led.record_paths()
    assert led.state_store.dir in rec and led.metrics_store.dir in rec and led.state_path in rec


def test_timing_reads_the_chunked_state(tmp_path):
    from models.demos.common.bringup import timing

    led = Ledger(tmp_path / "led")
    led.write_tasks({"tasks": [{"id": "A.1", "title": "a", "gate": {"cmd": "true"}}]})
    led.update("A.1", status="PASS", history_add={"t": "2026-11-12T10:00:00", "status": "PASS", "s": 5.0})
    out = timing.breakdown(led.dir, tmp_path / "run", {"A.1": {"step": "fix"}}, now=1.0e10)
    assert out is not None and not led.state_path.exists()


def test_a_gate_commit_stages_the_chunks_and_the_old_files_deletion(sandbox):
    from models.demos.common.bringup.core.gate import run_gate
    from models.demos.common.bringup.selftest.conftest import record_cmd

    led = sandbox.tasks(
        {"id": "A.1", "title": "task A.1", "deps": [], "gate": {"cmd": record_cmd(pcc_out=0.995), "metrics": {}}}
    )
    led.metrics_path.write_text(json.dumps({"OLD.1": {"pcc_out": 0.5}}, indent=1, sort_keys=True) + "\n")
    sandbox.git("add", "-f", str(led.metrics_path))
    sandbox.git("commit", "-q", "-m", "old single file")
    res = run_gate(sandbox.spec, led, "A.1", commit=True)
    assert res.verdict == "PASS" and res.commit, res.summary()
    shown = sandbox.git("show", "--name-status", "--no-renames", "--format=", "HEAD").split("\n")
    assert "D\tbringup/state_metrics.json" in shown
    assert "A\tbringup/state_metrics.chunks/0001.json" in shown and "A\tbringup/state.chunks/0001.json" in shown
    assert led.metrics_store.read() == {"OLD.1": {"pcc_out": 0.5}, "A.1": {"pcc_out": 0.995}}


# ---------------------------------------------------------------- append-only logs
def _log(n_sections: int, size: int, seed: int = 0) -> bytes:
    rnd = random.Random(seed)
    parts = [b"# Breadcrumbs\n\nAppend-only log.\n"]
    for i in range(n_sections):
        body = "".join(rnd.choice("abcdefgh \n") for _ in range(size))
        parts.append(f"\n## T.{i} attempt 1\n{body}\n".encode())
    return b"".join(parts)


def test_a_log_rolls_into_chunks_and_reads_back_byte_for_byte(tmp_path):
    log = tmp_path / "BREADCRUMBS.md"
    data = _log(400, 1500)
    log.write_bytes(data)
    written = roll_log(log)
    assert written and log_chunks(log) == written
    assert read_log(log).encode() == data
    assert all(p.stat().st_size <= CHUNK_BYTES for p in written) and log.stat().st_size <= CHUNK_BYTES // 4
    assert log.read_bytes().startswith(b"## T.")  # the live file keeps whole newest sections
    # appending past the bound again opens the next chunk numbers; nothing earlier changes
    before = {p.name: p.read_bytes() for p in written}
    more = _log(300, 1500, seed=1)
    with open(log, "ab") as f:
        f.write(more)
    written2 = roll_log(log)
    assert written2 and all(p.name not in before for p in written2)
    assert {p.name: p.read_bytes() for p in written} == before
    assert read_log(log).encode() == data + more
    assert roll_log(log) == []  # under the bound: no roll


def test_the_models_breadcrumbs_roll_losslessly(tmp_path):
    src = REPO / "models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md"
    if not src.exists():
        pytest.skip("no model breadcrumbs in this checkout")
    log = tmp_path / "BREADCRUMBS.md"
    log.write_bytes(src.read_bytes())
    roll_log(log, bound=64 * 1024, keep=16 * 1024)
    assert read_log(log).encode() == src.read_bytes()
    assert all(p.stat().st_size <= 64 * 1024 for p in log_chunks(log))


# ---------------------------------------------------------------- the live ledger
def _spec_dirs() -> list[tuple[Path, Path]]:
    from models.demos.common.bringup.core.spec import Spec

    path = os.environ.get("BRINGUP_SPEC")
    if not path or not Path(path).exists():
        return []
    spec = Spec.load(path)
    return [(Path(spec.ledger_dir), Path(spec.bringup_dir))]


def _committed_files(repo: Path, ledger_dir: Path, bringup_dir: Path) -> list[Path]:
    """The files this run's gate commits can take: the run's ledger (every file, tracked or not ignored) and the
    model's shared bring-up files (bringup_dir's own files: BREADCRUMBS.md and its chunks, known_issues.md, ...), without
    the generated dashboard pages (never committed, F210). A seeded run's bringup_dir/results/ is the seed's ledger,
    written once by the seed's gates and never by this run."""
    import fnmatch

    from models.demos.common.bringup.core.gate import GENERATED

    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", str(ledger_dir), str(bringup_dir)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    led = str(Path(ledger_dir).resolve().relative_to(repo.resolve())) + "/"
    top = str(Path(bringup_dir).resolve().relative_to(repo.resolve())) + "/"
    keep = [f for f in out if f.startswith(led) or (f.startswith(top) and "/" not in f[len(top) :])]
    return sorted({repo / f for f in keep if not any(fnmatch.fnmatch(f, g) for g in GENERATED)})


def test_the_live_ledger_is_migrated_losslessly_and_below_the_limit():
    """The migration of the spec's ledger (FIX.27): every key the old file held is in the chunks with the same content
    (manifest hashes; the old file itself while git keeps its object), unless its task has run since; no file a
    commit takes is over the repo's limit. Records ledger_files_over_limit."""
    from models.demos.common.bringup.core import metrics as M

    dirs = _spec_dirs()
    if not dirs:
        pytest.skip("BRINGUP_SPEC not set")
    ledger_dir, bringup_dir = dirs[0]
    led = Ledger(ledger_dir)
    state = led.state()
    checked = 0
    for st in led.stores:
        if not st.manifest.exists():
            continue
        man = json.loads(st.manifest.read_text())
        merged = st.read()
        ran_since = {  # a task with a gate or an agent attempt since the migration, and the task whose gate runs this
            t
            for t, e in state.items()
            if isinstance(e, dict)
            and (
                str(e.get("last_run") or "") > man["t"]
                or any(str(h.get("t", "")) > man["t"] for h in e.get("history") or [])
            )
        } | {M.task_id(""), M.record_task("")}
        missing = sorted(set(man["keys"]) - set(merged))
        assert not missing, f"{st.legacy.name}: keys dropped by the migration: {missing}"
        changed = sorted(k for k, h in man["keys"].items() if k not in ran_since and key_hash(merged[k]) != h)
        assert not changed, f"{st.legacy.name}: records changed although their task has not run since: {changed}"
        blob = man.get("git_blob")
        if blob and subprocess.run(["git", "cat-file", "-e", blob], cwd=REPO).returncode == 0:
            raw = subprocess.run(["git", "cat-file", "-p", blob], cwd=REPO, capture_output=True, check=True).stdout
            assert len(raw) == man["bytes"]
            old = json.loads(raw)
            assert {k: v for k, v in merged.items() if k in old and k not in ran_since} == {
                k: v for k, v in old.items() if k not in ran_since
            }
        checked += 1
    over = chunked.over_limit(_committed_files(REPO, ledger_dir, bringup_dir))
    M.record("ledger_files_over_limit", len(over))
    assert not over, f"over the repo's {REPO_LIMIT_BYTES // 1024} KB limit: {[(str(p), s) for p, s in over]}"
    assert checked or not any(st.legacy.exists() for st in led.stores), "an old single file was never migrated"
