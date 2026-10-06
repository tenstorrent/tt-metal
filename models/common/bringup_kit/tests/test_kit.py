"""Hardware-free tests for the bring-up kit. Run: pytest models/common/bringup_kit/tests -q --noconftest -p no:cacheprovider"""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(HERE, ".."))
import canary  # noqa: E402
import hangwatch  # noqa: E402
import host_health  # noqa: E402
import kitlib  # noqa: E402
import launch  # noqa: E402
import log_tool  # noqa: E402
import results  # noqa: E402
import scorecard  # noqa: E402
import status_report  # noqa: E402


def sh(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "src/a.py").write_text("1\n")
    (r / "other.txt").write_text("x\n")
    sh(r, "init", "-q", "-b", "main")
    sh(r, "config", "user.email", "t@t")
    sh(r, "config", "user.name", "t")
    sh(r, "add", "-A")
    sh(r, "commit", "-qm", "init")
    (tmp_path / "ev.log").write_text("evidence\n")
    return r


def exits(fn):
    """True if fn() calls sys.exit (the tools' refusal path)."""
    try:
        fn()
    except SystemExit:
        return True
    return False


def commit(r, name, content):
    (r / name).write_text(content)
    sh(r, "add", "-A")
    sh(r, "commit", "-qm", f"change {name}")


def add(repo, log, *extra, claim="TTFT 100 ms"):
    return log_tool.main(
        [
            "add",
            "--log",
            str(log),
            "--repo",
            str(repo),
            "--claim",
            claim,
            "--track",
            "prefill",
            "--evidence",
            str(repo.parent / "ev.log"),
            "--paths",
            "src",
            *extra,
        ]
    )


# ---------------- log_tool ----------------
def test_add_check_stale_by_paths_and_supersede(repo, tmp_path, capsys):
    log = repo / "bringup/m/LOG.md"
    assert add(repo, log, "--data", '{"ttft_ms": 100}') == 0
    e1 = log_tool.load(str(log))[0]
    assert e1["data_obj"] == {"ttft_ms": 100} and e1["status"] == "verified"
    ents = log_tool.load(str(log))
    assert log_tool.entry_state(e1, repo, ents)[0] == "FRESH"
    commit(repo, "other.txt", "y\n")  # unrelated file: still fresh
    assert log_tool.entry_state(e1, repo, ents)[0] == "FRESH"
    commit(repo, "src/a.py", "2\n")  # dependency changed
    s, why = log_tool.entry_state(e1, repo, ents)
    assert s == "STALE" and "src/a.py" in why[0]
    # re-measure at the new HEAD, supersede the old one
    assert add(repo, log, "--supersedes", e1["id"]) == 0
    ents = log_tool.load(str(log))
    assert log_tool.entry_state(ents[0], repo, ents)[0] == "SUPERSEDED"
    assert log_tool.entry_state(ents[1], repo, ents)[0] == "FRESH"
    capsys.readouterr()
    log_tool.main(["show", "--log", str(log), "--repo", str(repo), "--fresh-only"])
    out = capsys.readouterr().out
    assert ents[1]["id"] in out and ents[0]["id"] not in out
    assert log_tool.main(["check", "--log", str(log), "--repo", str(repo), "--strict"]) == 1


def test_not_ancestor_is_stale(repo):
    log = repo / "LOG.md"
    sh(repo, "checkout", "-qb", "side")
    commit(repo, "other.txt", "side\n")
    assert add(repo, log) == 0
    sh(repo, "checkout", "-q", "main")  # the entry's commit lives only on 'side'
    ents = log_tool.load(str(log))
    s, why = log_tool.entry_state(ents[0], repo, ents)
    assert s == "STALE" and "not an ancestor" in why[0]


def test_add_refuses_dirty_and_wrong_head(repo):
    log = repo / "bringup/LOG.md"
    (repo / "src/a.py").write_text("dirty\n")
    assert exits(lambda: add(repo, log))
    sh(repo, "checkout", "--", "src/a.py")
    # HEAD detached at an old commit that is not the branch tip
    commit(repo, "other.txt", "z\n")
    sh(repo, "checkout", "-q", "--detach", "HEAD~1")
    assert exits(lambda: add(repo, log, "--branch", "main"))
    old = kitlib.head(repo)
    commit(repo, "other.txt", "w\n")  # detached HEAD moved; --commit names an existing commit explicitly
    assert add(repo, log, "--commit", old) == 0
    assert log_tool.load(str(log))[0]["commit"] == old
    assert exits(lambda: add(repo, log, "--commit", "deadbeefdead"))
    # dirty only inside the log file is fine; evidence must exist for verified
    assert exits(
        lambda: log_tool.main()[
            "add",
            "--log",
            str(log),
            "--repo",
            str(repo),
            "--claim",
            "c",
            "--track",
            "t",
            "--paths",
            "src",
            "--evidence",
            "/nonexistent",
        ]
    )


def test_inferred_not_quotable(repo, capsys):
    log = repo / "LOG.md"
    log_tool.main(
        [
            "add",
            "--log",
            str(log),
            "--repo",
            str(repo),
            "--claim",
            "guess 3x",
            "--track",
            "perf",
            "--status",
            "inferred",
        ]
    )
    e = log_tool.load(str(log))[0]
    assert not log_tool.quotable(e, "FRESH")
    capsys.readouterr()
    log_tool.main(["show", "--log", str(log), "--repo", str(repo)])
    assert "NOT QUOTABLE" in capsys.readouterr().out


# ---------------- scorecard ----------------
def fixture_state(e):
    return ("SUPERSEDED", ["fixture"]) if e["id"].endswith("a06") else ("FRESH", [])


def test_scorecard_dsv41_fixture():
    entries = log_tool.load(os.path.join(FIX, "dsv41_LOG.md"))
    card = json.load(open(os.path.join(FIX, "dsv41_scorecard.json")))
    ok, rows = scorecard.run(
        card,
        entries,
        lambda e: ("FRESH", []) if not any(e["id"] in x["supersedes_list"] for x in entries) else ("SUPERSEDED", ["x"]),
    )
    st = {r["id"]: r["status"] for r in rows}
    assert not ok
    assert st["first_token"] == st["gsm8k"] == st["perf_prefill"] == st["perf_decode"] == st["layer_pcc.moe"] == "PASS"
    assert st["determinism"] == "MISSING"  # inferred evidence cannot satisfy a gate
    for g in (
        "layer_pcc.dense",
        "chain_prefix_pcc",
        "prefill_phase",
        "decode_phase",
        "memory_fit",
        "spec_exact",
        "hosted_oracle",
    ):
        assert st[g] == "MISSING"


def test_scorecard_stale_fail_and_exit(tmp_path):
    entries = [
        {"id": "E1", "status": "verified", "data_obj": {"pcc": 0.95}, "supersedes_list": []},
        {"id": "E2", "status": "verified", "data_obj": {"pcc": 0.999}, "supersedes_list": []},
    ]
    mk = lambda ev: {
        "gates": [{"id": "g", "kind": "single_layer_pcc", "threshold": 0.99, "evidence": [ev]}]
    }  # noqa: E731
    fresh = lambda e: ("FRESH", [])  # noqa: E731
    assert scorecard.run(mk("E1"), entries, fresh)[1][0]["status"] == "FAIL"
    assert scorecard.run(mk("E2"), entries, fresh)[0] is True
    assert scorecard.run(mk("E2"), entries, lambda e: ("STALE", ["why"]))[1][0]["status"] == "STALE"
    assert scorecard.run(mk("E9"), entries, fresh)[1][0]["status"] == "MISSING"
    optional = {
        "gates": [{"id": "g", "kind": "single_layer_pcc", "threshold": 0.99, "evidence": [], "required": False}]
    }
    assert scorecard.run(optional, entries, fresh)[0] is True


def test_scorecard_other_kinds():
    ev = lambda **d: [{"id": "E", "status": "verified", "data_obj": d, "supersedes_list": []}]  # noqa: E731
    fresh = lambda e: ("FRESH", [])  # noqa: E731
    one = lambda g, e: scorecard.evaluate_gate({**g, "evidence": ["E"]}, e, fresh)[0]  # noqa: E731
    assert (
        one(
            {"kind": "chain_pcc_curve", "min_final": 0.97, "max_drop_per_layer": 0.01},
            ev(pcc_curve=[1, 0.999, 0.99, 0.985]),
        )
        == "PASS"
    )
    assert (
        one({"kind": "chain_pcc_curve", "min_final": 0.97, "max_drop_per_layer": 0.01}, ev(pcc_curve=[1, 0.95, 0.94]))
        == "FAIL"
    )
    assert one({"kind": "determinism", "min_runs": 2}, ev(runs=2, bit_identical=False)) == "FAIL"
    tab = ev(table=[{"batch": 4, "isl": 4096, "free_gib": 2.0}, {"batch": 16, "isl": 4096, "free_gib": 0.4}])
    assert one({"kind": "memory_fit", "cells": [[4, 4096]], "min_free_gib": 1}, tab) == "PASS"
    assert one({"kind": "memory_fit", "cells": [[4, 4096], [16, 4096]], "min_free_gib": 1}, tab) == "FAIL"
    assert one({"kind": "memory_fit", "cells": [[64, 4096]], "min_free_gib": 1}, tab) == "MISSING"
    assert one({"kind": "metric", "key": "x", "op": "<=", "value": 3}, ev(x=2)) == "PASS"
    assert (
        one(
            {"kind": "prefill_check", "min_layer_pcc": 0.99, "min_kv_pcc": 0.99},
            ev(layer_pcc_min=0.995, kv_pcc_min=0.98),
        )
        == "FAIL"
    )
    assert one({"kind": "decode_check", "threshold": 0.98}, ev(pcc=0.99)) == "PASS"
    assert one({"kind": "hosted_oracle", "min_n": 2}, ev(n=3, prompts_file="/nonexistent")) == "FAIL"
    assert one({"kind": "spec_exact", "min_frac": 1.0}, ev(n=64, identical=60)) == "FAIL"


def test_scorecard_template_valid():
    t = scorecard.template("m", ("dense", "moe", "swa"))
    req = {g["id"] for g in t["gates"] if g["required"]}
    assert {"gsm8k", "determinism", "layer_pcc.swa", "prefill_phase", "decode_phase", "chain_prefix_pcc"} <= req
    assert not {"hosted_oracle", "spec_exact"} & req
    assert all(g["kind"] in scorecard.CHECKERS for g in t["gates"])
    ok, rows = scorecard.run(t, [], lambda e: ("FRESH", []))
    assert not ok and all(r["status"] == "MISSING" for r in rows)


def test_scorecard_cli_uses_git_staleness(repo, tmp_path):
    log = repo / "LOG.md"
    add(repo, log, "--data", '{"pcc": 0.999}')
    eid = log_tool.load(str(log))[0]["id"]
    card = tmp_path / "c.json"
    card.write_text(
        json.dumps({"gates": [{"id": "g", "kind": "single_layer_pcc", "threshold": 0.99, "evidence": [eid]}]})
    )
    args = ["run", "--card", str(card), "--log", str(log), "--repo", str(repo)]
    assert scorecard.main(args) == 0
    commit(repo, "src/a.py", "changed\n")
    assert scorecard.main(args) == 1  # evidence went STALE


# ---------------- results ----------------
LOG_A = textwrap.dedent(
    """\
    GRID_ENV DC16-2-BG-0204-u13-45 2026-10-06T09:32:02 head=a9593c4477a
    GRID_ENV DSV41_LAYERS=0-39
    GRID_ENV DSV41_SESSION=isl4k_b16
    2026 | INFO | built layer 0 (110s)
    2026-10-06 10:27:01.911 | INFO     | x - built layer 39 (3203s)
    === session scenario isl4k_b16 ===
    FIRSTTOK_IDS [427, 427]
    TTFT (whole batch of 16 users, ISL max 3720): 9358 ms -> prefill 6360 tok/s (585 ms/user amortised)
    Decode: 42.7 ms/token @ 23.40 tok/s/user (374.5 tok/s throughput), batch 16 (padded 16), host breakdown {}
    SPEC_RESULT_CYCLE {"B": 16, "speedup": 1.5}
    PASSED
    ===== 1 passed in 4216.16s =====
    """
)
LOG_B = LOG_A.replace("-45 ", "-46 ").replace("9358", "4679").replace("42.7", "40.0")


def test_results_extract_and_compare():
    r = results.extract(LOG_A)
    s = r["scenarios"][0]
    assert (r["host"], r["commit"], r["status"], r["built_layers"]) == ("45", "a9593c4477a", "PASSED", 40)
    assert (s["scenario"], s["B"], s["ttft_ms"], s["prefill_tok_s"], s["decode_ms_tok"], s["tok_s_user"]) == (
        "isl4k_b16",
        16,
        9358,
        6360,
        42.7,
        23.4,
    )
    assert r["first_tokens"] == [427, 427] and r["spec"][0]["speedup"] == 1.5
    assert "isl4k_b16" in results.fmt_md(r)
    c = results.compare(r, results.extract(LOG_A.replace("42.7", "40.0")))
    assert c["valid"] and c["rows"][0]["decode_ms_tok"]["b_over_a"] == pytest.approx(0.937, abs=1e-3)
    bad = results.compare(r, results.extract(LOG_B))
    assert not bad["valid"] and "cross-host" in bad["problems"][0]


def test_results_failures():
    oom = results.extract("GRID_ENV h-45 d head=abc\nTT_FATAL: Out of Memory: Not enough space to allocate\nFAILED x\n")
    assert oom["status"] == "FAILED" and oom["cause"] == "OOM"
    hang = results.extract(
        "GRID_ENV h-45 d head=abc\n[hangwatch] pid 5 log silent 45 min\n[hangwatch] killing process tree [5]\n"
    )
    assert hang["status"] == "HANG"
    assert results.extract("nothing")["status"] == "INCOMPLETE"


# ---------------- host_health ----------------
def test_host_health_verdicts(tmp_path):
    bad = host_health.load_bad(os.path.join(HERE, "..", "bad_hosts.txt"))
    assert "31" in bad and "wrong outputs" in bad["31"] and "42" in bad and "hangs" in bad["42"]
    info = host_health.parse_probe(
        "host=x\ndevices=32\nstray=0\nstray_cmd=\nlock=free\nload=1 2 3\ndisk_weights_gb=900\ndisk_logs_gb=900\n"
    )
    v = lambda h, i: host_health.verdict(h, i, bad, 50)[0]  # noqa: E731
    assert v("45", info) == "OK" and v("31", info) == "BAD" and v("46", None) == "UNREACHABLE"
    assert v("45", {**info, "stray": "1"}) == "BUSY" and v("45", {**info, "lock": "held"}) == "BUSY"
    assert v("45", {**info, "disk_logs_gb": "3"}) == "LOW_DISK" and v("45", {**info, "devices": "0"}) == "BAD"


def test_canary_evaluate():
    assert canary.evaluate(["a"] * 32, "a")[0]
    assert canary.evaluate(["a"] * 32, None)[0]
    ok, msg = canary.evaluate(["a"] * 16 + ["b"] * 16 + ["a"] * 0, "a")
    assert not ok
    ok, msg = canary.evaluate(["a"] * 30 + ["b", "b"], None)
    assert not ok and "[30, 31]" in msg
    assert not canary.evaluate(["a"] * 4, "z")[0]


# ---------------- hangwatch ----------------
def test_hangwatch_stage_rules():
    rules = hangwatch.parse_rules(hangwatch.DEFAULT_RULES)
    assert hangwatch.stall_limit_min("loading Engram table", rules, 45) == 60
    assert hangwatch.stall_limit_min("something else", rules, 45) == 45
    assert min(m for _, m in rules) >= 45


def test_hangwatch_kills_tree_and_records(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("start\n")
    # a 'wrapper' that spawns a python child (the real device holder), in its own process group
    child = subprocess.Popen(["bash", "-c", "python3 -c 'import time; time.sleep(600)' & wait"], start_new_session=True)
    time.sleep(1)
    kids = hangwatch.children_of(child.pid)
    assert kids
    old = time.time() - 120
    os.utime(log, (old, old))
    rc = hangwatch.main(
        [
            "--pid",
            str(child.pid),
            "--log",
            str(log),
            "--stall-seconds",
            "30",
            "--poll-s",
            "0.2",
            "--report-dir",
            str(tmp_path / "rep"),
            "--triage-cmd",
            "echo triage {report}",
            "--reset-cmd",
            "false;;true",
        ]
    )
    child.wait(timeout=10)
    assert rc == 3
    assert not any(hangwatch.alive(k) for k in kids)  # python grandchild is dead, not just the wrapper
    txt = log.read_text()
    assert "killed process tree" in txt and "reset rc=0" in txt and "triage" in txt
    ev = json.load(open(str(log) + ".hang.json"))
    assert ev["reset_rcs"] == [1, 0] and ev["triage_rc"] == 0


def test_hangwatch_exits_zero_when_job_ends(tmp_path):
    log = tmp_path / "l.log"
    log.write_text("x\n")
    p = subprocess.Popen(["true"])
    p.wait()
    assert hangwatch.main(["--pid", str(p.pid), "--log", str(log), "--poll-s", "0.1"]) == 0


# ---------------- launch ----------------
def test_launch_dry_run_and_refusals(repo, capsys, tmp_path):
    h = kitlib.head(repo)
    base = [
        "--worktree",
        str(repo),
        "--commit",
        h,
        "--log",
        str(tmp_path / "x.log"),
        "--dry-run",
        "--env",
        "DSV41_SESSION=a",
        "--",
    ]
    assert launch.main(base + ["pytest", "-x", "demo.py"]) == 0
    out = capsys.readouterr().out
    for needle in (
        "setsid nohup flock",
        "dsv4_dev.lock",
        "DSV41_BUILD_SLOTS=5",
        "DSV41_BUILD_STAGGER_S=480",
        "GRID_ENV",
        "hangwatch.py",
        "unset",
        "DSV41_SESSION=a",
        f"head={h}",
    ):
        assert needle in out
    # HEAD != commit
    assert exits(
        lambda: launch.main(["--worktree", str(repo), "--commit", "deadbeef", "--log", "l", "--dry-run", "--", "true"])
    )
    commit(repo, "other.txt", "n\n")
    assert exits(lambda: launch.main(base + ["true"]))  # HEAD moved
    # dirty tree
    h2 = kitlib.head(repo)
    (repo / "src/a.py").write_text("dirty\n")
    assert exits(
        lambda: launch.main(["--worktree", str(repo), "--commit", h2, "--log", "l", "--dry-run", "--", "true"])
    )
    assert (
        launch.main(
            [
                "--worktree",
                str(repo),
                "--commit",
                h2,
                "--log",
                str(tmp_path / "y"),
                "--dry-run",
                "--ignore-dirty",
                "src/",
                "--",
                "true",
            ]
        )
        == 0
    )


def test_launch_local_end_to_end(repo, tmp_path):
    """Runs the generated script for real (no device): env scrubbing, GRID_ENV header, hangwatch start, rc. Uses a private lock file."""
    log = tmp_path / "run.log"
    env = dict(
        os.environ, DSV41_STRAY="1", MOE_COMPUTE_STRAY="1", TT_METAL_STRAY="1", KIT_DEVICE_LOCK=str(tmp_path / "lock")
    )
    h = kitlib.head(repo)
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "..", "launch.py"),
            "--worktree",
            str(repo),
            "--commit",
            h,
            "--log",
            str(log),
            "--env",
            "DSV41_SESSION=s1",
            "--report-dir",
            str(tmp_path / "rep"),
            "--lock-wait-s",
            "5",
            "--",
            "bash",
            "-c",
            "env | grep -E '^(DSV41_|MOE_|TT_METAL_)' | sort; echo job-done",
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    for _ in range(100):
        if log.exists() and "LAUNCH_RC" in log.read_text():
            break
        time.sleep(0.2)
    t = log.read_text()
    assert "GRID_ENV" in t and f"head={h}" in t and "job-done" in t and "LAUNCH_RC=0" in t
    assert "STRAY" not in t
    assert "DSV41_SESSION=s1" in t and "TT_METAL_HOME=" in t


# ---------------- status_report ----------------
def test_status_parse_and_slots(tmp_path):
    jobs, lock = status_report.parse_jobs("JOB|123|7200|99.0|/l/a.log|42|pytest -x demo\nLOCK|held\n")
    assert jobs[0]["age_min"] == 120 and jobs[0]["silent_min"] == 42 and lock == "held"
    assert status_report.slot_occupancy(str(tmp_path / "none")) is None
    (tmp_path / "slot0.lock").write_text("")
    (tmp_path / "slot1.lock").write_text("")
    assert status_report.slot_occupancy(str(tmp_path)) == (0, 2)


def test_launch_refuses_bad_host():
    import subprocess
    import sys

    kit = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt = os.path.dirname(os.path.dirname(os.path.dirname(kit)))
    head = subprocess.check_output(["git", "-C", wt, "rev-parse", "--short=11", "HEAD"], text=True).strip()
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(kit, "launch.py"),
            "--worktree",
            wt,
            "--commit",
            head,
            "--host",
            "31",
            "--log",
            "/tmp/never.log",
            "--dry-run",
            "--",
            "true",
        ],
        capture_output=True,
        text=True,
    )
    if "worktree dirty" in r.stderr:
        import pytest

        pytest.skip("tree dirty: the bad-host check runs after the clean-tree check")
    assert r.returncode != 0 and "bad_hosts" in r.stderr
