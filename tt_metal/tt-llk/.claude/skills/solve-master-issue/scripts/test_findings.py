"""Tests for findings.py's multi-worker pieces: planned state, groups, next ordering, atomic claims,
the mkdir lock, arbitrary architecture names, and sync's pull-before-push. Run with:
    pytest -c /dev/null --noconftest -p no:cacheprovider test_findings.py
No device, no GitHub (reconcile is not exercised)."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "findings.py")
A10 = "tt_metal/tt-llk/tt_llk_wormhole/common/inc/a.h:10"
A12 = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/a.h:12"
B20 = "tt_metal/tt-llk/tt_llk_wormhole/llk_lib/b.h:20"
C30 = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/c.h:30"
D40 = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/d.h:40"
E50 = "ttnn/cpp/ttnn/operations/e.cpp:50"
F60 = "ttnn/cpp/ttnn/operations/f.cpp:60"


def run(*args, check=True, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(
        [sys.executable, "-I", FP, *args], capture_output=True, text=True, env=e
    )
    if check and r.returncode:
        raise AssertionError(
            f"findings.py {' '.join(args)} failed rc={r.returncode}\n{r.stdout}\n{r.stderr}"
        )
    return r


def rec(path, line, **kw):
    r = {
        "file": path,
        "line": line,
        "summary": f"bug at {path}:{line}",
        "severity": "medium",
    }
    r.update(kw)
    return r


@pytest.fixture
def ffile(tmp_path):
    recs = [
        rec("/x/wt/" + A10.rsplit(":", 1)[0], 10),
        rec("/x/wt/" + A12.rsplit(":", 1)[0], 12, duplicate_of=A10),
        rec("/x/wt/" + B20.rsplit(":", 1)[0], 20),
        rec("/x/wt/" + C30.rsplit(":", 1)[0], 30),
        rec("/x/wt/" + D40.rsplit(":", 1)[0], 40),
        rec("/x/wt/" + E50.rsplit(":", 1)[0], 50),
        rec(
            "/x/wt/" + F60.rsplit(":", 1)[0],
            60,
            disposition={"state": "already_filed", "issue": "https://g/1"},
        ),
    ]
    p = tmp_path / "CONFIRMED.json"
    p.write_text(json.dumps(recs, indent=1))
    return str(p)


def listed(f, all_=False):
    out = run("list", f, *(["--all"] if all_ else [])).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def nxt(f, owner, arch, claims, *extra):
    out = run(
        "next",
        f,
        "--owner",
        owner,
        "--arch",
        arch,
        "--claims",
        claims,
        "-n",
        "9",
        *extra,
    ).stdout
    return [json.loads(l) for l in out.splitlines()]


def test_list_arch_guess_and_workable(ffile):
    items = listed(ffile)
    keys = {i["key"]: i for i in items}
    assert F60 not in keys  # already_filed is not workable
    assert len(items) == 5  # the BH dup rides along, not listed on its own
    a = keys[A10]
    assert (
        a["arch_guess"] == "wormhole"
    )  # dup paths are not in also_at; the plan names both
    assert keys[C30]["arch_guess"] == "blackhole"
    assert keys[E50]["arch_guess"] == "neutral"
    assert a["duplicates"] == [A12]


def plan(f):
    run(
        "mark",
        f,
        A10,
        "--state",
        "planned",
        "--group",
        "g-a",
        "--arch",
        "wormhole,blackhole",
    )
    run("mark", f, B20, "--state", "planned", "--group", "g-b", "--arch", "wormhole")
    run(
        "mark",
        f,
        C30,
        D40,
        "--state",
        "planned",
        "--group",
        "g-cd",
        "--arch",
        "blackhole",
    )
    run("mark", f, E50, "--state", "planned", "--group", "g-e", "--arch", "neutral")


def test_planned_items_stay_workable_and_dup_follows(ffile):
    plan(ffile)
    items = listed(ffile)
    assert len(items) == 5 and all(i["state"] == "planned" for i in items)
    assert {i["key"]: i["arch"] for i in items}[
        A10
    ] == "blackhole,wormhole"  # canonical form
    # duplicates are never listed on their own (they ride along); check the file itself
    dup = [r for r in json.load(open(ffile)) if r.get("duplicate_of")][0]
    assert (
        dup["disposition"]["state"] == "planned"
        and dup["disposition"]["group"] == "g-a"
    )


def test_groups_and_next_order(ffile, tmp_path):
    plan(ffile)
    claims = str(tmp_path / "claims")
    groups = {
        json.loads(l)["group"]: json.loads(l)
        for l in run("groups", ffile, "--claims", claims).stdout.splitlines()
    }
    assert set(groups) == {"g-a", "g-b", "g-cd", "g-e"}
    assert (
        groups["g-a"]["arch"] == "blackhole,wormhole"
        and groups["g-cd"]["arch"] == "blackhole"
    )
    assert sorted(groups["g-cd"]["items"]) == [C30, D40]
    wh = [g["group"] for g in nxt(ffile, "wh", "wormhole", claims)]
    bh = [g["group"] for g in nxt(ffile, "bh", "blackhole", claims)]
    assert wh == [
        "g-b",
        "g-e",
        "g-a",
    ]  # all-local, neutral, partly-local; nothing-local excluded
    assert bh == ["g-cd", "g-e", "g-a"]
    both = [g["group"] for g in nxt(ffile, "bh", "blackhole", claims, "--take-other")]
    assert both[-1] == "g-b"
    # a box with both cards sees the mixed group as fully local (rank 0)
    dual = {
        g["group"]: g["rank"] for g in nxt(ffile, "x", "wormhole,blackhole", claims)
    }
    assert dual == {"g-a": 0, "g-b": 0, "g-cd": 0, "g-e": 1}


def test_any_architecture_and_many_workers(tmp_path):
    # a third architecture is just another name; n workers on one arch spread over the same rank
    recs = [
        rec(f"/x/tt_metal/tt-llk/tt_llk_quasar/llk_lib/q{i}.h", i) for i in range(1, 7)
    ]
    recs += [rec("/x/tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/w.h", 9)]
    p = tmp_path / "F.json"
    p.write_text(json.dumps(recs, indent=1))
    f = str(p)
    items = {i["key"]: i for i in listed(f)}
    assert (
        items["tt_metal/tt-llk/tt_llk_quasar/llk_lib/q1.h:1"]["arch_guess"] == "quasar"
    )
    assert (
        items["tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/w.h:9"]["arch_guess"]
        == "wormhole"
    )  # _b0 stripped
    for i in range(1, 7):
        run(
            "mark",
            f,
            f"tt_metal/tt-llk/tt_llk_quasar/llk_lib/q{i}.h:{i}",
            "--state",
            "planned",
            "--group",
            f"q{i}",
            "--arch",
            "quasar",
        )
    run(
        "mark",
        f,
        "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/w.h:9",
        "--state",
        "planned",
        "--group",
        "w",
        "--arch",
        "wormhole",
    )
    claims = str(tmp_path / "claims")
    orders = {}
    for w in ("qsr-1", "qsr-2", "qsr-3"):
        orders[w] = [g["group"] for g in nxt(f, w, "quasar", claims)]
        assert sorted(orders[w]) == [
            f"q{i}" for i in range(1, 7)
        ]  # the wormhole group is not offered
    assert (
        len({tuple(o) for o in orders.values()}) == 3
    )  # three workers, three different walk orders
    assert orders["qsr-1"] == [
        g["group"] for g in nxt(f, "qsr-1", "quasar", claims)
    ]  # and stable
    bad = run(
        "mark",
        f,
        "tt_metal/tt-llk/tt_llk_quasar/llk_lib/q1.h:1",
        "--state",
        "planned",
        "--arch",
        "Quasar!",
        check=False,
    )
    assert bad.returncode != 0


def test_claim_is_exclusive_and_release_returns(ffile, tmp_path):
    plan(ffile)
    claims = str(tmp_path / "claims")
    r1 = run("claim", ffile, "g-a", "--owner", "wh", "--claims", claims)
    assert r1.stdout.startswith(
        "CLAIMED g-a for wh: 2 record(s)"
    )  # canonical + its BH dup
    r2 = run("claim", ffile, "g-a", "--owner", "bh", "--claims", claims, check=False)
    assert r2.returncode == 1 and r2.stdout.startswith("TAKEN g-a by wh")
    allv = {i["key"]: i for i in listed(ffile, all_=True)}
    assert allv[A10]["state"] == "in-progress" and allv[A10]["owner"] == "wh"
    # a claimed/started group is no longer offered
    assert "g-a" not in [g["group"] for g in nxt(ffile, "bh", "blackhole", claims)]
    # an item that already got its issue goes back too, keeping the issue link
    run("mark", ffile, A10, "--state", "issue", "--issue", "https://g/7")
    run("release", ffile, "g-a", "--claims", claims)
    allv = {i["key"]: i for i in listed(ffile, all_=True)}
    assert allv[A10]["state"] == "planned" and allv[A10]["owner"] is None
    assert allv[A10]["disposition"]["issue"] == "https://g/7"
    assert [g["group"] for g in nxt(ffile, "bh", "blackhole", claims)].count(
        "g-a"
    ) == 1  # offered again
    assert not os.path.exists(os.path.join(claims, "g-a"))
    # and can be claimed again
    assert run(
        "claim", ffile, "g-a", "--owner", "bh", "--claims", claims
    ).stdout.startswith("CLAIMED")


def test_claim_race_one_winner(ffile, tmp_path):
    plan(ffile)
    claims = str(tmp_path / "claims")
    with ThreadPoolExecutor(8) as ex:
        rs = list(
            ex.map(
                lambda i: run(
                    "claim",
                    ffile,
                    "g-cd",
                    "--owner",
                    f"w{i}",
                    "--claims",
                    claims,
                    check=False,
                ),
                range(8),
            )
        )
    wins = [r for r in rs if r.returncode == 0]
    assert len(wins) == 1 and all(
        r.stdout.startswith("TAKEN") for r in rs if r.returncode
    )
    owner = json.load(open(os.path.join(claims, "g-cd", "owner.json")))["owner"]
    assert wins[0].stdout.startswith(f"CLAIMED g-cd for {owner}")


def test_concurrent_marks_do_not_lose_updates(ffile):
    keys = [i["key"] for i in listed(ffile)]
    with ThreadPoolExecutor(5) as ex:
        list(
            ex.map(
                lambda k: run("mark", ffile, k, "--state", "skipped", "--note", k), keys
            )
        )
    allv = {i["key"]: i for i in listed(ffile, all_=True)}
    for k in keys:
        assert allv[k]["state"] == "skipped" and allv[k]["disposition"]["note"] == k


def test_stale_lock_is_broken(ffile):
    lockd = ffile + ".lock.d"
    os.mkdir(lockd)
    open(os.path.join(lockd, "owner"), "w").write("dead-host pid=1\n")
    os.utime(lockd, (1, 1))  # epoch: ancient
    r = run("mark", ffile, E50, "--state", "skipped", env={"FINDINGS_LOCK_WAIT_S": "5"})
    assert "breaking stale lock" in r.stderr
    assert not os.path.exists(lockd)


def test_live_lock_is_waited_for(ffile):
    lockd = ffile + ".lock.d"
    os.mkdir(lockd)  # fresh: not stale
    r = run(
        "mark",
        ffile,
        E50,
        "--state",
        "skipped",
        check=False,
        env={"FINDINGS_LOCK_WAIT_S": "2", "FINDINGS_LOCK_STALE_S": "600"},
    )
    assert r.returncode != 0 and "could not take" in (r.stderr + r.stdout)
    os.rmdir(lockd)


def git(d, *a):
    return subprocess.run(
        ["git", "-C", d, *a], capture_output=True, text=True, check=True
    ).stdout


def test_sync_pulls_before_push(tmp_path):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    def clone(name):
        d = tmp_path / name
        subprocess.run(
            ["git", "clone", "-q", str(bare), str(d)], check=True, capture_output=True
        )
        git(str(d), "config", "user.email", "t@t")
        git(str(d), "config", "user.name", "t")
        return str(d)

    a = clone("a")
    f = os.path.join(a, "F.json")
    base = [rec(f"tt_metal/{c}.h", i) for i, c in enumerate("abcdefgh", 1)]
    open(f, "w").write(json.dumps(base, indent=1) + "\n")
    git(a, "add", "F.json")
    git(a, "commit", "-q", "-m", "init")
    git(a, "push", "-q", "-u", "origin", "HEAD:main")
    b = clone("b")
    git(b, "checkout", "-q", "main")
    # box B marks record 1 and pushes first (through the script, so it goes the same path)
    fb = os.path.join(b, "F.json")
    run("mark", fb, "tt_metal/a.h:1", "--state", "planned", "--group", "g1")
    assert "PUSH_OK" in run("sync", fb).stdout
    # box A marks record 8 on its stale copy; a plain push would be rejected — sync must rebase then push
    run("mark", f, "tt_metal/h.h:8", "--state", "planned", "--group", "g8")
    assert "PUSH_OK" in run("sync", f).stdout
    merged = {
        r["file"]: r.get("disposition", {}).get("group") for r in json.load(open(f))
    }
    assert (
        merged["tt_metal/a.h"] == "g1" and merged["tt_metal/h.h"] == "g8"
    )  # both marks survive
    assert (
        len(git(a, "log", "--oneline", "origin/main").splitlines()) == 3
    )  # linear history
    assert "no changes" in run("sync", f).stdout


def test_audit_key_regex_keeps_same_line_suffix():
    import importlib.util

    spec = importlib.util.spec_from_file_location("findings", FP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    body = "x\naudit-key: tt_metal/a.h:10#2\naudit-key: `tt_metal/b.h:5`\naudit-key: tt_metal/c.h:7#1 tail"
    assert m.AUDIT_KEY_RE.findall(body) == [
        "tt_metal/a.h:10#2",
        "tt_metal/b.h:5",
        "tt_metal/c.h:7#1",
    ]


def test_stale_lock_break_has_one_winner(ffile):
    # several contenders all see a stale lock; exactly one breaks it, and no mark is lost
    lockd = ffile + ".lock.d"
    os.mkdir(lockd)
    os.utime(lockd, (1, 1))
    keys = [i["key"] for i in listed(ffile)]
    with ThreadPoolExecutor(5) as ex:
        rs = list(
            ex.map(
                lambda k: run(
                    "mark",
                    ffile,
                    k,
                    "--state",
                    "skipped",
                    "--note",
                    k,
                    env={"FINDINGS_LOCK_WAIT_S": "20"},
                ),
                keys,
            )
        )
    assert sum("breaking stale lock" in r.stderr for r in rs) == 1
    allv = {i["key"]: i for i in listed(ffile, all_=True)}
    for k in keys:
        assert allv[k]["state"] == "skipped" and allv[k]["disposition"]["note"] == k
    assert not os.path.exists(lockd)
