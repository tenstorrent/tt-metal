# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shlex
from pathlib import Path

import gwdt_quad
import pytest
from gwdt_checks import Result, Status
from gwdt_quad import (
    build_quad_steps,
    chain_ok,
    mgd_path,
    producer_cmd,
    rank0_host,
    runner_cmd,
    setup_ranks,
    write_tokens,
)
from gwdt_steps import Ctx

REPO = Path("/data/u/tt-metal")
HOSTS = ["h0", "h1", "h2", "h3"]


def _inner(cmd: str) -> str:
    """The script tt-run / mpirun hands to `bash -lc` on every rank."""
    argv = shlex.split(cmd)
    return argv[argv.index("-lc") + 1]


def test_quad_keys_in_order():
    keys = [s.key for s in build_quad_steps(HOSTS, 900)]
    assert keys == ["quad:reset", "quad:passthrough-smoke", "quad:block-random-smoke", "quad:block-random-soak"]


def test_runner_cmd_exports_prefill_env_inside_bash():
    cmd = runner_cmd(REPO, HOSTS, "block_random", "2d_torus_xy", Path("/r/results"))
    inner = _inner(cmd)
    for kv in (
        "PREFILL_MANIFEST=/data/u/tt-metal/models/demos/deepseek_v3_d_p/tt/runners/manifests/quad_block_random.json",
        "PREFILL_FABRIC_MODE=2d_torus_xy",
        "PREFILL_QUAD_RESULTS_DIR=/r/results",
        "PREFILL_NUM_LAYERS=8",
        "PREFILL_H2D_SERVICE_ID=quad_screen",
    ):
        assert f"export {kv};" in inner
    assert inner.endswith("exec python3 -m models.demos.common.prefill.runners.prefill_runner")
    argv = shlex.split(cmd)
    assert argv[argv.index("--hosts") + 1] == "h0,h1,h2,h3"
    assert argv[argv.index("--mesh-graph-descriptor") + 1] == str(REPO / mgd_path(4, "2d_torus_xy"))


def test_mgd_by_host_count_and_fabric():
    base = "models/demos/common/prefill/runners/topology_configuration"
    assert mgd_path(4, "2d_torus_xy") == f"{base}/ci/sc4_mgd.textproto"
    assert mgd_path(4, "2d") == f"{base}/pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto"
    assert mgd_path(1, "2d_torus_xy") == f"{base}/ci/sc1_mgd.textproto"
    assert (
        mgd_path(1, "2d") == "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
    )
    with pytest.raises(ValueError):
        mgd_path(2, "2d")


def test_producer_cmd_budget_and_drain():
    inner = producer_cmd(REPO, "block_random", 4, Path("/r/tokens"), duration_s=300, max_requests=None)
    for kv in (
        "PREFILL_MODEL=quad_block_random",
        "PREFILL_NUM_LAYERS=8",
        "PREFILL_TRACE_DIR=/r/tokens",
        "PREFILL_PRODUCER_DURATION_S=300",
        "PREFILL_PRODUCER_MAX_REQUESTS=1000000000",
        "PREFILL_PRODUCER_DRAIN_ACKS=1",
        "PREFILL_SEND_SHUTDOWN=1",
        "PREFILL_PRODUCER_CHUNKS=2",
        "PREFILL_MAX_SEQ_LEN=40960",
        "PREFILL_H2D_SERVICE_ID=quad_screen",
    ):
        assert f"export {kv};" in inner
    assert "mpirun" not in inner
    fixed = producer_cmd(REPO, "passthrough", 1, Path("/t"), duration_s=None, max_requests=3)
    assert "export PREFILL_PRODUCER_MAX_REQUESTS=3;" in fixed and "PREFILL_PRODUCER_DURATION_S" not in fixed
    assert "export PREFILL_NUM_LAYERS=2;" in fixed


def test_producer_runs_on_rank0_host_over_ssh():
    argv = gwdt_quad._host_cmd("far-host", "echo hi")
    assert argv[0] == "ssh" and "far-host" in argv and argv[-1] == "bash -lc 'echo hi'"


def test_write_tokens_deterministic(tmp_path):
    a = json.loads((write_tokens(tmp_path / "a", 10240, 163840) / "metadata.json").read_text())["token_ids"]
    b = json.loads((write_tokens(tmp_path / "b", 10240, 163840) / "metadata.json").read_text())["token_ids"]
    assert a == b and len(a) == 10240 and all(0 <= t < 163840 for t in a) and len(set(a)) > 1000


def test_chain_ok():
    ring = ["h0", "h1", "h2", "h3"]
    assert chain_ok(["h0", "h1", "h2", "h3"], ring)
    assert chain_ok(["h2", "h1", "h0", "h3"], ring)
    assert chain_ok(["h3.exabox.tenstorrent.com", "h0", "h1", "h2"], ring)
    assert not chain_ok(["h0", "h2", "h1", "h3"], ring)


def test_setup_ranks():
    log = "[1,0]<stdout>: [pp rank 0] setup complete, entering request loop\nnoise\n[pp rank 2] setup complete, entering request loop\n"
    assert setup_ranks(log) == {0, 2}


def test_rank0_host_from_rankfile(tmp_path):
    d = tmp_path / "generated" / "ttrun" / "abc"
    d.mkdir(parents=True)
    (d / "rankfile").write_text("rank 1=h0 slot=0\nrank 0=h3 slot=0\nrank 2=h1 slot=0\nrank 3=h2 slot=0\n")
    assert rank0_host(tmp_path, HOSTS) == "h3"
    assert rank0_host(tmp_path / "none", HOSTS) == "h0"


def _ctx(tmp_path):
    return Ctx(tmp_path, tmp_path / "run", False)


def _scripted(results):
    calls = []

    def run_once(ctx, state, mode, fabric, hosts, duration_s, max_requests):
        calls.append((mode, fabric))
        return results[(mode, fabric)]

    return calls, run_once


def test_passthrough_blocks_only_when_both_modes_fail(tmp_path):
    fail, ok = Result(Status.FAIL, "x"), Result(Status.PASS, "ok")
    calls, run_once = _scripted(
        {
            ("passthrough", "2d_torus_xy"): fail,
            ("passthrough", "2d"): ok,
            ("block_random", "2d_torus_xy"): ok,
        }
    )
    steps = {s.key: s for s in build_quad_steps(HOSTS, 900, run_once=run_once)}
    ctx = _ctx(tmp_path)
    pt = steps["quad:passthrough-smoke"]
    assert pt.run(ctx, pt.state).status == Status.FAIL
    assert pt.state.fallbacks[0].result.status == Status.PASS
    assert pt.state.fallbacks[0].env == {"PREFILL_FABRIC_MODE": "2d"}
    smoke = steps["quad:block-random-smoke"]
    assert smoke.run(ctx, smoke.state).status == Status.PASS
    assert calls == [("passthrough", "2d_torus_xy"), ("passthrough", "2d"), ("block_random", "2d_torus_xy")]

    calls, run_once = _scripted({("passthrough", "2d_torus_xy"): fail, ("passthrough", "2d"): fail})
    steps = {s.key: s for s in build_quad_steps(HOSTS, 900, run_once=run_once)}
    pt = steps["quad:passthrough-smoke"]
    pt.run(ctx, pt.state)
    smoke = steps["quad:block-random-smoke"]
    r = smoke.run(ctx, smoke.state)
    assert r.status == Status.PENDING and "passthrough" in r.reason
    assert calls == [("passthrough", "2d_torus_xy"), ("passthrough", "2d")]


def test_soak_uses_soak_budget(tmp_path):
    seen = []

    def run_once(ctx, state, mode, fabric, hosts, duration_s, max_requests):
        seen.append((duration_s, max_requests))
        return Result(Status.PASS, "ok")

    steps = {s.key: s for s in build_quad_steps(HOSTS, 3600, run_once=run_once)}
    for key in ("quad:passthrough-smoke", "quad:block-random-smoke", "quad:block-random-soak"):
        steps[key].run(_ctx(tmp_path), steps[key].state)
    assert seen == [(None, 3), (300, None), (3600, None)]


def test_cli_quad_requires_hosts(capsys):
    from galaxy_workload_deployment_test import main

    assert main(["--quad", "--co"]) == 2
    assert "--hosts" in capsys.readouterr().out


def test_ring_order_mismatch_exits(monkeypatch, capsys):
    from galaxy_workload_deployment_test import main

    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: ["h0", "h1", "h2", "h3"])
    assert main(["--quad", "--co", "--hosts", "h0,h2,h1,h3"]) == 2
    out = capsys.readouterr().out
    assert "h0,h1,h2,h3" in out


def test_kmd_lock_blocker_named():
    from gwdt_quad import bringup_blocker

    log = (
        "[1,2]<stdout>: 01:08:05.938 | warning | UMD | Waiting for KMD lock 20 on /dev/tenstorrent/0, which is "
        "currently held by another handle (kmd_mutex.cpp:140)\n"
    )
    assert "held by another process" in bringup_blocker(log)
    assert bringup_blocker("[pp rank 0] setup complete, entering request loop\n") is None


def test_environment_fault_skips_fallback(tmp_path):
    from gwdt_quad import ENV_FAULT

    calls, run_once = _scripted({("passthrough", "2d_torus_xy"): Result(Status.FAIL, f"{ENV_FAULT}chips held")})
    pt = {s.key: s for s in build_quad_steps(HOSTS, 900, run_once=run_once)}["quad:passthrough-smoke"]
    assert pt.run(_ctx(tmp_path), pt.state).reason.startswith(ENV_FAULT)
    assert calls == [("passthrough", "2d_torus_xy")] and pt.state.fallbacks == []


def test_same_prompt_rotates_over_more_slots_than_chunks_in_flight():
    from gwdt_quad import CHUNKS, SLOTS

    run = _inner(runner_cmd(REPO, HOSTS, "passthrough", "2d_torus_xy", Path("/r")))
    feed = producer_cmd(REPO, "passthrough", 4, Path("/t"), duration_s=None, max_requests=3)
    assert f"export PREFILL_NUM_USERS={SLOTS};" in run and f"export PREFILL_NUM_USERS={SLOTS};" in feed
    # In flight on a quad: 8 forwards in the H2D FIFO, about two per stage in the D2D path, one being pushed.
    assert SLOTS * CHUNKS > 8 + 2 * 4 + 1


def test_ring_order_parses_the_helpers_comma_joined_string(monkeypatch, tmp_path):
    import subprocess

    out = json.dumps({"status": "success", "ordered_hosts": "h1,h2,h3,h0", "checked_at": "x"})
    monkeypatch.setattr(gwdt_quad, "DESCRIPTORS", tmp_path)
    for f in ("cabling_descriptor.textproto", "deployment_descriptor.textproto"):
        (tmp_path / f).write_text("")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    ring = gwdt_quad.ring_order(REPO, HOSTS)
    assert ring == ["h1", "h2", "h3", "h0"]
    assert chain_ok(HOSTS, ring)


def test_reset_timeout_is_a_failure(monkeypatch, tmp_path):
    import subprocess

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 900)

    monkeypatch.setattr(subprocess, "run", hang)
    r = gwdt_quad.reset_hosts(REPO, ["far-host"], tmp_path)
    assert r.status == Status.FAIL and "far-host" in r.reason


def test_rank_hosts_from_rankfile(tmp_path):
    d = tmp_path / "generated" / "ttrun" / "abc"
    d.mkdir(parents=True)
    (d / "rankfile").write_text("rank 0=h2 slot=0\nrank 1=h0 slot=0\nrank 2=h1 slot=0\nrank 3=h3 slot=0\n")
    assert gwdt_quad.rank_hosts(tmp_path, HOSTS) == ["h2", "h0", "h1", "h3"]
    assert gwdt_quad.rank_hosts(tmp_path / "none", HOSTS) == HOSTS


def test_bringup_rows_name_the_ranks_that_never_got_ready():
    rows = gwdt_quad.bringup_rows({0, 1, 3}, ["h0", "h1", "h2", "h3"], hung=True)
    assert [r.status for _, r in rows] == [Status.PASS, Status.PASS, Status.HANG, Status.PASS]
    assert rows[2][0] == "stage 2 h2" and "bring-up" in rows[2][1].reason
    rows = gwdt_quad.bringup_rows({0}, ["h0", "h1"], hung=False)
    assert rows[1][1].status == Status.FAIL


def test_old_kmd_lock_line_does_not_block_once_a_rank_got_ready():
    w = gwdt_quad.BringupWatch(n_ranks=2, now=0)
    w.feed("Waiting for KMD lock 20 on /dev/tenstorrent/0, which is currently held by another handle\n", now=10)
    w.feed("[pp rank 0] setup complete, entering request loop\n", now=20)
    assert w.blocker(now=20 + gwdt_quad.KMD_LOCK_S + 1) is None
    w.feed("Waiting for KMD lock 3 on /dev/tenstorrent/1, which is currently held by another handle\n", now=30)
    assert w.blocker(now=20 + gwdt_quad.KMD_LOCK_S + 1) is not None
    assert w.ready == {0}


def test_runner_has_no_device_op_timeout():
    run = _inner(runner_cmd(REPO, HOSTS, "block_random", "2d_torus_xy", Path("/r")))
    assert "export TT_METAL_OPERATION_TIMEOUT_SECONDS=0;" in run


def test_runner_waits_for_slow_inter_galaxy_handshake():
    run = _inner(runner_cmd(REPO, HOSTS, "passthrough", "2d_torus_xy", Path("/r")))
    assert "export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=300000;" in run


def test_tt_fatal_ends_bring_up_and_names_the_first_crashed_rank():
    w = gwdt_quad.BringupWatch(n_ranks=4, now=0)
    cause = (
        "[1,2]<stdout>: \x1b[90m2026-11-19 15:07:09.433\x1b[0m | critical | \x1b[35m         Always\x1b[0m | "
        "\x1b[37mTT_FATAL: D2DStreamService: the sender at coord MeshCoordinate([2, 0]) has no free fabric router "
        "toward its peer\x1b[0m \x1b[90m(assert.hpp:104)\x1b[0m\n"
    )
    echo = (
        "[1,3]<stdout>: 2026-11-19 15:07:09.442 | critical |          Always | TT_FATAL: D2DStreamService: the "
        "sender (rank 2) failed to build its service at coord MeshCoordinate([0, 0]); see its log (assert.hpp:104)\n"
    )
    # The follower hands over whatever is on disk, so a line can arrive in two reads.
    w.feed(cause[:80], now=100)
    assert w.fatal == {}
    w.feed(cause[80:] + echo, now=101)
    assert list(w.fatal) == [2, 3]
    assert (
        w.fatal[2]
        == "D2DStreamService: the sender at coord MeshCoordinate([2, 0]) has no free fabric router toward its peer"
    )
    assert not w.crashed(now=101 + gwdt_quad.FATAL_GRACE_S - 2)
    assert w.crashed(now=101 + gwdt_quad.FATAL_GRACE_S)


def test_runner_banner_mentioning_tt_fatal_is_not_a_crash():
    w = gwdt_quad.BringupWatch(n_ranks=2, now=0)
    w.feed("[1,0]<stderr>: comparison_mode_should_raise_exception=false TT_FATAL handling on\n", now=1)
    w.feed("[1,0]<stdout>: 15:02:11.221 | warning | Metal | Failed to create shared memory: Permission denied\n", now=2)
    assert w.fatal == {} and not w.crashed(now=10**6)


def test_python_traceback_ends_bring_up_with_its_exception_line():
    w = gwdt_quad.BringupWatch(n_ranks=4, now=0)
    log = (
        "[1,0]<stderr>: Traceback (most recent call last):\n"
        '[1,0]<stderr>:   File "prefill_runner.py", line 1290, in _unlink_stale_shm\n'
        "[1,2]<stderr>: 2026-11-20 06:16:40.848 | INFO     | __main__:_serve_request:1372 - routing up\n"
        "[1,0]<stderr>:     os.remove(path)\n"
        "[1,0]<stderr>: PermissionError: [Errno 1] Operation not permitted: "
        "'/dev/shm/tt_prefill_layer_completion_ring_0'\n"
    )
    w.feed(log, now=100)
    assert w.fatal == {
        0: "PermissionError: [Errno 1] Operation not permitted: '/dev/shm/tt_prefill_layer_completion_ring_0'"
    }
    assert w.crashed(now=100 + gwdt_quad.FATAL_GRACE_S)


def test_tt_fatal_wins_over_the_traceback_it_raises():
    w = gwdt_quad.BringupWatch(n_ranks=2, now=0)
    w.feed(
        "[1,1]<stdout>: 06:16:40.1 | critical |          Always | TT_FATAL: no free link (assert.hpp:104)\n"
        "[1,1]<stderr>: Traceback (most recent call last):\n"
        "[1,1]<stderr>: RuntimeError: TT_FATAL @ d2d_stream_service.cpp:666: !claimed.empty()\n",
        now=1,
    )
    assert w.fatal == {1: "no free link"}


def _fake_hosts(monkeypatch, sees: dict[str, bool]):
    """ssh to h reads the probe back only when sees[h]; the local host is h0."""
    monkeypatch.setattr(gwdt_quad.socket, "gethostname", lambda: "h0")

    def run(argv, **kw):
        host, path = argv[-2], shlex.split(shlex.split(argv[-1])[-1])[-1]
        out = Path(path).read_text() if sees[host] else ""
        return gwdt_quad.subprocess.CompletedProcess(argv, 0 if sees[host] else 1, out, "")

    monkeypatch.setattr(gwdt_quad.subprocess, "run", run)


def test_shared_dir_seen_by_every_host(tmp_path, monkeypatch):
    _fake_hosts(monkeypatch, {"h1": True, "h2": True, "h3": True})
    assert gwdt_quad.shared_dir_error(tmp_path, HOSTS) is None
    assert not list(tmp_path.glob(".shared_probe_*"))


def test_shared_dir_names_blind_hosts(tmp_path, monkeypatch):
    _fake_hosts(monkeypatch, {"h1": True, "h2": False, "h3": False})
    assert gwdt_quad.shared_dir_error(tmp_path, HOSTS) == f"{tmp_path} is not visible from h2, h3"
    assert not list(tmp_path.glob(".shared_probe_*"))


def test_shared_dir_unwritable(tmp_path):
    assert gwdt_quad.shared_dir_error(tmp_path / "missing", HOSTS).startswith("cannot write")


def test_quad_aborts_when_run_dir_not_shared(tmp_path, monkeypatch, capsys):
    import galaxy_workload_deployment_test as gwdt

    monkeypatch.setenv("GWDT_SHARED_FOLDER", str(tmp_path))
    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: None)
    monkeypatch.setattr(gwdt_quad, "shared_dir_error", lambda d, hosts: f"{d} is not visible from h3")
    monkeypatch.setattr(gwdt_quad, "checkout_error", lambda repo, hosts, **kw: None)
    monkeypatch.setattr(gwdt, "run_all", lambda *a, **k: pytest.fail("steps ran"))
    assert gwdt.main(["--quad", "--hosts", ",".join(HOSTS)]) == 2
    out = capsys.readouterr().out
    assert "not visible from h3" in out and "GWDT_SHARED_FOLDER" in out


A, B = "a" * 40, "b" * 40
CLEAN = "da39a3ee5e6b4b0d3255bfef95601890afd80709"


def _fake_checkouts(monkeypatch, outs: dict[str, str | None]):
    """The checkout script on h prints outs[h]; None is a host where it fails. The local host is h0."""
    monkeypatch.setattr(gwdt_quad.socket, "gethostname", lambda: "h0")

    def run(argv, **kw):
        out = outs["h0" if argv[0] == "bash" else argv[-2]]
        return gwdt_quad.subprocess.CompletedProcess(argv, 0 if out else 128, out or "", "")

    monkeypatch.setattr(gwdt_quad.subprocess, "run", run)


def _out(head: str, fp: str = CLEAN, *files: str) -> str:
    return "\n".join([head, fp, *files]) + "\n"


def _cells(table: str) -> list[list[str]]:
    """The table's cells per line; also checks every column starts at the same offset on every line."""
    lines = table.splitlines()
    for col in range(1, 3):
        starts = {m.start(col) for m in (re.match(r"  (\S+)\s+(\S+)\s+(\S)", l) for l in lines) if m}
        assert len(starts) == 1, table
    return [re.split(r"\s{2,}", l.strip()) for l in lines]


def test_checkout_same_everywhere(monkeypatch):
    _fake_checkouts(monkeypatch, dict.fromkeys(HOSTS, _out(A)))
    assert gwdt_quad.checkout_error(REPO, HOSTS) is None


def test_checkout_same_dirty_everywhere(monkeypatch):
    _fake_checkouts(monkeypatch, dict.fromkeys(HOSTS, _out(A, "f" * 40, "x.py")))
    assert gwdt_quad.checkout_error(REPO, HOSTS) is None


def test_checkout_names_hosts_that_differ(monkeypatch):
    _fake_checkouts(
        monkeypatch,
        {"h0": _out(A), "h1": _out(A, "f" * 40, "b.py", "a.py", "c.py", "d.py"), "h2": _out(B), "h3": None},
    )
    assert _cells(gwdt_quad.checkout_error(REPO, HOSTS)) == [
        ["host", "commit", "uncommitted changes"],
        ["h0", A[:11], "none", "this host"],
        ["h1", A[:11], "a.py, b.py, c.py +1 more", "DIFFERS"],
        ["h2", B[:11], "none", "DIFFERS"],
        ["h3", "-", "-", f"DIFFERS: no readable git checkout at {REPO}"],
    ]


def test_checkout_clean_remote_vs_dirty_here(monkeypatch):
    _fake_checkouts(
        monkeypatch,
        {
            "h0": _out(A, "f" * 40, "r.py"),
            "h1": _out(A),
            "h2": _out(A, "f" * 40, "r.py"),
            "h3": _out(A, "f" * 40, "r.py"),
        },
    )
    assert _cells(gwdt_quad.checkout_error(REPO, HOSTS)) == [
        ["host", "commit", "uncommitted changes"],
        ["h0", A[:11], "r.py", "this host"],
        ["h1", A[:11], "none", "DIFFERS"],
        ["h2", A[:11], "r.py"],
        ["h3", A[:11], "r.py"],
    ]


def test_checkout_unreadable_here(monkeypatch):
    _fake_checkouts(monkeypatch, dict.fromkeys(HOSTS, None))
    assert gwdt_quad.checkout_error(REPO, HOSTS).startswith("cannot read the git checkout")


def test_parse_checkout_rejects_garbage():
    assert gwdt_quad.parse_checkout("fatal: not a git repository\n") is None
    assert gwdt_quad.parse_checkout(_out(A)) == gwdt_quad.Checkout(A, CLEAN, ())


def test_checkout_script_on_a_real_repo(tmp_path):
    """The shell side: the fingerprint follows tracked edits and untracked files, and ignores LAUNCHER_ONLY."""
    git = lambda *a: gwdt_quad.subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "m.py").write_text("1\n")
    launcher = tmp_path / gwdt_quad.LAUNCHER_ONLY
    launcher.mkdir(parents=True)
    (launcher / "g.py").write_text("1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    state = lambda: gwdt_quad.parse_checkout(
        gwdt_quad.subprocess.run(
            ["bash", "-c", gwdt_quad.checkout_script(tmp_path)], capture_output=True, text=True
        ).stdout
    )
    clean = state()
    assert clean.fingerprint == CLEAN and clean.files == ()
    (launcher / "g.py").write_text("2\n")
    (launcher / "new.py").write_text("x\n")
    assert state() == clean
    (tmp_path / "m.py").write_text("2\n")
    edited = state()
    assert edited.files == ("m.py",) and edited.fingerprint != CLEAN
    (tmp_path / "m.py").write_text("3\n")
    assert state().fingerprint != edited.fingerprint
    (tmp_path / "m.py").write_text("1\n")
    (tmp_path / "u.py").write_text("u\n")
    untracked = state()
    assert untracked.files == ("u.py",)
    (tmp_path / "u.py").write_text("v\n")
    assert state().fingerprint != untracked.fingerprint


def test_quad_aborts_when_checkouts_differ(tmp_path, monkeypatch, capsys):
    import galaxy_workload_deployment_test as gwdt

    monkeypatch.setenv("GWDT_SHARED_FOLDER", str(tmp_path))
    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: None)
    monkeypatch.setattr(gwdt_quad, "shared_dir_error", lambda d, hosts: pytest.fail("shared-dir probe ran"))
    monkeypatch.setattr(gwdt_quad, "checkout_error", lambda repo, hosts, **kw: "h2 is at bbbbbbbbbbb")
    monkeypatch.setattr(gwdt, "run_all", lambda *a, **k: pytest.fail("steps ran"))
    assert gwdt.main(["--quad", "--hosts", ",".join(HOSTS)]) == 2
    out = capsys.readouterr().out
    assert "same tt-metal checkout" in out
    assert "GWDT_SKIP_CHECKOUT_CHECK=1" in out and "GWDT_SKIP_CHECKOUT_CHECK=2" in out


def test_quad_commit_only_mismatch_suggests_only_level_2(tmp_path, monkeypatch, capsys):
    import galaxy_workload_deployment_test as gwdt

    monkeypatch.setenv("GWDT_SHARED_FOLDER", str(tmp_path))
    monkeypatch.setenv("GWDT_SKIP_CHECKOUT_CHECK", "1")
    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: None)
    monkeypatch.setattr(gwdt_quad, "checkout_error", lambda repo, hosts, **kw: "  host  commit")
    assert gwdt.main(["--quad", "--hosts", ",".join(HOSTS)]) == 2
    out = capsys.readouterr().out
    assert "GWDT_SKIP_CHECKOUT_CHECK=2  skip" in out and "GWDT_SKIP_CHECKOUT_CHECK=1  compare" not in out
    assert not list(tmp_path.rglob("LOG_*")), "a commit mismatch must not leave a run dir behind"


def test_checkout_commit_only_ignores_uncommitted_changes(monkeypatch):
    _fake_checkouts(monkeypatch, {"h0": _out(A), "h1": _out(A, "f" * 40, "x.py"), "h2": _out(B), "h3": _out(A)})
    assert _cells(gwdt_quad.checkout_error(REPO, HOSTS, dirty=False)) == [
        ["host", "commit"],
        ["h0", A[:11], "this host"],
        ["h1", A[:11]],
        ["h2", B[:11], "DIFFERS"],
        ["h3", A[:11]],
    ]


def test_checkout_table_lists_this_host_first_when_not_in_hosts(monkeypatch):
    _fake_checkouts(monkeypatch, {"h0": _out(A), "h1": _out(B)})
    assert _cells(gwdt_quad.checkout_error(REPO, ["h1"])) == [
        ["host", "commit", "uncommitted changes"],
        ["h0", A[:11], "none", "this host"],
        ["h1", B[:11], "none", "DIFFERS"],
    ]


def test_checkout_skip_level():
    assert [gwdt_quad.checkout_skip_level(v) for v in (None, "", "0", "1", "2")] == [0, 0, 0, 1, 2]
    with pytest.raises(ValueError, match="GWDT_SKIP_CHECKOUT_CHECK='yes'"):
        gwdt_quad.checkout_skip_level("yes")


@pytest.mark.parametrize("level, dirty", [("0", True), ("1", False), ("2", None)])
def test_quad_checkout_skip_env(tmp_path, monkeypatch, capsys, level, dirty):
    """0 runs the full check, 1 compares commits only, 2 skips it; the run then stops at the shared-dir probe."""
    import galaxy_workload_deployment_test as gwdt

    seen = []
    monkeypatch.setenv("GWDT_SHARED_FOLDER", str(tmp_path))
    monkeypatch.setenv("GWDT_SKIP_CHECKOUT_CHECK", level)
    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: None)
    monkeypatch.setattr(gwdt_quad, "checkout_error", lambda repo, hosts, dirty: seen.append(dirty))
    monkeypatch.setattr(gwdt_quad, "shared_dir_error", lambda d, hosts: "not shared")
    assert gwdt.main(["--quad", "--hosts", ",".join(HOSTS)]) == 2
    assert seen == ([] if dirty is None else [dirty])
    assert ("note: GWDT_SKIP_CHECKOUT_CHECK" in capsys.readouterr().out) == (level != "0")


def test_quad_checkout_skip_env_rejects_bad_value(tmp_path, monkeypatch, capsys):
    import galaxy_workload_deployment_test as gwdt

    monkeypatch.setenv("GWDT_SHARED_FOLDER", str(tmp_path))
    monkeypatch.setenv("GWDT_SKIP_CHECKOUT_CHECK", "3")
    monkeypatch.setattr(gwdt_quad, "ring_order", lambda repo, hosts: None)
    monkeypatch.setattr(gwdt_quad, "checkout_error", lambda *a, **k: pytest.fail("check ran"))
    assert gwdt.main(["--quad", "--hosts", ",".join(HOSTS)]) == 2
    assert "use 0 (full check), 1 (commit only) or 2 (no check)" in capsys.readouterr().out
    assert not list(tmp_path.rglob("LOG_*"))
