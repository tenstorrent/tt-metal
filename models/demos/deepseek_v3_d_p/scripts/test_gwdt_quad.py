# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
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
    assert mgd_path(1, "2d") == "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
    with pytest.raises(ValueError):
        mgd_path(2, "2d")


def test_producer_cmd_budget_and_drain():
    cmd = producer_cmd(REPO, "h2", "block_random", 4, Path("/r/tokens"), duration_s=300, max_requests=None)
    argv = shlex.split(cmd)
    assert argv[argv.index("--host") + 1] == "h2:1"
    inner = _inner(cmd)
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
    fixed = _inner(producer_cmd(REPO, "h0", "passthrough", 1, Path("/t"), duration_s=None, max_requests=3))
    assert "export PREFILL_PRODUCER_MAX_REQUESTS=3;" in fixed and "PREFILL_PRODUCER_DURATION_S" not in fixed
    assert "export PREFILL_NUM_LAYERS=2;" in fixed


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
    (d / "rankfile").write_text("rank 1=h0 slot=0\nrank 0=h3 slot=0\n")
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
    feed = _inner(producer_cmd(REPO, "h0", "passthrough", 4, Path("/t"), duration_s=None, max_requests=3))
    assert f"export PREFILL_NUM_USERS={SLOTS};" in run and f"export PREFILL_NUM_USERS={SLOTS};" in feed
    # In flight on a quad: 8 forwards in the H2D FIFO, about two per stage in the D2D path, one being pushed.
    assert SLOTS * CHUNKS > 8 + 2 * 4 + 1
