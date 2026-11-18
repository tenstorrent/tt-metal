# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of dp_processes: the group chip sets, a group process's environment, mesh and serve loop, and request
dispatch over group processes (threads stand in for the processes). No device."""

from __future__ import annotations

import multiprocessing
import os
import re
import threading

import pytest

from ....pipelines.flux3_klein import dp_processes as dp
from ....pipelines.flux3_klein import pipeline_flux3_klein as pl


def test_groups_are_the_disjoint_stride_4_columns():
    assert len(dp.GROUP_PCIE_IDS) == 3
    assert all(len(g) == 8 and all(b - a == 4 for a, b in zip(g, g[1:])) for g in dp.GROUP_PCIE_IDS)
    ids = [i for g in dp.GROUP_PCIE_IDS for i in g]
    assert len(ids) == len(set(ids)) == 24
    assert not set(range(1, 32, 4)) & set(ids)


def test_group_env():
    env = dp.group_env([28, 0, 4, 8, 12, 16, 20, 24])
    assert env == {
        "TT_VISIBLE_DEVICES": "0,4,8,12,16,20,24,28",
        "TT_MESH_GRAPH_DESC_PATH": str(dp.MESH_GRAPH_DESCRIPTOR),
    }


def test_group_env_needs_8_distinct_chips(expect_error):
    with expect_error(ValueError, "8 distinct chips"):
        dp.group_env([0, 4, 8, 12, 16, 20, 24, 24])


def test_mesh_graph_descriptor_is_one_4x2_ring_line_mesh():
    text = dp.MESH_GRAPH_DESCRIPTOR.read_text()
    assert re.search(r"dims: \[ 4, 2 \] dim_types: \[ RING, LINE \]", text)
    assert text.count("mesh_descriptors {") == 1
    assert re.search(r"top_level_instance \{ mesh \{ mesh_descriptor: \"M0\" mesh_id: 0 \} \}", text)


def test_group_mesh_is_opened_as_in_one_process(monkeypatch):
    import ttnn

    calls = {}
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda config, **kwargs: calls.update(fabric=config, **kwargs))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape, **kwargs: calls.update(shape=shape, **kwargs) or "mesh")
    assert dp.open_group_mesh() == "mesh"
    assert calls["fabric"] == ttnn.FabricConfig.FABRIC_1D_RING
    assert calls["router_config"].max_packet_payload_size_bytes == 8192
    assert tuple(calls["shape"]) == dp.GROUP_SHAPE
    assert calls["l1_small_size"] == 65536


@pytest.mark.parametrize(("warmup", "sizes"), [(True, ()), ((1024,), ((1024,),))])
def test_serve_warms_before_ready_and_closes_the_mesh(monkeypatch, warmup, sizes):
    import ttnn

    calls = []

    class FakePipeline:
        def warmup(self, *args):
            calls.append(("warmup", args))

        def generate(self, request):
            return f"image {request}"

        def close(self):
            calls.append("close")

    monkeypatch.setattr(dp, "open_group_mesh", lambda: "mesh")
    monkeypatch.setattr(
        pl.Flux3KleinPipeline,
        "create",
        classmethod(lambda cls, mesh, **kw: calls.append(("create", kw)) or FakePipeline()),
    )
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda mesh: calls.append(("close_mesh", mesh)))
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(target=dp.serve, args=(child, {"upsample": False}, warmup), daemon=True)
    thread.start()
    assert parent.recv() == ("ready", None)
    assert calls == [("create", {"sp_axis": 0, "tp_axis": 1, "upsample": False}), ("warmup", sizes)]
    parent.send("r")
    assert parent.recv() == ("ok", "image r")
    parent.send(None)
    thread.join(5)
    assert calls[-2:] == ["close", ("close_mesh", "mesh")]


class ThreadProcess:
    """A group process as a thread; keeps the environment it was started with."""

    def __init__(self, target, args, daemon):
        self.args = args
        self._thread = threading.Thread(target=target, args=args, daemon=daemon)

    def start(self):
        self.env = {key: os.environ.get(key) for key in ("TT_VISIBLE_DEVICES", "TT_MESH_GRAPH_DESC_PATH")}
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def is_alive(self):
        return self._thread.is_alive()


class ThreadEnd:
    """A child pipe end the parent's close() leaves open (a thread shares it, a process has its own copy); die() closes it
    as a crashed process would."""

    def __init__(self, conn):
        self.send, self.recv, self.die = conn.send, conn.recv, conn.close

    def close(self):
        pass


class ThreadContext:
    Process = ThreadProcess

    @staticmethod
    def Pipe():
        parent, child = multiprocessing.Pipe()
        return parent, ThreadEnd(child)


def fake_serve(conn, pipeline_kwargs, warmup):
    if pipeline_kwargs.get("fail_start"):
        conn.send(("error", "RuntimeError: no mesh"))
        return
    conn.send(("ready", None))
    name = threading.current_thread().name
    while (request := conn.recv()) is not None:
        if request == "die":
            conn.die()
            return
        conn.send(("error", "ValueError: bad size") if request == "bad" else ("ok", (request, name)))


@pytest.fixture
def thread_groups(monkeypatch):
    monkeypatch.setattr(dp.multiprocessing, "get_context", lambda method: ThreadContext)
    monkeypatch.setattr(dp, "serve", fake_serve)


@pytest.fixture
def groups(thread_groups):
    with dp.Flux3KleinProcessGroups(warmup=(1024,)) as groups:
        yield groups


def test_each_group_process_sees_only_its_chips_and_gets_the_warmup(thread_groups, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "parent")
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    with dp.Flux3KleinProcessGroups(warmup=(1024,)) as groups:
        assert [p.env["TT_VISIBLE_DEVICES"] for p in groups._processes] == [
            ",".join(map(str, g)) for g in dp.GROUP_PCIE_IDS
        ]
        assert {p.env["TT_MESH_GRAPH_DESC_PATH"] for p in groups._processes} == {str(dp.MESH_GRAPH_DESCRIPTOR)}
        assert {p.args[2] for p in groups._processes} == {(1024,)}
    assert os.environ["TT_VISIBLE_DEVICES"] == "parent"
    assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


def test_requests_are_answered_in_order_across_groups(groups):
    results = groups.generate([f"r{i}" for i in range(9)])
    assert [r for r, _ in results] == [f"r{i}" for i in range(9)]
    assert len({worker for _, worker in results}) > 1


def test_a_failed_request_raises_and_the_group_keeps_serving(groups, expect_error):
    with expect_error(RuntimeError, "bad size"):
        groups.submit("bad").result()
    assert groups.alive_groups == 3
    assert [r for r, _ in groups.generate(["a", "b", "c", "d"])] == ["a", "b", "c", "d"]


def test_a_dead_group_fails_its_request_and_is_dropped(groups, expect_error):
    with expect_error(RuntimeError, "died"):
        groups.submit("die").result(timeout=10)
    assert groups.alive_groups == 2
    results = groups.generate([f"r{i}" for i in range(8)])
    assert [r for r, _ in results] == [f"r{i}" for i in range(8)]
    assert len({worker for _, worker in results}) == 2


def test_requests_fail_instead_of_hanging_once_every_group_is_dead(groups, expect_error):
    for _ in range(3):
        with expect_error(RuntimeError, "died"):
            groups.submit("die").result(timeout=10)
    assert groups.alive_groups == 0
    with expect_error(RuntimeError, "no group process is alive"):
        groups.submit("a").result(timeout=10)


def test_close_stops_every_group_process(thread_groups):
    groups = dp.Flux3KleinProcessGroups()
    processes = list(groups._processes)
    groups.close()
    assert not any(p.is_alive() for p in processes)


def test_a_group_that_fails_to_start_fails_the_constructor(thread_groups, expect_error):
    with expect_error(RuntimeError, "no mesh"):
        dp.Flux3KleinProcessGroups(fail_start=True)


def test_groups_may_not_share_chips(expect_error):
    with expect_error(ValueError, "share chips"):
        dp.Flux3KleinProcessGroups([dp.GROUP_PCIE_IDS[0], dp.GROUP_PCIE_IDS[0]])
