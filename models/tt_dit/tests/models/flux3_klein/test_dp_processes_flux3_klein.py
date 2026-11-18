# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of dp_processes: the process chip sets, a process's environment, mesh and serve loop, and request
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


def test_halves_are_the_two_4x4_column_pairs():
    assert dp.HALF_PCIE_IDS == (
        (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29),
        (2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31),
    )


@pytest.mark.parametrize(("pcie_ids", "groups"), [(dp.GROUP_PCIE_IDS[0][::-1], 1), (dp.HALF_PCIE_IDS[1][::-1], 2)])
def test_group_env(pcie_ids, groups):
    env = dp.group_env(pcie_ids)
    assert env == {
        "TT_VISIBLE_DEVICES": ",".join(map(str, sorted(pcie_ids))),
        "TT_MESH_GRAPH_DESC_PATH": str(dp.MESH_GRAPH_DESCRIPTORS[groups]),
    }


@pytest.mark.parametrize("pcie_ids", [[0, 4, 8, 12, 16, 20, 24, 24], list(range(12)), list(range(24))])
def test_a_process_needs_8_or_16_distinct_chips(pcie_ids, expect_error):
    with expect_error(ValueError, "8 or 16 distinct chips"):
        dp.group_env(pcie_ids)


@pytest.mark.parametrize("groups", [1, 2])
def test_mesh_graph_descriptor_is_one_ring_line_mesh(groups):
    text = dp.MESH_GRAPH_DESCRIPTORS[groups].read_text()
    assert re.search(rf"dims: \[ 4, {2 * groups} \] dim_types: \[ RING, LINE \]", text)
    assert text.count("mesh_descriptors {") == 1
    assert re.search(r"top_level_instance \{ mesh \{ mesh_descriptor: \"M0\" mesh_id: 0 \} \}", text)


@pytest.mark.parametrize(("groups", "fabric"), [(1, "FABRIC_1D_RING"), (2, "FABRIC_2D_TORUS_Y")])
def test_process_mesh_is_opened_as_in_one_process(monkeypatch, groups, fabric):
    import ttnn

    calls = {}
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda config, **kwargs: calls.update(fabric=config, **kwargs))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape, **kwargs: calls.update(shape=shape, **kwargs) or "mesh")
    assert dp.open_group_mesh(groups) == "mesh"
    assert calls["fabric"] == getattr(ttnn.FabricConfig, fabric)
    assert calls["router_config"].max_packet_payload_size_bytes == 8192
    assert tuple(calls["shape"]) == (4, 2 * groups)
    assert calls["l1_small_size"] == 65536


class FakeMesh:
    def create_submeshes(self, shape):
        assert tuple(shape) == dp.GROUP_SHAPE
        return ["sub0", "sub1"]


@pytest.mark.parametrize(
    ("groups", "warmup", "sizes"), [(1, True, ()), (1, (1024,), ((1024,),)), (2, (1024,), ((1024,),))]
)
def test_serve_warms_every_group_before_ready_and_closes_the_mesh(monkeypatch, groups, warmup, sizes):
    import ttnn

    calls = []

    class FakePipeline:
        def __init__(self, mesh):
            self.mesh = mesh

        def warmup(self, *args):
            calls.append(("warmup", self.mesh, args))

        def generate(self, request):
            return f"{request} on {self.mesh}"

        def close(self):
            calls.append(("close", self.mesh))

    mesh = FakeMesh() if groups == 2 else "mesh"
    monkeypatch.setattr(dp, "open_group_mesh", lambda n: mesh if n == groups else None)
    monkeypatch.setattr(
        pl.Flux3KleinPipeline,
        "create",
        classmethod(lambda cls, m, **kw: calls.append(("create", m, kw)) or FakePipeline(m)),
    )
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda m: calls.append(("close_mesh", m)))
    pipes = [multiprocessing.Pipe() for _ in range(groups)]
    thread = threading.Thread(target=dp.serve, args=([c for _, c in pipes], {"upsample": False}, warmup), daemon=True)
    thread.start()
    assert [parent.recv() for parent, _ in pipes] == [("ready", None)] * groups
    meshes = ["sub0", "sub1"] if groups == 2 else ["mesh"]
    kw = {"sp_axis": 0, "tp_axis": 1, "upsample": False}
    assert calls == [c for m in meshes for c in (("create", m, kw), ("warmup", m, sizes))]
    for (parent, _), m in zip(pipes, meshes):
        parent.send("r")
        assert parent.recv() == ("ok", f"r on {m}")
    for parent, _ in pipes:
        parent.send(None)
    thread.join(5)
    assert calls[-groups - 1 :] == [("close", m) for m in meshes] + [("close_mesh", mesh)]


class ThreadProcess:
    """A process as a thread; keeps the environment it was started with."""

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


def fake_serve(conns, pipeline_kwargs, warmup):
    if pipeline_kwargs.get("fail_start"):
        for conn in conns:
            conn.send(("error", "RuntimeError: no mesh"))
        return

    def answer(conn, group):
        conn.send(("ready", None))
        try:
            while (request := conn.recv()) is not None:
                if request in ("die", "die_process"):
                    for c in conns if request == "die_process" else [conn]:
                        c.die()
                    return
                conn.send(
                    ("error", "ValueError: bad size") if request == "bad" else ("ok", (request, f"{id(conns)}/{group}"))
                )
        except (OSError, TypeError):  # a sibling thread closed this end ("die_process"); recv then reads a None handle
            pass

    threads = [threading.Thread(target=answer, args=(conn, i), daemon=True) for i, conn in enumerate(conns)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.fixture
def thread_groups(monkeypatch):
    monkeypatch.setattr(dp.multiprocessing, "get_context", lambda method: ThreadContext)
    monkeypatch.setattr(dp, "serve", fake_serve)


@pytest.fixture
def groups(thread_groups):
    with dp.Flux3KleinProcessGroups(warmup=(1024,)) as groups:
        yield groups


@pytest.fixture
def halves(thread_groups):
    with dp.Flux3KleinProcessGroups(dp.HALF_PCIE_IDS, warmup=(1024,)) as groups:
        yield groups


def test_each_process_sees_only_its_chips_and_gets_the_warmup(thread_groups, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "parent")
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    with dp.Flux3KleinProcessGroups(warmup=(1024,)) as groups:
        assert [p.env["TT_VISIBLE_DEVICES"] for p in groups._processes] == [
            ",".join(map(str, g)) for g in dp.GROUP_PCIE_IDS
        ]
        assert {p.env["TT_MESH_GRAPH_DESC_PATH"] for p in groups._processes} == {str(dp.MESH_GRAPH_DESCRIPTORS[1])}
        assert {p.args[2] for p in groups._processes} == {(1024,)}
    assert os.environ["TT_VISIBLE_DEVICES"] == "parent"
    assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


def test_two_processes_of_two_groups(halves):
    assert [p.env["TT_MESH_GRAPH_DESC_PATH"] for p in halves._processes] == [str(dp.MESH_GRAPH_DESCRIPTORS[2])] * 2
    assert [len(p.args[0]) for p in halves._processes] == [2, 2]
    assert halves.alive_groups == 4
    results = halves.generate([f"r{i}" for i in range(12)])
    assert [r for r, _ in results] == [f"r{i}" for i in range(12)]
    assert len({group for _, group in results}) == 4


def test_requests_are_answered_in_order_across_groups(groups):
    results = groups.generate([f"r{i}" for i in range(9)])
    assert [r for r, _ in results] == [f"r{i}" for i in range(9)]
    assert len({group for _, group in results}) > 1


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
    assert len({group for _, group in results}) == 2


def test_a_dead_two_group_process_drops_both_groups_and_the_other_process_serves(halves, expect_error):
    with expect_error(RuntimeError, "died"):
        halves.submit("die_process").result(timeout=10)
    results = []
    for i in range(8):
        try:
            results.append(halves.submit(f"r{i}").result(timeout=10))
        except RuntimeError as e:
            assert "died" in str(e)
    assert halves.alive_groups == 2
    assert len({group for _, group in results}) == 2
    assert [r for r, _ in halves.generate(["a", "b", "c", "d"])] == ["a", "b", "c", "d"]


def test_requests_fail_instead_of_hanging_once_every_group_is_dead(groups, expect_error):
    for _ in range(3):
        with expect_error(RuntimeError, "died"):
            groups.submit("die").result(timeout=10)
    assert groups.alive_groups == 0
    with expect_error(RuntimeError, "no group process is alive"):
        groups.submit("a").result(timeout=10)


def test_close_stops_every_process(thread_groups):
    for layout in (dp.GROUP_PCIE_IDS, dp.HALF_PCIE_IDS):
        groups = dp.Flux3KleinProcessGroups(layout)
        processes = list(groups._processes)
        groups.close()
        assert not any(p.is_alive() for p in processes)


def test_a_process_that_fails_to_start_fails_the_constructor(thread_groups, expect_error):
    with expect_error(RuntimeError, "no mesh"):
        dp.Flux3KleinProcessGroups(dp.HALF_PCIE_IDS, fail_start=True)


def test_processes_may_not_share_chips(expect_error):
    with expect_error(ValueError, "share chips"):
        dp.Flux3KleinProcessGroups([dp.GROUP_PCIE_IDS[0], dp.HALF_PCIE_IDS[0]])
