# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of dp_processes: the process chip sets, a process's environment, mesh and serve loop, and request
dispatch over group processes (threads stand in for the processes). No device."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import re
import socket
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


@pytest.mark.parametrize(
    ("pcie_ids", "group_shape", "shape"),
    [
        (dp.GROUP_PCIE_IDS[0][::-1], (4, 2), (4, 2)),
        (dp.HALF_PCIE_IDS[1][::-1], (4, 2), (4, 4)),
        (dp.HALF_PCIE_IDS[1], (4, 4), (4, 4)),
        (dp.HALF_PCIE_IDS[0], (2, 8), (2, 8)),
    ],
)
def test_group_env(pcie_ids, group_shape, shape):
    env = dp.group_env(pcie_ids, group_shape)
    assert env == {
        "TT_VISIBLE_DEVICES": ",".join(map(str, sorted(pcie_ids))),
        "TT_MESH_GRAPH_DESC_PATH": str(dp.MESH_GRAPH_DESCRIPTORS[shape]),
    }


@pytest.mark.parametrize(
    ("pcie_ids", "group_shape"),
    [
        ([0, 4, 8, 12, 16, 20, 24, 24], (4, 2)),
        (list(range(12)), (4, 2)),
        (list(range(24)), (4, 2)),
        (list(range(8)), (2, 8)),
        (list(range(32)), (2, 8)),
        (list(range(32)), (4, 4)),
    ],
)
def test_a_process_needs_8_or_16_distinct_chips(pcie_ids, group_shape, expect_error):
    with expect_error(ValueError, "8 or 16 distinct chips"):
        dp.group_env(pcie_ids, group_shape)


@pytest.mark.parametrize(("shape", "types"), [((4, 2), "RING, LINE"), ((4, 4), "RING, LINE"), ((2, 8), "LINE, RING")])
def test_mesh_graph_descriptor_is_one_ring_line_mesh(shape, types):
    text = dp.MESH_GRAPH_DESCRIPTORS[shape].read_text()
    assert re.search(rf"dims: \[ {shape[0]}, {shape[1]} \] dim_types: \[ {types} \]", text)
    assert text.count("mesh_descriptors {") == 1
    assert re.search(r"top_level_instance \{ mesh \{ mesh_descriptor: \"M0\" mesh_id: 0 \} \}", text)


def test_a_group_runs_the_sequence_on_its_ring_axis():
    for shape, (sp_axis, tp_axis) in dp.GROUP_AXES.items():
        types = re.search(r"dim_types: \[ (\w+), (\w+) \]", dp.MESH_GRAPH_DESCRIPTORS[shape].read_text()).groups()
        assert (types[sp_axis], types[tp_axis]) == ("RING", "LINE")


def test_line_axes_are_the_descriptor_lines_longer_than_2_chips():
    for shape, line_axes in dp.GROUP_LINE_AXES.items():
        types = re.search(r"dim_types: \[ (\w+), (\w+) \]", dp.MESH_GRAPH_DESCRIPTORS[shape].read_text()).groups()
        assert line_axes == tuple(axis for axis, kind in enumerate(types) if kind == "LINE" and shape[axis] > 2)


@pytest.mark.parametrize(
    ("shape", "fabric"), [((4, 2), "FABRIC_1D_RING"), ((2, 8), "FABRIC_1D_RING"), ((4, 4), "FABRIC_2D_TORUS_Y")]
)
def test_process_mesh_is_opened_as_in_one_process(monkeypatch, shape, fabric):
    import ttnn

    calls = {}
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda config, **kwargs: calls.update(fabric=config, **kwargs))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape, **kwargs: calls.update(shape=shape, **kwargs) or "mesh")
    assert dp.open_group_mesh(shape) == "mesh"
    assert calls["fabric"] == getattr(ttnn.FabricConfig, fabric)
    assert calls["router_config"].max_packet_payload_size_bytes == 8192
    assert tuple(calls["shape"]) == shape
    assert calls["l1_small_size"] == 65536


class FakeMesh:
    def create_submeshes(self, shape):
        assert tuple(shape) == dp.GROUP_SHAPE
        return ["sub0", "sub1"]


@pytest.mark.parametrize(
    ("groups", "group_shape", "warmup", "sizes"),
    [
        (1, (4, 2), True, ()),
        (1, (4, 2), (1024,), ((1024,),)),
        (2, (4, 2), (1024,), ((1024,),)),
        (1, (2, 8), (1024,), ((1024,),)),
        (1, (4, 4), (1024,), ((1024,),)),
    ],
)
def test_serve_warms_every_group_before_ready_and_closes_the_mesh(monkeypatch, groups, group_shape, warmup, sizes):
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
    shape = (group_shape[0], group_shape[1] * groups)
    monkeypatch.setattr(dp, "open_group_mesh", lambda s: mesh if s == shape else None)
    monkeypatch.setattr(
        pl.Flux3KleinPipeline,
        "create",
        classmethod(lambda cls, m, **kw: calls.append(("create", m, kw)) or FakePipeline(m)),
    )
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda m: calls.append(("close_mesh", m)))
    pipes = [multiprocessing.Pipe() for _ in range(groups)]
    thread = threading.Thread(
        target=dp.serve, args=([c for _, c in pipes], {"upsample": False}, warmup, group_shape), daemon=True
    )
    thread.start()
    assert [parent.recv() for parent, _ in pipes] == [("ready", None)] * groups
    meshes = ["sub0", "sub1"] if groups == 2 else ["mesh"]
    sp_axis, tp_axis = (1, 0) if group_shape == (2, 8) else (0, 1)
    axis_topologies = {1: ttnn.Topology.Linear} if group_shape == (4, 4) else {}
    kw = {"sp_axis": sp_axis, "tp_axis": tp_axis, "axis_topologies": axis_topologies, "upsample": False}
    assert calls == [c for m in meshes for c in (("create", m, kw), ("warmup", m, sizes))]
    for (parent, _), m in zip(pipes, meshes):
        parent.send("r")
        assert parent.recv() == ("ok", f"r on {m}")
    for parent, _ in pipes:
        parent.send(None)
    thread.join(5)
    assert calls[-groups - 1 :] == [("close", m) for m in meshes] + [("close_mesh", mesh)]


def _explode():
    raise ValueError("cannot rebuild")


class Unpicklable:
    def __reduce__(self):
        return (_explode, ())


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)  # the loop's error, logged by the thread
def test_a_request_loop_that_dies_closes_its_pipe(monkeypatch, expect_error):
    import ttnn

    class FakePipeline:
        def generate(self, request):
            return request

        def close(self):
            pass

    monkeypatch.setattr(dp, "open_group_mesh", lambda shape: "mesh")
    monkeypatch.setattr(pl.Flux3KleinPipeline, "create", classmethod(lambda cls, m, **kw: FakePipeline()))
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda m: None)
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(target=dp.serve, args=([child], {}, False), daemon=True)
    thread.start()
    assert parent.recv() == ("ready", None)
    parent.send(Unpicklable())  # pickles here, fails to unpickle in the request loop
    assert parent.poll(5)
    with expect_error(EOFError, "^$"):
        parent.recv()
    thread.join(5)
    assert not thread.is_alive()


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
    """A child pipe end the parent's close() leaves open (a thread shares it, a process has its own copy); die() shuts it
    down as a crashed process's exit would (a plain close leaves the socket open under a recv blocked in another thread).
    """

    def __init__(self, conn):
        self._conn = conn
        self.send, self.recv = conn.send, conn.recv

    def die(self):
        with socket.socket(fileno=os.dup(self._conn.fileno())) as end:
            end.shutdown(socket.SHUT_RDWR)
        self._conn.close()

    def close(self):
        pass


class ThreadContext:
    Process = ThreadProcess

    @staticmethod
    def Pipe():
        parent, child = multiprocessing.Pipe()
        return parent, ThreadEnd(child)


def fake_serve(conns, pipeline_kwargs, warmup, group_shape):
    if pipeline_kwargs.get("fail_start"):
        for conn in conns:
            conn.send(("error", "RuntimeError: no mesh"))
        return

    def answer(conn, group):
        conn.send(("ready", None))
        try:
            while (request := conn.recv()) is not None:
                if request in ("die", "die_process"):
                    for c in [c for c in conns if c is not conn] if request == "die_process" else []:
                        c.die()
                    conn.die()  # last, so the parent sees this EOF only after the siblings are closed
                    return
                conn.send(
                    ("error", "ValueError: bad size") if request == "bad" else ("ok", (request, f"{id(conns)}/{group}"))
                )
        except (EOFError, OSError, TypeError):  # a sibling thread shut this end down ("die_process")
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
        assert {p.env["TT_MESH_GRAPH_DESC_PATH"] for p in groups._processes} == {str(dp.MESH_GRAPH_DESCRIPTORS[(4, 2)])}
        assert {p.args[2] for p in groups._processes} == {(1024,)}
    assert os.environ["TT_VISIBLE_DEVICES"] == "parent"
    assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


def test_two_processes_of_two_groups(halves):
    assert [p.env["TT_MESH_GRAPH_DESC_PATH"] for p in halves._processes] == [str(dp.MESH_GRAPH_DESCRIPTORS[(4, 4)])] * 2
    assert [len(p.args[0]) for p in halves._processes] == [2, 2]
    assert halves.alive_groups == 4
    results = halves.generate([f"r{i}" for i in range(12)])
    assert [r for r, _ in results] == [f"r{i}" for i in range(12)]
    assert len({group for _, group in results}) == 4


@pytest.mark.parametrize("group_shape", [(2, 8), (4, 4)])
def test_two_processes_of_one_16_chip_group(thread_groups, group_shape):
    with dp.Flux3KleinProcessGroups(dp.HALF_PCIE_IDS, group_shape=group_shape, warmup=(1024,)) as dp2:
        assert [p.env["TT_MESH_GRAPH_DESC_PATH"] for p in dp2._processes] == [
            str(dp.MESH_GRAPH_DESCRIPTORS[group_shape])
        ] * 2
        assert [(len(p.args[0]), p.args[3]) for p in dp2._processes] == [(1, group_shape)] * 2
        assert dp2.alive_groups == 2
        results = dp2.generate([f"r{i}" for i in range(6)])
        assert [r for r, _ in results] == [f"r{i}" for i in range(6)]
        assert len({group for _, group in results}) == 2


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


def test_a_dead_two_group_process_fails_only_the_request_it_held(halves, expect_error):
    with expect_error(RuntimeError, "died"):
        halves.submit("die_process").result(timeout=10)
    results = [
        halves.submit(f"r{i}").result(timeout=10) for i in range(8)
    ]  # the dead idle sibling's turn goes elsewhere
    assert [r for r, _ in results] == [f"r{i}" for i in range(8)]
    assert halves.alive_groups == 2
    assert len({group for _, group in results}) == 2


def test_a_request_that_does_not_pickle_keeps_its_group(groups, expect_error):
    with expect_error((AttributeError, pickle.PicklingError), "pickle"):
        groups.submit(lambda: None).result(timeout=10)
    assert groups.alive_groups == 3
    results = groups.generate([f"r{i}" for i in range(9)])
    assert len({group for _, group in results}) == 3


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
