# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein data parallel across processes: one Flux3KleinPipeline per group of a Blackhole Galaxy, (4, 2) by
default (DP4), a 16-chip (2, 8) / (4, 4) group (DP2) or a (2, 2) group (DP8). Each process sees only its chips (TT_VISIBLE_DEVICES), opens
them as one mesh (TT_MESH_GRAPH_DESC_PATH) and serves one group (GROUP_PCIE_IDS, the default) or two (4, 2) groups in
two threads (HALF_PCIE_IDS), so the groups' host dispatch does not share one Python GIL. With `depth` > 1 a group holds that
many requests and prepares the next while one runs (Flux3KleinPipeline.serve)."""

from __future__ import annotations

import math
import multiprocessing
import os
import queue
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from loguru import logger

GROUP_SHAPE = (4, 2)
# Each group shape's (sp_axis, tp_axis): the sequence runs on the group's ring axis. (2, 8) keeps the Galaxy's 8-chip
# ring (SP8 x TP2); (4, 4) is a 4x4 half (SP4 x TP4, the TP axis a line).
GROUP_AXES = {(4, 2): (0, 1), (2, 8): (1, 0), (4, 4): (0, 1), (2, 2): (0, 1)}
# Each group shape's longer axes that close no ring (the LINE dims of its mesh graph descriptor): their CCLs run as lines.
GROUP_LINE_AXES = {(4, 2): (), (2, 8): (), (4, 4): (1,), (2, 2): ()}
# DP8: (2, 2) groups, two per process on a 2 x 4 block (QUARTER_PCIE_IDS). No axis closes a ring: the block opens on
# FABRIC_1D and its pipelines run every CCL as a line (topology Linear).
LINE_GROUP_SHAPE = (2, 2)
# By process mesh shape: one (4, 2) group, two of them or one (4, 4) group, one (2, 8) group.
MESH_GRAPH_DESCRIPTORS = {
    shape: Path(__file__).with_name(f"dp_group_{shape[0]}x{shape[1]}_mesh_graph_descriptor.textproto")
    for shape in ((4, 2), (4, 4), (2, 8), (2, 4))
}

# The PCIe devices (/dev/tenstorrent/N) of the processes. One group each: FABRIC_1D_RING needs the 4-chip axis to close a
# ring, which the stride-4 columns do; the fourth (1, 5, ..., 29) does not, so its 8 chips stay idle.
GROUP_PCIE_IDS = tuple(tuple(range(first, 32, 4)) for first in (0, 2, 3))
# Two groups each: the two 4x4 halves. The (0, 1, 4, 5, ...) half closes no torus on its second axis, so these meshes run
# on FABRIC_2D_TORUS_Y, which every op of the pipeline must support.
HALF_PCIE_IDS = tuple(tuple(i for i in range(32) if i % 4 in columns) for columns in ((0, 1), (2, 3)))
# Two (2, 2) groups each: the four 2 x 4 blocks of the (4, 8) mesh (its submeshes' chips, job 231452).
QUARTER_PCIE_IDS = tuple(tuple(range(first, first + 8)) for first in range(0, 32, 8))


def process_shape(groups: int, group_shape: tuple[int, int] = GROUP_SHAPE) -> tuple[int, int]:
    return group_shape[0], group_shape[1] * groups


def process_groups(pcie_ids: Sequence[int], group_shape: tuple[int, int] = GROUP_SHAPE) -> int:
    groups, rest = divmod(len(set(pcie_ids)), math.prod(group_shape))
    if rest or len(set(pcie_ids)) != len(pcie_ids) or process_shape(groups, group_shape) not in MESH_GRAPH_DESCRIPTORS:
        raise ValueError(f"a process needs 8 or 16 distinct chips of whole {group_shape} groups, got {list(pcie_ids)}")
    return groups


def group_env(pcie_ids: Sequence[int], group_shape: tuple[int, int] = GROUP_SHAPE) -> dict[str, str]:
    """The variables that give a process only these chips, opened as one mesh of its groups side by side."""
    shape = process_shape(process_groups(pcie_ids, group_shape), group_shape)
    return {
        "TT_VISIBLE_DEVICES": ",".join(str(i) for i in sorted(pcie_ids)),
        "TT_MESH_GRAPH_DESC_PATH": str(MESH_GRAPH_DESCRIPTORS[shape]),
    }


def open_group_mesh(shape: tuple[int, int] = GROUP_SHAPE):
    """In a process: its mesh with 8 KB packets, as the one-process DP4. A 2-chip axis: FABRIC_1D_RING. (4, 4):
    FABRIC_2D_TORUS_Y, a ring on the 4-chip axis only."""
    import ttnn

    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    if shape == process_shape(2, LINE_GROUP_SHAPE):
        fabric = ttnn.FabricConfig.FABRIC_1D
    else:
        fabric = ttnn.FabricConfig.FABRIC_1D_RING if 2 in shape else ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    ttnn.set_fabric_config(fabric, router_config=router)
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=65536)


def _send_error(conns, e: Exception) -> None:
    for conn in conns:
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except OSError:
            pass


def _answer(conn, pipeline, depth: int = 1) -> None:
    def emit(answer) -> None:
        conn.send(("error", f"{type(answer).__name__}: {answer}") if isinstance(answer, Exception) else ("ok", answer))

    try:
        conn.send(("ready", None))
        if depth > 1:
            pipeline.serve(conn.recv, emit)
            return
        while (request := conn.recv()) is not None:
            try:
                answer = pipeline.generate(request)
            except Exception as e:  # noqa: BLE001
                answer = e
            emit(answer)
    finally:
        conn.close()  # a loop that dies (a request that does not unpickle) fails the parent's request instead of hanging it


def open_groups(
    groups: int,
    group_shape: tuple[int, int] = GROUP_SHAPE,
    sizes: Sequence[int] | None = None,
    **pipeline_kwargs,
):
    """In a process: its mesh and one pipeline per group, built one after another (they share the weight cache).
    Returns (mesh, pipelines); close each pipeline, then the mesh. Raises with nothing left open. Groups that serve
    more than one size (`sizes`, every size by default) free the other sizes' CCL pairs and plans on a size switch
    (free_other_sizes): a DP4 group holds one size's working set, not every size's (job 232721)."""
    import ttnn

    from .pipeline_flux3_klein import SIZES, Flux3KleinPipeline

    pipeline_kwargs.setdefault("free_other_sizes", len(set(SIZES if sizes is None else sizes)) > 1)
    mesh = open_group_mesh(process_shape(groups, group_shape))
    pipelines = []
    sp_axis, tp_axis = GROUP_AXES[group_shape]
    try:
        submeshes = mesh.create_submeshes(ttnn.MeshShape(*group_shape)) if groups > 1 else [mesh]
        for submesh in submeshes:
            pipelines.append(
                Flux3KleinPipeline.create(
                    submesh,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    axis_topologies={axis: ttnn.Topology.Linear for axis in GROUP_LINE_AXES[group_shape]},
                    **({"topology": ttnn.Topology.Linear} if group_shape == LINE_GROUP_SHAPE else {}),
                    **pipeline_kwargs,
                )
            )
    except BaseException:
        for pipeline in pipelines:
            pipeline.close()
        ttnn.close_mesh_device(mesh)
        raise
    return mesh, pipelines


def serve(
    conns: Sequence,
    pipeline_kwargs: Mapping,
    warmup: bool | Sequence[int],
    group_shape: tuple[int, int] = GROUP_SHAPE,
    depth: int = 1,
) -> None:
    """A process: one pipeline per group (`conns`, one per group), each warmed (every served shape, or `warmup` sizes);
    then each group answers its requests in its own thread until it gets None."""
    import ttnn

    try:
        sizes = None if isinstance(warmup, bool) else warmup
        mesh, pipelines = open_groups(len(conns), group_shape, sizes, **pipeline_kwargs)
    except Exception as e:  # noqa: BLE001
        _send_error(conns, e)
        return
    try:
        for pipeline in pipelines:
            if warmup is True:
                pipeline.warmup()
            elif warmup:
                pipeline.warmup(tuple(warmup))
        threads = [threading.Thread(target=_answer, args=(*pair, depth)) for pair in zip(conns, pipelines)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except Exception as e:  # noqa: BLE001
        _send_error(conns, e)
    finally:
        for pipeline in pipelines:
            pipeline.close()
        ttnn.close_mesh_device(mesh)


class Flux3KleinProcessGroups:
    """Warm pipeline processes of one or two groups each; each request goes to the next free group.

        with Flux3KleinProcessGroups() as groups:  # or Flux3KleinProcessGroups(HALF_PCIE_IDS)
            results = groups.generate(requests)

    DP2, one 16-chip group per process: Flux3KleinProcessGroups(sets, group_shape=(2, 8)), or (4, 4) on HALF_PCIE_IDS.

    A group whose process dies fails the requests it holds and is dropped (logged, `alive_groups` shrinks); once none
    is left, every request fails. A hung device hangs its request, as in one process.

    `depth` > 1: each group holds up to that many requests and answers them in order, preparing the next while one runs
    (throughput; a queued request waits for the one ahead of it)."""

    def __init__(
        self,
        processes: Iterable[Sequence[int]] = GROUP_PCIE_IDS,
        *,
        group_shape: tuple[int, int] = GROUP_SHAPE,
        warmup: bool | Sequence[int] = True,
        depth: int = 1,
        **pipeline_kwargs,
    ) -> None:
        processes = [tuple(p) for p in processes]
        used = [i for p in processes for i in p]
        if len(used) != len(set(used)):
            raise ValueError(f"processes share chips: {processes}")
        context = multiprocessing.get_context("spawn")
        self._processes, self._conns, self._names = [], [], {}
        self._lock = threading.Lock()
        try:
            # the JIT compiles on the executor's threads (one per host CPU by default): split them over the processes
            threads = os.environ.get("TT_METAL_EXECUTOR_THREADS") or str(
                max(1, (os.cpu_count() or 1) // len(processes))
            )
            for pcie_ids in processes:
                env = {**group_env(pcie_ids, group_shape), "TT_METAL_EXECUTOR_THREADS": threads}
                pipes = [context.Pipe() for _ in range(process_groups(pcie_ids, group_shape))]
                children = [child for _, child in pipes]
                process = context.Process(
                    target=serve, args=(children, pipeline_kwargs, warmup, tuple(group_shape), depth), daemon=True
                )
                saved = {key: os.environ.get(key) for key in env}
                os.environ.update(env)  # a spawned child starts with the parent's environment
                try:
                    process.start()
                finally:
                    for key, value in saved.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                self._processes.append(process)
                for i, (conn, child) in enumerate(pipes):
                    child.close()
                    self._conns.append(conn)
                    self._names[conn] = env["TT_VISIBLE_DEVICES"] + (f" group {i}" if len(pipes) > 1 else "")
            for conn in self._conns:
                self._receive(conn)
        except BaseException:
            self.close()
            raise
        self._alive = len(self._conns)
        self._dead, self._closing = set(), False
        self._waiting = {conn: deque() for conn in self._conns}
        self._send_locks = {conn: threading.Lock() for conn in self._conns}
        self._free = queue.SimpleQueue()
        for _ in range(depth):
            for conn in self._conns:
                self._free.put(conn)
        self._readers = [threading.Thread(target=self._read, args=(conn,), daemon=True) for conn in self._conns]
        for reader in self._readers:
            reader.start()
        self._pool = ThreadPoolExecutor(max_workers=len(self._conns) * depth, thread_name_prefix="flux3_klein_group")

    @property
    def alive_groups(self) -> int:
        return self._alive

    def submit(self, request) -> Future:
        return self._pool.submit(self._run, request)

    def generate(self, requests: Iterable) -> list:
        return [future.result() for future in [self.submit(request) for request in requests]]

    def _receive(self, conn):
        try:
            status, payload = conn.recv()
        except (EOFError, OSError) as e:
            raise RuntimeError(f"group process {self._names[conn]} died ({type(e).__name__})") from e
        if status == "error":
            raise RuntimeError(f"group process {self._names[conn]}: {payload}")
        return payload

    def _read(self, conn) -> None:
        """Answers this group's requests in the order they were sent."""
        while True:
            try:
                status, payload = conn.recv()
            except (EOFError, OSError) as e:
                if not self._closing:
                    self._drop(conn, RuntimeError(f"group process {self._names[conn]} died ({type(e).__name__})"))
                return
            future = self._waiting[conn].popleft()
            if status == "error":
                future.set_exception(RuntimeError(f"group process {self._names[conn]}: {payload}"))
            else:
                future.set_result(payload)

    def _run(self, request):
        conn = self._free.get()
        if conn is None:
            self._free.put(None)
            raise RuntimeError("no group process is alive")
        if conn in self._dead:
            return self._run(request)  # a dropped group's slot
        future = Future()
        with self._send_locks[conn]:
            self._waiting[conn].append(future)
            try:
                conn.send(request)
                delivered = True
            except OSError:
                self._waiting[conn].pop()
                delivered = False
            except BaseException:
                self._waiting[conn].pop()
                self._free.put(conn)  # the request did not pickle; nothing was written, the group is intact
                raise
        if not delivered:
            self._drop(conn, RuntimeError(f"group process {self._names[conn]} died"))
            return self._run(request)  # nothing was delivered: another group takes the request
        try:
            return future.result()
        finally:
            if conn not in self._dead:
                self._free.put(conn)

    def _drop(self, conn, error: Exception) -> None:
        with self._lock:
            if conn in self._dead:
                return
            self._dead.add(conn)
            self._alive -= 1
            alive = self._alive
        logger.error(f"FLUX 3 Klein group process {self._names[conn]} died; {alive} group(s) left")
        with self._send_locks[conn]:
            while self._waiting[conn]:
                self._waiting[conn].popleft().set_exception(error)
        if not alive:
            self._free.put(None)

    def close(self) -> None:
        """Stop every group. Waits for each process: a group still building its pipeline reaches recv and exits on
        None, so its mesh is closed."""
        if hasattr(self, "_pool"):
            self._pool.shutdown()
        self._closing = True
        for conn in self._conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for process in self._processes:
            process.join()
        self._processes, self._conns = [], []

    def __enter__(self) -> Flux3KleinProcessGroups:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
