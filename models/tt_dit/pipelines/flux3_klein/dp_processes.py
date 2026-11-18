# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein data parallel across processes: one Flux3KleinPipeline per (4, 2) group of a Blackhole Galaxy. Each
process sees only its chips (TT_VISIBLE_DEVICES), opens them as one mesh (TT_MESH_GRAPH_DESC_PATH) and serves one group
(GROUP_PCIE_IDS, the default) or two groups in two threads (HALF_PCIE_IDS), so the groups' host dispatch does not share one
Python GIL."""

from __future__ import annotations

import multiprocessing
import os
import queue
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from loguru import logger

GROUP_SHAPE = (4, 2)
GROUP_CHIPS = GROUP_SHAPE[0] * GROUP_SHAPE[1]
MESH_GRAPH_DESCRIPTORS = {
    groups: Path(__file__).with_name(f"dp_group_4x{2 * groups}_mesh_graph_descriptor.textproto") for groups in (1, 2)
}

# The PCIe devices (/dev/tenstorrent/N) of the processes. One group each: FABRIC_1D_RING needs the 4-chip axis to close a
# ring, which the stride-4 columns do; the fourth (1, 5, ..., 29) does not, so its 8 chips stay idle.
GROUP_PCIE_IDS = tuple(tuple(range(first, 32, 4)) for first in (0, 2, 3))
# Two groups each: the two 4x4 halves. The (0, 1, 4, 5, ...) half closes no torus on its second axis, so these meshes run
# on FABRIC_2D_TORUS_Y, which every op of the pipeline must support.
HALF_PCIE_IDS = tuple(tuple(i for i in range(32) if i % 4 in columns) for columns in ((0, 1), (2, 3)))


def process_groups(pcie_ids: Sequence[int]) -> int:
    groups, rest = divmod(len(set(pcie_ids)), GROUP_CHIPS)
    if rest or len(set(pcie_ids)) != len(pcie_ids) or groups not in MESH_GRAPH_DESCRIPTORS:
        raise ValueError(f"a process needs 8 or 16 distinct chips, got {list(pcie_ids)}")
    return groups


def group_env(pcie_ids: Sequence[int]) -> dict[str, str]:
    """The variables that give a process only these chips, opened as one (4, 2 x groups) mesh."""
    return {
        "TT_VISIBLE_DEVICES": ",".join(str(i) for i in sorted(pcie_ids)),
        "TT_MESH_GRAPH_DESC_PATH": str(MESH_GRAPH_DESCRIPTORS[process_groups(pcie_ids)]),
    }


def open_group_mesh(groups: int = 1):
    """In a process: its (4, 2 x groups) mesh with 8 KB packets, as the one-process DP4. One group: FABRIC_1D_RING. Two:
    FABRIC_2D_TORUS_Y, a ring on the 4-chip axis only."""
    import ttnn

    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if groups == 1 else ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    ttnn.set_fabric_config(fabric, router_config=router)
    return ttnn.open_mesh_device(ttnn.MeshShape(GROUP_SHAPE[0], GROUP_SHAPE[1] * groups), l1_small_size=65536)


def _send_error(conns, e: Exception) -> None:
    for conn in conns:
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except OSError:
            pass


def _answer(conn, pipeline) -> None:
    try:
        conn.send(("ready", None))
        while (request := conn.recv()) is not None:
            try:
                conn.send(("ok", pipeline.generate(request)))
            except Exception as e:  # noqa: BLE001
                conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()  # a loop that dies (a request that does not unpickle) fails the parent's request instead of hanging it


def serve(conns: Sequence, pipeline_kwargs: Mapping, warmup: bool | Sequence[int]) -> None:
    """A process: one pipeline per group (`conns`, one per group), each warmed (every served shape, or `warmup` sizes);
    then each group answers its requests in its own thread until it gets None."""
    import ttnn

    from .pipeline_flux3_klein import Flux3KleinPipeline

    try:
        mesh = open_group_mesh(len(conns))
    except Exception as e:  # noqa: BLE001
        _send_error(conns, e)
        return
    pipelines = []
    try:
        submeshes = mesh.create_submeshes(ttnn.MeshShape(*GROUP_SHAPE)) if len(conns) > 1 else [mesh]
        for submesh in submeshes:
            pipelines.append(Flux3KleinPipeline.create(submesh, sp_axis=0, tp_axis=1, **pipeline_kwargs))
            if warmup is True:
                pipelines[-1].warmup()
            elif warmup:
                pipelines[-1].warmup(tuple(warmup))
        threads = [threading.Thread(target=_answer, args=pair) for pair in zip(conns, pipelines)]
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

    A group whose process dies fails the request it holds and is dropped (logged, `alive_groups` shrinks); once none
    is left, every request fails. A hung device hangs its request, as in one process."""

    def __init__(
        self,
        processes: Iterable[Sequence[int]] = GROUP_PCIE_IDS,
        *,
        warmup: bool | Sequence[int] = True,
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
            for pcie_ids in processes:
                env = group_env(pcie_ids)
                pipes = [context.Pipe() for _ in range(process_groups(pcie_ids))]
                children = [child for _, child in pipes]
                process = context.Process(target=serve, args=(children, pipeline_kwargs, warmup), daemon=True)
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
        self._free = queue.SimpleQueue()
        for conn in self._conns:
            self._free.put(conn)
        self._pool = ThreadPoolExecutor(max_workers=len(self._conns), thread_name_prefix="flux3_klein_group")

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

    def _run(self, request):
        conn = self._free.get()
        if conn is None:
            self._free.put(None)
            raise RuntimeError("no group process is alive")
        try:
            conn.send(request)
        except OSError:
            self._drop(conn)  # nothing was delivered: another group takes the request
            return self._run(request)
        except BaseException:
            self._free.put(conn)  # the request did not pickle; nothing was written, the group is intact
            raise
        try:
            result = self._receive(conn)
        except RuntimeError as e:
            if isinstance(e.__cause__, (EOFError, OSError)):
                self._drop(conn)
            else:
                self._free.put(conn)
            raise
        self._free.put(conn)
        return result

    def _drop(self, conn) -> None:
        with self._lock:
            self._alive -= 1
            alive = self._alive
        logger.error(f"FLUX 3 Klein group process {self._names[conn]} died; {alive} group(s) left")
        if not alive:
            self._free.put(None)

    def close(self) -> None:
        """Stop every group. Waits for each process: a group still building its pipeline reaches recv and exits on
        None, so its mesh is closed."""
        if hasattr(self, "_pool"):
            self._pool.shutdown()
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
