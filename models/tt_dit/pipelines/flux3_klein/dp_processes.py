# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein data parallel across processes: one Flux3KleinPipeline per (4, 2) group of a Blackhole Galaxy, each in
its own process that sees only its group's chips (TT_VISIBLE_DEVICES) and opens them as one mesh
(TT_MESH_GRAPH_DESC_PATH), so the groups' host dispatch does not share one Python GIL."""

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
MESH_GRAPH_DESCRIPTOR = Path(__file__).with_name("dp_group_4x2_mesh_graph_descriptor.textproto")

# The PCIe devices (/dev/tenstorrent/N) of the groups that open as their own mesh: FABRIC_1D_RING needs the 4-chip axis to
# close a ring, which the stride-4 columns do. The fourth (1, 5, ..., 29) does not, so its 8 chips stay idle.
GROUP_PCIE_IDS = tuple(tuple(range(first, 32, 4)) for first in (0, 2, 3))


def group_env(pcie_ids: Sequence[int]) -> dict[str, str]:
    """The variables that give a process only these chips, opened as one (4, 2) mesh."""
    if len(set(pcie_ids)) != GROUP_SHAPE[0] * GROUP_SHAPE[1]:
        raise ValueError(f"a group needs {GROUP_SHAPE[0] * GROUP_SHAPE[1]} distinct chips, got {list(pcie_ids)}")
    return {
        "TT_VISIBLE_DEVICES": ",".join(str(i) for i in sorted(pcie_ids)),
        "TT_MESH_GRAPH_DESC_PATH": str(MESH_GRAPH_DESCRIPTOR),
    }


def open_group_mesh():
    """In a group's process: its (4, 2) mesh on FABRIC_1D_RING with 8 KB packets, as the one-process DP4 (the fused
    all-gather matmul takes only 1D packet headers)."""
    import ttnn

    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
    return ttnn.open_mesh_device(ttnn.MeshShape(*GROUP_SHAPE), l1_small_size=65536)


def serve(conn, pipeline_kwargs: Mapping, warmup: bool | Sequence[int]) -> None:
    """A group process: build the pipeline, warm it (every served shape, or `warmup` sizes), then answer each request
    from `conn` until it sends None."""
    import ttnn

    from .pipeline_flux3_klein import Flux3KleinPipeline

    try:
        mesh = open_group_mesh()
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    pipeline = None
    try:
        pipeline = Flux3KleinPipeline.create(mesh, sp_axis=0, tp_axis=1, **pipeline_kwargs)
        if warmup is True:
            pipeline.warmup()
        elif warmup:
            pipeline.warmup(tuple(warmup))
        conn.send(("ready", None))
        while (request := conn.recv()) is not None:
            try:
                conn.send(("ok", pipeline.generate(request)))
            except Exception as e:  # noqa: BLE001
                conn.send(("error", f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        if pipeline is not None:
            pipeline.close()
        ttnn.close_mesh_device(mesh)


class Flux3KleinProcessGroups:
    """One warm pipeline process per group; each request goes to the next free group.

        with Flux3KleinProcessGroups() as groups:
            results = groups.generate(requests)

    A group whose process dies fails the request it holds and is dropped (logged, `alive_groups` shrinks); once none
    is left, every request fails. A hung device hangs its request, as in one process."""

    def __init__(
        self,
        groups: Iterable[Sequence[int]] = GROUP_PCIE_IDS,
        *,
        warmup: bool | Sequence[int] = True,
        **pipeline_kwargs,
    ) -> None:
        groups = [tuple(g) for g in groups]
        used = [i for g in groups for i in g]
        if len(used) != len(set(used)):
            raise ValueError(f"groups share chips: {groups}")
        context = multiprocessing.get_context("spawn")
        self._processes, self._conns, self._names = [], [], {}
        self._lock = threading.Lock()
        try:
            for pcie_ids in groups:
                conn, child = context.Pipe()
                process = context.Process(target=serve, args=(child, pipeline_kwargs, warmup), daemon=True)
                env = group_env(pcie_ids)
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
                child.close()
                self._processes.append(process)
                self._conns.append(conn)
                self._names[conn] = env["TT_VISIBLE_DEVICES"]
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
        except OSError as e:
            self._drop(conn)
            raise RuntimeError(f"group process {self._names[conn]} died ({type(e).__name__})") from e
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
