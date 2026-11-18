# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein data parallel across processes: one Flux3KleinPipeline per (4, 2) group of a Blackhole Galaxy, each in
its own process that sees only its group's chips (TT_VISIBLE_DEVICES) and opens them as one mesh
(TT_MESH_GRAPH_DESC_PATH). The groups' host dispatch then does not share one Python GIL: at 1024 one process with four
groups serves 1.70 img/s, three group processes 2.18 img/s at the single-group latency."""

from __future__ import annotations

import multiprocessing
import os
import queue
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

GROUP_SHAPE = (4, 2)
MESH_GRAPH_DESCRIPTOR = Path(__file__).with_name("dp_group_4x2_mesh_graph_descriptor.textproto")

# The PCIe devices (/dev/tenstorrent/N) of the groups that open as their own mesh: FABRIC_1D_RING needs the 4-chip axis to
# close a ring, which the stride-4 columns do. The fourth (1, 5, ..., 29) does not.
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


def serve(conn, pipeline_kwargs: Mapping) -> None:
    """A group process: build the pipeline, then answer each request from `conn` until it sends None."""
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


def _receive(conn):
    status, payload = conn.recv()
    if status == "error":
        raise RuntimeError(f"group process: {payload}")
    return payload


class Flux3KleinProcessGroups:
    """One pipeline process per group; each request goes to the next free group.

    with Flux3KleinProcessGroups() as groups:
        results = groups.generate(requests)
    """

    def __init__(self, groups: Iterable[Sequence[int]] = GROUP_PCIE_IDS, **pipeline_kwargs) -> None:
        groups = [tuple(g) for g in groups]
        used = [i for g in groups for i in g]
        if len(used) != len(set(used)):
            raise ValueError(f"groups share chips: {groups}")
        context = multiprocessing.get_context("spawn")
        self._processes, self._conns = [], []
        try:
            for pcie_ids in groups:
                conn, child = context.Pipe()
                process = context.Process(target=serve, args=(child, pipeline_kwargs), daemon=True)
                saved = dict(os.environ)
                os.environ.update(group_env(pcie_ids))  # a spawned child starts with the parent's environment
                try:
                    process.start()
                finally:
                    os.environ.clear()
                    os.environ.update(saved)
                child.close()
                self._processes.append(process)
                self._conns.append(conn)
            for conn in self._conns:
                _receive(conn)
        except BaseException:
            self.close()
            raise
        self._free = queue.SimpleQueue()
        for conn in self._conns:
            self._free.put(conn)
        self._pool = ThreadPoolExecutor(max_workers=len(self._conns), thread_name_prefix="flux3_klein_group")

    def submit(self, request) -> Future:
        return self._pool.submit(self._run, request)

    def generate(self, requests: Iterable) -> list:
        return [future.result() for future in [self.submit(request) for request in requests]]

    def _run(self, request):
        conn = self._free.get()
        try:
            conn.send(request)
            return _receive(conn)
        finally:
            self._free.put(conn)

    def close(self) -> None:
        if hasattr(self, "_pool"):
            self._pool.shutdown()
        for conn in self._conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for process in self._processes:
            process.join(timeout=60)
            if process.is_alive():
                process.kill()
        self._processes, self._conns = [], []

    def __enter__(self) -> Flux3KleinProcessGroups:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
