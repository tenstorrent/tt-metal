# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image data parallelism within one Galaxy: n independent pipelines on n submeshes, run concurrently.

For each split of the 4x8 (1 x 4x8, 2 x 4x4, 4 x 4x2 -- TP stays 4 on axis 0 so the 66 GB DiT fits at 16.5 GB/chip),
every replica is warmed at the request shape, then each replica serves `requests_per_replica` requests from its own
host thread; images/s = all requests / wall time. Per-request latency is logged too, so one run gives both the
latency (DP=1) and the throughput (DP>1) picture.

Env: MINIMAX_H3_IMAGE_MODEL_PATH (any converted snapshot of the right shapes; a perf-tiled one is fine for timing).
"""

import os
import threading
import time

import pytest
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline
from .common import MESH_4X8_RING

PROMPT = (
    "The image is a photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a "
    "heavily weathered, dark brown wooden table with deep visible grain and scratches."
)


def _threads() -> tuple[int, list[tuple[str, int]]]:
    """Live threads in this process, and the most common thread names (comm), to see which component spawns them."""
    from collections import Counter

    names = Counter()
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                names[f.read().strip()] += 1
        except OSError:
            pass
    return sum(names.values()), names.most_common(8)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_dp_thread_budget(mesh_device):
    """Where do the threads come from? Count them after each stage of building two 4x4 replicas (nproc is 1024)."""
    weights = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not weights:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    logger.info(f"RESULT threads after mesh open: {_threads()}")
    devices = mesh_device.create_submeshes(ttnn.MeshShape(4, 4))
    logger.info(f"RESULT threads after 2 submeshes: {_threads()}")
    pipes = []
    for i, d in enumerate(devices):
        pipes.append(MiniMaxH3ImagePipeline(mesh_device=d, weights_dir=weights))
        logger.info(f"RESULT threads after building replica {i}: {_threads()}")
    for i, p in enumerate(pipes):
        p(PROMPT, width=2048, height=2048, seed=0)
        logger.info(f"RESULT threads after a request on replica {i}: {_threads()}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("submesh", [(4, 2), (4, 4)], ids=["alone_4x2", "alone_4x4"])
def test_single_replica_on_submesh(mesh_device, submesh):
    """One replica on one submesh with the rest of the Galaxy idle: if its 2K latency matches the DP run's, the DP
    slowdown is compute (fewer chips per replica); if it is much lower, it is host contention between replicas."""
    weights = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not weights:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    device = mesh_device.create_submeshes(ttnn.MeshShape(*submesh))[0]
    pipe = MiniMaxH3ImagePipeline(mesh_device=device, weights_dir=weights)
    pipe(PROMPT, width=2048, height=2048, seed=0)  # warm
    totals = [pipe(PROMPT, width=2048, height=2048, seed=1 + i).timings.total for i in range(3)]
    logger.info(
        f"RESULT alone_{submesh[0]}x{submesh[1]}_2048: latency mean {sum(totals) / 3:.2f}s "
        f"(min {min(totals):.2f} max {max(totals):.2f})"
    )


@pytest.mark.timeout(4 * 3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("submesh", [(4, 8), (4, 4), (4, 2)], ids=["dp1_4x8", "dp2_4x4", "dp4_4x2"])
@pytest.mark.parametrize("size", [2048, 4096], ids=["2K", "4K"])
def test_dp_throughput(mesh_device, submesh, size, requests_per_replica=3):
    weights = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not weights:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    devices = (
        [mesh_device]
        if tuple(submesh) == tuple(mesh_device.shape)
        else mesh_device.create_submeshes(ttnn.MeshShape(*submesh))
    )
    tag = f"dp{len(devices)}_{submesh[0]}x{submesh[1]}_{size}"
    mark = time.perf_counter()
    pipes = [MiniMaxH3ImagePipeline(mesh_device=d, weights_dir=weights) for d in devices]
    logger.info(f"RESULT {tag}: built {len(pipes)} replicas in {time.perf_counter() - mark:.0f}s")

    def warm(p):
        p(PROMPT, width=size, height=size, seed=0)

    mark = time.perf_counter()
    for p in pipes:  # serial: each replica compiles its rung (shared program cache per device set)
        warm(p)
    logger.info(f"RESULT {tag}: warmed in {time.perf_counter() - mark:.0f}s")

    latencies: list[float] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def serve(p, replica):
        try:
            for r in range(requests_per_replica):
                result = p(PROMPT, width=size, height=size, seed=100 + 10 * replica + r)
                with lock:
                    latencies.append(result.timings.total)
        except BaseException as err:  # surface in the main thread
            errors.append(err)

    threads = [threading.Thread(target=serve, args=(p, i)) for i, p in enumerate(pipes)]
    mark = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - mark
    if errors:
        raise errors[0]
    n = len(latencies)
    logger.info(
        f"RESULT {tag}: {n} images in {wall:.2f}s -> {n / wall:.3f} img/s | per-request latency "
        f"mean {sum(latencies) / n:.2f}s min {min(latencies):.2f}s max {max(latencies):.2f}s"
    )
