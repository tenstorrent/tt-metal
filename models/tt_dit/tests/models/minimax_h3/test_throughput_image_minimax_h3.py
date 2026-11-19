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
from ....utils.test import line_params_8k_req_exact_devices, ring_params_8k_req_exact_devices
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
@pytest.mark.parametrize("size", [2048, 4096], ids=["2K", "4K"])
def test_single_replica_on_submesh(mesh_device, submesh, size):
    """One replica on one submesh with the rest of the Galaxy idle: if its latency matches the DP run's, the DP
    slowdown is compute (fewer chips per replica); if it is much lower, it is host contention between replicas.
    n / (its latency) bounds DP=n throughput without contention; the stage split shows what each stage costs."""
    weights = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not weights:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    device = mesh_device.create_submeshes(ttnn.MeshShape(*submesh))[0]
    pipe = MiniMaxH3ImagePipeline(mesh_device=device, weights_dir=weights)
    tag = f"alone_{submesh[0]}x{submesh[1]}_{size}"
    pipe(PROMPT, width=size, height=size, seed=0)  # warm
    totals = []
    for i in range(3):
        t = pipe(PROMPT, width=size, height=size, seed=1 + i).timings
        totals.append(t.total)
        steps = ", ".join(f"{s:.2f}" for s in t.denoise_steps)
        logger.info(
            f"RESULT {tag} run {i}: total {t.total:.2f}s = prompt {t.encode_prompt:.2f} + denoise {t.denoise:.2f} "
            f"[{steps}] + decode {t.decode:.2f} (dynamic_length={pipe.dynamic_length})"
        )
    logger.info(f"RESULT {tag}: latency mean {sum(totals) / 3:.2f}s (min {min(totals):.2f} max {max(totals):.2f})")


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


# One DP replica per process: a launcher starts one pytest process per replica, each restricted to its chips with
# TT_VISIBLE_DEVICES and given its submesh's fabric with TT_MESH_GRAPH_DESC_PATH, so replicas share no host thread,
# GIL or dispatch queue. The replica runs like the 4x8 (ring, 2 links, dynamic length) where its fabric is a ring.
_L1_SMALL_DP = 65536
_DP_RING = {**ring_params_8k_req_exact_devices, "l1_small_size": _L1_SMALL_DP}
_DP_LINE = {**line_params_8k_req_exact_devices, "l1_small_size": _L1_SMALL_DP}


@pytest.mark.timeout(4 * 3600)
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param((4, 8), _DP_RING, id="replica_4x8_ring"),
        pytest.param((4, 4), _DP_RING, id="replica_4x4_ring"),
        pytest.param((4, 4), _DP_LINE, id="replica_4x4_line"),
        pytest.param((4, 2), _DP_RING, id="replica_4x2_ring"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dp_replica_process(mesh_device):
    """One replica of a multi-process DP run. Env: H3_DP_SIZE (2048 / 4096), H3_DP_REQUESTS, H3_DP_RANK, H3_DP_SYNC (a
    directory shared with the launcher: the replica writes `ready_<rank>` once warm, waits for `go`, serves its
    requests back to back with the host tail deferred to a thread (as the server does), then writes
    `result_<rank>.json`). Without H3_DP_SYNC it runs alone, which gives the contention-free per-replica number."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    from .common import fabric_topology

    weights = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not weights:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    size = int(os.environ.get("H3_DP_SIZE", 4096))
    n = int(os.environ.get("H3_DP_REQUESTS", 6))
    rank = int(os.environ.get("H3_DP_RANK", 0))
    sync = os.environ.get("H3_DP_SYNC")
    topology = fabric_topology()
    shape = tuple(mesh_device.shape)
    pipe = MiniMaxH3ImagePipeline(
        mesh_device=mesh_device, weights_dir=weights, tp_axis=0, sp_axis=1, num_links=2, topology=topology
    )
    tag = f"replica{rank}_{shape[0]}x{shape[1]}_{size}"
    logger.info(f"RESULT {tag}: topology {topology}, dynamic_length {pipe.dynamic_length}")
    for seed in (0, 1):  # warm: compile, then one warm request
        pipe(PROMPT, width=size, height=size, seed=seed)
    if sync:
        open(os.path.join(sync, f"ready_{rank}"), "w").close()
        deadline = time.time() + 3 * 3600
        while not os.path.exists(os.path.join(sync, "go")):
            if time.time() > deadline:
                raise TimeoutError("no go signal")
            time.sleep(0.05)
    finisher = ThreadPoolExecutor(1)
    pending, records = [], []
    start = time.time()
    for i in range(n):
        result = pipe(PROMPT, width=size, height=size, seed=100 + 10 * rank + i, defer_finish=True)
        t = result.timings
        records.append(
            {
                "total": t.total,
                "prompt": t.encode_prompt,
                "denoise": t.denoise,
                "steps": t.denoise_steps,
                "decode": t.decode,
            }
        )
        pending.append(finisher.submit(result.finish))
    for f in pending:
        f.result()
    end = time.time()
    finisher.shutdown()
    out = {"rank": rank, "shape": shape, "size": size, "n": n, "start": start, "end": end, "requests": records}
    if sync:
        with open(os.path.join(sync, f"result_{rank}.json"), "w") as f:
            json.dump(out, f)
    mean = sum(r["total"] for r in records) / n
    steps = ", ".join(f"{s:.2f}" for s in records[-1]["steps"])
    logger.info(
        f"RESULT {tag}: {n} images in {end - start:.2f}s -> {n / (end - start):.3f} img/s | mean pipeline "
        f"{mean:.2f}s (prompt {records[-1]['prompt']:.2f}, denoise {records[-1]['denoise']:.2f} [{steps}], decode "
        f"{records[-1]['decode']:.2f})"
    )
