# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-issue cost of the prefill's two collectives, and what moves it.

``prefill_opcount.py`` attributed the 128-token prefill's 60.4 ms of host issue by
op name.  The two async collectives are **19.1 ms of it on 209 calls** --
``reduce_scatter_minimal_async`` at 125.6 us/call and ``all_gather_async`` at
57.6 us/call -- against ~20 us and ~26 us of *device* time each in the committed
reduced-variant prefill profile.  Every other op in the prefill sits between 1.7 us
(``deallocate``) and 49 us.

That gap is the prefill's largest single lever, so this probe measures it directly,
away from the model: one mesh, one payload of the real prefill shape, N successive
calls with no synchronisation, so the number is host issue rather than device time.
Arms cross the persistent-buffer knob (``$optimize`` OPT-009), the worker counts the
decoder stage tuned, and the composite ``ttnn.reduce_scatter``/``ttnn.all_gather``
wrappers.

Usage::

    python doc/optimized_full_model/bench/ccl_host_probe.py --rows 128 --reps 40
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    CCL_TOPOLOGY,
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"
HIDDEN = 6656
TP = 4


def say(*args) -> None:
    print(*args, flush=True)


def make_sems(mesh, n: int):
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return [ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument(
        "--dtype",
        default="bfloat8_b",
        choices=("bfloat16", "bfloat8_b"),
        help="the collective payload dtype; the model's prefill reduction is BFLOAT8_B",
    )
    parser.add_argument(
        "--loaded-queue",
        action="store_true",
        help="issue a large matmul before each collective, so the arm includes the queue backpressure the in-model call sees",
    )
    parser.add_argument("--out", default="ccl_host_probe.json")
    args = parser.parse_args()

    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=0)
    rows, reps = args.rows, args.reps
    dtype = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[args.dtype]
    results: list[dict] = []

    def replicated(shape, tensor_dtype=None):
        return ttnn.from_torch(
            torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=tensor_dtype or dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def record(name: str, run, *, free: bool):
        """``run`` issues one call; ``free`` says whether its result is ours to free."""
        name = f"{name}[{args.dtype}{'+loaded' if args.loaded_queue else ''}]"
        for _ in range(2):  # compile + first-use
            loader()
            out = run()
            if free and out is not None:
                ttnn.deallocate(out)
        ttnn.synchronize_device(mesh)
        best_issue, best_drain = float("inf"), float("inf")
        for _ in range(3):
            started = time.perf_counter()
            produced = []
            for _ in range(reps):
                loader()
                produced.append(run())
            issue = (time.perf_counter() - started) / reps * 1e6
            ttnn.synchronize_device(mesh)
            drain = (time.perf_counter() - started) / reps * 1e6
            if free:
                for out in produced:
                    if out is not None:
                        ttnn.deallocate(out)
            best_issue, best_drain = min(best_issue, issue), min(best_drain, drain)
        results.append({"arm": name, "rows": rows, "issue_us": round(best_issue, 2), "drain_us": round(best_drain, 2)})
        say(f"CCL {name:<40} issue={best_issue:8.2f} us  issue+drain={best_drain:8.2f} us")

    try:
        full = replicated((1, 1, rows, HIDDEN))
        scattered = replicated((1, 1, rows, HIDDEN // TP))
        # A prefill-sized matmul to put in front of each collective when
        # ``--loaded-queue`` is set: the in-model reduce-scatter follows ``mlp_down``,
        # so its wall time can include waiting for a full command queue rather than
        # only its own dispatch.  This is the arm that separates the two.
        act = replicated((1, 1, rows, HIDDEN), ttnn.bfloat16)
        weight = replicated((1, 1, HIDDEN, HIDDEN // TP), ttnn.bfloat8_b)

        def loader():
            if args.loaded_queue:
                ttnn.deallocate(ttnn.linear(act, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG))

        rs_sems, rs_barrier = make_sems(mesh, 3), make_sems(mesh, 1)[0]
        ag_sems, ag_barrier = make_sems(mesh, 2), make_sems(mesh, 1)[0]

        def rs(buffers, workers):
            return lambda: ttnn.experimental.reduce_scatter_minimal_async(
                full,
                persistent_output_buffers=buffers,
                dim=3,
                multi_device_global_semaphore=rs_sems,
                barrier_semaphore=rs_barrier,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=workers,
            )

        def ag(buffer, workers):
            return lambda: ttnn.experimental.all_gather_async(
                scattered,
                persistent_output_buffer=buffer,
                dim=3,
                multi_device_global_semaphore=ag_sems,
                barrier_semaphore=ag_barrier,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                **({} if workers is None else {"num_workers_per_link": workers}),
            )

        record(
            "add_reference_same_payload", lambda: ttnn.add(full, full, memory_config=ttnn.DRAM_MEMORY_CONFIG), free=True
        )
        record(
            "clone_reference_same_payload", lambda: ttnn.clone(full, memory_config=ttnn.DRAM_MEMORY_CONFIG), free=True
        )

        record("rs_async_alloc_workers4", rs(None, 4), free=True)
        record("rs_async_alloc_workers1", rs(None, 1), free=True)
        record("ag_async_alloc_default", ag(None, None), free=True)
        record("ag_async_alloc_workers1", ag(None, 1), free=True)

        staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            full, dim=3, topology=CCL_TOPOLOGY
        )
        rs_out = ttnn.zeros(
            [1, 1, rows, HIDDEN // TP],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        buffers = [staging[0], rs_out, staging[1]]
        record("rs_async_persistent_workers4", rs(buffers, 4), free=False)
        record("rs_async_persistent_workers1", rs(buffers, 1), free=False)

        ag_out = ttnn.zeros(
            [1, 1, rows, HIDDEN],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        record("ag_async_persistent_default", ag(ag_out, None), free=False)

        # The wrapper forms, called exactly the way ``MultichipDecoder._all_reduce``
        # calls them.  These are what ``prefill_ccl_impl="wrapper"`` would issue.
        record(
            "rs_wrapper_workers4",
            lambda: ttnn.reduce_scatter(
                full,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=4,
                use_l1_small_for_semaphores=True,
            ),
            free=True,
        )
        record(
            "ag_wrapper",
            lambda: ttnn.all_gather(scattered, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            free=True,
        )
        record(
            "all_reduce_wrapper",
            lambda: ttnn.all_reduce(full, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY),
            free=True,
        )
        say("CCL_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(results, indent=2) + "\n")
        say(f"CCL summary -> {path}")
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
