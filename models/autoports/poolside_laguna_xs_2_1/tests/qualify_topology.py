# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded P150 mesh/fabric qualifier to run before loading Laguna weights.

Run each physical selection in a fresh process, for example::

    TT_VISIBLE_DEVICES=0,1 python -m \
      models.autoports.poolside_laguna_xs_2_1.tests.qualify_topology \
      --profile p150x2 --topology linear --num-links 1

The harness opens and closes the mesh for every cycle, checks eager and traced
all-reduce values, and prints one machine-readable ``QUALIFY_TOPOLOGY`` JSON line.
It never resets devices or changes persistent configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    VISIBLE_DEVICES_ENV,
    add_profile_args,
    close_mesh,
    open_mesh,
    print_memory_snapshot,
    profile_from_args,
    profile_summary,
)


def _topology_enum(name: str):
    return {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[name]


def _qualified_profile(args):
    profile = profile_from_args(args)
    if profile.num_devices == 1:
        if args.topology is not None or args.num_links is not None:
            raise ValueError("--topology/--num-links apply only to multi-device profiles")
        return profile, None, 0
    topology = args.topology or os.environ.get("TT_LAGUNA_CCL_TOPOLOGY", profile.ccl_topology)
    topology = topology.strip().lower()
    if topology not in {"linear", "ring"}:
        raise ValueError("topology must be linear or ring")
    raw_links = args.num_links if args.num_links is not None else os.environ.get("TT_LAGUNA_CCL_NUM_LINKS", "2")
    try:
        num_links = int(raw_links)
    except (TypeError, ValueError) as exc:
        raise ValueError("num-links must be 1 or 2") from exc
    if num_links not in {1, 2}:
        raise ValueError("num-links must be 1 or 2")
    fabric = "FABRIC_1D" if topology == "linear" else "FABRIC_1D_RING"
    # CLI selection wins over inherited shell values and keeps any decoder instantiated after this
    # topology probe aligned with the fabric that was actually qualified.
    os.environ["TT_LAGUNA_CCL_TOPOLOGY"] = topology
    os.environ["TT_LAGUNA_CCL_NUM_LINKS"] = str(num_links)
    return replace(profile, fabric_config=fabric), topology, num_links


def _device_tensor(mesh, devices):
    # Device d receives a tile filled with d+1. The all-reduce answer is therefore
    # D*(D+1)/2 on every device, which catches missing or duplicate participants.
    values = torch.arange(1, devices + 1, dtype=torch.bfloat16).reshape(devices, 1, 1, 1)
    values = values.expand(devices, 1, 32, 32).contiguous()
    return ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _to_host(mesh, value):
    return ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _run_cycle(profile, topology, num_links, trace_replays, cycle):
    started = time.perf_counter()
    mesh = open_mesh(ttnn, profile)
    trace_id = None
    try:
        x = _device_tensor(mesh, profile.num_devices)

        def op():
            if profile.num_devices == 1:
                return ttnn.add(x, ttnn.zeros_like(x))
            return ttnn.all_reduce(
                x,
                cluster_axis=1,
                topology=_topology_enum(topology),
                num_links=num_links,
            )

        eager = op()
        ttnn.synchronize_device(mesh)
        expected = profile.num_devices * (profile.num_devices + 1) / 2
        eager_host = _to_host(mesh, eager)
        if not torch.equal(eager_host, torch.full_like(eager_host, expected)):
            raise AssertionError(f"cycle {cycle}: eager reduction mismatch (expected {expected})")

        # Compile before capture, then capture exactly the operation exercised by Laguna decode.
        op()
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = op()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        for _ in range(trace_replays):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        traced_host = _to_host(mesh, traced)
        if not torch.equal(traced_host, torch.full_like(traced_host, expected)):
            raise AssertionError(f"cycle {cycle}: traced reduction mismatch (expected {expected})")
        memory = print_memory_snapshot(ttnn, mesh, f"topology_cycle_{cycle}", synchronize=False)
        return {
            "cycle": cycle,
            "eager_ok": True,
            "trace_ok": True,
            "trace_replays": trace_replays,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "memory": memory,
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        close_mesh(ttnn, mesh)


def main():
    parser = argparse.ArgumentParser()
    add_profile_args(parser, default_trace_region_size=200_000_000)
    parser.add_argument("--topology", choices=("linear", "ring"), default=None)
    parser.add_argument("--num-links", type=int, choices=(1, 2), default=None)
    parser.add_argument("--open-cycles", type=int, default=3)
    parser.add_argument("--trace-replays", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.open_cycles <= 10 or not 1 <= args.trace_replays <= 100:
        parser.error("open-cycles must be 1..10 and trace-replays must be 1..100")

    result = {
        "status": "failed",
        "visible_devices": os.environ.get(VISIBLE_DEVICES_ENV),
        "cycles": [],
    }
    try:
        profile, topology, num_links = _qualified_profile(args)
        result.update(profile_summary(profile))
        result.update({"ccl_topology": topology, "ccl_num_links": num_links})
        for cycle in range(1, args.open_cycles + 1):
            result["cycles"].append(_run_cycle(profile, topology, num_links, args.trace_replays, cycle))
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print("QUALIFY_TOPOLOGY", json.dumps(result, sort_keys=True))
        raise
    print("QUALIFY_TOPOLOGY", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
