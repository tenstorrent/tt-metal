#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QuietBox 1x4 mesh smoke with the pinned runtime, the way the launchers open the mesh.

Steps, each recorded as actual vs expected in the result JSON (nothing is
flashed or reset; the mesh is opened once and closed):

  1. runtime proof: the loaded ``ttnn._ttnn`` is the pinned extension (path + sha256);
  2. device discovery: (GetNumAvailableDevices, get_num_pcie_devices, get_num_devices) == (4, 4, 4);
  3. cluster descriptor: copied into the log directory; chips_with_mmio over the
     expected device nodes; the ethernet graph classified as line or ring; the
     route derived with ``physical_route.derive_canonical_line_route`` when the
     graph is a line, else the fabric's line order (``derive_fabric_line_route``,
     the ring walk from the lowest chip ID recorded next to it);
  4. SystemMeshDescriptor local shape (one local 4x1 line, all_local);
  5. fabric FABRIC_1D / STRICT_INIT, ``open_mesh_device`` with the server's
     arguments (l1_small_size 24576, trace_region_size 0), reshape to 1x4,
     per-coordinate mapping (ttnn id, device node, BDF, board link, fabric ids);
  6. one eager op (add) on a dim-0 sharded tensor, exact compare;
  7. one traced op (add captured with begin/end_trace_capture, executed once), exact compare;
  8. one four-device all_gather in the production call form (dim 3, cluster_axis 1, DRAM; Linear
     under FABRIC_1D), exact compare;
  9. synchronize, close, fabric DISABLED.

``--no-device`` stops after step 1 (import and argument proof on the CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCHEMA = "qwen38-qb-mesh-smoke/v1"
MESH_SHAPE = (1, 4)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_result(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True, default=str) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Checks:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.failures = 0

    def expect(self, name: str, actual: Any, expected: Any) -> bool:
        ok = actual == expected
        self.records.append({"check": name, "actual": actual, "expected": expected, "ok": ok})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: actual={actual!r} expected={expected!r}", flush=True)
        if not ok:
            self.failures += 1
        return ok


def _adjacency(document: dict[str, Any], chip_ids: set[int]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {chip: set() for chip in chip_ids}
    for link in document.get("ethernet_connections") or []:
        left, right = (int(endpoint["chip"]) for endpoint in link)
        if left in chip_ids and right in chip_ids and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def _ring_walk(adjacency: dict[int, set[int]]) -> tuple[int, ...]:
    start = min(adjacency)
    route = [start]
    previous = None
    current = start
    while len(route) < len(adjacency):
        candidates = sorted(adjacency[current] - ({previous} if previous is not None else set()) - set(route))
        if not candidates:
            break
        previous, current = current, candidates[0]
        route.append(current)
    return tuple(route)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--runtime-extension", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--expected-host", default="tt-quietbox")
    parser.add_argument("--expected-nodes", default="0,1,2,3")
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "start_utc": _utc(),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TT_VISIBLE_DEVICES",
                "TT_METAL_HOME",
                "TT_METAL_RUNTIME_ROOT",
                "TT_METAL_KERNEL_PATH",
                "TT_METAL_CACHE",
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "TTNN_CONFIG_PATH",
                "TTNN_CONFIG_OVERRIDES",
            )
        },
        "phases": [],
        "errors": [],
    }
    phases = report["phases"]

    def phase(name: str) -> None:
        phases.append({"phase": name, "utc": _utc(), "monotonic": time.monotonic()})
        print(f"[phase] {name} {phases[-1]['utc']}", flush=True)

    def finish(exit_code: int) -> int:
        report["end_utc"] = _utc()
        report["checks"] = checks.records
        report["check_failures"] = checks.failures
        report["exit_code"] = exit_code
        _write_result(args.result, report)
        print(f"[result] {args.result} exit={exit_code} failures={checks.failures}", flush=True)
        return exit_code

    checks.expect("hostname", socket.gethostname().split(".", 1)[0], args.expected_host)
    expected_nodes = {int(item) for item in args.expected_nodes.split(",")}
    checks.expect(
        "TT_VISIBLE_DEVICES", os.environ.get("TT_VISIBLE_DEVICES"), ",".join(str(n) for n in sorted(expected_nodes))
    )

    phase("before-ttnn-import")
    import ttnn  # noqa: E402

    phase("after-ttnn-import")
    loaded = Path(ttnn._ttnn.__file__).resolve(strict=True)
    digest = _sha256(loaded)
    report["runtime"] = {
        "python_module": str(Path(ttnn.__file__).resolve()),
        "extension": str(loaded),
        "sha256": digest,
    }
    ok_path = checks.expect("runtime.extension", str(loaded), str(args.runtime_extension.resolve(strict=True)))
    ok_sha = checks.expect("runtime.sha256", digest, args.runtime_sha256)
    if not (ok_path and ok_sha):
        return finish(2)
    if args.no_device:
        report["devices_opened"] = 0
        return finish(0 if checks.failures == 0 else 1)

    import torch
    import yaml

    from models.demos.blackhole.qwen38_flash_next.tools import physical_route

    phase("before-discovery")
    discovered = (int(ttnn.GetNumAvailableDevices()), int(ttnn.get_num_pcie_devices()), int(ttnn.get_num_devices()))
    checks.expect("discovered (available, pcie, total)", list(discovered), [4, 4, 4])

    descriptor_path = Path(ttnn.cluster.serialize_cluster_descriptor()).resolve(strict=True)
    descriptor_bytes = descriptor_path.read_bytes()
    copy_path = args.log_dir / "cluster-descriptor.yaml"
    copy_path.write_bytes(descriptor_bytes)
    document = yaml.safe_load(descriptor_bytes)
    report["cluster_descriptor"] = {
        "path": str(descriptor_path),
        "copy": str(copy_path),
        "sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "arch": document.get("arch"),
        "chips": document.get("chips"),
        "chips_with_mmio": document.get("chips_with_mmio"),
        "ethernet_connections": document.get("ethernet_connections"),
        "boardtype": document.get("boardtype"),
        "chip_to_boardtype": document.get("chip_to_boardtype"),
        "harvesting": document.get("harvesting"),
        "keys": sorted(document.keys()) if isinstance(document, dict) else None,
    }
    print(
        json.dumps(
            {k: v for k, v in report["cluster_descriptor"].items() if k not in ("harvesting",)}, indent=1, default=str
        ),
        flush=True,
    )

    chips_with_mmio = physical_route.parse_chips_with_mmio(document, expected_device_nodes=expected_nodes)
    checks.expect("chips_with_mmio device nodes", sorted(chips_with_mmio.values()), sorted(expected_nodes))
    adjacency = _adjacency(document, set(chips_with_mmio))
    degrees = sorted(len(neighbors) for neighbors in adjacency.values())
    link_count = sum(len(v) for v in adjacency.values()) // 2
    if degrees == [1, 1, 2, 2]:
        graph_kind = "line"
    elif degrees == [2, 2, 2, 2] and link_count == 4:
        graph_kind = "ring"
    else:
        graph_kind = f"other:{degrees}"
    report["topology"] = {
        "ethernet_adjacency": {str(chip): sorted(neighbors) for chip, neighbors in adjacency.items()},
        "ethernet_link_count_between_local_chips": link_count,
        "ethernet_connection_entries": len(document.get("ethernet_connections") or []),
        "graph_kind": graph_kind,
        "graph_kind_lab": "line",
    }
    print(
        f"[info] ethernet graph kind actual={graph_kind} expected=line adjacency={report['topology']['ethernet_adjacency']}",
        flush=True,
    )
    try:
        route = physical_route.derive_canonical_line_route(document, chips_with_mmio)
        report["topology"]["route_derivation"] = "physical_route.derive_canonical_line_route"
    except physical_route.PhysicalRouteError as error:
        report["topology"]["line_route_error"] = str(error)
        route = physical_route.derive_fabric_line_route(
            document,
            chips_with_mmio,
            lambda mesh_id, chip_id: int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_id, chip_id)),
        )
        walk = _ring_walk(adjacency)
        report["topology"]["ring_walk_route"] = list(walk)
        report["topology"]["ring_walk_agrees"] = walk == route
        report["topology"]["route_derivation"] = (
            "physical_route.derive_fabric_line_route (line derivation refused: see line_route_error; "
            "the ring walk from the lowest chip ID is recorded as ring_walk_route)"
        )
        print(f"[info] line route refused: {error}; fabric line order {route}; ring walk {walk}", flush=True)
    checks.expect("route covers four chips", sorted(route), sorted(chips_with_mmio))
    route_nodes = physical_route.route_device_nodes(route, chips_with_mmio)
    report["topology"].update(
        canonical_logical_route=list(route),
        canonical_device_node_route=list(route_nodes),
        fabric_config="FABRIC_1D",
        collective_topology="Linear",
    )

    descriptor = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    physical_shape = tuple(int(value) for value in descriptor.local_shape())
    all_local = bool(descriptor.all_local())
    report["topology"]["system_mesh_local_shape"] = list(physical_shape)
    report["topology"]["system_mesh_all_local"] = all_local
    # Auto-discovery on a larger host reports (4, 1); a 1x4 descriptor reports (1, 4).  Either is one 1D four-device mesh.
    checks.expect(
        "SystemMeshDescriptor.local_shape is one 1D four-device mesh (sorted)", sorted(physical_shape), [1, 4]
    )
    report["topology"]["system_mesh_local_shape_lab"] = [4, 1]
    checks.expect("SystemMeshDescriptor.all_local", all_local, True)
    if len(physical_shape) != 2 or physical_shape[0] * physical_shape[1] != 4:
        report["errors"].append(f"system mesh local shape {physical_shape} does not hold four devices; not opening")
        return finish(3)

    mesh = None
    fabric_may_be_enabled = False
    cleanup_errors: list[str] = []
    exit_code = 1
    try:
        phase("before-fabric-enable")
        fabric_may_be_enabled = True
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )
        phase("before-mesh-open")
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*physical_shape),
            physical_device_ids=list(route),
            l1_small_size=24576,
            trace_region_size=0,
        )
        phase("after-mesh-open")
        opened_shape = tuple(int(value) for value in mesh.shape)
        report["topology"]["opened_shape"] = list(opened_shape)
        if opened_shape != MESH_SHAPE:
            mesh.reshape(ttnn.MeshShape(*MESH_SHAPE))
        checks.expect("mesh.shape after reshape", [int(v) for v in mesh.shape], list(MESH_SHAPE))
        checks.expect("mesh.get_num_devices", int(mesh.get_num_devices()), 4)
        opened_ids = [int(v) for v in mesh.get_device_ids()]
        checks.expect("mesh.get_device_ids (1x4 order) == requested route", opened_ids, list(route))
        # The model's Linear collectives hop between neighbouring 1x4 coordinates:
        # every consecutive pair in the opened order must be one ethernet link.
        missing_links = [[a, b] for a, b in zip(opened_ids, opened_ids[1:]) if b not in adjacency.get(a, set())]
        order_is_path = checks.expect("opened 1x4 order is an ethernet path (missing links)", missing_links, [])
        report["topology"]["opened_order"] = opened_ids
        report["topology"]["opened_order_missing_links"] = missing_links
        mapping = []
        for column in range(4):
            coordinate = ttnn.MeshCoordinate(0, column)
            actual_id = int(mesh.get_device_id(coordinate))
            node = chips_with_mmio.get(actual_id)
            sysfs = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device").resolve(strict=True)
            by_id = {p.name: int(p.resolve().name) for p in Path("/dev/tenstorrent/by-id").iterdir() if p.is_symlink()}
            board_links = sorted(name for name, target in by_id.items() if target == node)
            fabric = mesh.get_fabric_node_id(coordinate)
            mapping.append(
                {
                    "mesh_coordinate": [0, column],
                    "ttnn_physical_id": actual_id,
                    "device_node": node,
                    "bdf": sysfs.name.lower(),
                    "numa_node": int((sysfs / "numa_node").read_text(encoding="ascii").strip()),
                    "board_links": board_links,
                    "fabric_mesh_id": int(fabric.mesh_id),
                    "fabric_chip_id": int(fabric.chip_id),
                }
            )
        report["topology"]["live_mapping"] = mapping
        print(json.dumps(mapping, indent=1), flush=True)
        mesh.enable_program_cache()

        # 6. eager op on a dim-0 sharded tensor: coordinate k holds the value k + 1.
        phase("before-eager-op")
        host = (
            torch.arange(1, 5, dtype=torch.float32)
            .view(4, 1, 1, 1)
            .expand(4, 1, 32, 32)
            .contiguous()
            .to(torch.bfloat16)
        )
        sharded = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, 0)),
        )
        composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(1, 0))
        doubled = ttnn.add(sharded, sharded)
        ttnn.synchronize_device(mesh)
        back = ttnn.to_torch(doubled, mesh_composer=composer).to(torch.float32)
        expected = host.to(torch.float32) * 2
        checks.expect("eager add: readback shape", list(back.shape), [4, 1, 32, 32])
        checks.expect(
            "eager add: per-device values",
            [float(back[k].mean()) for k in range(4)],
            [float(expected[k].mean()) for k in range(4)],
        )
        checks.expect("eager add: exact", bool(torch.equal(back, expected)), True)
        phase("after-eager-op")

        # 7. traced op: capture add(sharded, sharded) once, execute once, read back.
        trace_ok = False
        try:
            phase("before-trace-capture")
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = ttnn.add(sharded, sharded)
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            phase("before-trace-execute")
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            traced_back = ttnn.to_torch(traced, mesh_composer=composer).to(torch.float32)
            trace_ok = checks.expect("traced add: exact", bool(torch.equal(traced_back, expected)), True)
            ttnn.release_trace(mesh, trace_id)
            report["trace"] = {"trace_id": int(trace_id), "ok": trace_ok}
            phase("after-trace")
        except BaseException as error:  # noqa: BLE001
            report["trace"] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            checks.expect("traced add: completed", False, True)
            print(f"[FAIL] traced add raised {type(error).__name__}: {error}", flush=True)

        # 8. four-device all_gather along the mesh row.  Skipped (recorded) when the
        # opened order is not a physical path: a Linear collective over a missing
        # link could hang the fabric, and this box must not be reset.
        if not order_is_path:
            report["all_gather"] = {
                "skipped": True,
                "reason": f"opened order {opened_ids} is not an ethernet path: missing {missing_links}",
            }
            print(f"[skip] all_gather: {report['all_gather']['reason']}", flush=True)
        else:
            phase("before-all-gather")
            gathered = ttnn.all_gather(sharded, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(mesh)
            gathered_back = ttnn.to_torch(gathered, mesh_composer=composer).to(torch.float32)
            checks.expect("all_gather: readback shape", list(gathered_back.shape), [4, 1, 32, 128])
            blocks = [[float(gathered_back[k, 0, :, 32 * j : 32 * (j + 1)].mean()) for j in range(4)] for k in range(4)]
            checks.expect("all_gather: per-device blocks", blocks, [[1.0, 2.0, 3.0, 4.0]] * 4)
            expected_gathered = torch.cat([host.to(torch.float32)[k : k + 1] for k in range(4)], dim=3).expand(
                4, 1, 32, 128
            )
            checks.expect("all_gather: exact", bool(torch.equal(gathered_back, expected_gathered)), True)
            report["all_gather"] = {"skipped": False, "topology": "Linear", "blocks": blocks}
            phase("after-all-gather")
        exit_code = 0 if checks.failures == 0 else 1
    except BaseException as error:  # noqa: BLE001
        report["errors"].append(
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "phase": phases[-1]["phase"] if phases else None,
            }
        )
        print(f"[FAIL] {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        exit_code = 4
    finally:
        if mesh is not None:
            try:
                phase("before-mesh-close")
                ttnn.synchronize_device(mesh)
                ttnn.close_mesh_device(mesh)
                report["mesh_closed"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"mesh_close:{type(error).__name__}:{error}")
        if fabric_may_be_enabled:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                report["fabric_disabled"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"fabric_disable:{type(error).__name__}:{error}")
        report["cleanup_errors"] = cleanup_errors
        phase("after-cleanup")
    report["devices_opened"] = 4 if mesh is not None else 0
    if cleanup_errors and exit_code == 0:
        exit_code = 5
    return finish(exit_code)


if __name__ == "__main__":
    sys.exit(main())
