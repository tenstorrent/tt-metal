# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fabric node -> ASIC unique ID lookup through a throwaway 1x1 mesh in a child process.

``ttnn.cluster.get_chip_unique_id_from_fabric_node_id`` takes an open ``MeshDevice`` since PR #60180 (it resolves the
control plane through the mesh's MetalEnv).  The route lookup runs before the server opens its own mesh and sets the
fabric config, so the query runs in a child process: the server process keeps no device state before its own open.
The 1x1 mesh is opened with no fabric config; verified on QuietBox 2 that its mapping equals the old pre-open mapping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable

_CACHE: dict[tuple[int, tuple[int, ...]], dict[int, int]] = {}


def query_chip_unique_ids(mesh_id: int, chip_ids: Iterable[int], *, timeout_s: float = 300.0) -> dict[int, int]:
    """ASIC unique ID of each fabric node ``(mesh_id, chip_id)``, asked of a child process (cached per process)."""

    chip_ids = tuple(int(chip_id) for chip_id in chip_ids)
    key = (int(mesh_id), chip_ids)
    if key not in _CACHE:
        command = [
            sys.executable,
            "-m",
            "models.demos.blackhole.qwen38_flash_next.tools.fabric_node_probe",
            "--mesh-id",
            str(int(mesh_id)),
            "--chip-ids",
            ",".join(str(chip_id) for chip_id in chip_ids),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            tail = (error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr) or ""
            raise RuntimeError(f"fabric node probe timed out after {timeout_s}s: {tail[-2000:]}") from error
        if completed.returncode != 0:
            raise RuntimeError(f"fabric node probe exited {completed.returncode}: {completed.stderr[-2000:]}")
        try:
            # the runtime logs its device close after the JSON line: take the last line that is the JSON object
            result = json.loads([line for line in completed.stdout.splitlines() if line.startswith('{"mesh_id"')][-1])
            _CACHE[key] = {int(chip_id): int(unique_id) for chip_id, unique_id in result["chip_unique_ids"].items()}
        except (IndexError, KeyError, ValueError, AttributeError) as error:
            raise RuntimeError(
                f"fabric node probe printed no result line: stdout={completed.stdout[-500:]!r} "
                f"stderr={completed.stderr[-2000:]}"
            ) from error
    return dict(_CACHE[key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-id", type=int, required=True)
    parser.add_argument("--chip-ids", required=True, help="comma-separated fabric chip IDs, e.g. 0,1,2,3")
    args = parser.parse_args()
    try:
        chip_ids = [int(value) for value in args.chip_ids.split(",")]
        import ttnn

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        try:
            unique_ids = {
                str(chip_id): int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh, args.mesh_id, chip_id))
                for chip_id in chip_ids
            }
        finally:
            ttnn.close_mesh_device(mesh)
    except Exception as error:  # the parent reports the stderr tail
        print(f"fabric node probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"mesh_id": args.mesh_id, "chip_unique_ids": unique_ids}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
