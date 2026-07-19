#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import ttnn


def _resolve_layout(name: str):
    return getattr(ttnn, name)


def _resolve_dtype(name: str):
    return getattr(ttnn, name)


def _resolve_memory_config(name: str):
    return getattr(ttnn, name)


def _normalize_mesh_shape(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [int(dim) for dim in value]
    else:
        normalized = str(value).lower().replace("x", ",")
        parts = [int(part) for part in normalized.split(",") if part]
    if len(parts) == 1:
        parts = [1, parts[0]]
    if len(parts) != 2:
        raise ValueError(f"Mesh shape must have exactly two dimensions, got {value!r}")
    return tuple(parts)


def _resolve_within_base(raw_path: str, *, base_dir: Path, purpose: str) -> Path:
    """Canonicalize a user-supplied CLI path and confine it to ``base_dir``.

    Guards against path traversal: ``--manifest`` / ``--save-report`` values are
    resolved relative to (and must stay within) the working directory, so a crafted
    value cannot read or write files outside the intended scope. Absolute paths that
    already live inside the working directory are still allowed.
    """
    base = base_dir.resolve()
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"Refusing {purpose} path outside {base}: {raw_path!r}")
    return resolved


def _load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("Manifest must be a list of cases or an object with a 'cases' list.")
    return cases


def _resolve_mesh_mapper(device, case: dict, prefix: str):
    if getattr(device, "get_num_devices", lambda: 1)() <= 1:
        return None

    mesh_dims = case.get(f"{prefix}_mesh_dims")
    if mesh_dims is not None:
        normalized_dims = tuple(None if dim is None else int(dim) for dim in mesh_dims)
        return ttnn.ShardTensor2dMesh(device, mesh_shape=device.shape, dims=normalized_dims)

    shard_dim = case.get(f"{prefix}_shard_dim")
    if shard_dim is not None:
        return ttnn.ShardTensorToMesh(device, dim=int(shard_dim))

    return ttnn.ReplicateTensorToMesh(device)


def _make_device_tensor(shape, *, device, dtype, layout, memory_config, mesh_mapper=None):
    host = torch.randn(tuple(shape), dtype=torch.float32)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def _make_host_tensor(shape):
    return torch.randn(tuple(shape), dtype=torch.float32)


def _open_target_device(*, device_id: int, mesh_shape: tuple[int, int] | None):
    if mesh_shape is None:
        return ttnn.open_device(device_id=device_id), ttnn.close_device
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape)), ttnn.close_mesh_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-populate the measured auto-matmul cache from a manifest.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a JSON manifest containing matmul cases (resolved within the current working directory).",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to open for tuning.")
    parser.add_argument(
        "--mesh-shape",
        help="Optional mesh shape as ROWSxCOLS or ROWS,COLS (for example: 1x8). Opens a mesh device instead of a single device.",
    )
    parser.add_argument(
        "--rhs-on-host",
        action="store_true",
        help="Force the RHS tensor for every case to remain on host so host staging is exercised.",
    )
    parser.add_argument(
        "--save-report",
        help="Optional path to write the full per-case selection report as JSON, including candidate timings "
        "(resolved within the current working directory).",
    )
    args = parser.parse_args()

    base_dir = Path.cwd()
    manifest_path = _resolve_within_base(args.manifest, base_dir=base_dir, purpose="--manifest")
    cases = _load_manifest(manifest_path)
    mesh_shape = _normalize_mesh_shape(args.mesh_shape)
    report_cases = []

    device, close_device = _open_target_device(device_id=args.device_id, mesh_shape=mesh_shape)
    try:
        for case in cases:
            name = case.get("name", "<unnamed>")
            lhs_shape = case["lhs_shape"]
            rhs_shape = case["rhs_shape"]
            is_linear = bool(case.get("is_linear", False))
            rhs_on_host = bool(case.get("rhs_on_host", args.rhs_on_host))
            dtype = _resolve_dtype(case.get("dtype", "bfloat16"))
            layout = _resolve_layout(case.get("layout", "TILE_LAYOUT"))
            memory_config = _resolve_memory_config(case.get("memory_config", "DRAM_MEMORY_CONFIG"))

            if rhs_on_host and any(
                case.get(key) is not None
                for key in ("rhs_shard_dim", "rhs_mesh_dims", "bias_shard_dim", "bias_mesh_dims")
            ):
                raise ValueError(
                    f"{name}: host RHS/bias staging cannot be combined with device-side RHS/bias sharding manifest fields"
                )

            lhs = _make_device_tensor(
                lhs_shape,
                device=device,
                dtype=dtype,
                layout=layout,
                memory_config=memory_config,
                mesh_mapper=_resolve_mesh_mapper(device, case, "lhs"),
            )
            rhs = (
                _make_host_tensor(rhs_shape)
                if rhs_on_host
                else _make_device_tensor(
                    rhs_shape,
                    device=device,
                    dtype=dtype,
                    layout=layout,
                    memory_config=memory_config,
                    mesh_mapper=_resolve_mesh_mapper(device, case, "rhs"),
                )
            )
            bias = None
            if "bias_shape" in case:
                bias = (
                    _make_host_tensor(case["bias_shape"])
                    if rhs_on_host
                    else _make_device_tensor(
                        case["bias_shape"],
                        device=device,
                        dtype=dtype,
                        layout=layout,
                        memory_config=memory_config,
                        mesh_mapper=_resolve_mesh_mapper(device, case, "bias"),
                    )
                )

            result = ttnn.experimental.auto_config.explain_matmul(
                lhs,
                rhs,
                bias=bias,
                transpose_a=bool(case.get("transpose_a", False)),
                transpose_b=bool(case.get("transpose_b", False)),
                memory_config=memory_config,
                dtype=dtype,
                activation=case.get("activation"),
                is_linear=is_linear,
                allow_tuning=True,
            )

            summarized_result = {
                "name": name,
                "cache_hit": result["cache_hit"],
                "cache_path": result["cache_path"],
                "distributed_plan": result["distributed_plan"],
                "winner": result["winner"],
            }
            print(json.dumps(summarized_result, indent=2, sort_keys=True))
            report_cases.append(
                {
                    **summarized_result,
                    "candidate_timings_us": result["candidate_timings_us"],
                    "recommendations": result["recommendations"],
                    "signature": result["signature"],
                }
            )
    finally:
        close_device(device)

    if args.save_report:
        report_path = _resolve_within_base(args.save_report, base_dir=base_dir, purpose="--save-report")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"cases": report_cases}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
