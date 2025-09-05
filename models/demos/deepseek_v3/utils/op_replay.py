# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Helper to replay a single op call JSON record captured by op_capture_plugin.

Usage (programmatic):

    from models.demos.deepseek_v3.utils.op_replay import replay_op_record
    result = replay_op_record(record, mesh_device)

CLI (best-effort):

    python -m models.demos.deepseek_v3.utils.op_replay --jsonl /path/to/op_calls.jsonl --index 0

This will open a mesh device automatically (1xN if available) and attempt to replay the op.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any

import torch

from .serialize_configs import from_jsonable


def _infer_torch_dtype(ttnn_dtype_str: str | None):
    if not tttnn_dtype_str := (ttnn_dtype_str or "").lower():
        return torch.bfloat16
    if "bfloat16" in tttnn_dtype_str:
        return torch.bfloat16
    if "float32" in tttnn_dtype_str or "fp32" in tttnn_dtype_str:
        return torch.float32
    if "int32" in tttnn_dtype_str:
        return torch.int32
    if "uint32" in tttnn_dtype_str:
        return torch.int32  # torch lacks uint32 tensor; map to int32 content
    # Fallback
    return torch.bfloat16


def _infer_ttnn_dtype(ttnn_mod, tttnn_dtype_str: str | None):
    s = (tttnn_dtype_str or "").lower()
    return (
        getattr(ttnn_mod, "bfloat16") if "bfloat16" in s else
        getattr(ttnn_mod, "float32") if "float32" in s or "fp32" in s else
        getattr(ttnn_mod, "int32") if "int32" in s else
        getattr(ttnn_mod, "uint32") if "uint32" in s else
        getattr(ttnn_mod, "bfloat8_b", None) if "bfloat8" in s else
        getattr(ttnn_mod, "bfloat4_b", None) if "bfloat4" in s else
        getattr(ttnn_mod, "bfloat16")
    )


def _infer_layout(ttnn_mod, mem_spec: dict | None):
    layout = None
    if isinstance(mem_spec, dict):
        ml = mem_spec.get("memory_layout") or mem_spec.get("layout")
        if isinstance(ml, str):
            ml_up = ml.upper()
            if "TILE" in ml_up:
                layout = getattr(ttnn_mod, "TILE_LAYOUT", None)
            elif "ROW" in ml_up:
                layout = getattr(ttnn_mod, "ROW_MAJOR_LAYOUT", None)
    return layout or getattr(ttnn_mod, "TILE_LAYOUT", None)


def _make_tensor_from_spec(ttnn_mod, spec: dict, mesh_device, rng=None):
    shape = spec.get("shape") or []
    dtype_str = spec.get("dtype")
    mem_spec = spec.get("memory_config") or spec.get("mem")

    torch_dtype = _infer_torch_dtype(dtype_str)
    if rng is None:
        x = torch.randn(*shape, dtype=torch_dtype)
    else:
        x = torch.randn(*shape, dtype=torch_dtype, generator=rng)

    mem_cfg = from_jsonable(mem_spec, mesh_device=mesh_device) if isinstance(mem_spec, dict) else None

    # mesh_mapper: shard if dims present
    mesh_mapper = None
    try:
        shard = mem_spec.get("shard_spec") or mem_spec.get("shard") if isinstance(mem_spec, dict) else None
        dims = tuple(shard.get("dims")) if shard and shard.get("dims") is not None else None
        if dims is not None:
            mesh_mapper = ttnn_mod.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        else:
            mesh_mapper = ttnn_mod.ReplicateTensorToMesh(mesh_device)
    except Exception:
        # Fallback to replicate
        mesh_mapper = ttnn_mod.ReplicateTensorToMesh(mesh_device)

    layout = _infer_layout(ttnn_mod, mem_spec if isinstance(mem_spec, dict) else None)
    ttnn_dtype = _infer_ttnn_dtype(ttnn_mod, dtype_str)

    return ttnn_mod.from_torch(
        x,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        dtype=ttnn_dtype,
        memory_config=mem_cfg,
        layout=layout,
    )


def _materialize(ttnn_mod, obj: Any, mesh_device, rng=None):
    # TensorRef spec
    if isinstance(obj, dict) and obj.get("__type__") == "ttnn.TensorRef":
        return _make_tensor_from_spec(ttnn_mod, obj, mesh_device, rng=rng)
    # Containers
    if isinstance(obj, list):
        return [_materialize(ttnn_mod, x, mesh_device, rng=rng) for x in obj]
    if isinstance(obj, dict):
        return {k: _materialize(ttnn_mod, v, mesh_device, rng=rng) for k, v in obj.items()}
    # Try reconstructing other types (configs, enums, paths, primitives)
    return from_jsonable(obj, mesh_device=mesh_device)


def replay_op_record(record: dict[str, Any], mesh_device, rng_seed: int | None = 0):
    """
    Replays a single captured op record using the provided mesh_device.
    Returns the operator result.
    """
    ttnn_mod = importlib.import_module("ttnn")
    op_name = record.get("op_name")
    if not op_name or not op_name.startswith("ttnn"):
        raise ValueError(f"Unsupported op_name: {op_name}")
    mod_path, func_name = op_name.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, func_name)

    rng = torch.Generator().manual_seed(rng_seed) if rng_seed is not None else None

    args = _materialize(ttnn_mod, record.get("inputs", []), mesh_device, rng=rng)
    if not isinstance(args, list):
        args = [args]
    kwargs = _materialize(ttnn_mod, record.get("kwargs", {}), mesh_device, rng=rng)
    if not isinstance(kwargs, dict):
        kwargs = {}

    return fn(*args, **kwargs)


def _open_default_mesh():
    import ttnn

    device_ids = ttnn.get_device_ids()
    if len(device_ids) == 32:
        mesh_shape = ttnn.MeshShape(4, 8)
    else:
        mesh_shape = ttnn.MeshShape(1, len(device_ids))
    return ttnn.open_mesh_device(mesh_shape=mesh_shape)


def main():
    parser = argparse.ArgumentParser(description="Replay a captured TTNN op record")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file produced by op_capture_plugin")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index of the record to replay")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for input generation")
    args = parser.parse_args()

    import ttnn

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.index:
                record = json.loads(line)
                break
        else:
            raise IndexError(f"Index {args.index} out of range")

    mesh_device = _open_default_mesh()
    try:
        result = replay_op_record(record, mesh_device, rng_seed=args.seed)
        print("Replay success. Output shape:", getattr(result, "shape", None))
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()

