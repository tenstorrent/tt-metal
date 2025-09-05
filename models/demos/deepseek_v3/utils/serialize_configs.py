# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
JSON serialization/deserialization utilities for TTNN-related configs and DeepSeek op configs.

Design goals:
- Pure-Python; works even if ttnn is not importable (records specs as JSON).
- Best-effort deserialization to real objects when ttnn is available; otherwise returns spec dicts.
- Handles:
  - DeepSeek dataclass configs (OpConfigBase subclasses, SavedWeight, FromWeightConfig, MeshDeviceStub)
  - TTNN MemoryConfig, program configs (Matmul*ProgramConfig, LayerNorm program configs, etc.)
  - TTNN enums and simple python types
  - Placeholders for MeshDevice and Tensor (TensorRef)
"""

from __future__ import annotations

import importlib
import os
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Callable


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _is_dataclass_instance(obj: Any) -> bool:
    try:
        return is_dataclass(obj) and not isinstance(obj, type)
    except Exception:
        return False


def _fully_qualified_name(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _enum_to_json(e: Any) -> dict[str, Any]:
    return {"__enum__": _fully_qualified_name(e), "name": getattr(e, "name", str(e))}


def _enum_from_json(d: dict[str, Any]) -> Any:
    mod_cls = d.get("__enum__")
    name = d.get("name")
    if not mod_cls or not name:
        return d
    try:
        mod_name, cls_name = mod_cls.rsplit(".", 1)
        mod = _import_optional(mod_name)
        if mod is None:
            return d
        enum_cls = getattr(mod, cls_name)
        return getattr(enum_cls, name)
    except Exception:
        return d


def _memory_config_to_json(mem: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"__type__": "ttnn.MemoryConfig"}
    try:
        layout = getattr(mem, "memory_layout", None)
        buffer_type = getattr(mem, "buffer_type", None)
        out["memory_layout"] = getattr(layout, "name", str(layout)) if layout is not None else None
        out["buffer_type"] = getattr(buffer_type, "name", str(buffer_type)) if buffer_type is not None else None
    except Exception:
        pass
    try:
        shard_spec = getattr(mem, "shard_spec", None)
        if shard_spec is not None:
            shape = getattr(shard_spec, "shape", None)
            dims = getattr(shard_spec, "dims", None)
            out["shard_spec"] = {
                "shape": list(shape) if shape is not None else None,
                "dims": list(dims) if dims is not None else None,
            }
    except Exception:
        pass
    return out


def _memory_config_from_json(d: dict[str, Any]) -> Any:
    ttnn = _import_optional("ttnn")
    if ttnn is None:
        return d
    try:
        layout = d.get("memory_layout")
        buffer_type = d.get("buffer_type")
        shard = d.get("shard_spec")
        layout_enum = getattr(ttnn, layout) if isinstance(layout, str) and hasattr(ttnn, layout) else None
        buffer_enum = getattr(ttnn, buffer_type) if isinstance(buffer_type, str) and hasattr(ttnn, buffer_type) else None
        # Fallback to constants if matches known presets
        if shard is None and layout_enum is None and buffer_enum is None:
            # Not enough info: return as-is
            return d
        if hasattr(ttnn, "MemoryConfig"):
            if shard and hasattr(ttnn, "ShardSpec"):
                shard_spec = ttnn.ShardSpec(shape=tuple(shard.get("shape") or []), dims=tuple(shard.get("dims") or []))
            else:
                shard_spec = None
            return ttnn.MemoryConfig(memory_layout=layout_enum, buffer_type=buffer_enum, shard_spec=shard_spec)
    except Exception:
        pass
    return d


def _program_config_to_json(pc: Any) -> dict[str, Any]:
    out = {"__type__": _fully_qualified_name(pc)}
    try:
        # Extract public, non-callable attributes
        for name in dir(pc):
            if name.startswith("_"):
                continue
            try:
                val = getattr(pc, name)
            except Exception:
                continue
            if callable(val):
                continue
            out[name] = to_jsonable(val)
    except Exception:
        pass
    return out


def _program_config_from_json(d: dict[str, Any]) -> Any:
    mod_cls = d.get("__type__")
    if not mod_cls:
        return d
    mod_name, cls_name = mod_cls.rsplit(".", 1)
    ttnn = _import_optional(mod_name)
    if ttnn is None:
        return d
    try:
        cls = getattr(ttnn, cls_name)
        # Attempt to construct via kwargs matching keys
        kwargs = {k: from_jsonable(v) for k, v in d.items() if k not in {"__type__"}}
        try:
            return cls(**kwargs)
        except Exception:
            # Fallback: create instance without init then setattr
            obj = object.__new__(cls)
            for k, v in kwargs.items():
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
            return obj
    except Exception:
        return d


def _tensor_to_json(t: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"__type__": "ttnn.TensorRef"}
    try:
        info["shape"] = list(getattr(t, "shape", ()))
    except Exception:
        pass
    try:
        info["dtype"] = str(getattr(t, "dtype", "unknown"))
    except Exception:
        pass
    try:
        mem = getattr(t, "memory_config", None)
        mem = mem() if callable(mem) else mem
        info["memory_config"] = _memory_config_to_json(mem) if mem is not None else None
    except Exception:
        info["memory_config"] = None
    return info


def _tensor_from_json(d: dict[str, Any]) -> Any:
    # Not reconstructing actual tensors here; return spec for test generators
    return d


def _device_to_json(dev: Any) -> dict[str, Any]:
    try:
        shape = getattr(dev, "shape", None)
        if shape is not None:
            shape = tuple(shape)
    except Exception:
        shape = None
    return {"__type__": "ttnn.MeshDevice", "mesh_shape": list(shape) if shape else None}


def _device_from_json(d: dict[str, Any], mesh_device: Any | None) -> Any:
    # Return provided mesh_device, verify shape if present
    if mesh_device is not None:
        try:
            expected = tuple(d.get("mesh_shape") or [])
            actual = tuple(getattr(mesh_device, "shape", ()) or [])
            if expected and actual and expected != actual:
                # Mismatch; keep provided but annotate
                return mesh_device
        except Exception:
            pass
        return mesh_device
    return d


def to_jsonable(obj: Any) -> Any:
    """Convert supported objects into JSON-serializable structures."""
    # Simple types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Path
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}

    # Enums
    try:
        import enum

        if isinstance(obj, enum.Enum):
            return _enum_to_json(obj)
    except Exception:
        pass

    # DeepSeek dataclasses
    if _is_dataclass_instance(obj):
        out = {"__type__": _fully_qualified_name(obj)}
        out["fields"] = {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
        return out

    # TTNN-specifics (guard import)
    ttnn = _import_optional("ttnn")
    if ttnn is not None:
        # Tensor
        try:
            if isinstance(obj, getattr(ttnn, "Tensor", ())):
                return _tensor_to_json(obj)
        except Exception:
            pass
        # MemoryConfig
        try:
            if isinstance(obj, getattr(ttnn, "MemoryConfig", ())):
                return _memory_config_to_json(obj)
        except Exception:
            pass
        # Device / MeshDevice
        try:
            if obj.__class__.__name__ in {"Device", "MeshDevice"}:
                return _device_to_json(obj)
        except Exception:
            pass
        # ProgramConfig (heuristic by name)
        try:
            if "ProgramConfig" in obj.__class__.__name__:
                return _program_config_to_json(obj)
        except Exception:
            pass

    # Containers
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Fallback
    return {"__type__": "repr", "value": repr(obj)}


def from_jsonable(obj: Any, *, mesh_device: Any | None = None) -> Any:
    """Reconstruct objects from JSON-serializable structures. Best-effort when ttnn is unavailable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [from_jsonable(x, mesh_device=mesh_device) for x in obj]
    if isinstance(obj, dict):
        # Enum
        if "__enum__" in obj:
            return _enum_from_json(obj)
        t = obj.get("__type__")
        if not t:
            return {k: from_jsonable(v, mesh_device=mesh_device) for k, v in obj.items()}
        # Path
        if t == "path":
            return Path(obj.get("value", ""))
        # TensorRef
        if t == "ttnn.TensorRef":
            return _tensor_from_json(obj)
        # MemoryConfig
        if t == "ttnn.MemoryConfig":
            return _memory_config_from_json(obj)
        # MeshDevice
        if t == "ttnn.MeshDevice":
            return _device_from_json(obj, mesh_device)
        # ProgramConfig-like
        if t.startswith("ttnn.") and t.endswith("ProgramConfig"):
            return _program_config_from_json(obj)
        # DeepSeek dataclasses
        try:
            mod_name, cls_name = t.rsplit(".", 1)
            mod = _import_optional(mod_name)
            if mod is None:
                return obj
            cls = getattr(mod, cls_name)
            field_values = {k: from_jsonable(v, mesh_device=mesh_device) for k, v in obj.get("fields", {}).items()}
            return cls(**field_values)
        except Exception:
            return obj
    return obj

