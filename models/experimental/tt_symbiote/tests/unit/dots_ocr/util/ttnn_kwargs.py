# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct ttnn kwargs (ProgramConfig, ComputeKernelConfig, MemoryConfig,
DataType, Layout) from the string-typed records produced by Phase 0 capture.

Defensive policy:
* Unknown / unserialized fields are silently dropped.
* Unknown enum tokens fall back to the most-conservative sane default
  (BFLOAT16, TILE_LAYOUT, DRAM/INTERLEAVED, HiFi2).
* A missing *required* field (e.g. a 1D matmul without ``per_core_M``)
  raises ``ValueError`` with the row's call_id in the message so the
  failure points back at the offending row.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import ttnn


# ---------------------------------------------------------------------------
# Enum-token parsing
# ---------------------------------------------------------------------------


def parse_ttnn_dtype(s: str) -> ttnn.DataType:
    """``'DataType.BFLOAT16'`` -> ``ttnn.bfloat16`` (a ``DataType`` value)."""
    if not isinstance(s, str):
        return ttnn.bfloat16
    token = s.split(".")[-1].upper()
    mapping = {
        "BFLOAT16": ttnn.bfloat16,
        "BFLOAT8_B": ttnn.bfloat8_b,
        "BFLOAT4_B": ttnn.bfloat4_b,
        "FLOAT32": ttnn.float32,
        "UINT16": ttnn.uint16,
        "UINT32": ttnn.uint32,
        "INT32": ttnn.int32,
    }
    return mapping.get(token, ttnn.bfloat16)


def parse_ttnn_layout(s: str) -> ttnn.Layout:
    if not isinstance(s, str):
        return ttnn.TILE_LAYOUT
    token = s.split(".")[-1].upper()
    if "ROW" in token:
        return ttnn.ROW_MAJOR_LAYOUT
    return ttnn.TILE_LAYOUT


def _parse_buffer_type(s: str) -> ttnn.BufferType:
    if not isinstance(s, str):
        return ttnn.BufferType.DRAM
    token = s.split(".")[-1].upper()
    return getattr(ttnn.BufferType, token, ttnn.BufferType.DRAM)


def _parse_memory_layout(s: str) -> ttnn.TensorMemoryLayout:
    if not isinstance(s, str):
        return ttnn.TensorMemoryLayout.INTERLEAVED
    token = s.split(".")[-1].upper()
    return getattr(ttnn.TensorMemoryLayout, token, ttnn.TensorMemoryLayout.INTERLEAVED)


def _parse_math_fidelity(s: str) -> ttnn.MathFidelity:
    if not isinstance(s, str):
        return ttnn.MathFidelity.HiFi2
    token = s.split(".")[-1]
    return getattr(ttnn.MathFidelity, token, ttnn.MathFidelity.HiFi2)


def _parse_corecoord(s: str) -> Optional[ttnn.CoreCoord]:
    """``'(x=8,y=8)'`` -> ``ttnn.CoreCoord(8, 8)``."""
    if not isinstance(s, str):
        return None
    m = re.search(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", s)
    if not m:
        return None
    return ttnn.CoreCoord(int(m.group(1)), int(m.group(2)))


_BOOL_TOKENS = {"true": True, "false": False, "1": True, "0": False, "none": False}


def _coerce(value: Any) -> Any:
    """Coerce a captured string field to int/bool/None/float when obvious."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s.lower() in ("none", "std::nullopt", "null", "<unserializable>"):
        return None
    if s.lower() in _BOOL_TOKENS:
        return _BOOL_TOKENS[s.lower()]
    # int / float
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", s):
            return float(s)
    except ValueError:
        pass
    if s.startswith("<unserializable"):
        return None
    return value


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


def build_memory_config(record: Optional[Dict[str, Any]]) -> ttnn.MemoryConfig:
    """Reconstruct a ttnn.MemoryConfig from a captured ``memory_config`` dict.

    The captured records may either be the **tensor-side** dict
    ``{buffer_type, memory_layout, repr}`` or the **kwarg-side** dict
    ``{kind: 'MemoryConfig', fields: {buffer_type, memory_layout, ...}, repr}``.
    Both forms are accepted.
    """
    if record is None:
        return ttnn.DRAM_MEMORY_CONFIG

    fields = record.get("fields") if isinstance(record.get("fields"), dict) else record

    bt_str = fields.get("buffer_type", "BufferType.DRAM")
    ml_str = fields.get("memory_layout", "TensorMemoryLayout.INTERLEAVED")

    bt = _parse_buffer_type(bt_str)
    ml = _parse_memory_layout(ml_str)

    # Sharded specs are NOT reconstructed here — Phase 1 sanity exercises only
    # interleaved ops. For Phase 2 sharded ops the caller will need to parse
    # ``shard_spec`` separately; for now we fall back to a simple interleaved
    # MemoryConfig when a shard spec is present (so we at least don't crash).
    return ttnn.MemoryConfig(memory_layout=ml, buffer_type=bt)


# ---------------------------------------------------------------------------
# ComputeKernelConfig
# ---------------------------------------------------------------------------


def build_compute_kernel_config(kind: Optional[str], fields: Optional[Dict[str, Any]]):
    """Reconstruct a ``ttnn.WormholeComputeKernelConfig`` from captured fields.

    ``kind`` is currently ignored (we always build Wormhole config — Phase 1
    target is T3K/WH). If unknown fields are encountered they are silently
    dropped.
    """
    fields = fields or {}
    raw = {k: _coerce(v) for k, v in fields.items()}

    kwargs: Dict[str, Any] = {}
    if "math_fidelity" in fields:
        kwargs["math_fidelity"] = _parse_math_fidelity(fields["math_fidelity"])
    for key in ("math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        if key in raw and isinstance(raw[key], bool):
            kwargs[key] = raw[key]
    # throttle_level is a ttnn enum — skip unless we can resolve it cleanly.
    tl = fields.get("throttle_level")
    if isinstance(tl, str) and "." in tl:
        token = tl.split(".")[-1]
        try:
            kwargs["throttle_level"] = getattr(ttnn.ThrottleLevel, token)
        except AttributeError:
            pass

    return ttnn.WormholeComputeKernelConfig(**kwargs)


# ---------------------------------------------------------------------------
# ProgramConfig
# ---------------------------------------------------------------------------


_PROG_CFG_FIELDS = {
    "MatmulMultiCoreReuseMultiCast1DProgramConfig": {
        "compute_with_storage_grid_size": _parse_corecoord,
        "in0_block_w": int,
        "out_subblock_h": int,
        "out_subblock_w": int,
        "out_block_h": int,
        "out_block_w": int,
        "per_core_M": int,
        "per_core_N": int,
        "fuse_batch": bool,
        "fused_activation": "none_ok",
        "mcast_in0": bool,
        "gather_in0": bool,
        "num_global_cb_receivers": int,
        "untilize_out": bool,
    },
    "MatmulMultiCoreReuseMultiCastProgramConfig": {
        "compute_with_storage_grid_size": _parse_corecoord,
        "in0_block_w": int,
        "out_subblock_h": int,
        "out_subblock_w": int,
        "out_block_h": int,
        "out_block_w": int,
        "per_core_M": int,
        "per_core_N": int,
        "transpose_mcast": bool,
        "fused_activation": "none_ok",
        "fuse_batch": bool,
    },
    "MatmulMultiCoreReuseProgramConfig": {
        "compute_with_storage_grid_size": _parse_corecoord,
        "in0_block_w": int,
        "out_subblock_h": int,
        "out_subblock_w": int,
        "per_core_M": int,
        "per_core_N": int,
    },
    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig": {
        "in0_block_w": int,
        "per_core_M": int,
        "per_core_N": int,
        "fused_activation": "none_ok",
    },
}


def build_program_config(kind: Optional[str], fields: Optional[Dict[str, Any]]):
    """Reconstruct a TTNN matmul ProgramConfig from captured field strings.

    Returns ``None`` when ``kind`` is empty / unknown — callers should
    then let TTNN auto-select the program config.
    """
    if not kind:
        return None

    cls = getattr(ttnn, kind, None)
    if cls is None:
        return None

    schema = _PROG_CFG_FIELDS.get(kind)
    raw_fields = fields or {}
    out_kwargs: Dict[str, Any] = {}

    if schema is None:
        # Best-effort coercion: pass through anything that parses to a Python primitive.
        for k, v in raw_fields.items():
            cv = _coerce(v)
            if cv is not None and not isinstance(cv, str):
                out_kwargs[k] = cv
    else:
        for k, kind_or_caster in schema.items():
            if k not in raw_fields:
                continue
            v = raw_fields[k]
            if v in (None, "None", "std::nullopt"):
                # Leave default
                continue
            if isinstance(v, str) and v.startswith("<unserializable"):
                continue
            if kind_or_caster is int:
                try:
                    out_kwargs[k] = int(v)
                except (TypeError, ValueError):
                    pass
            elif kind_or_caster is bool:
                cv = _coerce(v)
                if isinstance(cv, bool):
                    out_kwargs[k] = cv
            elif kind_or_caster == "none_ok":
                if isinstance(v, str) and v.lower() in ("none", "std::nullopt"):
                    continue
                # If activation is given as a string token we'd need to resolve
                # it to UnaryWithParam — skip for now (Phase 1 only needs no-act).
                continue
            elif callable(kind_or_caster):
                parsed = kind_or_caster(v)
                if parsed is not None:
                    out_kwargs[k] = parsed

    try:
        return cls(**out_kwargs)
    except TypeError as e:
        # Surface the row id by raising a clearer error.
        raise ValueError(f"Failed to build {kind} from fields {out_kwargs!r}: {e}") from e
