# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""YAML/JSON-friendly field types for ttnn objects used in `TTMoEDecodeConfig`.

Each alias is `Annotated[ttnn.X, BeforeValidator, PlainSerializer(when_used="json")]`
so:
  - `model_dump()` (default `mode="python"`) returns the raw ttnn objects — keeps
    `**cfg.x.model_dump()` splatting into ttnn ops working unchanged.
  - `model_dump(mode="json")` (and JSON/YAML serialization) returns dicts/strings
    produced by each type's `to_json()`. Combined with `exclude_defaults=True`
    + `exclude_none=True`, only user-set non-default values appear.
  - Loading the same dict back through `model_validate` rehydrates the ttnn
    objects via `from_json`.

Most ttnn types here already expose `to_json` / `from_json` on the C++ side; we
just hook those into pydantic. Enums fall back to name-based round-tripping.
"""

from __future__ import annotations

import enum
import json
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer
from ttnn.operations.ccl import MoEActivationFunction

import ttnn


def _ttnn_json_annotated(cls):
    """Annotated alias for any ttnn type that exposes `to_json` / `from_json`."""

    def _validate(v: Any) -> Any:
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            return cls.from_json(v)
        if isinstance(v, (dict, list)):
            return cls.from_json(json.dumps(v))
        raise ValueError(f"cannot convert {v!r} to {cls.__name__}")

    def _serialize(v: Any) -> Any:
        # Parse to dict/list so yaml/json downstream handle it natively
        return json.loads(v.to_json())

    return Annotated[cls, BeforeValidator(_validate), PlainSerializer(_serialize, when_used="json")]


def _enum_annotated(enum_cls: type[enum.Enum]):
    """Annotated alias for an enum: serialize as name, accept name/int/instance."""

    def _validate(v: Any) -> Any:
        if isinstance(v, enum_cls):
            return v
        if isinstance(v, str):
            return enum_cls[v]
        if isinstance(v, int):
            return enum_cls(v)
        raise ValueError(f"cannot convert {v!r} to {enum_cls.__name__}")

    def _serialize(v: Any) -> str:
        return v.name

    return Annotated[enum_cls, BeforeValidator(_validate), PlainSerializer(_serialize, when_used="json")]


# Most ttnn types round-trip through to_json / from_json cleanly:
CoreCoord = _ttnn_json_annotated(ttnn.CoreCoord)
CoreRange = _ttnn_json_annotated(ttnn.CoreRange)
CoreRangeSet = _ttnn_json_annotated(ttnn.CoreRangeSet)


# MemoryConfig needs a small workaround: when it carries an `NdShardSpec`, ttnn's
# `to_json` can't serialize the `Shape` (it emits an error string for the
# `span<const unsigned int>` field). Pull the shape out via the live object and
# overwrite that field with a plain list before yielding the dict. `from_json`
# already accepts a list there.
def _validate_memory_config(v: Any) -> Any:
    if isinstance(v, ttnn.MemoryConfig):
        return v
    if isinstance(v, str):
        return ttnn.MemoryConfig.from_json(v)
    if isinstance(v, (dict, list)):
        return ttnn.MemoryConfig.from_json(json.dumps(v))
    raise ValueError(f"cannot convert {v!r} to ttnn.MemoryConfig")


def _serialize_memory_config(v: ttnn.MemoryConfig) -> dict:
    d = json.loads(v.to_json())
    nd = d.get("nd_shard_spec")
    if isinstance(nd, dict) and isinstance(nd.get("shard_shape"), str):
        # the bad string is `ttsl::json::to_json_t: Unsupported type span<const unsigned int>`
        nd["shard_shape"] = list(v.nd_shard_spec.shard_shape)
    return d


MemoryConfig = Annotated[
    ttnn.MemoryConfig,
    BeforeValidator(_validate_memory_config),
    PlainSerializer(_serialize_memory_config, when_used="json"),
]

# enums (name-based)
Topology = _enum_annotated(ttnn.Topology)
WorkerMode = _enum_annotated(ttnn.WorkerMode)
DispatchAlgorithm = _enum_annotated(ttnn.DispatchAlgorithm)
ActivationFunction = _enum_annotated(MoEActivationFunction)
