# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, Sequence, Type, TypeVar

import yaml
from pydantic import BaseModel

from .trainer import SFTTrainerConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def apply_override(data: dict, override: str) -> None:
    """Apply one ``dotted.key=value`` override to ``data`` in place; ``value`` is parsed as YAML."""
    key, sep, raw = override.partition("=")
    if not sep or not key:
        raise ValueError(f"override {override!r}: expected dotted.key=value")
    *parents, leaf = key.split(".")
    node = data
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"override {override!r}: {part!r} is not a section")
        node = child
    node[leaf] = yaml.safe_load(raw)


def load_config(
    path: str | os.PathLike,
    overrides: Sequence[str] = (),
    config_cls: Type[ConfigT] = SFTTrainerConfig,
) -> ConfigT:
    """Read a YAML recipe, apply ``dotted.key=value`` overrides, and validate it as ``config_cls``.

    Raises ``pydantic.ValidationError`` (a ``ValueError``) on an unknown field or an invalid value.
    """
    with open(path) as f:
        data: Any = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of config fields, got {type(data).__name__}")
    for override in overrides:
        apply_override(data, override)
    return config_cls.model_validate(data)
