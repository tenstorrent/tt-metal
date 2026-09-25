# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loading a table that lives in YAML rather than in Python.

The budget registry is the one such table. What lives here is the two ways a YAML table
fails silently. A *duplicate mapping key* is the dangerous one: YAML keeps the last, so
a copy-pasted op name drops the earlier op's whole table with nothing downstream able to
notice. A *scalar that is not an enum member* is the other; both the quoted and the bare
spelling are accepted because YAML 1.1 reads a bare ``No`` as ``False`` and
``ApproximationMode.No`` is spelled ``False`` too, so the two must not disagree.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

E = TypeVar("E", bound=Enum)


class StrictLoader(yaml.SafeLoader):
    """A loader that refuses a duplicate mapping key instead of keeping the last one."""


def _no_duplicate_keys(loader: StrictLoader, node: yaml.MappingNode) -> Dict[Any, Any]:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ValueError(
                f"duplicate entry for {key!r}. YAML keeps only the last, so the earlier "
                "one would vanish with nothing to catch it."
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=True)


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def load_yaml_table(path: Path) -> Dict[str, Any]:
    """*path* as a mapping, with every failure naming the file it came from. An empty
    file is an empty mapping rather than ``None``, so a caller can iterate the result
    without guarding it."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=StrictLoader)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{path.name}: expected a mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def enum_member(enum_cls: Type[E], value: Any, where: str) -> E:
    """One YAML scalar as an enum member, by name or by value."""
    try:
        if isinstance(value, str) and value in enum_cls.__members__:
            return enum_cls[value]
        return enum_cls(value)
    except (KeyError, ValueError):
        raise ValueError(
            f"{where}: {value!r} is not a {enum_cls.__name__}; expected one of "
            f"{', '.join(m.name for m in enum_cls)}"
        ) from None
