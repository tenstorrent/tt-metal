# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Looks up values in a parsed fabric manifest and checks their JSON types.
"""

from __future__ import annotations


class ManifestError(ValueError):
    """The manifest lacks a value its reader needs, or holds one the reader can't use."""


# JSON's name for each type json.load returns. bool comes first: a JSON true or false is also a Python int.
_JSON_TYPES = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (type(None), "null"),
    (list, "array"),
    (dict, "object"),
)


def _json_type(value) -> str:
    return next((name for cls, name in _JSON_TYPES if isinstance(value, cls)), type(value).__name__)


def _a(json_type: str) -> str:
    """A JSON type's name with its article, such as "an integer"."""
    if json_type == "null":
        return json_type
    return f"{'an' if json_type[0] in 'aeiou' else 'a'} {json_type}"


def lookup(node, path: str, kind: str | tuple[str, ...], of: str | tuple[str, ...] | None = None, where: str = ""):
    """
    The value at a dotted path under `node`, such as "routing_encodings.1d.hop". `kind` is the JSON type the value must
    have, or a tuple of them, and `of` the type of each item of an array or each value of an object, or a tuple of
    them. `where` is the node's own dotted path in the manifest, for errors, when the node isn't the manifest.
    """
    full_path = f"{where}.{path}" if where else path
    value = node
    keys = path.split(".")
    for depth, key in enumerate(keys):
        if not isinstance(value, dict):
            parent = ".".join(([where] if where else []) + keys[:depth]) or "the manifest"
            raise ManifestError(f"{parent} is {_a(_json_type(value))}; expected an object")
        if key not in value:
            raise ManifestError(f"manifest has no {full_path}")
        value = value[key]

    kinds = (kind,) if isinstance(kind, str) else kind
    item_kinds = () if of is None else (of,) if isinstance(of, str) else of
    need = " or ".join(_a(name) for name in kinds)
    if item_kinds:
        need += " of " + " or ".join(f"{name}s" for name in item_kinds)
    if _json_type(value) not in kinds:
        raise ManifestError(f"{full_path} is {_a(_json_type(value))}; expected {need}")
    if item_kinds:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for item_key, item in items:
            if _json_type(item) not in item_kinds:
                item_path = f"{full_path}.{item_key}" if isinstance(value, dict) else f"{full_path}[{item_key}]"
                raise ManifestError(f"{item_path} is {_a(_json_type(item))}; expected {full_path} to be {need}")
    return value
