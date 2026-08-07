#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
import capnp
import os

# Import Cap'n Proto schema.
# `capnp` is a compiled (Cython) extension module whose public API is re-exported via a star
# import from a `.so`, so static checkers cannot see these attributes that exist at runtime.
capnp.remove_import_hook()  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
script_dir = os.path.dirname(os.path.abspath(__file__))
capnp_scheme = capnp.load(  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    os.path.join(script_dir, "inspector.capnp"), imports=[os.path.dirname(p) for p in capnp.__path__]
)

# Dynamically import all types from capnp_scheme so that users can reference them in other scripts.
# Stub file should be generated during compile time, but here we just need to expose python types.
__all__ = []
for name, obj in capnp_scheme.__dict__.items():
    if hasattr(obj, "schema"):  # Cap'n Proto struct/interface
        if name == "Inspector":
            continue
        setattr(__import__(__name__), name, obj)
        # __all__ is built dynamically from the runtime-loaded capnp schema; the static list of
        # exported names lives in the generated stub instead.
        __all__.append(name)  # pyright: ignore[reportUnsupportedDunderAll]


# This is empty class representing the Inspector interface that will be defined in stub file.
class Inspector:
    pass
