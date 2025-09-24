#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import capnp
import os

# Import Cap'n Proto schema
capnp.remove_import_hook()
script_dir = os.path.dirname(os.path.abspath(__file__))
capnp_scheme = capnp.load(
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
        __all__.append(name)


# This is empty class representing the Inspector interface that will be defined in stub file.
class Inspector:
    pass
