#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import capnp
from pathlib import Path

# Import Cap'n Proto schema
capnp.remove_import_hook()
script_dir = Path(__file__).resolve().parent
capnp_scheme = capnp.load(str(script_dir / "inspector.capnp"), imports=[str(Path(p).parent) for p in capnp.__path__])

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
