#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import capnp
import os

# Import Cap'n Proto schema
capnp.remove_import_hook()
__all__ = []


def load_capnp_schemas(files: list[str], all: list[str]):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file in files:
        schema = capnp.load(os.path.join(script_dir, file), imports=[os.path.dirname(p) for p in capnp.__path__])
        for name, obj in schema.__dict__.items():
            if hasattr(obj, "schema"):  # Cap'n Proto struct/interface
                setattr(__import__(__name__), name, obj)
                all.append(name)


load_capnp_schemas(["rpc.capnp", "runtime_rpc.capnp"], __all__)
