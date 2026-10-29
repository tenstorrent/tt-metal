# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Builders for the hand-written ``test_*_small_grid.py`` companions.

The generated ``test_<op>.py`` files replay the model's captured calls, most of which are
prefill-sized, bfloat8_b, or sharded over grids a 1- or 2-compute-node device does not
have, so ``-m emulator`` deselects or skips them (see conftest.py). A small-grid companion
keeps the captured call's signature (op, memory layouts, kwargs) and shrinks it until it
fits that device, so every op the model runs has a case there:

  * bfloat16 instead of bfloat8_b / bfloat4_b (Quasar has no block-float formats);
  * at most ``conftest._EMU_MAX_ROWS`` rows in the primary input and well under
    ``conftest._EMU_MAX_INPUT_BYTES`` in total;
  * L1 shard grids starting at core (0, 0) and spanning at most 2 cores.

The builders return the same dicts the generator writes, so ``graph_case.run_case``
materializes and checks these cases exactly as it does the captured ones.
"""

import pytest


def dram():
    return {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def l1():
    return {"layout": "INTERLEAVED", "buffer": "L1", "shard": None}


def sharded(layout, shard_shape, cores_x=1, cores_y=1, buffer="L1"):
    """``layout`` sharding over the core rectangle (0, 0)..(cores_x - 1, cores_y - 1)."""
    return {
        "layout": layout,
        "buffer": buffer,
        "shard": {
            "grid": [[0, 0, cores_x - 1, cores_y - 1]],
            "shape": list(shard_shape),
            "orientation": "ROW_MAJOR",
        },
    }


def t(shape, dtype="BFLOAT16", layout="TILE", mem=None):
    """An input tensor spec."""
    return {"k": "t", "shape": list(shape), "dtype": dtype, "layout": layout, "mem": mem or dram()}


def out(shape, dtype="BFLOAT16", layout="TILE", mem=None):
    """An expected-output spec."""
    return {"k": "t", "shape": list(shape), "dtype": dtype, "layout": layout, "mem": mem or dram()}


def lit(value):
    return {"k": "lit", "v": value}


def dtype(name):
    return {"k": "dtype", "v": name}


def mem_arg(mem):
    """A memory config passed as an argument (rather than describing a tensor)."""
    return dict(mem, k="mem")


def acts(*names):
    return {"k": "acts", "v": list(names)}


def cfg(kind, **fields):
    """A program config: ``kind`` is the ttnn class name, ``fields`` its constructor arguments."""
    return {"k": "cfg", "kind": kind, "fields": fields}


def skip_if_grid_too_small(case_spec, mesh_device):
    """Skip when a program config in the case names a compute grid larger than the device's.

    Tensor shard grids are already gated by graph_case.build_memory_config; a program
    config's compute_with_storage_grid_size is not, and the op FATALs on it instead.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    for spec in case_spec["kwargs"].values():
        if spec.get("k") != "cfg":
            continue
        want = spec["fields"].get("compute_with_storage_grid_size")
        if want is not None and (want[0] > grid.x or want[1] > grid.y):
            pytest.skip(f"case needs a {want[0]}x{want[1]} compute grid; device grid is {grid.x}x{grid.y}")


def case(case_id, op, args, kwargs=None, outs=None):
    """One case. ``outs`` holds one spec per returned tensor; None entries fall back to what the
    call itself implies (graph_case._derived_spec)."""
    return {
        "id": case_id,
        "op": op,
        "count": 1,
        "args": list(args),
        "kwargs": dict(kwargs or {}),
        "outs": list(outs) if outs is not None else [None],
    }
