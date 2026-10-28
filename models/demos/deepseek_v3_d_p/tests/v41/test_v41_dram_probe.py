# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The allocator's view of a chip's DRAM at the served trace region: the capacity the
allocator manages (per bank x banks), what is allocated and free right after the mesh opens, and the trace
region.
No weights; 20 s. ``V41_DRAM_TRACE_REGION`` (bytes, default 2 GiB, the served catalog's)."""

import json
import os

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes

TRACE_REGION = int(os.environ.get("V41_DRAM_TRACE_REGION", 2 * 1024**3))


def _view(mesh_device, kind):
    v = ttnn.get_memory_view(mesh_device, kind)
    out = {}
    for attr in dir(v):
        if attr.startswith("_") or attr in ("blocks", "block_table"):
            continue
        try:
            val = getattr(v, attr)
            if isinstance(val, (int, float)):
                out[attr] = int(val)
        except Exception:
            pass
    out["fields"] = [a for a in dir(v) if not a.startswith("_")]
    return out


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_dram_probe(mesh_device, device_params):
    out = {"trace_region_bytes": TRACE_REGION}
    for attr in ("dram_size_per_channel", "num_dram_channels", "dram_grid_size", "l1_size_per_core"):
        try:
            v = getattr(mesh_device, attr)()
            out[attr] = int(v) if not hasattr(v, "x") else [int(v.x), int(v.y)]
        except Exception as e:  # not every binding exposes it
            out[attr] = str(e)
    for name, kind in (("dram", ttnn.BufferType.DRAM), ("trace", ttnn.BufferType.TRACE), ("l1", ttnn.BufferType.L1)):
        try:
            out[name] = _view(mesh_device, kind)
        except Exception as e:  # the API moves
            out[name] = {"error": str(e)}
    for name in ("dram", "trace"):
        v = out[name]
        if "num_banks" in v:
            for k in list(v):
                if k.endswith("_per_bank"):
                    out[f"{name}_{k[:-9]}_bytes"] = v[k] * v["num_banks"]
    if isinstance(out.get("dram_size_per_channel"), int) and isinstance(out.get("num_dram_channels"), int):
        out["dram_physical_bytes"] = out["dram_size_per_channel"] * out["num_dram_channels"]
    logger.info(f"V41_DRAM_PROBE {json.dumps(out)}")
    path = os.environ.get("V41_DRAM_OUT")
    if path:
        with open(os.path.expanduser(path), "w") as f:
            json.dump(out, f, indent=1)
