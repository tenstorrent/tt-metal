# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does ttnn.as_tensor(..., cache_file_name=...) with a mesh mapper write a correct cache on this build?

Writes a fresh cache for a random bf16 TILE tensor replicated over the mesh,
reads the device tensor back, reloads it from the cache in a second as_tensor
call, and checks both against the host tensor.
"""
import os
import tempfile

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("shape", [(256, 512), (6912, 3840)], ids=["small", "dense_weight_size"])
def test_as_tensor_cache_probe(mesh_device, shape):
    torch.manual_seed(0)
    host = torch.randn(1, 1, *shape).to(torch.bfloat16)
    tmp = tempfile.mkdtemp(prefix="as_tensor_probe_")
    name = os.path.join(tmp, "probe.weight")
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
    kw = dict(
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    first = ttnn.as_tensor(host, cache_file_name=name, **kw)
    back1 = ttnn.to_torch(ttnn.get_device_tensors(first)[0]).float() if mapper else ttnn.to_torch(first).float()
    files = sorted(os.listdir(tmp))
    sizes = {f: os.path.getsize(os.path.join(tmp, f)) for f in files}
    nz = {
        f: sum(1 for x in open(os.path.join(tmp, f), "rb").read() if x) / max(1, os.path.getsize(os.path.join(tmp, f)))
        for f in files
    }
    second = ttnn.as_tensor(host, cache_file_name=name, **kw)  # must hit the cache now
    back2 = ttnn.to_torch(ttnn.get_device_tensors(second)[0]).float() if mapper else ttnn.to_torch(second).float()
    d1 = (back1 - host.float()).abs().max().item()
    d2 = (back2 - host.float()).abs().max().item()
    logger.info(
        f"cache files {sizes} nonzero-fraction {nz}; first write max diff {d1}; reload from cache max diff {d2}"
    )
    assert d1 == 0.0, "the device tensor written through the cache path is wrong"
    assert d2 == 0.0, "the tensor reloaded from the cache is wrong"
