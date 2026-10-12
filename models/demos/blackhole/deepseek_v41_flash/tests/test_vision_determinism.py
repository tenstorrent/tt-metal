# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vision tower is deterministic: the same image encoded repeatedly (and interleaved with other images, other allocator states) gives bit-identical rows."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision import DSV41Vision, VisionArgs
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def _img(w, h, seed):
    r = np.random.RandomState(seed)
    base = r.randint(0, 255, (h // 14 + 1, w // 14 + 1, 3)).astype(np.uint8)
    return Image.fromarray(base).resize((w, h), Image.BICUBIC)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_vision_deterministic(mesh_device):
    md = mesh_device
    va = VisionArgs()
    vision = DSV41Vision(
        md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va, shards=_Shards()
    )
    if (
        os.environ.get("VIS_DIRTY", "0") == "1"
    ):  # fill the free DRAM with random data and free it again: any read of uninitialised memory then sees different garbage
        junk = [
            ttnn.from_torch(
                torch.randn(1, 1, 16384, 8192).to(torch.bfloat16),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            for _ in range(int(os.environ.get("VIS_JUNK", "12")))
        ]
        ttnn.synchronize_device(md)
        for t in junk:
            ttnn.deallocate(t)
    n = int(os.environ.get("VIS_REPEAT", "6"))
    sizes = [(640, 480), (1100, 700), (300, 900)]
    ims = [VP.patchify_pil(_img(w, h, i), va) for i, (w, h) in enumerate(sizes)]
    first = {}
    bad = 0
    held = []
    for it in range(n):
        if os.environ.get("VIS_SHIFT", "0") == "1":
            # a different allocator state at every iteration (tensors of varying size kept alive): an op that caches a buffer address instead of updating it reads garbage
            held.append(
                ttnn.from_torch(
                    torch.randn(1, 1, 32 * (3 + 7 * it), 1024).to(torch.bfloat16),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                )
            )
            if it % 2:
                ttnn.deallocate(held.pop(0))
        for j, (patches, nh, nw, _, _) in enumerate(ims):
            rows = vision.encode(patches, nh, nw)
            if j not in first:
                first[j] = rows
            else:
                eq = torch.equal(first[j], rows)
                d = (first[j].float() - rows.float()).abs().max().item()
                print(f"iter {it} image {j}: rows bit-equal {eq} (max|diff| {d:.5f})", flush=True)
                bad += not eq
    assert bad == 0, f"{bad} encodes differ from the first"
