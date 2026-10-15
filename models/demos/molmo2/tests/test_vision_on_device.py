# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vision backbone keeps ViT features on device (all-gather + on-device pooling table). Check it matches the
host round-trip path (ViT -> CPU -> upload -> pooling) used before, and report both timings.

    MOLMO2_VERIF_DIR=<verification dir> pytest models/demos/molmo2/tests/test_vision_on_device.py -s
"""

import time

import torch
from loguru import logger

import ttnn
from models.demos.molmo2.tests.profile_forward_pass import (  # noqa: F401
    _load_first_video_test,
    build_inputs,
    mesh_device,
    processor,
    tt_model,
)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _reference(model, pv, pool_idx):
    """Previous implementation: ViT features through the host, then the CPU-table pooling path."""
    B, n_crops, n_patches, px = pv.shape
    nd = model.mesh_device.get_num_devices()
    pv4 = pv.reshape(B * n_crops, 1, n_patches, px).to(torch.bfloat16)
    parts = []
    for s in range(0, n_crops, nd):
        c = pv4[s : s + nd]
        real = c.shape[0]
        if real < nd:
            c = torch.cat([c, torch.zeros(nd - real, 1, n_patches, px, dtype=torch.bfloat16)])
        x = ttnn.from_torch(
            c,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(model.mesh_device, dim=0),
        )
        f = model.vit_encoder.forward(x, n_crops_per_device=1)
        ttnn.deallocate(x)
        parts.append(torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(f)], dim=0)[:real].squeeze(1))
        ttnn.deallocate(f)
    vit_cpu = torch.cat(parts, dim=0).reshape(B, n_crops, n_patches, -1)
    pooled = model._run_chunked_ttnn_pooling(vit_cpu, pool_idx)
    valid = (pool_idx[0] >= 0).any(dim=-1)
    return pooled[0][valid]


def test_vision_matches_host_roundtrip(mesh_device, tt_model, processor):
    model, _ = tt_model
    vp, prompt = _load_first_video_test()
    ids, pv, pool_idx, _ = build_inputs(processor, vp, prompt)
    ref = _reference(model, pv, pool_idx)
    for _ in range(2):
        t = time.perf_counter()
        feats = model.run_vision_backbone(pv, pool_idx)
        dt = time.perf_counter() - t
    # feats are projected to 4096; project the reference the same way
    p = ttnn.from_torch(
        ref.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ref_proj = ttnn.to_torch(ttnn.get_device_tensors(model.image_projector.forward(p))[0]).float().squeeze(0).squeeze(0)
    pcc = _pcc(feats, ref_proj)
    logger.info(
        f"vision on-device path: {dt*1e3:.0f} ms, PCC vs host round-trip {pcc:.6f}, shapes {tuple(feats.shape)}"
    )
    assert feats.shape == ref_proj.shape
    assert pcc > 0.999
