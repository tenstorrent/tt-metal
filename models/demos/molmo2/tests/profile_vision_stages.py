# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock split of the vision backbone (ViT chunks: upload / device / readback; pooling) for one video.

    MOLMO2_VERIF_DIR=<verification dir> pytest models/demos/molmo2/tests/profile_vision_stages.py -s
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


def test_vision_stages(mesh_device, tt_model, processor):
    model, cfg = tt_model
    vp, prompt = _load_first_video_test()
    ids, pv, pool_idx, _ = build_inputs(processor, vp, prompt)
    B, n_crops, n_patches, px = pv.shape
    logger.info(f"S={ids.shape[1]} crops={n_crops} pooled windows={pool_idx.shape[1]} pool={pool_idx.shape[2]}")
    for it in range(3):  # 0 = compile
        pv4 = pv.reshape(B * n_crops, 1, n_patches, px).to(torch.bfloat16)
        nd = mesh_device.get_num_devices()
        up = dev = rb = 0.0
        for s in range(0, n_crops, nd):
            c = pv4[s : s + nd]
            if c.shape[0] < nd:
                c = torch.cat([c, torch.zeros(nd - c.shape[0], 1, n_patches, px, dtype=torch.bfloat16)])
            t0 = time.perf_counter()
            x = ttnn.from_torch(
                c,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            f = model.vit_encoder.forward(x, n_crops_per_device=1)
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()
            _ = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(f)]
            t3 = time.perf_counter()
            ttnn.deallocate(f)
            up += t1 - t0
            dev += t2 - t1
            rb += t3 - t2
        n_chunks = (n_crops + nd - 1) // nd
        vit_cpu = torch.randn(B, n_crops, n_patches, 2304)
        t4 = time.perf_counter()
        model._run_chunked_ttnn_pooling(vit_cpu, pool_idx)
        t5 = time.perf_counter()
        logger.info(
            f"iter {it}: {n_chunks} ViT chunks: upload {up*1e3:.0f} ms, device {dev*1e3:.0f} ms "
            f"({dev*1e3/n_chunks:.0f}/chunk), readback+float {rb*1e3:.0f} ms; pooling {1e3*(t5-t4):.0f} ms"
        )
