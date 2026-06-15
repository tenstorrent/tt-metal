# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A6 localization: does the compressed-latent paged cache WRITE round-trip? Writes a known latent
per user at pos 0 via paged_update_cache, reads the cache back, compares. Isolates the cache-write
from the flash-MLA op (the step-0 PCC was ~0.018, a single-step bug)."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4MLA

B = 32


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_mla_cache_rt(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)
    d = mla.kvl + mla.rope
    cache, page_table = mla.init_compressed_cache(B, max_seq=64)  # [nb,1,32,d], pt[B,bpu]

    # a distinct known latent per user (so a mis-mapped write is obvious)
    torch.manual_seed(0)
    lat = torch.randn(B, 1, 1, d, dtype=torch.bfloat16) * 0.1
    cores = ttnn.num_cores_to_corerangeset(B, mesh_device.compute_with_storage_grid_size(), row_wise=True)
    mem = ttnn.create_sharded_memory_config(
        shape=[32, d],
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    lat_in = ttnn.to_memory_config(
        ttnn.from_torch(
            lat.permute(1, 0, 2, 3),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        mem,
    )  # [1,B,1,d]
    pos = ttnn.from_torch(
        torch.zeros(B, dtype=torch.int32), device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ttnn.experimental.paged_update_cache(cache, lat_in, update_idxs_tensor=pos, page_table=page_table)

    pt = ttnn.to_torch(page_table, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B]
    c = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))  # device0 copy is [nb,1,32,d]
    nb = c.shape[0]
    # user u, pos 0 -> physical block pt[u,0], row 0
    read = torch.stack([c[int(pt[u, 0]), 0, 0, :] for u in range(B)]).float()  # [B,d]
    passing, msg = comp_pcc(lat.reshape(B, d).float(), read, 0.99)
    logger.info(f"compressed-cache write round-trip PCC (B={B}): {msg}")
    assert passing, f"cache write did not round-trip: {msg}"
