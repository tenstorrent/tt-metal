# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0227: the seed server's MTP layer with its latent rows in HOST memory (``HostLatentCache``) exports the
same bytes as with the replicated device cache (``TtLatentCache``).

Two cache-only MTP blocks (``GLM53F_MTP_HOST_LATENT`` 1 and 0) take the same normed hidden + next ids, chunk by chunk as
the server's ``_mtp_hook`` feeds them (this mesh row's slab of each): a full 16 K chunk at 0 and a ragged one after it.
The exported state (``_dsa_state_host``: latent rows, pooled-key index, index rings) must be bitwise equal.

    GLM53F_HL_LEN=20624 pytest models/demos/glm53_flash/tests/test_mtp_host_latent.py
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, shard_headers
from models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden import _dsa_state_host
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import HostLatentCache, TtLatentCache
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.mtp import TtMTPBlock

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
LEN = int(os.environ.get("GLM53F_HL_LEN", "20624"))  # 16384 + 4240: a full chunk and the 250000 prompt's ragged tail
CHUNK = 16384
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mtp_host_latent_bitwise(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    headers = shard_headers(snapshot)
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    t0 = time.time()
    emb = TtEmbedding.from_checkpoint(mesh_device, cfg, snapshot, headers)
    head = TtModelHead.from_checkpoint(mesh_device, cfg, snapshot, headers, mesh=mesh)
    blocks = {}
    for mode in ("1", "0"):
        os.environ["GLM53F_MTP_HOST_LATENT"] = mode
        blocks[mode] = TtMTPBlock.from_checkpoint(
            mesh_device,
            cfg,
            snapshot,
            headers,
            embedding=emb,
            head=head,
            weight_cache_root=None if _cache_env.lower() == "none" else _cache_env,
            mesh=mesh,
            max_batch=1,
            max_seq_len=LEN + CHUNK,
            cache_only=True,
        )
    os.environ.pop("GLM53F_MTP_HOST_LATENT", None)
    assert isinstance(blocks["1"].cache, HostLatentCache) and isinstance(blocks["0"].cache, TtLatentCache)
    logger.info(f"embedding + head + 2 cache-only MTP blocks built in {time.time() - t0:.0f} s")

    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(LEN, cfg.hidden_size, generator=g) * 0.5  # stands in for the normed final hidden H
    nxt = torch.randint(0, cfg.vocab_size, (LEN,), generator=g)
    host_s = {"1": 0.0, "0": 0.0}
    for s0 in range(0, LEN, CHUNK):
        e = min(LEN, s0 + CHUNK)
        n = e - s0
        unit = 32 * rows
        T_pad = -(-n // unit) * unit
        h = torch.zeros(1, 1, T_pad, cfg.hidden_size)
        h[0, 0, :n] = hidden[s0:e]
        for mode, blk in blocks.items():
            hn = (
                mesh.upload(h, ttnn.bfloat16, rows=2)
                if rows > 1
                else ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            )
            ids_host = emb.tokens_to_host(nxt[s0:e], pad_to=T_pad)
            ids_dev = mesh.upload(ids_host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            blk.prefill_state_rows(ids_dev, hn, user=0, seq_len=n, start=s0)
            ttnn.synchronize_device(mesh_device)
            host_s[mode] += time.perf_counter() - t1
            ttnn.deallocate(ids_dev)
            ttnn.deallocate(hn)
    model = SimpleNamespace(mesh=mesh)
    st = {mode: _dsa_state_host(model, SimpleNamespace(cache=blk.cache), LEN) for mode, blk in blocks.items()}
    for k in st["0"]:
        a, b = st["1"][k], st["0"][k]
        same = a.shape == b.shape and torch.equal(
            a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
            b.view(torch.int16) if b.dtype == torch.bfloat16 else b,
        )
        logger.info(
            f"{k}: host-latent {tuple(a.shape)} vs device {tuple(b.shape)}: {'BITWISE EQUAL' if same else 'DIFFER'}"
        )
        assert same, k
    logger.info(
        f"[G53F-0227] host latent == device cache over {LEN} rows ({-(-LEN // CHUNK)} chunks); prefill_state_rows wall "
        f"host-latent {host_s['1']:.3f} s vs device {host_s['0']:.3f} s"
    )
