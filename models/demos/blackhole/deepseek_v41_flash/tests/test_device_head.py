# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device embedding and LM head vs the host versions / the reference chain.
  embedding: decode tokens of the chain -> streams, pre (exact: a table lookup)
  head     : the reference LAST layer output (+ its pre) -> logits PCC vs final.pt, argmax agreement with the CPU head."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEmbedding, HostHead
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_device_embedding_and_head(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    sh = _Shards()
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))["decode_tokens"].reshape(-1)  # [16]
    B = toks.shape[0]

    emb = DSV41DeviceEmbedding(md, sh.get("embed.weight"))
    streams, pre = emb.forward(emb.upload_tokens(toks))
    dev = lambda t: torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]) for r in range(rows)]).float()
    got_s, got_p = dev(streams).reshape(B, 1, 4, -1), dev(pre).reshape(B, 1, 4)
    ref_s, ref_p = HostEmbedding()(toks.reshape(B, 1))
    print(
        "EMB streams PCC",
        R.pcc(got_s, ref_s.float()),
        "max|diff|",
        (got_s - ref_s.float()).abs().max().item(),
        "pre equal",
        torch.equal(got_p, ref_p.float()),
        flush=True,
    )
    assert torch.equal(got_s, ref_s.float()) and torch.equal(got_p, ref_p.float())

    last = torch.load(os.path.join(CHAIN, "layer_39.pt"))
    h, p = last["dec_out"].float().reshape(B, 1, 4, -1), last["pre_out"].float().reshape(B, 1, 4)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    logits = head.forward(up(h), up(p.reshape(B, 1, 1, 4)))
    got = head.gather_logits(logits)[:B]
    fin = torch.load(os.path.join(CHAIN, "final.pt"))
    host_logits = HostHead()(h, p)
    print("HEAD logits PCC vs host head", R.pcc(got, host_logits), "vs final.pt", R.pcc(got, fin["logits"]), flush=True)
    am_dev, am_host = head.argmax(logits)[:B], host_logits.argmax(-1)
    print(
        "HEAD argmax device==host:",
        int((am_dev == am_host).sum()),
        "/",
        B,
        " device-gathered argmax==device argmax:",
        int((got.argmax(-1) == am_dev).sum()),
        "/",
        B,
        flush=True,
    )
    assert R.pcc(got, host_logits) > 0.999
