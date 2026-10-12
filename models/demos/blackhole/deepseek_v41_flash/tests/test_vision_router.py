# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Image-aware routing (DSV41Gate._forward_fused(x, vl=...)): tokens inside an image span rank their experts with ``bias_vl`` instead of ``bias`` (checkpoint ``Gate``).
Real gate weights of layer VIS_ROUTER_LAYER (default 0), random activations, half of the tokens flagged as image tokens:
  * rows with vl = 0 are bit-identical to the plain call (no vl),
  * expert sets of the vl rows agree with the host computation using bias_vl as often as the plain rows agree with the host computation using bias (the device router
    differs from the fp32 host only by the matmul's tf32 operand truncation), and clearly more often than with the wrong bias."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYER = int(os.environ.get("VIS_ROUTER_LAYER", "0"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1600000000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
def test_vision_router(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()
        raise


@torch.no_grad()
def _body(mesh_device):
    os.environ["DSV41_VISION"] = "1"
    md = mesh_device
    U, B = 4, 16
    pp = default_page_params(1024, U)
    _, m, _, _ = create_tt_model(
        md,
        B,
        1024,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=[LAYER],
        log=lambda s: None,
    )
    gate = m.prefill_model.layers[0][1].L.moe.gate
    sh = _Shards()
    p = f"layers.{LAYER}.ffn.gate."
    W = sh.get(p + "weight").float()  # [384, 5120]
    bias, bias_vl = sh.get(p + "bias").float(), sh.get(p + "bias_vl").float()
    print(
        f"|bias_vl - bias|: max {(bias_vl - bias).abs().max():.3f}, mean {(bias_vl - bias).abs().mean():.3f}",
        flush=True,
    )
    T = 256
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 1, T, 5120, generator=g).to(torch.bfloat16)
    mask = (torch.arange(T) % 2).bool()  # alternate tokens

    up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    xd = up(x, ttnn.bfloat16)
    vl = up(mask.float().reshape(1, 1, T, 1), ttnn.float32)

    def idx_of(ix):  # [T,1,1,k] uint16 row-major, replicated: device 0
        return ttnn.to_torch(ttnn.get_device_tensors(ix)[0]).reshape(T, -1).long()

    _, ix_plain = gate._forward_fused(xd, grid=(2, 8))
    _, ix_vl = gate._forward_fused(xd, grid=(2, 8), vl=vl)
    ip, iv = idx_of(ix_plain), idx_of(ix_vl)
    srt = lambda a: a.sort(-1).values
    assert torch.equal(srt(ip[~mask]), srt(iv[~mask])), "text rows changed by the vl path"

    score = torch.sqrt(torch.nn.functional.softplus(x.reshape(T, 5120).float() @ W.T))
    host = lambda b: (score + b).topk(6, dim=-1).indices
    agree = lambda a, b: (srt(a) == srt(b)).all(-1).float().mean().item()
    a_text = agree(ip[~mask], host(bias)[~mask])
    a_img = agree(iv[mask], host(bias_vl)[mask])
    a_wrong = agree(iv[mask], host(bias)[mask])
    print(
        f"expert-set agreement with the host: text rows (bias) {a_text:.3f}, image rows (bias_vl) {a_img:.3f}, image rows vs the WRONG bias {a_wrong:.3f}",
        flush=True,
    )
    assert a_img > a_text - 0.05 and a_img > a_wrong + 0.1
