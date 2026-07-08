# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh (QB2, flat TP=4) PCC test for `hunyuan_video15_transformer_block`.

Validates the Megatron-style column/row-parallel sharding added for QB2 (see
`real_weights/README.md` "RESUME ON QB2") against the same torch reference the
single-device test (`test_hunyuan_video15_transformer_block.py`) uses: column-
parallel QKV/ff1 (no communication), row-parallel out-proj/ff2 with an
all-reduce (reduce_scatter + all_gather) across all 4 mesh devices.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5._stubs import hunyuan_video15_transformer_block as stub
from models.demos.hf_eager.hunyuanvideo_1_5.tests.pcc._reference_loader import load_reference_model
from models.tt_dit.parallel.manager import CCLManager

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"
PCC_TARGET = 0.99


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_hunyuan_video15_transformer_block_mesh_tp4(mesh_device):
    torch.manual_seed(0)
    model = load_reference_model(HF_MODEL_ID)
    blk = model.transformer_blocks[0]
    C = model.config.num_attention_heads * model.config.attention_head_dim

    B, Limg, Ltxt = 1, 64, 32
    h = torch.randn(B, Limg, C)
    e = torch.randn(B, Ltxt, C)
    temb = torch.randn(B, C)
    # HunyuanVideo15AttnProcessor2_0 unconditionally F.pad()s attention_mask;
    # an all-ones (all-valid) mask gives unmasked joint attention, matching
    # what the ttnn stub always runs (it never applies attention_mask).
    attn_mask = torch.ones(B, Limg + Ltxt, dtype=torch.bool)

    with torch.no_grad():
        h_ref, e_ref = blk(hidden_states=h, encoder_hidden_states=e, temb=temb, attention_mask=attn_mask)

    tp = mesh_device.get_num_devices()
    assert tp > 1, f"expected a real multi-device mesh, got tp={tp}"
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    fwd = stub.build(mesh_device, blk, ccl_manager=ccl_manager, tp=tp)

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    h_tt = ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper)
    e_tt = ttnn.from_torch(e, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper)
    t_tt = ttnn.from_torch(
        temb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper
    )

    h_out, e_out = fwd(hidden_states=h_tt, encoder_hidden_states=e_tt, temb=t_tt)

    ttnn.synchronize_device(mesh_device)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    # h_out/e_out are REPLICATED across the mesh (identical on every device
    # after the block's internal all-reduces); take the first of the `tp`
    # concatenated copies.
    h_out_t = ttnn.to_torch(h_out, mesh_composer=composer).float()[:B]
    e_out_t = ttnn.to_torch(e_out, mesh_composer=composer).float()[:B]

    ok_h, pcc_h = comp_pcc(h_ref, h_out_t, PCC_TARGET)
    ok_e, pcc_e = comp_pcc(e_ref, e_out_t, PCC_TARGET)
    print(f"[mesh tp={tp}] hidden-stream PCC={pcc_h} encoder-stream PCC={pcc_e} target={PCC_TARGET}", flush=True)
    assert ok_h, f"hidden-stream PCC {pcc_h} below target {PCC_TARGET}"
    assert ok_e, f"encoder-stream PCC {pcc_e} below target {PCC_TARGET}"
