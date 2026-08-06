# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware PCC / bit-identity gate for ``HY_DIT_RS_DOMAIN_BIAS``.

The gate moves the row-parallel bias add of ``to_out``, ``to_add_out`` and both
FF down-projections into the scattered domain, between the TP reduce-scatter and
the all-gather.  The reduction is untouched, so the block output must come back
*bit-identical* to the legacy path, not merely within PCC tolerance.  The host
proof is in ``test_row_parallel_bias_dataflow.py``; this is the on-device
confirmation.

Run on 4 devices with the production Linear/FABRIC_1D fabric:

    pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_transformer_block_rs_domain_bias_pcc.py -q
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
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_rs_domain_bias_is_bit_identical_to_legacy_tp4(mesh_device, monkeypatch, bf16):
    torch.manual_seed(0)
    monkeypatch.setenv("HY_DIT_MMRS_OVERLAP", "0")
    monkeypatch.setenv("HY_DIT_BF16", "1" if bf16 else "0")

    model = load_reference_model(HF_MODEL_ID)
    blk = model.transformer_blocks[0]
    C = model.config.num_attention_heads * model.config.attention_head_dim

    B, Limg, Ltxt = 1, 64, 32
    h = torch.randn(B, Limg, C)
    e = torch.randn(B, Ltxt, C)
    temb = torch.randn(B, C)
    attn_mask = torch.ones(B, Limg + Ltxt, dtype=torch.bool)
    with torch.no_grad():
        h_ref, e_ref = blk(hidden_states=h, encoder_hidden_states=e, temb=temb, attention_mask=attn_mask)

    tp = mesh_device.get_num_devices()
    assert tp > 1, f"expected a real multi-device mesh, got tp={tp}"
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Linear)

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)

    h_tt, e_tt, t_tt = to_device(h), to_device(e), to_device(temb)

    monkeypatch.setenv("HY_DIT_RS_DOMAIN_BIAS", "0")
    legacy = stub.build(mesh_device, blk, ccl_manager=ccl, tp=tp)
    monkeypatch.setenv("HY_DIT_RS_DOMAIN_BIAS", "1")
    scattered = stub.build(mesh_device, blk, ccl_manager=ccl, tp=tp)

    legacy_out = legacy(hidden_states=h_tt, encoder_hidden_states=e_tt, temb=t_tt)
    scattered_out = scattered(hidden_states=h_tt, encoder_hidden_states=e_tt, temb=t_tt)
    ttnn.synchronize_device(mesh_device)

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    def gather(t):
        return ttnn.to_torch(t, mesh_composer=composer).float()[:B]

    for name, reference, legacy_t, scattered_t in (
        ("hidden", h_ref, gather(legacy_out[0]), gather(scattered_out[0])),
        ("context", e_ref, gather(legacy_out[1]), gather(scattered_out[1])),
    ):
        ok, achieved = comp_pcc(reference, scattered_t, PCC_TARGET)
        print(f"RS_DOMAIN_BIAS_{name.upper()}_REF_PCC={achieved}", flush=True)
        assert ok, f"{name} reference PCC {achieved} below {PCC_TARGET}"

        max_abs = (legacy_t - scattered_t).abs().max().item()
        print(f"RS_DOMAIN_BIAS_{name.upper()}_MAX_ABS_VS_LEGACY={max_abs}", flush=True)
        # Moving a per-column constant across an all-gather cannot change any
        # bit. A nonzero delta means the fractured bias landed on the wrong
        # device or the reduction was reassociated -- both are real bugs.
        assert max_abs == 0.0, f"{name} differs from legacy by {max_abs}; expected bit-identical"


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_mmrs_overlap_fails_fast_on_the_production_linear_fabric(mesh_device, monkeypatch):
    """The build must reject MMRS on Linear rather than TT_FATAL mid-generation."""
    monkeypatch.setenv("HY_DIT_MMRS_OVERLAP", "1")
    monkeypatch.setenv("HY_DIT_BF16", "1")
    model = load_reference_model(HF_MODEL_ID)
    blk = model.transformer_blocks[0]
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    with pytest.raises(ValueError, match="Ring CCL topology"):
        stub.build(mesh_device, blk, ccl_manager=ccl, tp=mesh_device.get_num_devices())
