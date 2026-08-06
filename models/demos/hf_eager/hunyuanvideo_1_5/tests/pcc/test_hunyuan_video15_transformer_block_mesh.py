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

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5._stubs import hunyuan_video15_transformer_block as stub
from models.demos.hf_eager.hunyuanvideo_1_5.tests.pcc._reference_loader import load_reference_model
from models.tt_dit.parallel.manager import CCLManager

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"
PCC_TARGET = 0.99


_LINEAR_PARAMS = {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
# The fused MM+RS kernel TT_FATALs on anything but a Ring topology
# (minimal_matmul_strided_reduce_scatter_async_op.cpp), and a Ring needs the
# wrap-around fabric. The production Hunyuan pipeline runs FABRIC_1D/Linear, so
# this parametrization is a component test of the kernel, not of production.
_RING_PARAMS = {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}


@pytest.mark.parametrize("mesh_device", [4], indirect=True)
@pytest.mark.parametrize(
    "device_params,mmrs_overlap,bf16,topology",
    [
        (_LINEAR_PARAMS, False, False, ttnn.Topology.Linear),
        (_LINEAR_PARAMS, False, True, ttnn.Topology.Linear),
        (_RING_PARAMS, True, True, ttnn.Topology.Ring),
    ],
    ids=["legacy-fp32-linear", "legacy-bf16-linear", "hunyuan-mmrs-bf16-ring"],
    indirect=["device_params"],
)
def test_hunyuan_video15_transformer_block_mesh_tp4(mesh_device, monkeypatch, mmrs_overlap, bf16, topology):
    torch.manual_seed(0)
    monkeypatch.setenv("HY_DIT_MMRS_OVERLAP", "1" if mmrs_overlap else "0")
    monkeypatch.setenv("HY_DIT_BF16", "1" if bf16 else "0")
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
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=topology)
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
    print(
        f"[mesh tp={tp} mmrs={mmrs_overlap}] hidden-stream PCC={pcc_h} "
        f"encoder-stream PCC={pcc_e} target={PCC_TARGET}",
        flush=True,
    )
    assert ok_h, f"hidden-stream PCC {pcc_h} below target {PCC_TARGET}"
    assert ok_e, f"encoder-stream PCC {pcc_e} below target {PCC_TARGET}"


@pytest.mark.parametrize("device_params", [_RING_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_hunyuan_mmrs_representative_121f_local_shape(mesh_device, monkeypatch):
    """Opt-in per-block A/B at the 480p/121f SP8 local latent shape.

    Run only on an idle Galaxy with ``HY_DIT_MMRS_121F_BENCH=1``. Set
    ``HY_DIT_MMRS_BENCH_TOKENS=13952`` for the 720p local shape.

    Both arms run on a Ring fabric because the fused kernel requires it. That
    is NOT the production configuration (the pipeline builds FABRIC_1D/Linear),
    so treat the delta as an upper bound on what MMRS could buy if the whole
    pipeline were moved to a ring.
    """
    if os.environ.get("HY_DIT_MMRS_121F_BENCH", "0") != "1":
        pytest.skip("set HY_DIT_MMRS_121F_BENCH=1 only after confirming hardware is idle")

    torch.manual_seed(0)
    model = load_reference_model(HF_MODEL_ID)
    blk = model.transformer_blocks[0]
    C = model.config.num_attention_heads * model.config.attention_head_dim
    B = 1
    Limg = int(os.environ.get("HY_DIT_MMRS_BENCH_TOKENS", "6176"))
    Ltxt = int(os.environ.get("HY_DIT_MMRS_BENCH_CONTEXT_TOKENS", "768"))
    iters = int(os.environ.get("HY_DIT_MMRS_BENCH_ITERS", "3"))

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    h = ttnn.from_torch(
        torch.randn(B, Limg, C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )
    e = ttnn.from_torch(
        torch.randn(B, Ltxt, C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )
    temb = ttnn.from_torch(
        torch.randn(B, C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)

    monkeypatch.setenv("HY_DIT_BF16", "1")
    monkeypatch.setenv("HY_DIT_MMRS_OVERLAP", "0")
    legacy = stub.build(mesh_device, blk, ccl_manager=ccl, tp=4)
    monkeypatch.setenv("HY_DIT_MMRS_OVERLAP", "1")
    overlap = stub.build(mesh_device, blk, ccl_manager=ccl, tp=4)

    def run_and_time(forward):
        output = forward(h, encoder_hidden_states=e, temb=temb)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iters):
            output = forward(h, encoder_hidden_states=e, temb=temb)
        ttnn.synchronize_device(mesh_device)
        return output, (time.perf_counter() - start) * 1000.0 / iters

    legacy_out, legacy_ms = run_and_time(legacy)
    overlap_out, overlap_ms = run_and_time(overlap)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    for name, expected, actual in (
        ("hidden", legacy_out[0], overlap_out[0]),
        ("context", legacy_out[1], overlap_out[1]),
    ):
        expected_t = ttnn.to_torch(expected, mesh_composer=composer).float()[:B]
        actual_t = ttnn.to_torch(actual, mesh_composer=composer).float()[:B]
        ok, achieved = comp_pcc(expected_t, actual_t, PCC_TARGET)
        assert ok, f"{name} overlap/legacy PCC {achieved} below {PCC_TARGET}"
        print(f"MMRS_{name.upper()}_PCC={achieved}", flush=True)

    delta = 100.0 * (legacy_ms - overlap_ms) / legacy_ms
    print(
        f"MMRS_LOCAL_TOKENS={Limg} LEGACY_BLOCK_MS={legacy_ms:.4f} "
        f"OVERLAP_BLOCK_MS={overlap_ms:.4f} DELTA_PERCENT={delta:.2f}",
        flush=True,
    )
