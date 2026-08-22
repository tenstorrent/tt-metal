# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B parity for the decode CCL diet (QWEN36_CCL_DIET): fused out-proj matmul+reduce-scatter
vs the unfused matmul + tt_all_reduce pair, at the exact TP decode shapes of each site.

Per-site op-level parity covers all three sites (attn_rs / gdn_rs / mlp_rs) including the
fp32 activation the fused-GDN decode feeds its out-proj (fp32 partial sums are PCC-load-bearing
for the row-parallel reduce — see matmul_reduce_scatter_prefill). Module-level parity covers the
MLP forward and the GDN forward_decode in BOTH baseline modes (composite and
QWEN36_GDN_FUSED_DECODE=1), so the diet is validated on the path the campaign actually runs.

Run (P150x8):
    MESH_DEVICE=P150x8 HF_MODEL=Qwen/Qwen3.6-27B \
        pytest models/demos/blackhole/qwen36/tests/test_ccl_diet.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_gdn_layer,
    load_mlp_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

# Fused-vs-unfused arms run the same matmul math and the same RS reduction; only op scheduling
# differs, so parity is held tight. The torch check is the usual bf8-weight envelope.
_AB_PCC = 0.9999
_TORCH_PCC = 0.99


def _mlp_decode_ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )


# (K_local getter, activation dtype, compute kernel config factory) per diet site.
_SITES = {
    "attn_rs": (lambda args, nd: args.attn_out_dim_tp, ttnn.bfloat16, lambda: tpc.COMPUTE_HIFI2),
    "gdn_rs": (lambda args, nd: args.gdn_value_dim_tp, ttnn.float32, lambda: tpc.COMPUTE_HIFI2),
    "mlp_rs": (lambda args, nd: args.hidden_dim // nd, ttnn.bfloat16, _mlp_decode_ckc),
}


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("site", sorted(_SITES), ids=sorted(_SITES))
@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_ccl_diet_out_proj_op(mesh_device, site, B, reset_seeds, ensure_gc):
    """Op-level A/B: matmul_reduce_scatter_out_proj_decode vs linear + tt_all_reduce."""
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("CCL diet is TP-only")
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    k_fn, act_dtype, ckc_fn = _SITES[site]
    K_local, N = k_fn(args, nd), args.dim
    ckc = ckc_fn()
    tt_ccl = TT_CCL(mesh_device)
    topology = args.ccl_topology()
    logger.info(f"site={site} B={B} K_local={K_local} N={N} act_dtype={act_dtype}")

    torch_dtype = torch.float32 if act_dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(1, 1, B, K_local * nd, dtype=torch_dtype)
    w = (torch.randn(K_local * nd, N, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    x_tt = ttnn.from_torch(
        x,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    fused = tpc.matmul_reduce_scatter_out_proj_decode(x_tt, w_tt, tt_ccl, ckc, topology, nd)

    partial = ttnn.linear(x_tt, w_tt, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    unfused = tt_all_reduce(
        partial,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert list(fused.shape) == list(unfused.shape), f"shape mismatch {fused.shape} vs {unfused.shape}"
    composer = tp_composer(mesh_device)
    fused_t = ttnn.to_torch(fused, mesh_composer=composer)[0, 0].to(torch.float32)
    unfused_t = ttnn.to_torch(unfused, mesh_composer=composer)[0, 0].to(torch.float32)
    ref = x[0, 0].to(torch.float32) @ w.to(torch.float32)

    ab_ok, ab_pcc = comp_pcc(unfused_t, fused_t, _AB_PCC)
    ref_ok, ref_pcc = comp_pcc(ref, fused_t, _TORCH_PCC)
    logger.info(f"site={site} B={B}: fused-vs-unfused PCC={ab_pcc}, fused-vs-torch PCC={ref_pcc}")
    assert ab_ok, f"{site} B={B}: fused vs unfused PCC too low: {ab_pcc}"
    assert ref_ok, f"{site} B={B}: fused vs torch PCC too low: {ref_pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_ccl_diet_mlp_forward(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Module-level A/B: Qwen36MLP decode forward with mlp_rs on vs off."""
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("CCL diet is TP-only")
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=256)
    mlp_state = load_mlp_layer(args.CKPT_DIR, 0)
    tt_ccl = TT_CCL(mesh_device)

    B = 8
    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)

    monkeypatch.delenv("QWEN36_CCL_DIET", raising=False)
    mlp_base = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)
    out_base = mlp_base.forward(replicate_to_device(mesh_device, x))

    monkeypatch.setenv("QWEN36_CCL_DIET", "mlp_rs")
    mlp_diet = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)
    assert mlp_diet._ccl_diet_w2, "diet flag did not engage"
    out_diet = mlp_diet.forward(replicate_to_device(mesh_device, x))

    composer = tp_composer(mesh_device)
    base_t = ttnn.to_torch(out_base, mesh_composer=composer)[0, 0].to(torch.float32)
    diet_t = ttnn.to_torch(out_diet, mesh_composer=composer)[0, 0].to(torch.float32)
    ok, pcc = comp_pcc(base_t, diet_t, _AB_PCC)
    logger.info(f"MLP decode diet-vs-base PCC = {pcc}")
    assert ok, f"MLP decode diet vs base PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("gdn_mode", ["composite", "fused"], ids=["composite", "fused"])
def test_ccl_diet_gdn_forward_decode(mesh_device, gdn_mode, monkeypatch, reset_seeds, ensure_gc):
    """Module-level A/B: TPGatedDeltaNet.forward_decode with gdn_rs on vs off, in both the
    composite path and the QWEN36_GDN_FUSED_DECODE=1 path the campaign baseline runs."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("CCL diet is TP-only")
    B = 8
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    sd = load_gdn_layer(args.CKPT_DIR, 0)
    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args, cache_dir=None)

    if gdn_mode == "fused":
        monkeypatch.setenv("QWEN36_GDN_FUSED_DECODE", "1")
    else:
        monkeypatch.delenv("QWEN36_GDN_FUSED_DECODE", raising=False)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)

    def run(diet):
        if diet:
            monkeypatch.setenv("QWEN36_CCL_DIET", "gdn_rs")
        else:
            monkeypatch.delenv("QWEN36_CCL_DIET", raising=False)
        gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        if diet:
            assert gdn._ccl_diet_out, "diet flag did not engage"
        if gdn_mode == "fused":
            assert gdn._fused_decode, "fused-decode baseline did not engage"
        gdn.reset_state()
        out = gdn.forward_decode(replicate_to_device(mesh_device, x))
        return ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].to(torch.float32)

    base_t = run(diet=False)
    diet_t = run(diet=True)
    ok, pcc = comp_pcc(base_t, diet_t, _AB_PCC)
    logger.info(f"GDN decode ({gdn_mode}) diet-vs-base PCC = {pcc}")
    assert ok, f"GDN decode ({gdn_mode}) diet vs base PCC too low: {pcc}"
