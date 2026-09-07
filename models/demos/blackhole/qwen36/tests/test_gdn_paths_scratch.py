# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: in-layer A/B of the GDN prefill variants (conv2d chain vs KDA fused conv; MMRS fp32 vs AGMM
out-proj) on one real GDN layer, same input, T=2048, TP=4. Reports PCC of the layer output and of the
conv q/k/v against the default path. Run with MESH_DEVICE=P150x4 and the HF_MODEL env of the checkpoint."""
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_paths(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    T = 2048
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    # Force both out-proj weights to exist.
    os.environ["QWEN36_GDN_OUT_MODE"] = "agmm"
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    os.environ["QWEN36_GDN_OUT_MODE"] = "mmrs_fp32"
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)  # works for the 3D conv tensors and the 4D layer output

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    # ---- conv q/k/v: conv2d chain vs KDA on the same qkv ----
    qkv, z, a, b = gdn._project_qkvzab(x_tt, T, out_mc=ttnn.L1_MEMORY_CONFIG)
    conv, ns_a = gdn._conv1d_prefill(qkv, T, None)
    kd = gdn.key_dim_tp
    q_a = ttnn.to_torch(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), mesh_composer=comp).float()
    v_a = ttnn.to_torch(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, gdn.qkv_dim_tp)), mesh_composer=comp).float()
    (q_b, k_b, v_b), ns_b = gdn._conv1d_prefill_kda(qkv, T, None)
    q_bt = ttnn.to_torch(q_b, mesh_composer=comp).float()
    v_bt = ttnn.to_torch(v_b, mesh_composer=comp).float()
    ns_at = ttnn.to_torch(ns_a, mesh_composer=comp).float()
    ns_bt = ttnn.to_torch(ns_b, mesh_composer=comp).float()
    for name, ref, out in (("conv q", q_a, q_bt), ("conv v", v_a, v_bt), ("new_state", ns_at, ns_bt)):
        _, p = comp_pcc(ref, out, 0.99)
        logger.info(
            f"GDN_PATHS {name:10s} conv2d vs KDA: PCC={p} max|d|={float((ref - out).abs().max()):.4f} shapes {tuple(ref.shape)} {tuple(out.shape)}"
        )
    # Also: KDA with a nonzero history vs conv2d with the same carry
    carry = torch.randn(1, 3, gdn.qkv_dim_tp * 4, dtype=torch.bfloat16)
    carry_tt = shard_to_device(mesh_device, carry, dim=-1)  # [1,3,C] TILE per device
    conv2, _ = gdn._conv1d_prefill(qkv, T, carry_tt)
    (q_c, _, _), _ = gdn._conv1d_prefill_kda(qkv, T, carry_tt)
    q_2 = ttnn.to_torch(ttnn.slice(conv2, (0, 0, 0), (1, T, kd)), mesh_composer=comp).float()
    q_ct = ttnn.to_torch(q_c, mesh_composer=comp).float()
    _, p = comp_pcc(q_2, q_ct, 0.99)
    logger.info(f"GDN_PATHS conv q with carry: conv2d vs KDA PCC={p} max|d|={float((q_2 - q_ct).abs().max()):.4f}")
    logger.info(
        f"GDN_PATHS first-3-rows diff (carry effect region): {float((q_2[..., :3, :] - q_ct[..., :3, :]).abs().max()):.4f}"
    )

    # ---- KDA TILE input (QWEN36_KDA_TILE_IN) vs ROW_MAJOR input: must be BIT-identical ----
    # The TILE kernels shift rows by matmul against exact 0/1 matrices, so every tap product and the bf16
    # partial accumulation are unchanged. Anything but max|d| == 0 is a bug, not a rounding difference.
    _saved_tile_in = gdn._kda_tile_in
    _saved_zero_hist = gdn._kda_zero_hist
    try:
        for label, cstate in (("zero-hist", None), ("carry", carry_tt)):
            gdn._kda_tile_in, gdn._kda_zero_hist = False, None
            (q_rm, k_rm, v_rm), ns_rm = gdn._conv1d_prefill_kda(qkv, T, cstate)
            gdn._kda_tile_in, gdn._kda_zero_hist = True, None
            (q_ti, k_ti, v_ti), ns_ti = gdn._conv1d_prefill_kda(qkv, T, cstate)
            for name, a_t, b_t in (
                ("q", q_rm, q_ti),
                ("k", k_rm, k_ti),
                ("v", v_rm, v_ti),
                ("new_state", ns_rm, ns_ti),
            ):
                a_h = ttnn.to_torch(a_t, mesh_composer=comp).float()
                b_h = ttnn.to_torch(b_t, mesh_composer=comp).float()
                d = float((a_h - b_h).abs().max())
                nmm = int((a_h != b_h).sum())
                logger.info(
                    f"GDN_PATHS TILE_IN {label:9s} {name:9s} RM vs TILE: max|d|={d:.6g} mismatches={nmm}"
                    f" {'BIT-IDENTICAL' if nmm == 0 else 'DIFFERS'}"
                )
            for t in (q_rm, k_rm, v_rm, ns_rm, q_ti, k_ti, v_ti, ns_ti):
                ttnn.deallocate(t)
    finally:
        gdn._kda_tile_in, gdn._kda_zero_hist = _saved_tile_in, _saved_zero_hist

    # ---- full layer output: default vs KDA conv vs AGMM out-proj ----
    def run(conv_kda, out_mode):
        gdn._gdn_conv_kda = conv_kda
        gdn._gdn_out_mode = out_mode
        gdn._fuse_out_mmrs_prefill = out_mode.startswith("mmrs")
        gdn._mmrs_dtype = ttnn.float32
        gdn.reset_state()
        out = gdn.forward_prefill(x_tt, chunk_size=128)
        return ttnn.to_torch(out, mesh_composer=comp)[0, 0].float()

    ref = run(False, "mmrs_fp32")
    for label, ck, om in (
        ("kda", True, "mmrs_fp32"),
        ("agmm", False, "agmm"),
        ("ag_mm", False, "ag_mm"),
        ("rs_bf16", False, "rs_bf16"),
    ):
        out = run(ck, om)
        _, p = comp_pcc(ref, out, 0.99)
        logger.info(
            f"GDN_PATHS layer out {label:8s} vs default: PCC={p} max|d|={float((ref - out).abs().max()):.4f} ref_max={float(ref.abs().max()):.3f} out_max={float(out.abs().max()):.3f}"
        )
