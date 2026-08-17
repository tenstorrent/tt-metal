# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# block_A_conv_blocks3_clip  —  FULL IR reproduction (const-eval + forward + torch golden)
#
# Source: BEV_TRACY_CONV3_CLIP/ttnn_block_A_conv_blocks3_clip_annotated.mlir
#
# Reproduces the ENTIRE mlir, not just @forward:
#   * const-eval weight/bias prep:
#       - yuv adapter weight  (%0)  : block-diagonal [3,3]->[96,96]   (cpu_hoisted, K=32) reproduced in torch
#       - yuv adapter bias    (%6)  : repeat_interleave(32)  [3]->[96]
#       - packed conv weight  (%7)  : block-diagonal [64,24]->[256,96] (cpu_hoisted, K=4) reproduced in torch
#       - packed conv bias    (%3)  : repeat_interleave(4)   [64]->[256]
#       - avgpool-11 weight   (%1)  : repeat_interleave(16, dim0) [2,1,2,1]->[32,1,2,1] + prepare_conv2d_weights
#       - avgpool-23 weight   (%5)  : prepare_conv2d_weights  [64,1,2,2]
#       - conv %45 / %47 W,b  (%2,%4,%8,%9) : prepare_conv2d_weights / prepare_conv2d_bias
#   * @forward %11..%50 op-by-op (linear, slices, 2x pixel_unshuffle, concat, packed conv,
#       B3/B2/B1 unpack permutes, avgpool, conv g2, conv, relu6, permute)
#   * torch golden (logical model — the packing/unpack is a lossless rearrangement) + PCC check
#
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest <this file>

import pytest
import torch
import torch.nn.functional as F
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
BF8 = ttnn.bfloat8_b
SPATIAL = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR
RELU6 = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6)
SLICE = ttnn.Conv2dL1FullSliceConfig


# --------------------------------------------------------------------------- const-eval helpers
def _pack_w_pointwise(w, ic, oc, K):
    """Block-diagonal 1x1 packing: [OC,IC,1,1] -> [IC*K, OC*K] (cpu_hoisted const-eval reproduction)."""
    w_bc = w.float().reshape(oc, ic, 1, 1).expand(oc, ic, K, K)
    k = torch.arange(K)
    diag = (k.reshape(1, 1, K, 1) == k.reshape(1, 1, 1, K)).float()
    return (w_bc * diag).permute(1, 2, 0, 3).reshape(ic * K, oc * K).to(torch.bfloat16)


def _pack_b_pointwise(b, oc, K):
    return b.reshape(oc).repeat_interleave(K).to(torch.bfloat16)  # [OC] -> [OC*K]


def _conv_cfg(act, adb, wdb, abh):
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,  # IR uses bfp_bf8; bf16 here so standalone prepare_conv_* works
        activation=(RELU6 if act else None),
        deallocate_activation=False,
        act_block_h_override=abh,
        config_tensors_in_dram=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_act_double_buffer=adb,
        enable_weights_double_buffer=wdb,
        enable_kernel_stride_folding=False,
    )


def _compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False
    )


def _run(device):
    torch.manual_seed(42)
    cc = _compute(device)

    # ---- raw ONNX weights (arg*) ----
    x = torch.randn(1, 3, 1536, 1536, dtype=torch.bfloat16)  # arg0 input
    yw = torch.randn(3, 3, 1, 1, dtype=torch.bfloat16) * 0.2  # arg3 yuv weight [OC=3,IC=3]
    yb = torch.randn(3, dtype=torch.bfloat16) * 0.1  # arg4 yuv bias
    w_ap11 = torch.full((2, 1, 2, 1), 0.5, dtype=torch.bfloat16)  # arg1 avgpool11 (k=2x1 avg)
    w_ap23 = torch.full((64, 1, 2, 2), 0.25, dtype=torch.bfloat16)  # arg2 avgpool23 (k=2x2 avg)
    fw = torch.randn(64, 24, 1, 1, dtype=torch.bfloat16) * 0.1  # arg5 packed conv [OC=64,IC=24]
    fb = torch.randn(64, dtype=torch.bfloat16) * 0.1  # arg6 packed conv bias
    w45 = torch.randn(64, 32, 3, 3, dtype=torch.bfloat16) * 0.05  # arg7 conv g2
    b45 = torch.randn(64, dtype=torch.bfloat16) * 0.1  # arg8
    w47 = torch.randn(96, 64, 3, 3, dtype=torch.bfloat16) * 0.05  # arg9 conv
    b47 = torch.randn(96, dtype=torch.bfloat16) * 0.1  # arg10

    # ================= TORCH GOLDEN (logical model) =================
    xf = x.float()
    yuv = F.conv2d(xf, yw.float(), bias=yb.float())  # [1,3,1536,1536]
    y_us = torch.nn.PixelUnshuffle(4)(yuv[:, 0:1])  # [1,16,384,384]
    uv_avg = F.avg_pool2d(yuv[:, 1:3], kernel_size=(2, 1), stride=(2, 2))  # [1,2,768,768]
    uv_us = torch.nn.PixelUnshuffle(2)(uv_avg)  # [1,8,384,384]
    cat = torch.cat([y_us, uv_us], dim=1)  # [1,24,384,384]
    c1 = F.conv2d(cat, fw.float(), bias=fb.float()).clamp(0, 6)  # packed conv relu6 -> [1,64,384,384]
    c2 = F.avg_pool2d(c1, kernel_size=(2, 2), stride=(2, 2))  # avgpool23 -> [1,64,192,192]
    c3 = F.conv2d(c2, w45.float(), bias=b45.float(), groups=2, padding=1).clamp(0, 6)  # [1,64,192,192]
    golden = F.conv2d(c3, w47.float(), bias=b47.float(), padding=1).clamp(0, 6)  # [1,96,192,192]

    # ================= const-eval weight prep =================
    # yuv adapter: block-diag 96x96 linear weight + repeat_interleave(32) bias  (%0, %6)
    ywp = _pack_w_pointwise(yw, 3, 3, 32).reshape(1, 1, 96, 96)  # [96,96] (IC,OC)
    ybp = _pack_b_pointwise(yb, 3, 32).reshape(1, 1, 96, 1)
    tt_ywp = ttnn.from_torch(ywp, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
    tt_ybp = ttnn.from_torch(ybp, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
    lin_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=3,
        out_block_w=36,
        per_core_M=3,
        per_core_N=36,
        fuse_batch=True,
        mcast_in0=True,
        gather_in0=False,
    )

    # packed conv: block-diag [256,96,1,1] weight (%7) + repeat_interleave(4) bias (%3)
    packed_w = _pack_w_pointwise(fw, 24, 64, 4).t().contiguous().reshape(256, 96, 1, 1)  # [OC=256,IC=96]
    packed_b = _pack_b_pointwise(fb, 64, 4)  # [256]

    # avgpool-11 weight: repeat_interleave(16, dim0) [2,1,2,1]->[32,1,2,1]  (%1)
    w_ap11_rep = w_ap11.repeat_interleave(16, dim=0)  # [32,1,2,1]

    cfg_ap11 = _conv_cfg(act=False, adb=False, wdb=False, abh=64)
    cfg_pack = _conv_cfg(act=True, adb=True, wdb=True, abh=64)
    cfg_ap23 = _conv_cfg(act=False, adb=False, wdb=False, abh=64)
    cfg_45 = _conv_cfg(act=True, adb=True, wdb=True, abh=0)
    cfg_47 = _conv_cfg(act=True, adb=True, wdb=True, abh=0)

    # ---- const-eval: standalone prepare_conv2d_weights / prepare_conv2d_bias for every conv ----
    def _prep(w_t, b_t, ic, oc, g, k, s, p, H, W, cfg, in_mem):
        common = dict(
            input_memory_config=in_mem,
            input_layout=TILE,
            in_channels=ic,
            out_channels=oc,
            batch_size=1,
            input_height=H,
            input_width=W,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=(1, 1),
            groups=g,
            device=device,
            input_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            conv_config=cfg,
            compute_config=cc,
            slice_config=SLICE,
        )
        tw = ttnn.prepare_conv_weights(
            weight_tensor=ttnn.from_torch(w_t, dtype=ttnn.bfloat16, layout=RM),
            weights_format="OIHW",
            has_bias=(b_t is not None),
            **common,
        )
        tb = None
        if b_t is not None:
            tb = ttnn.prepare_conv_bias(
                bias_tensor=ttnn.from_torch(b_t.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=RM), **common
            )
        return tw, tb

    #                     w,          b,        ic, oc,  g,  k,      s,      p,             H,   W,    cfg,      in_mem
    tw_ap11, _ = _prep(w_ap11_rep, None, 32, 32, 32, (2, 1), (2, 2), (0, 0, 0, 0), 96, 1536, cfg_ap11, L1)  # %1
    tw_pack, tb_pack = _prep(
        packed_w, packed_b, 96, 256, 1, (1, 1), (1, 1), (0, 0, 0, 0), 96, 384, cfg_pack, DRAM
    )  # %7,%3
    tw_ap23, _ = _prep(w_ap23, None, 64, 64, 64, (2, 2), (2, 2), (0, 0, 0, 0), 384, 384, cfg_ap23, L1)  # %5
    tw45, tb45 = _prep(w45, b45, 64, 64, 2, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_45, DRAM)  # %2,%9
    tw47, tb47 = _prep(w47, b47, 64, 96, 1, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_47, DRAM)  # %4,%8

    def conv(x, tw, tb, ic, oc, g, k, s, p, H, W, cfg):
        return ttnn.conv2d(
            input_tensor=x,
            weight_tensor=tw,
            bias_tensor=tb,
            device=device,
            in_channels=ic,
            out_channels=oc,
            batch_size=1,
            input_height=H,
            input_width=W,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=(1, 1),
            groups=g,
            dtype=ttnn.bfloat16,
            conv_config=cfg,
            compute_config=cc,
            slice_config=SLICE,
        )

    # ================= FORWARD (%11..%50) =================
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    t = ttnn.reshape(t, [1, 1, 96, 73728])  # %11
    t = ttnn.to_layout(t, TILE)  # %12
    t = ttnn.linear(tt_ywp, t, bias=tt_ybp, program_config=lin_pc, compute_kernel_config=cc)  # %13 yuv linear
    t = ttnn.to_memory_config(t, DRAM)  # %14
    t = ttnn.to_layout(t, RM)  # %15
    t = ttnn.reshape(t, [1, 3, 1536, 1536])  # %16
    full = ttnn.to_memory_config(t, DRAM)  # %19

    y = ttnn.slice(t, [0, 0, 0, 0], [1, 1, 1536, 1536], [1, 1, 1, 1])  # %17
    y = ttnn.to_memory_config(y, DRAM)  # %18
    pu4 = ttnn.pixel_unshuffle(y, downscale_factor=4, channel_order=SPATIAL, memory_config=L1)  # %20

    uv = ttnn.slice(full, [0, 1, 0, 0], [1, 3, 1536, 1536], [1, 1, 1, 1])  # %21
    uv = ttnn.reshape(uv, [1, 32, 96, 1536])  # %22
    uv = ttnn.permute(uv, (0, 2, 3, 1))  # %23
    uv = ttnn.reshape(uv, [1, 1, 147456, 32])  # %24
    uv = conv(uv, tw_ap11, None, 32, 32, 32, (2, 1), (2, 2), (0, 0, 0, 0), 96, 1536, cfg_ap11)  # %25
    uv = ttnn.reshape(uv, [1, 48, 768, 32])  # %26
    uv = ttnn.to_memory_config(uv, L1)  # %27
    uv = ttnn.permute(uv, (0, 3, 1, 2))  # %28
    uv = ttnn.reshape(uv, [1, 2, 768, 768])  # %29
    uv = ttnn.to_memory_config(uv, DRAM)  # %30
    pu2 = ttnn.pixel_unshuffle(uv, downscale_factor=2, channel_order=SPATIAL, memory_config=L1)  # %31

    t = ttnn.concat([pu4, pu2], dim=1)  # %32 [1,24,384,384]
    # reshape -> B3 -> reshape -> packed conv -> reshape -> B2 -> reshape -> B1 -> reshape
    t = ttnn.reshape(t, [1, 96, 96, 384])  # %33
    t = ttnn.permute(t, (0, 2, 3, 1))  # %34 B3
    t = ttnn.reshape(t, [1, 1, 36864, 96])  # %35
    t = ttnn.to_memory_config(t, DRAM)  # %36
    t = conv(t, tw_pack, tb_pack, 96, 256, 1, (1, 1), (1, 1), (0, 0, 0, 0), 96, 384, cfg_pack)  # %37 packed
    t = ttnn.reshape(t, [1, 96, 384, 256])  # %38
    t = ttnn.permute(t, (0, 3, 1, 2))  # %39 B2
    t = ttnn.reshape(t, [1, 64, 384, 384])  # %40
    t = ttnn.permute(t, (0, 2, 3, 1))  # %41 B1
    t = ttnn.reshape(t, [1, 1, 147456, 64])  # %42 -> NHWC flat
    t = conv(t, tw_ap23, None, 64, 64, 64, (2, 2), (2, 2), (0, 0, 0, 0), 384, 384, cfg_ap23)  # %43 avgpool
    t = ttnn.to_memory_config(t, DRAM)  # %44
    t = conv(t, tw45, tb45, 64, 64, 2, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_45)  # %45
    t = ttnn.to_memory_config(t, DRAM)  # %46
    t = conv(t, tw47, tb47, 64, 96, 1, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_47)  # %47
    t = ttnn.reshape(t, [1, 192, 192, 96])  # %48
    t = ttnn.permute(t, (0, 3, 1, 2))  # %49 -> [1,96,192,192]
    t = ttnn.to_memory_config(t, DRAM)  # %50
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(ttnn.to_layout(t, RM, memory_config=DRAM)).float()
    assert list(result.shape) == [1, 96, 192, 192]
    pcc = torch.corrcoef(torch.stack([result.flatten(), golden.flatten()]))[0, 1].item()
    print(f"\nPCC(golden, ttnn) = {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} < 0.97"


def test_block_A_conv3_clip_full(device):
    """Baseline: full MLIR reproduction with B3/packed-conv/B2/B1 unpack chain + torch golden."""
    _run(device)
