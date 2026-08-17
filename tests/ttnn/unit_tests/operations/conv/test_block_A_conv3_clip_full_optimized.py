# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# block_A_conv_blocks3_clip  —  OPTIMIZED variant (permute-bottleneck elimination)
#
# Same whole graph as the baseline (tests/.../test_block_A_conv3_clip_full.py) EXCEPT the packed
# channel-mix sub-pattern %33..%42 is replaced.
#
# Baseline sub-pattern (3 permute bottlenecks):
#     concat -> reshape -> B3 permute -> reshape -> to_mem -> packed conv2d(96->256)
#            -> reshape -> B2 permute -> reshape -> B1 permute -> reshape -> [1,1,147456,64] NHWC
#     (B3 %34, B2 %39, B1 %41 are the three data-movement bottlenecks)
#
# OPTIMIZED approach — SKIP the spatial packing entirely.  The whole chain is logically a 1x1 conv
# IC=24 -> OC=64 (+relu6) that must produce NHWC for the next conv.  Do it directly:
#     concat -> permute {0,2,3,1} (NCHW->NHWC, 24 ch) -> reshape -> conv2d(24->64, 1x1, relu6)
#            -> [1,1,147456,64] NHWC   (feeds the avgpool conv directly)
#   => eliminates BOTH unpack permutes (B2 + B1) and the packed conv's 96-ch B3; only ONE 24-ch
#      permute remains (unavoidable: NCHW concat must become NHWC for conv2d).
#
# Same torch golden as the baseline.
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
SPATIAL = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR
RELU6 = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6)
SLICE = ttnn.Conv2dL1FullSliceConfig


def _pack_w_pointwise(w, ic, oc, K):
    w_bc = w.float().reshape(oc, ic, 1, 1).expand(oc, ic, K, K)
    k = torch.arange(K)
    diag = (k.reshape(1, 1, K, 1) == k.reshape(1, 1, 1, K)).float()
    return (w_bc * diag).permute(1, 2, 0, 3).reshape(ic * K, oc * K).to(torch.bfloat16)


def _pack_b_pointwise(b, oc, K):
    return b.reshape(oc).repeat_interleave(K).to(torch.bfloat16)


def _conv_cfg(act, adb, wdb, abh):
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
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


def test_block_A_conv3_clip_full_optimized(device):
    """Whole graph with the packed-conv/B3/B2/B1 chain replaced by a direct 1x1 conv2d in NHWC."""
    torch.manual_seed(42)
    cc = _compute(device)

    # ---- raw ONNX weights (arg*) ----
    x = torch.randn(1, 3, 1536, 1536, dtype=torch.bfloat16)
    yw = torch.randn(3, 3, 1, 1, dtype=torch.bfloat16) * 0.2
    yb = torch.randn(3, dtype=torch.bfloat16) * 0.1
    w_ap11 = torch.full((2, 1, 2, 1), 0.5, dtype=torch.bfloat16)
    w_ap23 = torch.full((64, 1, 2, 2), 0.25, dtype=torch.bfloat16)
    fw = torch.randn(64, 24, 1, 1, dtype=torch.bfloat16) * 0.1  # logical 1x1 conv [64,24]
    fb = torch.randn(64, dtype=torch.bfloat16) * 0.1
    w45 = torch.randn(64, 32, 3, 3, dtype=torch.bfloat16) * 0.05
    b45 = torch.randn(64, dtype=torch.bfloat16) * 0.1
    w47 = torch.randn(96, 64, 3, 3, dtype=torch.bfloat16) * 0.05
    b47 = torch.randn(96, dtype=torch.bfloat16) * 0.1

    # ================= TORCH GOLDEN (identical to baseline) =================
    xf = x.float()
    yuv = F.conv2d(xf, yw.float(), bias=yb.float())
    y_us = torch.nn.PixelUnshuffle(4)(yuv[:, 0:1])
    uv_avg = F.avg_pool2d(yuv[:, 1:3], kernel_size=(2, 1), stride=(2, 2))
    uv_us = torch.nn.PixelUnshuffle(2)(uv_avg)
    cat = torch.cat([y_us, uv_us], dim=1)
    c1 = F.conv2d(cat, fw.float(), bias=fb.float()).clamp(0, 6)  # the 1x1 conv 24->64 + relu6
    c2 = F.avg_pool2d(c1, kernel_size=(2, 2), stride=(2, 2))
    c3 = F.conv2d(c2, w45.float(), bias=b45.float(), groups=2, padding=1).clamp(0, 6)
    golden = F.conv2d(c3, w47.float(), bias=b47.float(), padding=1).clamp(0, 6)  # [1,96,192,192]

    # ================= const-eval weight prep =================
    ywp = _pack_w_pointwise(yw, 3, 3, 32).reshape(1, 1, 96, 96)
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

    w_ap11_rep = w_ap11.repeat_interleave(16, dim=0)

    cfg_ap11 = _conv_cfg(act=False, adb=False, wdb=False, abh=64)
    cfg_direct = _conv_cfg(act=True, adb=True, wdb=True, abh=64)  # the direct 24->64 1x1 conv (relu6)
    cfg_ap23 = _conv_cfg(act=False, adb=False, wdb=False, abh=64)
    cfg_45 = _conv_cfg(act=True, adb=True, wdb=True, abh=0)
    cfg_47 = _conv_cfg(act=True, adb=True, wdb=True, abh=0)

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

    tw_ap11, _ = _prep(w_ap11_rep, None, 32, 32, 32, (2, 1), (2, 2), (0, 0, 0, 0), 96, 1536, cfg_ap11, L1)
    tw_dir, tb_dir = _prep(fw, fb, 24, 64, 1, (1, 1), (1, 1), (0, 0, 0, 0), 384, 384, cfg_direct, DRAM)  # direct 24->64
    tw_ap23, _ = _prep(w_ap23, None, 64, 64, 64, (2, 2), (2, 2), (0, 0, 0, 0), 384, 384, cfg_ap23, L1)
    tw45, tb45 = _prep(w45, b45, 64, 64, 2, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_45, DRAM)
    tw47, tb47 = _prep(w47, b47, 64, 96, 1, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_47, DRAM)

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

    # ================= FORWARD =================
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    t = ttnn.reshape(t, [1, 1, 96, 73728])
    t = ttnn.to_layout(t, TILE)
    t = ttnn.linear(tt_ywp, t, bias=tt_ybp, program_config=lin_pc, compute_kernel_config=cc)  # yuv linear
    t = ttnn.to_memory_config(t, DRAM)
    t = ttnn.to_layout(t, RM)
    t = ttnn.reshape(t, [1, 3, 1536, 1536])
    full = ttnn.to_memory_config(t, DRAM)

    y = ttnn.slice(t, [0, 0, 0, 0], [1, 1, 1536, 1536], [1, 1, 1, 1])
    y = ttnn.to_memory_config(y, DRAM)
    pu4 = ttnn.pixel_unshuffle(y, downscale_factor=4, channel_order=SPATIAL, memory_config=L1)

    uv = ttnn.slice(full, [0, 1, 0, 0], [1, 3, 1536, 1536], [1, 1, 1, 1])
    uv = ttnn.reshape(uv, [1, 32, 96, 1536])
    uv = ttnn.permute(uv, (0, 2, 3, 1))
    uv = ttnn.reshape(uv, [1, 1, 147456, 32])
    uv = conv(uv, tw_ap11, None, 32, 32, 32, (2, 1), (2, 2), (0, 0, 0, 0), 96, 1536, cfg_ap11)
    uv = ttnn.reshape(uv, [1, 48, 768, 32])
    uv = ttnn.to_memory_config(uv, L1)
    uv = ttnn.permute(uv, (0, 3, 1, 2))
    uv = ttnn.reshape(uv, [1, 2, 768, 768])
    uv = ttnn.to_memory_config(uv, DRAM)
    pu2 = ttnn.pixel_unshuffle(uv, downscale_factor=2, channel_order=SPATIAL, memory_config=L1)

    # ===== OPTIMIZED channel-mix: direct 1x1 conv 24->64 in NHWC (no packing, no B2/B1) =====
    t = ttnn.concat([pu4, pu2], dim=1)  # [1,24,384,384] NCHW
    t = ttnn.permute(t, (0, 2, 3, 1))  # NCHW->NHWC [1,384,384,24]  (only permute)
    t = ttnn.reshape(t, [1, 1, 147456, 24])  # NHWC flat
    t = conv(
        t, tw_dir, tb_dir, 24, 64, 1, (1, 1), (1, 1), (0, 0, 0, 0), 384, 384, cfg_direct
    )  # 24->64 relu6 -> [1,1,147456,64]
    # ==========================================================================================
    t = conv(t, tw_ap23, None, 64, 64, 64, (2, 2), (2, 2), (0, 0, 0, 0), 384, 384, cfg_ap23)  # %43 avgpool
    t = ttnn.to_memory_config(t, DRAM)
    t = conv(t, tw45, tb45, 64, 64, 2, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_45)  # %45
    t = ttnn.to_memory_config(t, DRAM)
    t = conv(t, tw47, tb47, 64, 96, 1, (3, 3), (1, 1), (1, 1, 1, 1), 192, 192, cfg_47)  # %47
    t = ttnn.reshape(t, [1, 192, 192, 96])  # %48
    t = ttnn.permute(t, (0, 3, 1, 2))  # %49 -> [1,96,192,192]
    t = ttnn.to_memory_config(t, DRAM)  # %50
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(ttnn.to_layout(t, RM, memory_config=DRAM)).float()
    assert list(result.shape) == [1, 96, 192, 192]
    pcc = torch.corrcoef(torch.stack([result.flatten(), golden.flatten()]))[0, 1].item()
    print(f"\nPCC(golden, optimized) = {pcc:.6f}")
    assert pcc >= 0.97, f"PCC {pcc:.6f} < 0.97"
