# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# block_A_conv_blocks3_clip — the packed channel-mix sub-pattern (%33..%42), isolated.
#
# Source: BEV_TRACY_CONV3_CLIP/ttnn_block_A_conv_blocks3_clip_annotated.mlir
#
# Logical op: a 1x1 conv IC=24 -> OC=64 (+ relu6) applied to the concat output [1,24,384,384],
# yielding NCHW [1,64,384,384].
#
#   test_baseline_conv2d   — the AS-SHIPPED chain (%33..%40):
#       reshape -> B3 permute -> reshape -> to_mem -> packed conv2d(96->256) -> reshape
#       -> B2 permute -> reshape           (ends NCHW [1,64,384,384]; B1 permute %41 + reshape %42 dropped)
#       (the packed 1x1 conv2d lowers to a MatmulDeviceOperation; the permutes B3/B2 are the
#        pack/unpack bottlenecks)
#
#   test_optimized_linear  — the NCHW-native rewrite (mirrors the yuv adapter %11..%16):
#       reshape -> to_layout(tile) -> linear(24->64) + relu6 -> to_layout(rm) -> reshape
#       (packed conv + B3 + B2 + B1  ->  one linear; result already NCHW [1,64,384,384])
#
# Both emit NCHW [1,64,384,384] and are checked against the same torch golden.
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
RELU6 = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6)
SLICE = ttnn.Conv2dL1FullSliceConfig

IC, OC, H, W = 24, 64, 384, 384  # logical 1x1 conv IC=24 OC=64 at 384x384
P = H * W  # 147456
PACK = 4  # spatial-pack factor (IC 24->96, OC 64->256)


def _pack_w_pointwise(w, ic, oc, K):
    """Block-diagonal 1x1 packing: [OC,IC,1,1] -> [IC*K, OC*K] (cpu_hoisted const-eval reproduction)."""
    w_bc = w.float().reshape(oc, ic, 1, 1).expand(oc, ic, K, K)
    k = torch.arange(K)
    diag = (k.reshape(1, 1, K, 1) == k.reshape(1, 1, 1, K)).float()
    return (w_bc * diag).permute(1, 2, 0, 3).reshape(ic * K, oc * K).to(torch.bfloat16)


def _pack_b_pointwise(b, oc, K):
    return b.reshape(oc).repeat_interleave(K).to(torch.bfloat16)


def _compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False
    )


def _packed_conv_cfg():
    # matches %37 conv2d_config (weights_dtype bf16 here so standalone prepare works; IR uses bfp_bf8)
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=RELU6,
        deallocate_activation=False,
        act_block_h_override=64,
        config_tensors_in_dram=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        enable_kernel_stride_folding=False,
    )


def _inputs():
    torch.manual_seed(7)
    x = torch.randn(1, IC, H, W, dtype=torch.bfloat16) * 0.3  # concat output [1,24,384,384]
    fw = torch.randn(OC, IC, 1, 1, dtype=torch.bfloat16) * 0.1  # logical 1x1 conv weight [64,24]
    fb = torch.randn(OC, dtype=torch.bfloat16) * 0.1  # logical bias [64]
    return x, fw, fb


def _golden_nchw(x, fw, fb):
    """Logical 1x1 conv 24->64 + relu6 -> [1,64,384,384] NCHW."""
    return F.conv2d(x.float(), fw.float(), bias=fb.float()).clamp(0, 6)


def _pcc(result, golden, shape, tag):
    assert list(result.shape) == shape, f"{tag}: shape {list(result.shape)} != {shape}"
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.flatten()]))[0, 1].item()
    print(f"\nPCC(golden, {tag}) = {pcc:.6f}")
    assert pcc >= 0.97, f"[{tag}] PCC {pcc:.6f} < 0.97"


# ===========================================================================
# BASELINE — the packed conv2d chain (%33..%42)
# ===========================================================================
def test_baseline_conv2d(device):
    x, fw, fb = _inputs()
    golden = _golden_nchw(x, fw, fb)  # NCHW [1,64,384,384]
    cc = _compute(device)
    cfg = _packed_conv_cfg()

    # const-eval: packed weight [256,96,1,1] (block-diag K=4) + repeat_interleave(4) bias [256]
    packed_w = _pack_w_pointwise(fw, IC, OC, PACK).t().contiguous().reshape(OC * PACK, IC * PACK, 1, 1)  # [256,96,1,1]
    packed_b = _pack_b_pointwise(fb, OC, PACK)  # [256]
    common = dict(
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=96,
        out_channels=256,
        batch_size=1,
        input_height=96,
        input_width=384,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=cc,
        slice_config=SLICE,
    )
    tw = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(packed_w, dtype=ttnn.bfloat16, layout=RM),
        weights_format="OIHW",
        has_bias=True,
        **common,
    )
    tb = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(packed_b.reshape(1, 1, 1, 256), dtype=ttnn.bfloat16, layout=RM), **common
    )

    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)  # concat [1,24,384,384]
    t = ttnn.reshape(t, [1, 96, 96, 384])  # %33 pack IC 24->96, H 384->96
    t = ttnn.permute(t, (0, 2, 3, 1))  # %34 B3  -> [1,96,384,96]
    t = ttnn.reshape(t, [1, 1, 36864, 96])  # %35
    t = ttnn.to_memory_config(t, DRAM)  # %36
    t = ttnn.conv2d(
        input_tensor=t,
        weight_tensor=tw,
        bias_tensor=tb,
        device=device,
        in_channels=96,
        out_channels=256,
        batch_size=1,
        input_height=96,
        input_width=384,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg,
        compute_config=cc,
        slice_config=SLICE,
    )  # %37 packed conv -> [1,1,36864,256]
    t = ttnn.reshape(t, [1, 96, 384, 256])  # %38
    t = ttnn.permute(t, (0, 3, 1, 2))  # %39 B2  -> [1,256,96,384]
    t = ttnn.reshape(t, [1, 64, 384, 384])  # %40 -> NCHW result [1,64,384,384] (done)
    ttnn.synchronize_device(device)

    _pcc(ttnn.to_torch(ttnn.to_layout(t, RM, memory_config=DRAM)), golden, [1, OC, H, W], "baseline_conv2d")


# ===========================================================================
# OPTIMIZED — reshape -> linear -> reshape (mirrors yuv adapter %11..%16)
# ===========================================================================
def test_optimized_linear(device):
    x, fw, fb = _inputs()
    golden = _golden_nchw(x, fw, fb)  # NCHW [1,64,384,384]
    cc = _compute(device)

    # channel-mix linear weight = logical fw[64,24] (A[o,c]); column bias fb[64] (per-o, broadcast over P)
    cm_w = ttnn.from_torch(
        fw.reshape(1, 1, OC, IC), dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM
    )
    cm_b = ttnn.from_torch(fb.reshape(1, 1, OC, 1), dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)

    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)  # concat [1,24,384,384]
    t = ttnn.reshape(t, [1, 1, IC, P])  # R2[C=24,P] NCHW flat  [1,1,24,147456]
    t = ttnn.to_layout(t, TILE)  # tilize
    mm = ttnn.matmul(cm_w, t, compute_kernel_config=cc)  # A[64,24] @ R2[24,P] -> [1,1,64,147456]
    # fuse the per-channel bias add AND relu6 into one eltwise op (saves a separate ~206us Unary)
    t = ttnn.add(mm, cm_b, activations=[RELU6])  # + bias, then relu6
    t = ttnn.to_layout(t, RM)  # untilize
    t = ttnn.reshape(t, [1, OC, H, W])  # NCHW result [1,64,384,384] (done)
    ttnn.synchronize_device(device)

    _pcc(ttnn.to_torch(ttnn.to_layout(t, RM, memory_config=DRAM)), golden, [1, OC, H, W], "optimized_linear")
