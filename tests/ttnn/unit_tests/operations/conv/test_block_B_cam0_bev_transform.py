# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# block_B_cam0_bev_transform  —  baseline IR reproduction
#
# Source: BLOCKBD_BOTTLENECK/ir/ttnn_block_B_cam0_bev_transform.mlir
#
# Pipeline:
#
#   feat  [1,192,96,96]  RM DRAM  bf16
#      │
#   ┌──┴──────────────────────────────────────┐
#   │  to_layout TILE                          │
#   │  ★ P1 permute NCHW→NHWC  ~0.146 ms      │
#   │  reshape [1,1,9216,192]                  │
#   │  conv2d IC=192 OC=64 k=1×1 ReLU6        │
#   │  reshape [1,96,96,64]                    │
#   │  to_DRAM  →  to_layout RM               │
#   │  [1,96,96,64]  RM DRAM                  │
#   └──┬──────────────────────────────────────┘
#      │feat_enc
#   LUT [1,128,64,8,2] RM DRAM
#      │
#   ┌──┴─────────────────────────────────────────────────────────────────────┐
#   │  ★ R1 reshape [1,128,64,8,2]→[1,128,64,16]  ~0.293 ms                 │
#   │     stride d3*8+d4 → d3*16+d4 forces full DRAM copy                   │
#   │  Note: ttnn nearest mode requires precomputed grid (prepare on host);   │
#   │        R1 still reproduced as device op via separate lut reshape below  │
#   └──┬─────────────────────────────────────────────────────────────────────┘
#      │
#   precomputed LUT [1,128,64,16] RM L1_HS  64 cores [128×16]
#      │LUT_l1
#      ▼
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  ★ GS grid_sample(nearest,use_precomputed=True,batch_output_ch=True)    │
#   │     ~0.230 ms  8192 BEV pos × 8 coord pairs → 512 output channels       │
#   │     output [1,128,64,512]  RM L1_HS  64 cores [128×512]                 │
#   └──┬───────────────────────────────────────────────────────────────────────┘
#      │
#   to_layout TILE  →  to_DRAM  →  reshape [1,1,8192,512]
#   to_L1_HS 43 cores [192×512]
#   conv2d IC=512 OC=64 k=1×1
#   reshape [1,128,64,64]  →  permute NHWC→NCHW  →  to_DRAM
#      │
#   OUTPUT [1,64,128,64]  TILE DRAM  bf16
#
# Bottlenecks (★):
#   R1: reshape [1,128,64,8,2]→[1,128,64,16]  stride-change DRAM copy  ~0.293 ms
#   GS: grid_sample nearest  8192 non-sequential DRAM reads             ~0.230 ms
#   P1: permute NCHW→NHWC  3.54 MB DRAM read                           ~0.146 ms

import pytest
import torch
import torch.nn.functional as F
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


def _conv1_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
        output_layout=TILE,
    )


def _conv2_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
        output_layout=TILE,
    )


def _l1_hs(core_ranges, shard_shape):
    """Build L1 HEIGHT_SHARDED MemoryConfig.
    core_ranges: list of ((row0,col0),(row1,col1)) tuples.
    """
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(c0, r0), ttnn.CoreCoord(c1, r1)) for (r0, c0), (r1, c1) in core_ranges}
            ),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def test_block_B_cam0_bev_transform(device):
    """
    Reproduces block_B_cam0_bev_transform TTNN IR op-by-op on Wormhole N150.

    Inputs:
      feat  [1, 192, 96, 96]    ROW_MAJOR DRAM bf16  — camera feature map (NCHW)
      LUT   [1, 128, 64, 8, 2]  ROW_MAJOR DRAM bf16  — BEV→camera coord LUT

    Output: [1, 64, 128, 64]  TILE DRAM bf16

    Bottlenecks (matching ops_perf_block_B_cam0_derived.csv):
      ★ R1: reshape (1,128,64,8,2)→(1,128,64,16)  stride-change DRAM copy  ~0.293 ms
      ★ GS: grid_sample nearest  8192 random DRAM reads                    ~0.230 ms
      ★ P1: permute NCHW→NHWC  3.54 MB DRAM read  FPU=9.79%               ~0.146 ms

    Note on nearest mode: TTNN grid_sample requires use_precomputed_grid=True for
    nearest interpolation.  prepare_grid_sample_grid (host-side) converts raw
    [1,128,512,2] float32 → precomputed [1,128,64,16] bf16 before device execution.
    (nearest: 2 raw pixel indices per pair × 8 pairs = 16; 48 would be bilinear)
    The device-side R1 reshape is reproduced separately to match profiler ops.
    """
    torch.manual_seed(42)

    # ── Torch tensors ──────────────────────────────────────────────────────────
    feat = torch.randn(1, 192, 96, 96, dtype=torch.bfloat16)
    lut_5d = torch.rand(1, 128, 64, 8, 2, dtype=torch.bfloat16) * 2 - 1  # coords in [-1,1]

    w1 = torch.randn(64, 192, 1, 1, dtype=torch.bfloat16)
    b1 = torch.randn(64, dtype=torch.bfloat16)
    w2 = torch.randn(64, 512, 1, 1, dtype=torch.bfloat16)
    b2 = torch.randn(64, dtype=torch.bfloat16)

    # ── CPU golden ──────────────────────────────────────────────────────────────
    # Stage 1: 1×1 conv IC=192 OC=64 + ReLU6
    stage1 = F.conv2d(feat.float(), w1.float(), b1.float()).clamp(0, 6)  # [1,64,96,96]

    # Stage 2: grid_sample K=8 → [1,128,64,512]
    gathered = []
    for k in range(8):
        coords_k = lut_5d[:, :, :, k, :].float()  # [1,128,64,2]
        g = F.grid_sample(stage1, coords_k, mode="nearest", align_corners=True, padding_mode="zeros")
        gathered.append(g.permute(0, 2, 3, 1))  # [1,128,64,64]
    sampled = torch.cat(gathered, dim=3)  # [1,128,64,512]

    # Stage 3: 1×1 conv IC=512 OC=64
    golden = F.conv2d(sampled.permute(0, 3, 1, 2).float(), w2.float(), b2.float())  # [1,64,128,64]

    # ── Host: precompute LUT for ttnn nearest grid_sample ────────────────────
    # ttnn nearest mode requires use_precomputed_grid=True.
    # For nearest mode, precomputed format = 2 raw-integer pixel coords per pair (h,w).
    # The kernel stores them as raw 16-bit integers in bf16 memory (subnormal bit pattern).
    # prepare_grid_sample_grid(mode="nearest") generates this format correctly.
    # Treat K=8 coord pairs as extra width: [1,128,64,8,2] → [1,128,512,2].
    lut_wide_f32 = lut_5d.float().reshape(1, 128, 512, 2)  # [1,128,512,2] normalized (x,y)
    tt_lut_wide_host = ttnn.from_torch(lut_wide_f32, dtype=ttnn.float32)  # host float32
    pre_host = ttnn.prepare_grid_sample_grid(
        tt_lut_wide_host,
        [1, 96, 96, 64],
        mode="nearest",
        align_corners=True,
        padding_mode="zeros",
        output_dtype=ttnn.bfloat16,
    )  # → [1,128,512,2] nearest precomputed (raw 16-bit pixel indices in bf16 memory)
    # Stack K=8 into last dim: [1,128,64,16] where K=16/2=8 ✓  C_out=64×8=512
    pre_k8 = ttnn.to_torch(pre_host).reshape(1, 128, 64, 16)

    # ── Compute and conv configs ───────────────────────────────────────────────
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    cfg1 = _conv1_config()
    cfg2 = _conv2_config()

    # ── Memory configs ────────────────────────────────────────────────────────
    # conv2 input: 43 cores [0,0→7,4]+[0,5→2,5], shard [192,512]
    l1_hs_43_conv2 = _l1_hs([((0, 0), (4, 7)), ((5, 0), (5, 2))], [192, 512])
    # LUT precomputed: 64 cores [0,0→7,7], shard [128,16]  (nearest: 8 pairs × 2 indices = 16)
    l1_hs_64_lut = _l1_hs([((0, 0), (7, 7))], [128, 16])
    # grid_sample output: 64 cores [0,0→7,7], shard [128,512]
    l1_hs_64_gs_out = _l1_hs([((0, 0), (7, 7))], [128, 512])

    # ── Device weights ────────────────────────────────────────────────────────
    tt_w1 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=L1,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=192,
        out_channels=64,
        batch_size=1,
        input_height=96,
        input_width=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b1 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b1.reshape(1, 1, 1, 64), dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=L1,
        input_layout=TILE,
        in_channels=192,
        out_channels=64,
        batch_size=1,
        input_height=96,
        input_width=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_w2 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=l1_hs_43_conv2,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=64,
        batch_size=1,
        input_height=128,
        input_width=64,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b2 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b2.reshape(1, 1, 1, 64), dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=l1_hs_43_conv2,
        input_layout=TILE,
        in_channels=512,
        out_channels=64,
        batch_size=1,
        input_height=128,
        input_width=64,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # ── Forward pass — matching IR op by op ──────────────────────────────────
    tt_feat = ttnn.from_torch(feat, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    # Raw LUT on device for R1 bottleneck reproduction
    tt_lut = ttnn.from_torch(lut_5d, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    # Precomputed LUT on device for grid_sample
    tt_pre = ttnn.from_torch(pre_k8, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)

    # %5 = to_layout(feat, TILE)  →  [1,192,96,96] TILE DRAM
    tt5 = ttnn.to_layout(tt_feat, TILE, memory_config=DRAM)
    ttnn.deallocate(tt_feat)

    # %6 = permute({0,2,3,1})  →  [1,96,96,192] TILE L1   ★ P1 BOTTLENECK
    tt6 = ttnn.permute(tt5, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt5)

    # %7 = reshape([1,1,9216,192])
    tt7 = ttnn.reshape(tt6, (1, 1, 9216, 192))

    # %8 = conv2d IC=192 OC=64 k=1×1 ReLU6  →  [1,1,9216,64] TILE L1 HS
    tt8 = ttnn.conv2d(
        input_tensor=tt7,
        weight_tensor=tt_w1,
        bias_tensor=tt_b1,
        in_channels=192,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=96,
        input_width=96,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt6)
    ttnn.deallocate(tt_w1)
    ttnn.deallocate(tt_b1)

    # %9 = reshape([1,96,96,64])
    tt9 = ttnn.reshape(tt8, (1, 96, 96, 64))
    ttnn.deallocate(tt8)

    # %10 = to_memory_config(DRAM)  →  [1,96,96,64] TILE DRAM
    tt10 = ttnn.to_memory_config(tt9, DRAM)
    ttnn.deallocate(tt9)

    # %11 = to_layout(RM)  →  [1,96,96,64] RM DRAM
    tt11 = ttnn.to_layout(tt10, RM, memory_config=DRAM)
    ttnn.deallocate(tt10)

    # ★ R1 BOTTLENECK: reshape (1,128,64,8,2)→(1,128,64,16)
    # Stride change d3*8+d4 → d3 forces a full DRAM copy (ReshapeViewDeviceOperation).
    # ttnn nearest grid_sample requires precomputed grid, so we use tt_pre for grid_sample
    # and deallocate tt12 after — the DRAM copy still appears in the profiler.
    tt12 = ttnn.reshape(tt_lut, (1, 128, 64, 16))  # forces DRAM copy
    ttnn.deallocate(tt_lut)
    ttnn.deallocate(tt12)  # R1 result consumed; precomputed grid used for grid_sample

    # %13 — precomputed LUT in DRAM: prepare_grid_sample_grid([1,128,512,2]) reshaped to [1,128,64,16]
    # tt-mlir runtime passes the raw [1,128,64,16] R1 output to prepare then caches result on device.
    tt13 = tt_pre  # [1,128,64,16] RM DRAM

    # %14 = grid_sample  ★ GS BOTTLENECK  →  [1,128,64,512] RM DRAM
    tt14 = ttnn.grid_sample(
        tt11,
        tt13,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        use_precomputed_grid=True,
        batch_output_channels=True,
    )
    ttnn.deallocate(tt11)
    ttnn.deallocate(tt13)
    # Note: tt-mlir runtime (grid_sample.cpp) explicitly deshards output to DRAM
    # if it is HEIGHT_SHARDED L1, before subsequent layout ops.
    # In this test the implicit chain to_layout(TILE) → to_memory_config(DRAM)
    # achieves the same result: L1_HS_RM → L1_HS_TILE → real DRAM copy, which
    # keeps tt16 and tt15 as distinct buffers (no alias double-free hazard).

    # %15 = to_layout(TILE)  →  [1,128,64,512] TILE L1 HS
    tt15 = ttnn.to_layout(tt14, TILE)
    ttnn.deallocate(tt14)

    # %16 = to_memory_config(DRAM)  →  [1,128,64,512] TILE DRAM
    tt16 = ttnn.to_memory_config(tt15, DRAM)
    ttnn.deallocate(tt15)

    # %17 = reshape([1,1,8192,512])  →  [1,1,8192,512] TILE DRAM
    tt17 = ttnn.reshape(tt16, (1, 1, 8192, 512))
    ttnn.deallocate(tt16)

    # %18 = to_memory_config(L1 HS 43 cores shard=[192,512])
    tt18 = ttnn.to_memory_config(tt17, l1_hs_43_conv2)
    ttnn.deallocate(tt17)

    # %19 = conv2d IC=512 OC=64 k=1×1  →  [1,1,8192,64] TILE L1 HS
    tt19 = ttnn.conv2d(
        input_tensor=tt18,
        weight_tensor=tt_w2,
        bias_tensor=tt_b2,
        in_channels=512,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=128,
        input_width=64,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt18)
    ttnn.deallocate(tt_w2)
    ttnn.deallocate(tt_b2)

    # %20 = reshape([1,128,64,64])
    tt20 = ttnn.reshape(tt19, (1, 128, 64, 64))
    ttnn.deallocate(tt19)

    # %21 = permute({0,3,1,2}) NHWC→NCHW  →  [1,64,128,64] TILE L1
    tt21 = ttnn.permute(tt20, (0, 3, 1, 2), memory_config=L1)
    ttnn.deallocate(tt20)

    # %22 = to_memory_config(DRAM)  →  [1,64,128,64] TILE DRAM
    tt22 = ttnn.to_memory_config(tt21, DRAM)
    ttnn.deallocate(tt21)

    # ── Verify ────────────────────────────────────────────────────────────────
    result = ttnn.to_torch(ttnn.to_layout(tt22, RM, memory_config=DRAM))
    ttnn.deallocate(tt22)

    assert list(result.shape) == [1, 64, 128, 64], f"Shape mismatch: {result.shape}"
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


def test_block_B_cam0_bev_transform_rm_l1hs(device):
    """
    RM L1_HS optimized flow: keep grid_sample inputs in L1 to eliminate random DRAM reads.

    MatMul (conv2d) stays TILE L1_HS (hardware requirement).
    R1 reshape (DRAM→DRAM) unchanged — 4-byte rows cannot satisfy L1_HS 16B page constraint.
    Post-GS S→I→S bounce unchanged — direct reshard 64c→43c is not supported; applies
    TILE shard spec to RM tensor and corrupts data.
    Feat permute path unchanged — TILE L1_HS permute decomposes into 2 transpose ops
    and is slower (146 us) than TILE DRAM permute (102 us); no benefit from moving feat to L1.

    Effective changes vs baseline (21.7% device kernel speedup, 197 us saved)
    --------------------------------------------------------------------------
    B) conv1 out: TILE L1_HS → RM L1 → RM L1_HS [144,64]  (baseline S→I+Untilize DRAM removed)
    C) precomputed LUT: RM DRAM → RM L1_HS [128,16] before grid_sample
    D) grid_sample with feat RM L1_HS + LUT RM L1_HS  →  238 us → 37 us  ★ main gain
    """
    torch.manual_seed(42)

    feat = torch.randn(1, 192, 96, 96, dtype=torch.bfloat16)
    lut_5d = torch.rand(1, 128, 64, 8, 2, dtype=torch.bfloat16) * 2 - 1
    w1 = torch.randn(64, 192, 1, 1, dtype=torch.bfloat16)
    b1 = torch.randn(64, dtype=torch.bfloat16)
    w2 = torch.randn(64, 512, 1, 1, dtype=torch.bfloat16)
    b2 = torch.randn(64, dtype=torch.bfloat16)

    # CPU golden (identical to baseline)
    stage1 = F.conv2d(feat.float(), w1.float(), b1.float()).clamp(0, 6)
    gathered = []
    for k in range(8):
        coords_k = lut_5d[:, :, :, k, :].float()
        g = F.grid_sample(stage1, coords_k, mode="nearest", align_corners=True, padding_mode="zeros")
        gathered.append(g.permute(0, 2, 3, 1))
    sampled = torch.cat(gathered, dim=3)
    golden = F.conv2d(sampled.permute(0, 3, 1, 2).float(), w2.float(), b2.float())

    # Host: precompute LUT for ttnn nearest grid_sample (identical to baseline)
    lut_wide_f32 = lut_5d.float().reshape(1, 128, 512, 2)
    tt_lut_wide_host = ttnn.from_torch(lut_wide_f32, dtype=ttnn.float32)
    pre_host = ttnn.prepare_grid_sample_grid(
        tt_lut_wide_host,
        [1, 96, 96, 64],
        mode="nearest",
        align_corners=True,
        padding_mode="zeros",
        output_dtype=ttnn.bfloat16,
    )
    pre_k8 = ttnn.to_torch(pre_host).reshape(1, 128, 64, 16)

    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    cfg1 = _conv1_config()
    cfg2 = _conv2_config()

    # ── Memory configs ─────────────────────────────────────────────────────────
    # conv1 out [1,96,96,64] RM NHWC → 2D 9216×64; 64c → shard [144,64]  page=128B ✓
    l1_hs_64_conv1_out = _l1_hs([((0, 0), (7, 7))], [144, 64])
    # precomputed LUT [1,128,64,16] RM → 2D 8192×16; 64c → shard [128,16]  page=32B ✓
    l1_hs_64_lut = _l1_hs([((0, 0), (7, 7))], [128, 16])
    # conv2 input: 43 cores [0,0→7,4]+[0,5→2,5], shard [192,512]  (same as baseline)
    l1_hs_43_conv2 = _l1_hs([((0, 0), (4, 7)), ((5, 0), (5, 2))], [192, 512])

    # ── Device weights (identical to baseline) ─────────────────────────────────
    tt_w1 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=L1,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=192,
        out_channels=64,
        batch_size=1,
        input_height=96,
        input_width=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b1 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b1.reshape(1, 1, 1, 64), dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=L1,
        input_layout=TILE,
        in_channels=192,
        out_channels=64,
        batch_size=1,
        input_height=96,
        input_width=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_w2 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=l1_hs_43_conv2,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=64,
        batch_size=1,
        input_height=128,
        input_width=64,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b2 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b2.reshape(1, 1, 1, 64), dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=l1_hs_43_conv2,
        input_layout=TILE,
        in_channels=512,
        out_channels=64,
        batch_size=1,
        input_height=128,
        input_width=64,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_feat = ttnn.from_torch(feat, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    tt_lut = ttnn.from_torch(lut_5d, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    tt_pre = ttnn.from_torch(pre_k8, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)

    # ── STAGE 1: feat → conv1 ─────────────────────────────────────────────────
    # Feat path same as baseline (permute on TILE L1_HS decomposes into 2 transpose
    # ops and is slower than permute on TILE DRAM; no benefit from change A).
    tt5 = ttnn.to_layout(tt_feat, TILE, memory_config=DRAM)  # tilize RM DRAM → TILE DRAM
    ttnn.deallocate(tt_feat)
    tt6 = ttnn.permute(tt5, (0, 2, 3, 1), memory_config=L1)  # P1 permute reads TILE DRAM
    ttnn.deallocate(tt5)

    # ops 3,4,5 unchanged
    tt7 = ttnn.reshape(tt6, (1, 1, 9216, 192))
    tt8 = ttnn.conv2d(
        input_tensor=tt7,
        weight_tensor=tt_w1,
        bias_tensor=tt_b1,
        in_channels=192,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=96,
        input_width=96,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg1,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt6)
    ttnn.deallocate(tt_w1)
    ttnn.deallocate(tt_b1)
    tt9 = ttnn.reshape(tt8, (1, 96, 96, 64))
    ttnn.deallocate(tt8)

    # B) conv1 out: TILE L1_HS → RM L1 interleaved → RM L1_HS  (no DRAM trip)
    tt_conv1_rm_l1 = ttnn.to_layout(tt9, RM, memory_config=L1)  # Untilize → RM L1
    ttnn.deallocate(tt9)
    tt11 = ttnn.to_memory_config(tt_conv1_rm_l1, l1_hs_64_conv1_out)  # I→S → RM L1_HS
    ttnn.deallocate(tt_conv1_rm_l1)

    # ── STAGE 2: LUT reshape (R1 unchanged — DRAM→DRAM, 275 us) ───────────────
    tt12 = ttnn.reshape(tt_lut, (1, 128, 64, 16))
    ttnn.deallocate(tt_lut)
    ttnn.deallocate(tt12)  # raw coords; precomputed LUT used for grid_sample

    # C) precomputed LUT: RM DRAM → RM L1_HS  (32B rows → fast sequential shard)
    tt_pre_l1 = ttnn.to_memory_config(tt_pre, l1_hs_64_lut)  # I→S → RM L1_HS
    ttnn.deallocate(tt_pre)

    # ── STAGE 3: grid_sample — both inputs RM L1_HS ───────────────────────────
    tt14 = ttnn.grid_sample(
        tt11,
        tt_pre_l1,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        use_precomputed_grid=True,
        batch_output_channels=True,
    )
    ttnn.deallocate(tt11)
    ttnn.deallocate(tt_pre_l1)

    # ── STAGE 4: GS output → conv2 (baseline post-GS path, l1_hs_43_conv2 expects TILE) ──

    tt15 = ttnn.to_layout(tt14, TILE)  # tilize RM → TILE L1_HS 64c
    ttnn.deallocate(tt14)
    tt16 = ttnn.to_memory_config(tt15, DRAM)  # S→I → TILE DRAM
    ttnn.deallocate(tt15)
    tt17 = ttnn.reshape(tt16, (1, 1, 8192, 512))  # reshape view in DRAM
    ttnn.deallocate(tt16)
    tt18 = ttnn.to_memory_config(tt17, l1_hs_43_conv2)  # I→S → TILE L1_HS 43c
    ttnn.deallocate(tt17)

    tt19 = ttnn.conv2d(
        input_tensor=tt18,
        weight_tensor=tt_w2,
        bias_tensor=tt_b2,
        in_channels=512,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=128,
        input_width=64,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=cfg2,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt18)
    ttnn.deallocate(tt_w2)
    ttnn.deallocate(tt_b2)

    # ── Output (identical to baseline) ────────────────────────────────────────
    tt20 = ttnn.reshape(tt19, (1, 128, 64, 64))
    ttnn.deallocate(tt19)
    tt21 = ttnn.permute(tt20, (0, 3, 1, 2), memory_config=L1)
    ttnn.deallocate(tt20)
    tt22 = ttnn.to_memory_config(tt21, DRAM)
    ttnn.deallocate(tt21)

    result = ttnn.to_torch(ttnn.to_layout(tt22, RM, memory_config=DRAM))
    ttnn.deallocate(tt22)

    assert list(result.shape) == [1, 64, 128, 64], f"Shape mismatch: {result.shape}"
    pcc_val = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc_val >= 0.99, f"PCC {pcc_val:.6f} < 0.99"
