# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import pytest
import torch
import torch.nn.functional as F
import ttnn

# ---------------------------------------------------------------------------
# UV AveragePool — Spatial Packing Fix
#
# Bottleneck recap (test_avgpool_dram_bottleneck.py):
#   NCHW→NHWC permute with C=2 writes 150.9/188.7 MB DRAM (93.75% zeros)
#   because C=2 in the tile-column dim is padded to TILE_WIDTH=32.
#
# Fix: pack K=16 spatial rows into channels BEFORE the permute.
#   C×K = 2×16 = 32 = TILE_WIDTH → 0% tile-column padding waste.
#   Pack permute writes only 9.4 MB  (16× less than baseline).
#
# Key insight: the reshape [N,C,H,W] → [N,C×K,H/K,W] maps:
#   original (c, h) → packed channel c_new = c×K + h//(H/K), packed row h_new = h%(H/K)
#   e.g. K=16: c_new=0 holds rows 0..95 of orig_c=0; c_new=1 holds rows 96..191; etc.
#
# The original avgpool (kH=2, kW=1, stride=[2,2], groups=C) on [N,C,H,W] is
# IDENTICAL to running the same kernel with groups=C×K on the packed [N,C×K,H/K,W].
# Every packed channel is an independent group — no cross-channel mixing.
#
# Full pipeline:
#
#   [N, 2, H, W]   ROW_MAJOR DRAM
#     ↓ reshape [N, 32, H/16, W]              ← FREE VIEW (no data copy)
#     ↓ permute {0,2,3,1}                      ← writes 9.4 MB (0% waste: C×K=32=TILE_WIDTH)
#   [N, H/16, W, 32]  NHWC ROW_MAJOR DRAM
#     ↓ reshape [1, 1, H/16×W, 32]            ← FREE VIEW
#     ↓ ttnn.conv2d(kH=2, kW=1, s=[2,2],
#                   groups=32, weight=0.5)     ← same op, same kernel — just more groups
#   [1, 1, H/32×W/2, 32]
#     ↓ reshape [N, H/32, W/2, 32]            ← FREE VIEW
#     ↓ permute {0,3,1,2}                      ← [N, 32, H/32, W/2]
#     ↓ reshape [N, 2, H/2, W/2]              ← FREE VIEW (32×H/32 = 2×H/2)
#
# Verification (CPU, exact arithmetic):
#   PCC = 1.000000, max_abs_diff = 0.000000
#
# DRAM comparison:
#   Baseline pack permute: 150.9 MB (Block A) / 188.7 MB (Block C) — OUT_X_PAD=32[2]
#   Packed  pack permute:    9.4 MB (Block A) /   9.4 MB (Block C) — OUT_X_PAD=32[32]
# ---------------------------------------------------------------------------

TILE_WIDTH = 32

_CONFIGS = [
    pytest.param(1, 2, 2, 1536, 1536, id="avgpool_1_1x2x1536x1536"),
    pytest.param(1, 2, 2, 1280, 2304, id="avgpool_2_1x2x1280x2304"),
]

_KERNEL_SIZE = (2, 1)
_STRIDE = (2, 2)
_PADDING = (0, 0, 0, 0)
_DILATION = (1, 1)


def _pack_factor(in_channels: int) -> int:
    return TILE_WIDTH // math.gcd(in_channels, TILE_WIDTH)


def _make_packed_weight(original_weight: torch.Tensor, K: int) -> torch.Tensor:
    # original_weight: [groups, IC/groups, kH, kW]  e.g. [2, 1, 2, 1]
    # Packed weight:   [groups*K, IC/groups, kH, kW] e.g. [32, 1, 2, 1]
    #
    # Spatial packing maps K rows from each original channel into K independent
    # packed channels.  Each packed channel applies the SAME kernel, so we
    # replicate each group's weight K times via repeat_interleave(K, dim=0):
    #   group 0 → packed groups 0..K-1  (all identical)
    #   group 1 → packed groups K..2K-1 (all identical)
    return original_weight.repeat_interleave(K, dim=0)


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_avgpool_spatial_packing(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    """
    Spatial packing fix for the UV avgpool DRAM bottleneck.

    Packs K=16 rows into the channel dimension so C*K=32=TILE_WIDTH.
    The pack permute writes 9.4 MB (0% tile padding waste) vs 150.9/188.7 MB
    in the baseline. ttnn.conv2d is kept as the underlying compute op with the
    same kernel parameters (kH=2, kW=1, stride=[2,2]) — only groups increases
    from C=2 to C*K=32 since each packed channel is independent.
    """
    K = _pack_factor(in_channels)  # 16
    assert input_height % K == 0, f"input_height={input_height} not divisible by K={K}"
    assert (input_height // K) >= _KERNEL_SIZE[0], f"packed_h too small for kH={_KERNEL_SIZE[0]}"

    packed_ic = in_channels * K  # 32 = TILE_WIDTH
    packed_h = input_height // K  # 96 (Block A) / 80 (Block C)
    packed_spatial = packed_h * input_width
    groups_packed = packed_ic  # 32 (fully depthwise on packed channels)
    out_h = input_height // _STRIDE[0]  # H/2
    out_w = input_width // _STRIDE[1]  # W/2
    packed_out_h = packed_h // _STRIDE[0]  # H/32
    packed_out_w = input_width // _STRIDE[1]  # W/2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    # output_layout=ROW_MAJOR_LAYOUT: conv2d outputs ROW_MAJOR directly,
    # eliminating the ShardedToInterleaved + unpack permute + Copy + Untilize
    # chain that the TILE output path requires.
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)

    # Original avgpool weight: [groups=2, IC/groups=1, kH=2, kW=1]
    # Each group weight = 0.5 per kH element (1/(kH*kW) = 0.5 for kH=2, kW=1)
    torch_w_orig = torch.full(
        (in_channels, 1, _KERNEL_SIZE[0], _KERNEL_SIZE[1]),
        1.0 / (_KERNEL_SIZE[0] * _KERNEL_SIZE[1]),
        dtype=torch.bfloat16,
    )
    # Pack weight: [2,1,2,1] → [32,1,2,1] via repeat_interleave(K=16, dim=0)
    # Packed group g*K+k uses the same kernel as original group g.
    torch_w_packed = _make_packed_weight(torch_w_orig, K)

    tt_weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_w_packed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW",
        in_channels=packed_ic,
        out_channels=groups_packed,
        batch_size=batch,
        input_height=packed_h,
        input_width=input_width,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        has_bias=False,
        groups=groups_packed,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight),
    )

    # ── Pack path ─────────────────────────────────────────────────────────────

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # Step 1: reshape [N, C=2, H, W] → [N, C×K=32, H/K, W]  — FREE VIEW
    # Mapping: original (c, h) → packed (c_new = c×K + h//(H/K), h_new = h%(H/K))
    tt_packed_nchw = ttnn.reshape(tt_input, (batch, packed_ic, packed_h, input_width))

    # Step 2: permute {0,2,3,1} NCHW→NHWC  ← KEY IMPROVEMENT
    #   C×K=32 = TILE_WIDTH → OUT_X_PAD[LOGICAL] = 32[32] (0% tile-column waste)
    #   Block A: writes 9.4 MB vs 150.9 MB baseline  (16× reduction)
    #   Block C: writes 9.4 MB vs 188.7 MB baseline  (20× reduction)
    tt_nhwc = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_packed_nchw)

    # Step 3: flatten [N, H/K, W, 32] → [1, 1, H/K×W, 32]  — FREE VIEW
    tt_flat = ttnn.reshape(tt_nhwc, (batch, 1, packed_spatial, packed_ic))
    ttnn.deallocate(tt_nhwc)

    # Step 4: ttnn.conv2d — same kernel as baseline, more groups
    #   kH=2, kW=1, stride=(2,2): UNCHANGED from baseline
    #   groups=32 (one group per packed channel = fully depthwise on packed dims)
    #   input_height=H/K (16× smaller), no change to conv2d kernel logic
    tt_out = ttnn.conv2d(
        input_tensor=tt_flat,
        weight_tensor=tt_weight,
        in_channels=packed_ic,
        out_channels=groups_packed,
        device=device,
        bias_tensor=None,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        batch_size=batch,
        input_height=packed_h,
        input_width=input_width,
        groups=groups_packed,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight),
    )
    ttnn.deallocate(tt_flat)
    ttnn.deallocate(tt_weight)

    # ── Unpack path ───────────────────────────────────────────────────────────
    # conv2d output: [1, 1, packed_out_h×out_w, 32]  ROW_MAJOR (output_layout=ROW_MAJOR)
    #
    # Unpack to [N, 2, H/2, W/2] via:
    #   reshape [N, packed_out_h, out_w, 32] → NHWC
    #   permute {0,3,1,2}                    → NCHW [N, 32, packed_out_h, out_w]
    #   reshape [N, 2, H/2, W/2]             → FREE VIEW (32×packed_out_h = 2×(H/2))
    #
    # No untilize needed — output is already ROW_MAJOR.

    # Step 5: reshape flat → NHWC [N, packed_out_h, out_w, 32]
    tt_out_nhwc = ttnn.reshape(tt_out, (batch, packed_out_h, packed_out_w, groups_packed))
    ttnn.deallocate(tt_out)

    # Step 6: permute NHWC→NCHW [N, 32, packed_out_h, out_w] — to L1 (small tensor)
    tt_out_nchw_packed = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_out_nhwc)

    # Step 7: copy to DRAM, then untilize to ROW_MAJOR
    # Note: output_layout=ROW_MAJOR_LAYOUT is overridden to TILE internally for the
    # DRAMSliceHeight path (conv2d.cpp:686 forces TILE to reduce CB memory usage).
    # Reshape [N,32,packed_out_h,W/2] → [N,2,H/2,W/2] is a free view ONLY in ROW_MAJOR;
    # in TILE it reorders 32×32 blocks incorrectly. Untilize first.
    tt_out_dram = ttnn.to_memory_config(tt_out_nchw_packed, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_nchw_packed)
    tt_out_rm = ttnn.to_layout(tt_out_dram, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_dram)

    # Step 8: reshape [N, 32, packed_out_h, out_w] → [N, 2, H/2, W/2]  — FREE VIEW
    # Correctness: 32×packed_out_h = 32×(H/32) = H = 2×(H/2), memory order preserved ✓
    tt_output = ttnn.reshape(tt_out_rm, (batch, out_channels, out_h, out_w))
    result = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_out_rm)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size(
        [batch, out_channels, out_h, out_w]
    ), f"Shape mismatch: got {tuple(result.shape)}, expected ({batch},{out_channels},{out_h},{out_w})"

    # Golden: depthwise avgpool using F.conv2d with the original (unpacked) weight
    golden = F.conv2d(
        torch_input.float(),
        torch_w_orig.float(),
        bias=None,
        stride=_STRIDE,
        padding=(0, 0),
        dilation=_DILATION,
        groups=in_channels,
    )

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, (
        f"PCC {pcc:.6f} < 0.99 for config " f"({batch},{in_channels},{out_channels},{input_height},{input_width})"
    )


# ---------------------------------------------------------------------------
# Full L1 HEIGHT_SHARDED pipeline
#
# Every intermediate tensor lives in L1 HEIGHT_SHARDED (no DRAM round-trips
# between pack-permute and conv2d output).
#
# Per-core L1 budget check (64 cores, 1,362 KB usable each):
#   Peak tensor (packed input 9.4 MB / 11.8 MB): 144 KB / 184 KB per core — 10.6% L1
#   Conv output  (2.36 MB / 2.95 MB):             36 KB  / 46 KB  per core — 2.7% L1
#
# Pipeline (DRAM ops are only the first read and the final write):
#
#   [N,2,H,W]  ROW_MAJOR  DRAM                    ← initial input
#     ↓ reshape [N,32,H/16,W]                      FREE VIEW  (DRAM)
#     ↓ permute {0,2,3,1}  → L1 HEIGHT_SHARDED ROW_MAJOR
#                                                  reads 9.4 MB DRAM, writes L1
#     ↓ reshape [1,1,sp,32]                        FREE VIEW  (L1)
#     ↓ to_layout(TILE, L1_HS)                     L1 ROW_MAJOR → L1 TILE  (no DRAM)
#     ↓ conv2d(groups=32, k=[2,1], s=[2,2])        L1 TILE → DRAM TILE
#     ↓ reshape [N,packed_out_h,out_w,32]
#     ↓ permute {0,3,1,2}  → L1 TILE             DRAM TILE → L1
#     ↓ to_memory_config(DRAM)
#     ↓ to_layout(ROW_MAJOR, DRAM)                 DRAM TILE → DRAM ROW_MAJOR
#     ↓ reshape [N,2,H/2,W/2]                      FREE VIEW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_avgpool_spatial_packing_l1(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    """
    Full L1 HEIGHT_SHARDED spatial packing pipeline.

    All intermediate tensors are kept in L1 HEIGHT_SHARDED — no DRAM
    round-trips between the pack permute and the conv2d output.
    Peak per-core L1 usage: 144 KB / 184 KB (10-13% of 1,362 KB available).
    """
    K = _pack_factor(in_channels)  # 16
    assert input_height % K == 0
    assert (input_height // K) >= _KERNEL_SIZE[0]

    packed_ic = in_channels * K  # 32
    packed_h = input_height // K  # 96 / 80
    packed_sp = packed_h * input_width
    groups_packed = packed_ic  # 32 (fully depthwise)
    out_h = input_height // _STRIDE[0]
    out_w = input_width // _STRIDE[1]
    packed_out_h = packed_h // _STRIDE[0]
    packed_out_w = input_width // _STRIDE[1]

    # Shared L1 HEIGHT_SHARDED config for all intermediate tensors
    l1_hs = ttnn.create_sharded_memory_config(
        shape=(packed_sp, packed_ic),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_w_orig = torch.full(
        (in_channels, 1, _KERNEL_SIZE[0], _KERNEL_SIZE[1]),
        1.0 / (_KERNEL_SIZE[0] * _KERNEL_SIZE[1]),
        dtype=torch.bfloat16,
    )
    torch_w_packed = _make_packed_weight(torch_w_orig, K)

    tt_weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_w_packed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=l1_hs,
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="OIHW",
        in_channels=packed_ic,
        out_channels=groups_packed,
        batch_size=batch,
        input_height=packed_h,
        input_width=input_width,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        has_bias=False,
        groups=groups_packed,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # ── Pack path ─────────────────────────────────────────────────────────────

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # Reshape: free view, stays in DRAM
    tt_packed_nchw = ttnn.reshape(tt_input, (batch, packed_ic, packed_h, input_width))

    # Pack permute NCHW→NHWC → L1 HEIGHT_SHARDED ROW_MAJOR
    # Reads 9.4 MB from DRAM, writes directly to L1 (no DRAM write for packed tensor)
    tt_nhwc_l1 = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=l1_hs)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_packed_nchw)

    # Reshape: free view, stays in L1
    tt_flat_l1 = ttnn.reshape(tt_nhwc_l1, (batch, 1, packed_sp, packed_ic))
    ttnn.deallocate(tt_nhwc_l1)

    # Tilize L1 ROW_MAJOR → L1 TILE HEIGHT_SHARDED (no DRAM write — L1 to L1)
    tt_tile_l1 = ttnn.to_layout(tt_flat_l1, ttnn.TILE_LAYOUT, memory_config=l1_hs)
    ttnn.deallocate(tt_flat_l1)

    # Conv2d reads from L1 HEIGHT_SHARDED TILE
    # L1FullSliceConfig: input is already in L1, no DRAM slicing, no InterleavedToSharded
    tt_out = ttnn.conv2d(
        input_tensor=tt_tile_l1,
        weight_tensor=tt_weight,
        in_channels=packed_ic,
        out_channels=groups_packed,
        device=device,
        bias_tensor=None,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        batch_size=batch,
        input_height=packed_h,
        input_width=input_width,
        groups=groups_packed,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_tile_l1)
    ttnn.deallocate(tt_weight)

    # ── Unpack path ───────────────────────────────────────────────────────────

    tt_out_nhwc = ttnn.reshape(tt_out, (batch, packed_out_h, packed_out_w, groups_packed))
    ttnn.deallocate(tt_out)

    tt_out_nchw_packed = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_out_nhwc)

    tt_out_dram = ttnn.to_memory_config(tt_out_nchw_packed, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_nchw_packed)
    tt_out_rm = ttnn.to_layout(tt_out_dram, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_dram)

    tt_output = ttnn.reshape(tt_out_rm, (batch, out_channels, out_h, out_w))
    result = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_out_rm)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size([batch, out_channels, out_h, out_w])

    golden = F.conv2d(
        torch_input.float(),
        torch_w_orig.float(),
        bias=None,
        stride=_STRIDE,
        padding=(0, 0),
        dilation=_DILATION,
        groups=in_channels,
    )

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, (
        f"PCC {pcc:.6f} < 0.99 for config " f"({batch},{in_channels},{out_channels},{input_height},{input_width})"
    )
