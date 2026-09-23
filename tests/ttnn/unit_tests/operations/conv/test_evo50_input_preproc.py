# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
EVO50 input-preprocessing block, reproduced op-for-op from the TTNN IR.

Source IR:
  FULL_EVO50_MODEL/INPUT_PREPROC/WITH_MLIR_TRACE/IR/ttnn_evo50_input_preproc.mlir
  function @trace_0_forward

Reference device profile (scoped rows 62-82, between the evo50_inference-start /
evo50_inference-end signposts of):
  .../TRACY/reports/preproc_opt2_with_trace/2026_09_23_09_06_40/
  ops_perf_results_preproc_opt2_with_trace_2026_09_23_09_06_40.csv

Every attribute, layout, shape and dtype below is transcribed from the IR:

  arg0 input_luv            [1,3,384,1664] bf16  ROW_MAJOR DRAM interleaved   (#ttnn_layout1)
  arg1 fused_7x7_weights    [16,3,7,7]     bf16  host                         (#ttnn_layout2)
  arg2 grouped_conv_weights [16,8,3,3]     bf16  host                         (#ttnn_layout3)
  arg3 fused_7x7_bias       [1,1,1,16]     bf16  host                         (#ttnn_layout)
  arg4 grouped_conv_bias    [1,1,1,16]     bf16  host                         (#ttnn_layout)

  %1  to_layout            -> TILE DRAM interleaved      memref<36x52 tile>     (#ttnn_layout5)
  %2  permute [0,2,3,1]    -> TILE L1 interleaved  <8x8> memref<1x312 tile>     (#ttnn_layout6)
  %3  reshape  [1,1,638976,3]                                                   (#ttnn_layout7)
  %4  to_memory_config     -> TILE DRAM interleaved      memref<19968x1 tile>   (#ttnn_layout8)
  %5  conv2d (7x7 s2, relu, dram_width slicing)
                           -> TILE DRAM interleaved      memref<4992x1 tile>    (#ttnn_layout9)
  %6  to_memory_config     -> TILE L1 height_sharded <64x1> memref<78x1 tile>   (#ttnn_layout10)
  %7  conv2d (3x3 s1 g2, relu, l1_full)
                           -> TILE L1 height_sharded                            (#ttnn_layout10)
  %8  reshape  [1,192,832,16]                                                   (#ttnn_layout11)
  %9  permute [0,3,1,2]    -> TILE L1 interleaved  <8x8> memref<1x39 tile>      (#ttnn_layout12)
  %10 to_memory_config     -> TILE DRAM interleaved      memref<96x26 tile>     (#ttnn_layout4)

Second test, test_evo50_input_preproc_s2d, is the stem rewritten per
docs/evo50_stem_space_to_depth.md (tt-forge-onnx, --ttir-strided-conv-space-to-depth-opt):
the 7x7/s2 conv becomes pixel_unshuffle(2) + a 4x4/s1 conv on re-packed weights, so the
conv input is [1,1,159744,12] in L1 (no tilize of the image, 4x smaller permute, no DRAM
slicing, no to_memory_config before the grouped conv). Everything from the grouped conv on
is identical between the two tests.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*args, **kwargs):
        pass


# ---------------------------------------------------------------------------- shapes
BATCH = 1
IN_C = 3
IN_H = 384
IN_W = 1664

# conv1: fused 7x7
C1_OUT = 16
C1_K = (7, 7)
C1_STRIDE = (2, 2)
C1_PAD = (2, 3, 2, 3)  # [pad_top, pad_bottom, pad_left, pad_right]
C1_GROUPS = 1

# conv2: grouped 3x3
C2_OUT = 16
C2_K = (3, 3)
C2_STRIDE = (1, 1)
C2_PAD = (1, 1, 1, 1)
C2_GROUPS = 2

MID_H = (IN_H + C1_PAD[0] + C1_PAD[1] - C1_K[0]) // C1_STRIDE[0] + 1  # 192
MID_W = (IN_W + C1_PAD[2] + C1_PAD[3] - C1_K[1]) // C1_STRIDE[1] + 1  # 832

# space-to-depth form of conv1 (docs/evo50_stem_space_to_depth.md, 2A.1 / 2A.3)
S2D_R = C1_STRIDE[0]  # 2
S2D_C = IN_C * S2D_R * S2D_R  # 12 packed channels
S2D_K = (-(-C1_K[0] // S2D_R), -(-C1_K[1] // S2D_R))  # ceil(7/2) = 4 -> (4, 4)
S2D_H, S2D_W = IN_H // S2D_R, IN_W // S2D_R  # 192, 832
# pT' = pT/2, pB' = outH + kHp - 1 - H/2 - pT'   (same for width)  -> (1, 2, 1, 2) top,bottom,left,right
S2D_PAD = (
    C1_PAD[0] // S2D_R,
    MID_H + S2D_K[0] - 1 - S2D_H - C1_PAD[0] // S2D_R,
    C1_PAD[2] // S2D_R,
    MID_W + S2D_K[1] - 1 - S2D_W - C1_PAD[2] // S2D_R,
)

TILE = 32
# #ttnn_layout10: <64x1> grid, memref<78x1 x tile<32x32>> per core
HS_CORE_GRID = (8, 8)
HS_SHARD_SHAPE = (MID_H * MID_W // (HS_CORE_GRID[0] * HS_CORE_GRID[1]), TILE)  # (2496, 32)


def torch_golden(x, w1, b1, w2, b2):
    """The two relu-fused convolutions, in NCHW."""
    # F.pad order is (left, right, top, bottom); C1_PAD is (top, bottom, left, right).
    xp = F.pad(x, (C1_PAD[2], C1_PAD[3], C1_PAD[0], C1_PAD[1]))
    y1 = F.relu(F.conv2d(xp, w1, b1, stride=C1_STRIDE, padding=0, groups=C1_GROUPS))
    y2 = F.relu(F.conv2d(y1, w2, b2, stride=C2_STRIDE, padding=C2_PAD[0], groups=C2_GROUPS))
    return y1, y2


def pack_weight_s2d(w, r=S2D_R):
    """[OC, C, kH, kW] -> [OC, C*r*r, ceil(kH/r), ceil(kW/r)], CHANNEL_MAJOR order
    (c' = c*r*r + dy*r + dx), i.e. W'[oc, c', i, j] = Wpad[oc, c, r*i + dy, r*j + dx].
    docs/evo50_stem_space_to_depth.md 2A.4."""
    OC, C, kH, kW = w.shape
    kHp, kWp = -(-kH // r), -(-kW // r)
    wp = F.pad(w, (0, kWp * r - kW, 0, kHp * r - kH))  # zero-pad kernel to (kHp*r, kWp*r): 7x7 -> 8x8
    wp = wp.reshape(OC, C, kHp, r, kWp, r)  # [oc, c, i, dy, j, dx]
    wp = wp.permute(0, 1, 3, 5, 2, 4)  # [oc, c, dy, dx, i, j]
    return wp.reshape(OC, C * r * r, kHp, kWp).contiguous()


def torch_stem_s2d(x, w1, b1):
    """conv1 in its space-to-depth form; must equal torch_golden()[0]."""
    s = F.pixel_unshuffle(x, S2D_R)  # [N, 12, 192, 832], channel_major
    s = F.pad(s, (S2D_PAD[2], S2D_PAD[3], S2D_PAD[0], S2D_PAD[1]))
    return F.relu(F.conv2d(s, pack_weight_s2d(w1), b1, stride=1))


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (torch.dot(a, b) / denom).item()


def _free_unless_alias(src, view):
    """ttnn.reshape that only collapses tile-aligned dims returns a zero-copy view of `src`.
    Freeing `src` would then free the buffer under `view` (the next L1 allocation overwrites it),
    so `src` is released only when the reshape really produced a new buffer; an alias is
    released through `view` by its consumer (here conv2d's deallocate_activation)."""
    if src.buffer_address() != view.buffer_address():
        ttnn.deallocate(src)


def _stem_baseline(device, tt_in, tt_w1, tt_b1, hs_mem_cfg, conv1_cfg, compute_cfg, slice1):
    """@trace_0_forward %1..%6: tilize, permute, reshape, spill to DRAM, 7x7/s2 DRAM-sliced conv,
    re-shard into L1. Returns the <64x1> height-sharded L1 tensor the grouped conv consumes."""
    # %1 = ttnn.to_layout(%arg0)  -> TILE DRAM
    t1 = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # %2 = ttnn.permute(%1) {permutation = [0, 2, 3, 1]}  -> TILE L1 interleaved
    t2 = ttnn.permute(t1, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t1)

    # %3 = ttnn.reshape(%2) {shape = [1, 1, 638976, 3]}
    t3 = ttnn.reshape(t2, (1, 1, IN_H * IN_W, IN_C))
    _free_unless_alias(t2, t3)

    # %4 = ttnn.to_memory_config(%3)  -> TILE DRAM
    t4 = ttnn.to_memory_config(t3, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t3)  # frees the t2/t3 buffer

    # %5 = ttnn.conv2d(%4, %arg1, %arg3, %0)
    t5 = ttnn.conv2d(
        input_tensor=t4,
        weight_tensor=tt_w1,
        bias_tensor=tt_b1,
        device=device,
        in_channels=IN_C,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=IN_H,
        input_width=IN_W,
        kernel_size=C1_K,
        stride=C1_STRIDE,
        padding=C1_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_cfg,
        compute_config=compute_cfg,
        slice_config=slice1,
    )
    ttnn.deallocate(t4)

    # %6 = ttnn.to_memory_config(%5)  -> TILE L1 height_sharded <64x1>
    t6 = ttnn.to_memory_config(t5, hs_mem_cfg)
    ttnn.deallocate(t5)
    return t6


def _stem_s2d(device, tt_in, tt_w1_packed, tt_b1, conv1_s2d_cfg, compute_cfg, slice_l1_full):
    """The stem after --ttir-strided-conv-space-to-depth-opt (doc section 2 / 2A.6 'After').
    Returns the same <64x1> height-sharded L1 tensor as _stem_baseline."""
    # (1') ttnn.pixel_unshuffle(%arg0) {downscale_factor = 2, channel_order = channel_major,
    #      memory_config = <#l1, interleaved>}  -> [1,12,192,832] TILE L1
    #      input stays ROW_MAJOR (no tilize of the image); kernel emits ROW_MAJOR, tilizes on request
    s1 = ttnn.pixel_unshuffle(
        tt_in,
        S2D_R,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        channel_order=ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR,
    )

    # (2') ttnn.permute {permutation = [0, 2, 3, 1]}  -> [1,192,832,12] TILE L1  (4x fewer bytes than %2)
    s2 = ttnn.permute(s1, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(s1)

    # (3') ttnn.reshape {shape = [1, 1, 159744, 12]}
    s3 = ttnn.reshape(s2, (1, 1, S2D_H * S2D_W, S2D_C))
    _free_unless_alias(s2, s3)  # s3 is a view of s2; conv2d(deallocate_activation=True) frees it below

    # (4') ttnn.conv2d 4x4 / s1 / pad [1,2,1,2] on the re-packed weights, l1_full, height sharded.
    #      deallocate_activation=true frees s3.  Output: [1,1,159744,16] TILE L1 height_sharded <64x1>
    y1 = ttnn.conv2d(
        input_tensor=s3,
        weight_tensor=tt_w1_packed,
        bias_tensor=tt_b1,
        device=device,
        in_channels=S2D_C,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=S2D_H,
        input_width=S2D_W,
        kernel_size=S2D_K,
        stride=(1, 1),
        padding=S2D_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_s2d_cfg,
        compute_config=compute_cfg,
        slice_config=slice_l1_full,
    )
    # (5') the to_memory_config is gone: the next conv consumes the sharded L1 result directly
    return y1


# channels_last stem: conv1's packed input channels zero-padded to a 16-element (32 B) stick
S2D_CP = 16


def _stem_s2d_nhwc(device, tt_in, tt_w1_packed16, tt_b1, conv1_cfg, compute_cfg, slice_l1_full, hs_rm_cfg):
    """Space-to-depth stem with pixel_unshuffle emitting the conv's input directly:
    NHWC [1,192,832,16] ROW_MAJOR, height-sharded over the 64-core grid (2496 pixels per core).
    No tilize, no permute, no re-shard; conv2d's halo reads the row-major shards as they are."""
    # pixel_unshuffle(%arg0) {downscale_factor = 2, channel_order = channel_major, channels_last}
    s1 = ttnn.pixel_unshuffle(
        tt_in,
        S2D_R,
        memory_config=hs_rm_cfg,
        channel_order=ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR,
        channels_last=True,
        padded_channels=S2D_CP,
    )
    # reshape [1, 1, 159744, 16] (view of the sharded tensor)
    s3 = ttnn.reshape(s1, (1, 1, S2D_H * S2D_W, S2D_CP))
    _free_unless_alias(s1, s3)

    # 4x4 / s1 / pad [1,2,1,2], in_channels = 16 (4 zero taps), l1_full, height sharded on the input's grid
    y1 = ttnn.conv2d(
        input_tensor=s3,
        weight_tensor=tt_w1_packed16,
        bias_tensor=tt_b1,
        device=device,
        in_channels=S2D_CP,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=S2D_H,
        input_width=S2D_W,
        kernel_size=S2D_K,
        stride=(1, 1),
        padding=S2D_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_cfg,
        compute_config=compute_cfg,
        slice_config=slice_l1_full,
    )
    return y1


def _tail(device, t6, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice2):
    """@trace_0_forward %7..%10 plus the runtime's output untilize. Identical for both stems."""
    # %7 = ttnn.conv2d(%6, %arg2, %arg4, %0)
    t7 = ttnn.conv2d(
        input_tensor=t6,
        weight_tensor=tt_w2,
        bias_tensor=tt_b2,
        device=device,
        in_channels=C2_OUT,
        out_channels=C2_OUT,
        batch_size=BATCH,
        input_height=MID_H,
        input_width=MID_W,
        kernel_size=C2_K,
        stride=C2_STRIDE,
        padding=C2_PAD,
        dilation=(1, 1),
        groups=C2_GROUPS,
        conv_config=conv2_cfg,
        compute_config=compute_cfg,
        slice_config=slice2,
    )

    # %8 = ttnn.reshape(%7) {shape = [1, 192, 832, 16]}
    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))
    _free_unless_alias(t7, t8)

    # %9 = ttnn.permute(%8) {permutation = [0, 3, 1, 2]}  -> TILE L1 interleaved
    t9 = ttnn.permute(t8, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t8)

    # %10 = ttnn.to_memory_config(%9)  -> TILE DRAM
    t10 = ttnn.to_memory_config(t9, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t9)

    # The runtime untilizes the TILE DRAM result (#ttnn_layout4) on device before handing it
    # back to the host; this is the trailing UntilizeDeviceOperation of the reference report.
    out_rm = ttnn.to_layout(t10, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t10)
    return out_rm


def _make_inputs():
    torch.manual_seed(0)
    x = torch.randn(BATCH, IN_C, IN_H, IN_W, dtype=torch.bfloat16)
    w1 = torch.randn(C1_OUT, IN_C // C1_GROUPS, *C1_K, dtype=torch.bfloat16)
    b1 = torch.randn(C1_OUT, dtype=torch.bfloat16)
    w2 = torch.randn(C2_OUT, C2_OUT // C2_GROUPS, *C2_K, dtype=torch.bfloat16)
    b2 = torch.randn(C2_OUT, dtype=torch.bfloat16)
    return x, w1, b1, w2, b2


def _common_configs(device):
    # #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
    )
    # #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>,
    #                     deallocate_activation = true, act_block_h_override = 64,
    #                     shard_layout = height_sharded, enable_kernel_stride_folding = false,
    #                     config_tensors_in_dram = true>
    conv2_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),  # activation = <op_type = relu>
        deallocate_activation=True,
        act_block_h_override=64,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    # #ttnn.conv2d_slice_config<l1_full, 0>
    slice_l1_full = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0)
    return compute_cfg, conv2_cfg, slice_l1_full


def _check(result, golden):
    assert list(result.shape) == [BATCH, C2_OUT, MID_H, MID_W], f"shape mismatch: {result.shape}"
    pcc = _pcc(golden, result)
    print(f"\nEVO50 input-preproc output {list(result.shape)}  PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc} below 0.99"
    return pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc(device):
    """@trace_0_forward as compiled: the 2.55 ms baseline."""
    x, w1, b1, w2, b2 = _make_inputs()
    _, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    # %arg0: [1,3,384,1664] bf16 ROW_MAJOR DRAM interleaved (#ttnn_layout1)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # %arg1/%arg2 weights, %arg3/%arg4 biases: host (#system_memory)
    tt_w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, conv2_cfg, slice_l1_full = _common_configs(device)

    # #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>,
    #                     act_block_h_override = 1024, enable_kernel_stride_folding = false,
    #                     config_tensors_in_dram = true>
    conv1_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),  # activation = <op_type = relu>
        act_block_h_override=1024,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    # #ttnn.conv2d_slice_config<dram_width, 0>
    slice1 = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0)

    # #ttnn_layout10: <64x1>, memref<78x1 x tile<32x32,bf16>, #l1>, height_sharded,
    #                 core_ranges = <[(0,0) - (7,7)]>
    hs_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HS_CORE_GRID[0] - 1, HS_CORE_GRID[1] - 1))}
            ),
            HS_SHARD_SHAPE,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run():
        t6 = _stem_baseline(device, tt_in, tt_w1, tt_b1, hs_mem_cfg, conv1_cfg, compute_cfg, slice1)
        return _tail(device, t6, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    # warm-up fills the program cache
    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_inference-end")

    _check(result, golden)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc_s2d(device):
    """Same block with the stem in space-to-depth form (docs/evo50_stem_space_to_depth.md)."""
    x, w1, b1, w2, b2 = _make_inputs()
    y1_ref, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    # The re-packed stem is the same arithmetic regrouped: check that in torch before touching the device.
    y1_s2d = torch_stem_s2d(x.float(), w1.float(), b1.float())
    assert list(y1_s2d.shape) == [BATCH, C1_OUT, MID_H, MID_W], f"packed conv shape {y1_s2d.shape}"
    assert torch.allclose(
        y1_s2d, y1_ref, atol=1e-3, rtol=1e-4
    ), f"packed-weight stem differs from the 7x7/s2 conv: max |diff| {(y1_s2d - y1_ref).abs().max().item():.3e}"
    assert S2D_PAD == (1, 2, 1, 2), S2D_PAD

    # %arg0 unchanged: ROW_MAJOR DRAM; the stem no longer tilizes it
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # conv1 weights re-packed on host to [16,12,4,4] (the compiler const-evals this chain)
    tt_w1_packed = ttnn.from_torch(pack_weight_s2d(w1.float()).to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, conv2_cfg, slice_l1_full = _common_configs(device)

    # conv2d_config = <relu, shard_layout = height_sharded, act_block_h_override = 64,
    #                  deallocate_activation = true, config_tensors_in_dram = true>   (doc 2A.6)
    conv1_s2d_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        act_block_h_override=64,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    def run():
        y1 = _stem_s2d(device, tt_in, tt_w1_packed, tt_b1, conv1_s2d_cfg, compute_cfg, slice_l1_full)
        return _tail(device, y1, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_s2d_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_s2d_inference-end")

    _check(result, golden)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc_s2d_nhwc(device):
    """S2D stem with pixel_unshuffle(channels_last=True) feeding conv2d directly, plus the
    conv2d config found by sweep for BOTH convs (act_block_h_override=832, act double buffer,
    no halo realloc): conv1 291 -> 131 us, the grouped conv 229 -> 123 us. Same arithmetic as
    the IR's config; only blocking / buffering knobs differ."""
    x, w1, b1, w2, b2 = _make_inputs()
    _, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # packed [16,12,4,4] -> [16,16,4,4]: input channels 12..15 are the zero pad channels of the activation
    w1_packed16 = F.pad(pack_weight_s2d(w1.float()), (0, 0, 0, 0, 0, S2D_CP - S2D_C))
    tt_w1_packed16 = ttnn.from_torch(w1_packed16.to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, _, slice_l1_full = _common_configs(device)

    # sweep result (docs/evo50_input_preproc_ttnn_repro.md section 7): the two convs are
    # reader-bound at the IR's act_block_h_override=64; 832 (= one 26-tile output row block,
    # 2496/832 = 3 blocks per core) is the floor, double-buffered activations add a little,
    # and reallocate_halo_output=False removes a Move.
    tuned = dict(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=832,
        enable_act_double_buffer=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    conv1_cfg = ttnn.Conv2dConfig(**tuned)
    conv2_cfg = ttnn.Conv2dConfig(**tuned)

    # pixel_unshuffle output: [1,192,832,16] ROW_MAJOR, <64x1> height-sharded, shard [2496, 16]
    hs_rm_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HS_CORE_GRID[0] - 1, HS_CORE_GRID[1] - 1))}
            ),
            (S2D_H * S2D_W // (HS_CORE_GRID[0] * HS_CORE_GRID[1]), S2D_CP),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run():
        y1 = _stem_s2d_nhwc(device, tt_in, tt_w1_packed16, tt_b1, conv1_cfg, compute_cfg, slice_l1_full, hs_rm_cfg)
        return _tail(device, y1, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_s2d_nhwc_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_s2d_nhwc_inference-end")

    _check(result, golden)
