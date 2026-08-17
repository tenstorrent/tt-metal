# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Block C (optimized run, FORGE_CONV_OPT_ENABLE=1) top bottleneck conv2ds, standalone.
# Source: BEV_OPT_ANALYSIS/block_C/block_C_conv2d_bottleneck_analysis.md + optimized IR/CSV.
# weights = bfloat8_b (FIXED), output = bf16 (FIXED).
#
#   Shape 1 (ranks #1/#2): IC=192 OC=192 160x288 3x3 BLOCK 48c abh=12t adb+wdb
#       reference: 1234 us, 28.4% FPU, PM-ideal 349.9 us
#   Shape 2 (ranks #3/#4): IC=64  OC=96  320x576 3x3 HEIGHT 64c abh=2t  (double-buffers OFF)
#       reference:  598 us, 39.0% FPU, PM-ideal 233.3 us
#
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest <this file>

import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
BF8 = ttnn.bfloat8_b
_FID = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hi2": ttnn.MathFidelity.HiFi2,
    "hi3": ttnn.MathFidelity.HiFi3,
    "hi4": ttnn.MathFidelity.HiFi4,
}

# shape -> geometry + the optimized-run sharding
SHAPES = {
    "s1_192_block": dict(ic=192, oc=192, h=160, w=288, layout="block", gx=5, gy=7, shard=[5760, 32]),  # 6x8=48
    "s2_64_96_height": dict(ic=64, oc=96, h=320, w=576, layout="height", gx=7, gy=7, shard=[2880, 64]),  # 8x8=64
}

# variant -> (shape_key, overrides). Baselines reproduce the MD; the rest try to improve.
_V = {
    # --- Shape 1 (192->192 BLOCK): baseline is already the good 48c/abh384/wadb config ---
    "s1_baseline": ("s1_192_block", dict(abh=384, adb=True, wdb=True)),  # ref 1234us/28.4%
    "s1_hi4": ("s1_192_block", dict(abh=384, adb=True, wdb=True, fid="hi4")),  # fill idle compute
    # --- Shape 2 (64->96 HEIGHT): baseline has DOUBLE-BUFFERS OFF (the MD's flagged miss) ---
    "s2_baseline": ("s2_64_96_height", dict(abh=64, adb=False, wdb=False)),  # ref 598us/39.0%
    "s2_db": ("s2_64_96_height", dict(abh=64, adb=True, wdb=True)),  # enable double-buffers
    "s2_db_abh192": ("s2_64_96_height", dict(abh=192, adb=True, wdb=True)),  # 6t -> 15 blocks
    "s2_db_hi4": ("s2_64_96_height", dict(abh=64, adb=True, wdb=True, fid="hi4")),
    "s2_db_abh192_hi4": ("s2_64_96_height", dict(abh=192, adb=True, wdb=True, fid="hi4")),
    # (abh>=288 on the height path OOMs under the fixed bf16 output)
    # --- height-sharding im2col booster (split reader) + lower fidelity ---
    # (enable_activation_reuse is infeasible here: it needs act_block_h > output-width-in-tiles = 18t,
    #  i.e. abh>=608, but abh>=288 already OOMs under the fixed bf16 output.)
    "s2_split": ("s2_64_96_height", dict(abh=64, split=True)),  # force split reader
    "s2_abh192_split": ("s2_64_96_height", dict(abh=192, split=True)),
    "s2_abh192_lofi": ("s2_64_96_height", dict(abh=192, fid="lofi")),  # less compute
    "s2_abh192_hi2": ("s2_64_96_height", dict(abh=192, fid="hi2")),
    "s2_abh192_split_lofi": ("s2_64_96_height", dict(abh=192, split=True, fid="lofi")),
    # --- probe the ~450us data-movement floor exposed at LoFi (last untested knobs) ---
    # (config_tensors_in_dram=False and reallocate_halo_output=False both OOM under bf16 output)
    "s2_lofi_abh96": ("s2_64_96_height", dict(abh=96, fid="lofi")),  # 3t, 30 blocks
    "s2_lofi_abh160": ("s2_64_96_height", dict(abh=160, fid="lofi")),  # 5t, 18 blocks
    "s2_lofi_awdiv2": ("s2_64_96_height", dict(abh=192, fid="lofi", awdiv=2)),  # split act block width
}


def _shard_cfg(s):
    layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED if s["layout"] == "block" else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(s["gx"], s["gy"]))})
    spec = ttnn.ShardSpec(grid, s["shard"], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, spec)


@pytest.mark.parametrize("variant", list(_V.keys()))
def test_top_bottleneck(device, variant):
    shape_key, o = _V[variant]
    s = SHAPES[shape_key]
    ic, oc, H, W = s["ic"], s["oc"], s["h"], s["w"]
    nhw = H * W
    torch.manual_seed(0)

    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=_FID[o.get("fid", "hi3")],
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    conv_cfg = ttnn.Conv2dConfig(
        weights_dtype=BF8,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        act_block_h_override=o["abh"],
        act_block_w_div=o.get("awdiv", 1),
        config_tensors_in_dram=o.get("config_dram", True),
        reallocate_halo_output=o.get("realloc_halo", True),
        enable_act_double_buffer=o.get("adb", True),
        enable_weights_double_buffer=o.get("wdb", True),
        enable_kernel_stride_folding=False,
        enable_activation_reuse=o.get("reuse", False),
        force_split_reader=(True if o.get("split") else None),
        shard_layout=(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED if s["layout"] == "block" else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        ),
        output_layout=TILE,
    )
    # prepare uses a sharded-L1 (TILE) input_memory_config so the bf8 weight/bias prep
    # succeeds (bf8 bias prep on a DRAM-interleaved config raises "Only TILE for bf8").
    prep_mem = _shard_cfg(s)
    # The actual conv is fed from where the IR feeds it: block shapes stay sharded-L1;
    # height shapes are fed DRAM-interleaved (per the IR) so the conv auto-shards
    # internally — feeding a pre-sharded L1 input there would double L1 pressure and OOM.
    conv_in_mem = DRAM if s["layout"] == "height" else prep_mem

    w = torch.randn(oc, ic, 3, 3, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, oc, dtype=torch.bfloat16)
    common = dict(
        input_memory_config=prep_mem,
        input_layout=TILE,
        in_channels=ic,
        out_channels=oc,
        batch_size=1,
        input_height=H,
        input_width=W,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM), weights_format="OIHW", has_bias=True, **common
    )
    tt_b = ttnn.prepare_conv_bias(bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM), **common)

    x = torch.randn(1, 1, nhw, ic, dtype=torch.bfloat16)
    out = None
    for _ in range(5):
        if out is not None:
            ttnn.deallocate(out)  # free prior iteration's output before the next conv allocates
            out = None
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
        if conv_in_mem is not DRAM:
            tt_x = ttnn.to_memory_config(tt_x, conv_in_mem)
        out = ttnn.conv2d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            device=device,
            in_channels=ic,
            out_channels=oc,
            batch_size=1,
            input_height=H,
            input_width=W,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1, 1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=compute,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, nhw, oc]
