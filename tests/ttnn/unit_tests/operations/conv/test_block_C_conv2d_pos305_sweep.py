# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# EXHAUSTIVE option sweep for Block C's #1 bottleneck conv (pos=305 / %275):
# 192->192, 160x288, k3x3, bf8 weights (FIXED), bf16 output (FIXED).  Every
# option from docs/conv2d_all_options_reference.md that is applicable under the
# bf8-weights / bf16-output constraint.  Goal: reduce duration + raise FPU vs the
# baseline (2394 us / 14.6% FPU / 6.84x PM-ideal).
#
# KEY: block-sharded grid = 6 OC-cols x N NHW-rows (N<=8).  Per-core out height
# (tiles) = ceil(1440/N); its divisibility gates act_block_h.  N=8 ->180 (good),
# N=6 ->240 (good), N=5 ->288 (good), N=7 ->206=2x103 (hostile -> abh clamps to 2).
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

N, C_IN, H, W = 1, 192, 160, 288
C_OUT, KH, KW = 192, 3, 3
NHW = N * H * W  # 46080

_FID = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hi2": ttnn.MathFidelity.HiFi2,
    "hi3": ttnn.MathFidelity.HiFi3,
    "hi4": ttnn.MathFidelity.HiFi4,
}
_THR = {
    0: ttnn.ThrottleLevel.NO_THROTTLE,
    1: ttnn.ThrottleLevel.LEVEL_1,
    3: ttnn.ThrottleLevel.LEVEL_3,
    5: ttnn.ThrottleLevel.LEVEL_5,
}
# 6 OC-cols x N rows -> (rows, per-core out-height tiles, shard_h elements)
_GRID = {30: (5, 288, 9216), 36: (6, 240, 7680), 42: (7, 206, 6592), 48: (8, 180, 5760)}


def _V():
    v = {}
    # ---- (1) baseline (exact IR: 42 cores, abh=384 -> clamped to 2 tiles) ----
    v["baseline_42c_abh384"] = dict(cores=42, abh=384)
    # ---- (2) grid x act_block_h: divisor-friendly grids unclamp abh ----
    #   48c/180t divisors(*32): 64(2) 96(3) 128(4) 192(6) 320(10) 384(12) 576(18) 640(20)
    for abh in (64, 96, 128, 192, 320, 384, 576, 640):
        v[f"g48_abh{abh}"] = dict(cores=48, abh=abh)
    #   36c/240t: 64(2) 384(12) 480(15) 640(20)
    for abh in (64, 384, 480, 640):
        v[f"g36_abh{abh}"] = dict(cores=36, abh=abh)
    #   30c/288t: 64(2) 384(12) 576(18) 768(24)
    for abh in (64, 384, 576, 768):
        v[f"g30_abh{abh}"] = dict(cores=30, abh=abh)
    # ---- (3) compute-config knobs on the best base (48c, abh=384) ----
    base = dict(cores=48, abh=384)
    v["g48_abh384_hi4"] = {**base, "fid": "hi4"}
    v["g48_abh384_hi2"] = {**base, "fid": "hi2"}
    v["g48_abh384_lofi"] = {**base, "fid": "lofi"}
    v["g48_abh384_nofp32"] = {**base, "fp32": False}
    v["g48_abh384_packer"] = {**base, "pk": True}
    v["g48_abh384_approx"] = {**base, "approx": True}
    v["g48_abh384_dstsync"] = {**base, "dst": True}
    v["g48_abh384_thr1"] = {**base, "thr": 1}
    v["g48_abh384_thr3"] = {**base, "thr": 3}
    v["g48_abh384_thr5"] = {**base, "thr": 5}
    # ---- (4) Conv2dConfig flags on the best base ----
    v["g48_abh384_full_inner"] = {**base, "fi": True}
    v["g48_abh384_transpose"] = {**base, "transpose": True}
    v["g48_abh384_no_config_dram"] = {**base, "config_dram": False}
    v["g48_abh384_no_realloc_halo"] = {**base, "realloc_halo": False}
    v["g48_abh384_reshard"] = {**base, "reshard": True}
    v["g48_abh384_override_out"] = {**base, "override_out": True}
    v["g48_abh384_no_db"] = {**base, "db": False}
    v["g48_abh384_actdb_only"] = {**base, "adb": True, "wdb": False}
    v["g48_abh384_wdb_only"] = {**base, "adb": False, "wdb": True}
    # ---- (5) best-guess stacked combos aimed at max FPU / min duration ----
    v["g48_abh384_hi4_packer"] = {**base, "fid": "hi4", "pk": True}
    v["g48_abh384_hi4_fi"] = {**base, "fid": "hi4", "fi": True}
    v["g48_abh320_hi4"] = dict(cores=48, abh=320, fid="hi4")
    v["g48_abh576_hi4"] = dict(cores=48, abh=576, fid="hi4")
    # ---- (6) alternative regime: HEIGHT_SHARDED (feed DRAM, conv auto-shards) ----
    v["height_abh0"] = dict(layout="height", abh=0)
    v["height_abh0_hi4"] = dict(layout="height", abh=0, fid="hi4")
    return v


# Variants that OOM (fewer-core grids, abh>=576, full_inner, packer, config-dram-off,
# height) are pruned — under the bf8-weights/bf16-output constraint their L1 doesn't
# fit, and their partial failures would break the profiler row alignment.
_OOM = {
    "g48_abh576",
    "g48_abh640",
    "g48_abh576_hi4",
    "g48_abh384_full_inner",
    "g48_abh384_packer",
    "g48_abh384_hi4_packer",
    "g48_abh384_hi4_fi",
    "g48_abh384_no_config_dram",
    "height_abh0",
    "height_abh0_hi4",
    "g36_abh64",
    "g36_abh384",
    "g36_abh480",
    "g36_abh640",
    "g30_abh64",
    "g30_abh384",
    "g30_abh576",
    "g30_abh768",
}
_VARIANTS = {k: v for k, v in _V().items() if k not in _OOM}


def _block_input_cfg(cores):
    rows, _tiles, shard_h = _GRID[cores]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, rows - 1))})
    shard = ttnn.ShardSpec(grid, [shard_h, 32], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


@pytest.mark.parametrize("variant", list(_VARIANTS.keys()))
def test_pos305_sweep(device, variant):
    v = _VARIANTS[variant]
    layout = v.get("layout", "block")
    cores = v.get("cores", 48)
    torch.manual_seed(0)

    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=_FID[v.get("fid", "hi3")],
        fp32_dest_acc_en=v.get("fp32", True),
        math_approx_mode=v.get("approx", False),
        packer_l1_acc=v.get("pk", False),
        dst_full_sync_en=v.get("dst", False),
        throttle_level=_THR[v.get("thr", 0)],
    )

    in_mem = DRAM if layout == "height" else _block_input_cfg(cores)
    core_grid = None
    if v.get("override_out"):
        rows = _GRID[cores][0]
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, rows - 1))})

    conv_cfg = ttnn.Conv2dConfig(
        weights_dtype=BF8,  # FIXED
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        reallocate_halo_output=v.get("realloc_halo", True),
        act_block_h_override=v.get("abh", 0),
        config_tensors_in_dram=v.get("config_dram", True),
        enable_act_double_buffer=v.get("adb", v.get("db", True)),
        enable_weights_double_buffer=v.get("wdb", v.get("db", True)),
        full_inner_dim=v.get("fi", False),
        transpose_shards=v.get("transpose", False),
        reshard_if_not_optimal=v.get("reshard", False),
        override_output_sharding_config=v.get("override_out", False),
        core_grid=core_grid,
        enable_kernel_stride_folding=False,
        shard_layout=(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if layout == "height" else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ),
        output_layout=TILE,  # bf16 output (FIXED)
    )

    w = torch.randn(C_OUT, C_IN, KH, KW, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, C_OUT, dtype=torch.bfloat16)
    common = dict(
        input_memory_config=in_mem,
        input_layout=TILE,
        in_channels=C_IN,
        out_channels=C_OUT,
        batch_size=N,
        input_height=H,
        input_width=W,
        kernel_size=(KH, KW),
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

    x = torch.randn(1, 1, NHW, C_IN, dtype=torch.bfloat16)
    out = None
    for _ in range(5):
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
        if in_mem is not DRAM:
            tt_x = ttnn.to_memory_config(tt_x, in_mem)
        out = ttnn.conv2d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            device=device,
            in_channels=C_IN,
            out_channels=C_OUT,
            batch_size=N,
            input_height=H,
            input_width=W,
            kernel_size=(KH, KW),
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
    assert list(out.shape) == [1, 1, NHW, C_OUT]
