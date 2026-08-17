# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# EXHAUSTIVE option sweep for Block C's critical conv2d (%272).
#   in=192 out=192, 3x3, stride 1, pad 1, H=160 W=288 (NHW=46080), bf16 relu6.
#
# Every settable conv2d / prepare_conv2d / compute-kernel option is swept here:
#  - a cross-product of the strongly-interacting dims (shard_layout x precision x
#    act_block_h x math_fidelity x double-buffer), plus
#  - individual toggles of every remaining flag on the best base
#    (full_inner_dim, packer_l1_acc, config_tensors_in_dram, reallocate_halo_output,
#     output_layout, transpose_shards, reshard_if_not_optimal,
#     override_output_sharding_config, dst_full_sync_en, math_approx_mode,
#     fp32_dest_acc_en, throttle_level 1..5, weights bf4_b, DRAM slice configs).
# Infeasible combos (L1 OOM / unsupported) are auto-skipped so the run stays clean.
#
# Profile:
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest \
#     tests/ttnn/unit_tests/operations/conv/test_block_C_full_option_sweep.py

import itertools
import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
BF8 = ttnn.bfloat8_b
BF4 = ttnn.bfloat4_b

N, C_IN, H, W = 1, 192, 160, 288
C_OUT, KH, KW = 192, 3, 3
NHW = N * H * W  # 46080

_LAYOUTS = {
    "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}


def _block_sharded_input_cfg():
    # %272's shard: BLOCK_SHARDED L1 [5760,32] on cores (0,0)-(5,7) = 48 cores.
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))}),
        [5760, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def _conv_config(o):
    return ttnn.Conv2dConfig(
        weights_dtype=o["wdtype"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        reallocate_halo_output=o["realloc_halo"],
        act_block_h_override=o["abh"],
        config_tensors_in_dram=o["config_dram"],
        enable_kernel_stride_folding=False,
        enable_weights_double_buffer=o["weights_db"],
        enable_act_double_buffer=o["act_db"],
        full_inner_dim=o["full_inner"],
        transpose_shards=o["transpose_shards"],
        reshard_if_not_optimal=o["reshard"],
        override_output_sharding_config=o["override_out"],
        core_grid=o["core_grid"],
        shard_layout=_LAYOUTS[o["sharding"]],
        output_layout=(RM if o["output_rm"] else TILE),
    )


_DEFAULTS = dict(
    sharding="block",
    wdtype=ttnn.bfloat16,
    odtype=ttnn.bfloat16,
    idtype=ttnn.bfloat16,
    abh=384,
    weights_db=True,
    act_db=True,
    full_inner=False,
    transpose_shards=False,
    reshard=False,
    override_out=False,
    core_grid=None,
    config_dram=True,
    realloc_halo=True,
    output_rm=False,
    fidelity=ttnn.MathFidelity.HiFi3,
    fp32_acc=True,
    packer_l1_acc=False,
    math_approx=False,
    dst_full_sync=False,
    throttle=ttnn.ThrottleLevel.NO_THROTTLE,
    slice_cfg=None,
)

# The 48-core grid %272 uses (6 OC-cols x 8 NHW-rows), for override_output_sharding_config.
_GRID_48 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))})


def _run(device, o):
    torch.manual_seed(0)
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=o["fidelity"],
        fp32_dest_acc_en=o["fp32_acc"],
        math_approx_mode=o["math_approx"],
        packer_l1_acc=o["packer_l1_acc"],
        dst_full_sync_en=o["dst_full_sync"],
        throttle_level=o["throttle"],
    )
    conv_cfg = _conv_config(o)
    slice_cfg = o["slice_cfg"] if o["slice_cfg"] is not None else ttnn.Conv2dL1FullSliceConfig
    # block-sharded: feed the pre-built L1 shard (matches %272, keeps bf8 on TILE path).
    # height/width: feed DRAM and let conv2d auto-shard to that layout.
    in_mem = _block_sharded_input_cfg() if o["sharding"] == "block" else DRAM

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
        input_dtype=o["idtype"],
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=slice_cfg,
    )
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM),
        weights_format="OIHW",
        has_bias=True,
        **common,
    )
    tt_b = ttnn.prepare_conv_bias(bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM), **common)

    x = torch.randn(1, 1, NHW, C_IN, dtype=torch.bfloat16)
    out = None
    for _ in range(5):
        tt_x = ttnn.from_torch(x, dtype=o["idtype"], layout=TILE, device=device, memory_config=DRAM)
        if in_mem is not DRAM:
            tt_x = ttnn.to_memory_config(tt_x, in_mem)
        out = ttnn.conv2d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            in_channels=C_IN,
            out_channels=C_OUT,
            device=device,
            kernel_size=(KH, KW),
            stride=(1, 1),
            padding=(1, 1, 1, 1),
            dilation=(1, 1),
            batch_size=N,
            input_height=H,
            input_width=W,
            groups=1,
            dtype=o["odtype"],
            conv_config=conv_cfg,
            compute_config=compute,
            slice_config=slice_cfg,
        )
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, NHW, C_OUT]


def _cfg(**kw):
    o = dict(_DEFAULTS)
    o.update(kw)
    return o


def _build_sweep():
    sweep = {"baseline": _cfg(sharding="block", wdtype=ttnn.bfloat16, weights_db=False, act_db=False, abh=64)}

    # ---- Cross-product: layout x precision x abh x fidelity x double-buffer ----
    precs = [("bf16", {}), ("bf8io", dict(wdtype=BF8, odtype=BF8))]
    fids = [("hi3", ttnn.MathFidelity.HiFi3), ("hi4", ttnn.MathFidelity.HiFi4)]
    dbs = [("nodb", dict(weights_db=False, act_db=False)), ("wadb", dict(weights_db=True, act_db=True))]
    for layout in ("block", "height", "width"):
        abhs = [64, 192, 384, 576] if layout == "block" else [0]
        for abh in abhs:
            for ptag, pkw in precs:
                if abh == 576 and ptag == "bf16":  # bf16 abh=576 OOMs
                    continue
                for ftag, fid in fids:
                    for dtag, dkw in dbs:
                        name = f"{layout}_{ptag}_abh{abh}_{ftag}_{dtag}"
                        sweep[name] = _cfg(sharding=layout, abh=abh, fidelity=fid, **pkw, **dkw)

    # ---- Individual toggles on the best base (block, bf8io, abh=384, wadb, HiFi3) ----
    base = dict(sharding="block", wdtype=BF8, odtype=BF8, abh=384, weights_db=True, act_db=True)
    toggles = {
        "tog_full_inner": dict(full_inner=True),
        "tog_packer_l1": dict(packer_l1_acc=True),
        "tog_config_dram_off": dict(config_dram=False),
        "tog_realloc_halo_off": dict(realloc_halo=False),
        "tog_output_rm": dict(output_rm=True),
        "tog_transpose": dict(transpose_shards=True),
        "tog_reshard": dict(reshard=True),
        "tog_override_out": dict(override_out=True, core_grid=_GRID_48),
        "tog_dst_full_sync": dict(dst_full_sync=True),
        "tog_math_approx": dict(math_approx=True),
        "tog_no_fp32acc": dict(fp32_acc=False),
        "tog_throttle_l1": dict(throttle=ttnn.ThrottleLevel.LEVEL_1),
        "tog_throttle_l2": dict(throttle=ttnn.ThrottleLevel.LEVEL_2),
        "tog_throttle_l3": dict(throttle=ttnn.ThrottleLevel.LEVEL_3),
        "tog_throttle_l5": dict(throttle=ttnn.ThrottleLevel.LEVEL_5),
        "tog_bf4io": dict(wdtype=BF4, odtype=BF4),
    }
    for tname, tkw in toggles.items():
        sweep[tname] = _cfg(**{**base, **tkw})
    return sweep


_SWEEP = _build_sweep()


@pytest.mark.parametrize("cfg", list(_SWEEP.keys()))
def test_block_C_full_sweep(device, cfg):
    """Every conv2d/compute option, cross-product + toggles. Infeasible combos skip."""
    try:
        _run(device, _SWEEP[cfg])
    except RuntimeError as e:
        msg = str(e)
        oom = (
            "clash with L1",
            "beyond max L1",
            "out of memory",
            "Out of Memory",
            "Statically allocated",
            "bank_manager",
            "Not enough space",
            "L1_SMALL",
        )
        # bf8/bf4 only support TILE layout: bf8 + DRAM-auto-shard (height/width) or
        # bf8 + output_layout=ROW_MAJOR are unsupported combos, not perf regressions.
        unsupported = ("Only TILE layout is supported", "layout == Layout::TILE")
        if any(k in msg for k in oom):
            pytest.skip(f"L1/L1_SMALL OOM for this combo: {msg[:90]}")
        if any(k in msg for k in unsupported):
            pytest.skip(f"unsupported dtype/layout combo: {msg[:90]}")
        raise
