# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Replication of Block A's critical conv2d (%400/%587/%774/%961 in
# ttnn_block_A_deformed_backbone — 4 identical instances, one per camera).
#
#   in=384, out=384, k=3x3, stride=1, pad=1, H=24, W=24, batch=1
#   bf16 · HiFi3 · fp32_dest_acc · relu6 · act_block_h_override=0
#   config_tensors_in_dram=true · HEIGHT_SHARDED L1 (shard [32,384] on 18 cores:
#   ranges (0,0)-(7,1) + (0,2)-(1,2))
#
# Bottleneck (from block_A_critical_conv2d_analysis.md): each instance runs
# ~254 us but PM-ideal is only 17.5 us -> 14.5x slower, 6.9% FPU. The spatial map
# is tiny (24x24 = 576 px = 18 tiles, 1 tile/core over 18 cores). In the 1D
# height-sharded path ONE sender core reads the ENTIRE 2.65 MB bf16 weight tensor
# from DRAM once and multicasts it to the others (receivers never touch DRAM for
# weights). That single serial weight stream, single-buffered with no prefetch and
# only 1 tile of compute to hide behind, IS the kernel: BRISC ~227 us, FPU 6.9%.
#
# Levers (batch=1; see docs/block_A_conv2d_perf_improvement.md; measured, median of 5):
#   height bf16 (baseline) .......... 253 us  6.9% FPU  14.5x  1.00x  single serial weight sender
#   height bf8 + weights_db ......... 146 us 12.0% FPU   8.3x  1.74x  caps ~12% (sender is the wall)
#   block bf16 ...................... 93 us 18.8% FPU   5.3x  2.71x  36 PARALLEL weight readers (identical)
#   block bf16 + weights_db+act_db .. 62 us 28.3% FPU   3.5x  4.10x  (numerically identical)
#   block bf8  + weights_db+act_db .. 58 us 30.4% FPU   3.3x  4.40x  >30% FPU  (bf8 weights)
# KEY: the sharding LAYOUT is the dominant lever. HEIGHT->BLOCK turns the one serial
# weight sender into 36 cores each reading only their OC slice from DRAM in parallel
# (2.71x, no precision change). On block-sharded, act_double_buffer is the multiplier
# (weights_double_buffer alone does nothing); bf8 weights add the final push past 30%.
# What does NOT help (verified): enable_activation_reuse (fatal: needs act_block_h>1),
# act_block_h/fewer cores on height-sharded (weights already loaded once, not per-core),
# width-sharded (391 us, worse), full_inner_dim / act_block_w_div / split_reader.
#
# Run with the Tracy device profiler to reproduce the bottleneck row:
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest \
#     tests/ttnn/unit_tests/operations/conv/test_block_A_critical_conv2d.py
# then read DEVICE KERNEL DURATION / PM IDEAL / PM FPU UTIL from
#   generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv

import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT

# Op geometry (matches %400).
N, C_IN, H, W = 1, 384, 24, 24
C_OUT, KH, KW = 384, 3, 3
NHW = N * H * W  # 576


def _height_sharded_input_cfg(shard_h=32):
    # #ttnn_layout173: HEIGHT_SHARDED L1, shard [shard_h, 384] on the same 18 cores:
    # ranges (0,0)-(7,1) [16 cores] + (0,2)-(1,2) [2 cores]. shard_h=32 -> 1 tile/core
    # (batch=1); shard_h=32*batch keeps 18 cores but puts batch tiles per core.
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(1, 2)),
        }
    )
    shard = ttnn.ShardSpec(grid, [shard_h, 384], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)


def _block_sharded_input_cfg():
    # BLOCK_SHARDED L1 on a 6x6 grid: NHW=576 split into 6 row-groups of 96 (3 tiles),
    # C=384 split into 6 col-groups of 64 (2 tiles) -> 36 cores each reading only its
    # own OC slice from DRAM (parallel weight reads, vs the single height-sharded sender).
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))})
    shard = ttnn.ShardSpec(grid, [96, 64], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


_LAYOUTS = {
    "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}


def _conv_config(
    weights_db=False,
    act_db=False,
    abh=0,
    wdtype=ttnn.bfloat16,
    act_reuse=False,
    sharding="height",
    dram_config=True,
    full_inner=False,
    transpose_shards=False,
    abw_div=1,
    split_reader=None,
    realloc_halo=False,
):
    return ttnn.Conv2dConfig(
        weights_dtype=wdtype,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        act_block_h_override=abh,
        act_block_w_div=abw_div,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=dram_config,
        enable_weights_double_buffer=weights_db,
        enable_act_double_buffer=act_db,
        enable_activation_reuse=act_reuse,
        full_inner_dim=full_inner,
        transpose_shards=transpose_shards,
        reallocate_halo_output=realloc_halo,
        force_split_reader=split_reader,
        shard_layout=_LAYOUTS[sharding],
        output_layout=TILE,
    )


def _run(
    device,
    weights_db,
    act_db,
    abh=0,
    wdtype=ttnn.bfloat16,
    odtype=ttnn.bfloat16,
    act_reuse=False,
    batch=1,
    sharding="height",
    fidelity=ttnn.MathFidelity.HiFi3,
    dram_config=True,
    fp32_acc=True,
    packer_l1_acc=False,
    math_approx=False,
    dst_full_sync=False,
    full_inner=False,
    transpose_shards=False,
    abw_div=1,
    split_reader=None,
    realloc_halo=False,
):
    """Prepare weights/bias and run %400's conv2d 5x under the given config knobs."""
    torch.manual_seed(0)
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=math_approx,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=dst_full_sync,
    )
    conv_cfg = _conv_config(
        weights_db=weights_db,
        act_db=act_db,
        abh=abh,
        wdtype=wdtype,
        act_reuse=act_reuse,
        sharding=sharding,
        dram_config=dram_config,
        full_inner=full_inner,
        transpose_shards=transpose_shards,
        abw_div=abw_div,
        split_reader=split_reader,
        realloc_halo=realloc_halo,
    )
    nhw_b = batch * NHW
    if batch == 1 and sharding == "height":
        in_mem = _height_sharded_input_cfg()
    elif batch == 1 and sharding == "block":
        # Feed a pre-built block-sharded L1 input: parallel per-core weight reads,
        # and (unlike DRAM auto-shard) it keeps bf8 weights on the legal TILE path.
        in_mem = _block_sharded_input_cfg()
    else:
        # Feed DRAM and let conv2d auto-shard to the requested layout on its own grid.
        in_mem = DRAM

    w = torch.randn(C_OUT, C_IN, KH, KW, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, C_OUT, dtype=torch.bfloat16)

    # prepare_conv2d_weights  (@forward_const_eval_149)
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=in_mem,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=C_IN,
        out_channels=C_OUT,
        batch_size=batch,
        input_height=H,
        input_width=W,
        kernel_size=(KH, KW),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # prepare_conv2d_bias  (@forward_const_eval_10)
    tt_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=in_mem,
        input_layout=TILE,
        in_channels=C_IN,
        out_channels=C_OUT,
        batch_size=batch,
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

    # activation: [1,1,576,384] (the conv input). deallocate_activation=True frees
    # it each call, so upload a fresh copy per iteration (host cost, not device time).
    x = torch.randn(1, 1, nhw_b, C_IN, dtype=torch.bfloat16)

    out = None
    for _ in range(5):
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM)
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
            batch_size=batch,
            input_height=H,
            input_width=W,
            groups=1,
            dtype=odtype,
            conv_config=conv_cfg,
            compute_config=compute,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, nhw_b, C_OUT], f"unexpected output shape {list(out.shape)}"


def test_block_A_baseline(device):
    """%400 as shipped: height-sharded, no double-buffering. ~254 us, 6.9% FPU."""
    _run(device, weights_db=False, act_db=False, abh=0)


# ---------------------------------------------------------------------------
# FPU-maximization sweep (batch=1 fixed). The height-sharded single-sender weight
# stream caps FPU at ~12%; the route past 30% is to PARALLELIZE the weight DRAM
# read across cores via BLOCK/WIDTH sharding (each core reads only its OC slice),
# then stack bf8 weights + double-buffering on top.
# ---------------------------------------------------------------------------
import itertools

# Combinatorial cross-product sweep (batch=1). Rather than change one option at a
# time, enumerate the full product of the knobs that actually move the needle and
# find the global optimum. Fixed: BLOCK_SHARDED (the dominant layout lever) + bf8
# weights (bf4 was proven to give nothing — the weight read is latency-bound) +
# act double-buffer (the required multiplier on block-sharded).
_DIMS = {
    "fidelity": [("hi3", ttnn.MathFidelity.HiFi3), ("hi4", ttnn.MathFidelity.HiFi4)],
    "full_inner": [("", False), ("fi", True)],
    "packer_l1_acc": [("", False), ("pk", True)],
    "weights_db": [("", False), ("wdb", True)],
    "abh": [("", 0), ("abh96", 96)],
    "transpose_shards": [("", False), ("tr", True)],
}


def _build_sweep():
    sweep = {"height_bf16": dict(sharding="height", weights_db=False, act_db=False)}
    keys = list(_DIMS.keys())
    for combo in itertools.product(*[_DIMS[k] for k in keys]):
        cfg = dict(sharding="block", act_db=True, wdtype=ttnn.bfloat8_b)
        tags = []
        for k, (tag, val) in zip(keys, combo):
            cfg[k] = val
            if tag:
                tags.append(tag)
        name = "_".join(tags) if tags else "min"
        sweep[name] = cfg
    return sweep


_SWEEP = _build_sweep()


@pytest.mark.parametrize("cfg", list(_SWEEP.keys()))
def test_block_A_fpu_sweep(device, cfg):
    """Full cross-product of the impactful conv2d/compute options (batch=1)."""
    _run(device, **_SWEEP[cfg])
