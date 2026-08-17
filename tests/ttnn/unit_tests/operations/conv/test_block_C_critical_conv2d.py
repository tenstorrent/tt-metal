# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Replication of Block C's critical conv2d (%272 in ttnn_block_C_cylinder_backbone).
#
#   in=192, out=192, k=3x3, stride=1, pad=1, H=160, W=288, batch=1
#   bf16 · HiFi3 · fp32_dest_acc · relu6 · act_block_h_override=64
#   config_tensors_in_dram=true · BLOCK_SHARDED L1 (shard [5760,32] on cores (0,0)-(5,7), 48 cores)
#
# This op is compute-bound in the PM model (PM ideal ~350us) but runs ~2.965 ms
# on device — 8.47x slower, ~11.8% FPU util. Root cause (verified): the prepared
# weights are ALWAYS DRAM-streamed tile-by-tile by the writer kernel (this has
# nothing to do with config_tensors_in_dram, which only relocates the sliding-
# window index tensor out of L1_SMALL). With act_block_h_override=64 the output
# height is split into 90 blocks, so the whole weight set is re-streamed from
# DRAM 90 times, and with the weights CB single-buffered TRISC idle-waits on the
# writer's DRAM loads block-for-block (~88% idle).
#
# The bottleneck is the weight *sender* (BRISC), which is transaction-latency-
# bound (halving weight *bytes* via bf8 alone did NOT help) — it re-streams the
# whole weight set once per output-height block, so the only real lever is fewer
# blocks (larger act_block_h), gated by L1. Two tiers of fix:
#
# Tier 1 (numerically identical, pure perf/L1 knobs):
#   enable_weights_double_buffer + enable_act_double_buffer  (writer prefetches the
#   next block while TRISC matmuls the current), plus the largest act_block_h that
#   fits bf16 -> abh=384 (15 blocks). abh>=480 clashes with the bf16 sharded tensors.
# Tier 2 (trades precision for headroom): bf8_b weights + output shrink the
#   globally-allocated activation/output CBs, freeing L1 for abh=576 (10 blocks).
#   abh>=640 and act_db=off both regress, so 576 is the measured ceiling.
#
# Measured (Tracy, median of 5, this shape, 48 cores):
#   baseline (abh=64) .............. 2962 us  11.8% FPU  8.46x PM-ideal  1.00x
#   tier1  wadb + abh=384 (bf16) ... 1240 us  28.2% FPU  3.54x PM-ideal  2.39x
#   tier2  bf8 in/out + abh=576 ....  888 us  39.4% FPU  2.54x PM-ideal  3.34x
# BRISC stays ~98% of wall throughout: still weight-transaction-bound, pushed as
# far as L1 allows. Beyond this needs a resident-weights schedule (doesn't fit L1).
#
# Run with the Tracy device profiler to reproduce the rows:
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#     python -m tracy -r -p -m pytest \
#     tests/ttnn/unit_tests/operations/conv/test_block_C_critical_conv2d.py
# then read DEVICE KERNEL DURATION / PM IDEAL / PM FPU UTIL from
#   generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv

import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT

# Op geometry (matches %272).
N, C_IN, H, W = 1, 192, 160, 288
C_OUT, KH, KW = 192, 3, 3
NHW = N * H * W  # 46080


def _block_sharded_input_cfg():
    # #ttnn_layout214: BLOCK_SHARDED L1, shard [5760, 32] on cores (0,0)-(5,7).
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))}),
        [5760, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def _conv_config(
    weights_db=False,
    act_db=False,
    dram_weights=True,
    sharding="block",
    abh=64,
    full_inner=False,
    wdtype=ttnn.bfloat16,
    transpose_shards=False,
):
    layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if sharding == "height" else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    return ttnn.Conv2dConfig(
        weights_dtype=wdtype,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        deallocate_activation=True,
        act_block_h_override=abh,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=dram_weights,
        enable_weights_double_buffer=weights_db,
        enable_act_double_buffer=act_db,
        full_inner_dim=full_inner,
        transpose_shards=transpose_shards,
        shard_layout=layout,
        output_layout=TILE,
    )


# Config variants to compare under the profiler. "baseline" reproduces %272's stall.
# The weights are always DRAM-streamed; the levers are double-buffering the weight/act
# CBs and reducing the number of output-height blocks (bigger act_block_h) so weights
# are re-streamed fewer times.
_BF8 = ttnn.bfloat8_b


def _run(
    device,
    weights_db,
    act_db,
    abh,
    wdtype=ttnn.bfloat16,
    odtype=ttnn.bfloat16,
    idtype=ttnn.bfloat16,
    full_inner=False,
    transpose_shards=False,
    fidelity=ttnn.MathFidelity.HiFi3,
    fp32_acc=True,
    packer_l1_acc=False,
    math_approx=False,
    dst_full_sync=False,
):
    """Prepare weights/bias and run %272's conv2d 5x under the given config knobs."""
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
        full_inner=full_inner,
        transpose_shards=transpose_shards,
    )
    in_mem = _block_sharded_input_cfg()

    w = torch.randn(C_OUT, C_IN, KH, KW, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, C_OUT, dtype=torch.bfloat16)

    # prepare_conv2d_weights  (@forward_const_eval_80)
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=in_mem,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=C_IN,
        out_channels=C_OUT,
        batch_size=N,
        input_height=H,
        input_width=W,
        kernel_size=(KH, KW),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=idtype,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # prepare_conv2d_bias  (@forward_const_eval_70)
    tt_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=RM),
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
        input_dtype=idtype,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # activation: [1,1,46080,192] (the conv input). deallocate_activation=True frees
    # it each call, so upload a fresh copy per iteration (host cost, not device time).
    x = torch.randn(1, 1, NHW, C_IN, dtype=torch.bfloat16)

    out = None
    for _ in range(5):
        tt_x = ttnn.from_torch(x, dtype=idtype, layout=TILE, device=device, memory_config=DRAM)
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
            dtype=odtype,
            conv_config=conv_cfg,
            compute_config=compute,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
    ttnn.synchronize_device(device)
    assert list(out.shape) == [1, 1, NHW, C_OUT], f"unexpected output shape {list(out.shape)}"


def test_block_C_baseline(device):
    """%272 as shipped: no double-buffering, act_block_h=64 (90 blocks).
    ~2962 us, 11.8% FPU, 8.46x PM-ideal — the compute-bound stall."""
    _run(device, weights_db=False, act_db=False, abh=64)


def test_block_C_double_buffer_abh384(device):
    """Tier 1 (numerically identical): weights + act double-buffer, act_block_h=384
    (15 blocks). ~1240 us, 28.2% FPU, 2.39x faster than baseline."""
    _run(device, weights_db=True, act_db=True, abh=384)


def test_block_C_bf8_abh576(device):
    """Tier 2 (bf8_b weights + output, precision trade): frees L1 for act_block_h=576
    (10 blocks). ~888 us, 39.4% FPU, 3.34x faster than baseline."""
    _run(device, weights_db=True, act_db=True, abh=576, wdtype=_BF8, odtype=_BF8)


import itertools


# Combinatorial cross-product sweep (block-sharded, weights+act double-buffer fixed
# on — both are required here). Unlike Block A (single-sender height-sharded), Block C
# is bound by re-streaming the weights once per output-height block, so the primary
# lever is act_block_h (fewer blocks). bf16 caps at abh=384; bf8 in/out frees L1 for
# abh=576. NOTE: full_inner_dim AND packer_l1_acc both OOM on Block C — its activation
# and output tensors are far larger than Block A's, leaving no L1 for the extra buffers
# (verified: CB clash at abh=192 even). So the feasible space is abh x precision x
# fidelity; those two knobs are omitted from the product.
def _bc_sweep():
    sweep = {"baseline": dict(weights_db=False, act_db=False, abh=64)}
    # (abh, dtype-tag) pairs that fit L1: bf16<=384, bf8io up to 576
    prec = [("bf16", {}), ("bf8io", dict(wdtype=_BF8, odtype=_BF8))]
    for ptag, pkw in prec:
        abhs = [384, 576] if ptag == "bf8io" else [192, 384]
        for abh in abhs:
            for fid in [("", ttnn.MathFidelity.HiFi3), ("hi4", ttnn.MathFidelity.HiFi4)]:
                tags = [ptag, f"abh{abh}"] + ([fid[0]] if fid[0] else [])
                sweep["_".join(tags)] = dict(
                    weights_db=True,
                    act_db=True,
                    abh=abh,
                    fidelity=fid[1],
                    **pkw,
                )
    return sweep


_SWEEP = _bc_sweep()


@pytest.mark.parametrize("cfg", list(_SWEEP.keys()))
def test_block_C_fpu_sweep(device, cfg):
    """Full cross-product of act_block_h / precision / full_inner / packer / fidelity."""
    _run(device, **_SWEEP[cfg])
