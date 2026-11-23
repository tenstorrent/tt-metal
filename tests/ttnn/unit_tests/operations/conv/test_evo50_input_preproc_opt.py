# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""EVO50 input-preproc, optimisation round on top of test_evo50_input_preproc_s2d_nhwc.

Same stem (pixel_unshuffle channels_last -> 4x4 S2D conv) and grouped 3x3 conv; the variants
differ in how the conv2 output [1,1,159744,16] (TILE, height-sharded, 2496 pixels = 3 image rows
per core) becomes the model output [1,16,192,832] ROW_MAJOR in DRAM.

  ir : the IR tail - reshape, permute(0,3,1,2) to L1 TILE, move to DRAM, untilize.
  tw : transpose the last two dims on the shards (each core: [2496,16] -> [16,2496]), untilize to
       DRAM, reshape. A core's 2496 pixels are 3 whole image rows, so for every channel they are
       contiguous in NCHW memory: [16, 159744] row-major is [1,16,192,832] row-major.
"""

import os

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.unit_tests.operations.conv.test_evo50_input_preproc import (
    BATCH,
    C1_OUT,
    C2_OUT,
    C2_GROUPS,
    C2_K,
    C2_PAD,
    C2_STRIDE,
    HS_CORE_GRID,
    MID_H,
    MID_W,
    S2D_C,
    S2D_CP,
    S2D_H,
    S2D_W,
    _check,
    _common_configs,
    _free_unless_alias,
    _make_inputs,
    _stem_s2d_nhwc,
    pack_weight_s2d,
    signpost,
    torch_golden,
)


def _conv2(device, t6, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_cfg):
    return ttnn.conv2d(
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
        slice_config=slice_cfg,
    )


def _tail_ir(t7):
    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))
    _free_unless_alias(t7, t8)
    t9 = ttnn.permute(t8, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t8)
    t10 = ttnn.to_memory_config(t9, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t9)
    out = ttnn.to_layout(t10, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t10)
    return out


def _tail_tw(t7):
    # [1,1,159744,16] HS tile -> [1,1,16,159744]: transpose on the shards, output width-sharded
    t8 = ttnn.transpose(t7, 2, 3)
    ttnn.deallocate(t7)
    out = ttnn.to_layout(t8, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t8)
    return ttnn.reshape(out, (BATCH, C2_OUT, MID_H, MID_W))


def _tail_rowt(t7, rm_mem=ttnn.L1_MEMORY_CONFIG):
    # [1,1,159744,16] -> view [1,192,832,16] -> WH transpose per image row: [1,192,16,832] TILE
    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))
    _free_unless_alias(t7, t8)
    t9 = ttnn.transpose(t8, 2, 3)
    ttnn.deallocate(t8)
    if rm_mem == "same":
        print(f"rowt transpose out: {t9.shape} {t9.memory_config()}")
        rm_mem = t9.memory_config()
    # untilize: every (h, c) row of 832 elements is one 1664 B stick
    t10 = ttnn.to_layout(t9, ttnn.ROW_MAJOR_LAYOUT, memory_config=rm_mem)
    ttnn.deallocate(t9)
    # swap H and C on whole sticks, straight into DRAM
    out = ttnn.transpose(t10, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t10)
    return out


def _tail_rowt_dram(t7):
    return _tail_rowt(t7, ttnn.DRAM_MEMORY_CONFIG)


def _tail_rowt_sharded(t7):
    return _tail_rowt(t7, "same")


TAILS = {
    "ir": _tail_ir,
    "tw": _tail_tw,
    "rowt": _tail_rowt,
    "rowtd": _tail_rowt_dram,
    "rowts": _tail_rowt_sharded,
}


# conv knob sets on top of the sweep-tuned s2d_nhwc config: (conv1 overrides, conv2 overrides)
CONVS = {
    "base": ({}, {}),
    "fid1": ({"full_inner_dim": True}, {}),
    "fid2": ({}, {"full_inner_dim": True}),
    "fid12": ({"full_inner_dim": True}, {"full_inner_dim": True}),
}
for _abh in (416, 192, 96):
    _f = {"full_inner_dim": True, "act_block_h_override": _abh}
    CONVS[f"fid1a{_abh}"] = (_f, {})
    CONVS[f"fid2a{_abh}"] = ({}, _f)
_nd = {"full_inner_dim": True, "enable_act_double_buffer": False}
CONVS["fid1nd"] = (_nd, {})
CONVS["fid2nd"] = ({}, _nd)


@pytest.mark.parametrize("conv", list(CONVS))
@pytest.mark.parametrize("tail", list(TAILS))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc_opt(device, tail, conv):
    x, w1, b1, w2, b2 = _make_inputs()
    _, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w1_packed16 = F.pad(pack_weight_s2d(w1.float()), (0, 0, 0, 0, 0, S2D_CP - S2D_C))
    tt_w1_packed16 = ttnn.from_torch(w1_packed16.to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, _, slice_l1_full = _common_configs(device)
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
    conv1_cfg = ttnn.Conv2dConfig(**{**tuned, **CONVS[conv][0]})
    conv2_cfg = ttnn.Conv2dConfig(**{**tuned, **CONVS[conv][1]})
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
    tail_fn = TAILS[tail]

    def run():
        y1 = _stem_s2d_nhwc(device, tt_in, tt_w1_packed16, tt_b1, conv1_cfg, compute_cfg, slice_l1_full, hs_rm_cfg)
        t7 = _conv2(device, y1, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)
        return tail_fn(t7)

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header=f"evo50_opt_{tail}_{conv}-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header=f"evo50_opt_{tail}_{conv}-end")

    print(f"tail={tail} out {out.shape} {out.layout} {out.memory_config()}")
    _check(result, golden)
