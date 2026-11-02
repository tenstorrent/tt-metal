# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Smaller repro attempts for #60551 (ResNet-50 on Horizon: the layer1 3x3 conv2 gives wrong output on the 2nd and later
calls, test_conv2d_layer1_conv2_repeat.py).

Each case runs one op N times on one device with the same input and checks every call against torch and against
call 0. Both ops use the halo step (untilize_with_halo) on a TILE, HEIGHT_SHARDED input, 64 channels, 3x3 window,
stride 1, padding 1:
  conv     the model's conv2 config (as test_conv2d_layer1_conv2_repeat.py "model"), at a chosen size
  maxpool  max_pool2d with the same window: the halo step without the conv's matmul
If maxpool also changes from call to call, the matmul is not needed for the bug.

Results on Horizon (2026-11-02, L1 bank 2787776 B): call 0 is always correct. Every maxpool case and the conv at 16
and 32 give the same correct output on every call. The conv at 40 (1 core) and 56 (1 and 2 cores) give a wrong output
(PCC ~0.6) on a later call. Those are exactly the convs whose per-core tilized activation is over conv2d.cpp's L1-fit
threshold, which sends the split conv through DRAM height-slicing: the smallest repro is conv-40-one.

For each conv case it logs [HALO] with the L1 bank size and whether the conv goes through DRAM height-slicing. For
each call it logs [HALO] with the PCC, the number of output pixels that differ from call 0, and the bad
(32-row block, channel half) pairs: a pixel is bad in channel half k when its max error over channels
32k..32k+31 is above 0.1.

RESNET_REPEAT_N sets the number of calls (default 3). RESNET_NO_PROGRAM_CACHE=1 turns the program cache off.

Run (Horizon emulator, slow dispatch):
  pytest -q -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_halo_tile_repeat.py -k conv-40-one
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

CH = 64


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _bad_blocks(out, golden):
    """out/golden [H*W, C] -> sorted (32-row block, channel half) pairs with an error above 0.1."""
    err = (out - golden).abs().reshape(out.shape[0], CH // 32, 32).amax(dim=-1)  # [HW, 2]
    bad = (err > 0.1).nonzero().tolist()
    return sorted({(r // 32, c) for r, c in bad})


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "op, spatial, cores",
    [
        ("conv", 56, "one"),
        ("conv", 56, "all"),
        ("conv", 16, "one"),
        ("conv", 16, "all"),
        # Either side of the Quasar split conv's L1-fit limit on 1 core (conv2d.cpp: per-core tilized activation
        # above ~60 % of a small L1 bank goes through DRAM height-slicing): 32x32 stays in L1, 40x40 slices.
        ("conv", 32, "one"),
        ("conv", 40, "one"),
        ("maxpool", 56, "one"),
        ("maxpool", 56, "all"),
        ("maxpool", 16, "one"),
        ("maxpool", 16, "all"),
    ],
)
def test_halo_tile_repeat(mesh_device, op, spatial, cores, monkeypatch):
    device = mesh_device
    torch.manual_seed(0)
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # the model's split path for HS conv2
    n_iter = int(os.environ.get("RESNET_REPEAT_N", "3"))
    if os.environ.get("RESNET_NO_PROGRAM_CACHE"):
        device.disable_and_clear_program_cache()  # every call builds new programs

    torch_input = torch.relu(torch.randn((1, CH, spatial, spatial))).bfloat16().float()
    if op == "conv":
        torch_weight = (torch.randn((CH, CH, 3, 3)) * (2.0 / (CH * 9)) ** 0.5).bfloat16().float()
        torch_bias = (torch.randn((CH,)) * 0.1).bfloat16().float()
        golden = torch.relu(torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias, padding=1))
    else:
        golden = torch.nn.functional.max_pool2d(torch_input, kernel_size=3, stride=1, padding=1)
    nhw = spatial * spatial
    golden = golden[0].permute(1, 2, 0).reshape(nhw, CH)

    flat = torch.permute(torch_input, (0, 2, 3, 1)).reshape(1, 1, nhw, CH).contiguous()
    grid = device.compute_with_storage_grid_size()
    num_cores = 1 if cores == "one" else max(c for c in range(1, grid.x * grid.y + 1) if nhw % (c * 32) == 0)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, nhw // num_cores, CH),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    logger.info(f"[HALO] op={op} spatial={spatial} cores={num_cores} calls={n_iter}")
    if op == "conv":
        # The same L1-fit check conv2d.cpp makes for the split conv: the per-core tilized activation
        # [per_core_M, full_K] (bfloat16 tiles) against 60 % of a small (< 3 MB) L1 bank, else 80 %.
        bank = ttnn._ttnn.reports.get_device_info(device).l1_bank_size
        tilized = (nhw // num_cores // 32) * (CH * 9 // 32) * 2048
        threshold = bank * (60 if bank < 3 * 1024 * 1024 else 80) // 100
        logger.info(
            f"[HALO] l1_bank={bank} B, per-core tilized activation={tilized} B, fit threshold={threshold} B "
            f"-> {'DRAM height-sliced' if tilized > threshold else 'in L1'}"
        )

    if op == "conv":
        tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
        tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, CH), dtype=ttnn.bfloat16)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            reallocate_halo_output=False,
            act_block_h_override=32,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
        )

    outs = []
    for it in range(n_iter):
        tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, in_mem)
        if op == "conv":
            out = ttnn.experimental.quasar.conv2d(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=tt_bias,
                in_channels=CH,
                out_channels=CH,
                batch_size=1,
                input_height=spatial,
                input_width=spatial,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                device=device,
                conv_config=conv_config,
                compute_config=compute_config,
                dtype=ttnn.bfloat16,
            )
        else:
            out = ttnn.experimental.quasar.max_pool2d(
                input_tensor=tt_input,
                batch_size=1,
                input_h=spatial,
                input_w=spatial,
                channels=CH,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding=[1, 1],
                dilation=[1, 1],
            )
            ttnn.deallocate(tt_input)
        res = ttnn.to_torch(ttnn.from_device(out)).float()
        ttnn.deallocate(out)
        res = res.reshape(-1, res.shape[-1])[:nhw, :CH]
        outs.append(res)
        changed = int((res != outs[0]).any(dim=-1).sum())
        logger.info(
            f"[HALO] op={op} spatial={spatial} cores={num_cores} call={it} pcc={_pcc(res, golden):.6f} "
            f"pixels_changed_vs_call0={changed} bad(block32,ch_half)={_bad_blocks(res, golden)}"
        )

    diffs = [it for it in range(1, n_iter) if not torch.equal(outs[it], outs[0])]
    assert not diffs, f"the output changed between calls: calls {diffs} differ from call 0"
    pcc0 = _pcc(outs[0], golden)
    assert pcc0 >= 0.99, f"PCC {pcc0:.6f} < 0.99"
