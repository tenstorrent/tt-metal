# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for #60551: on Horizon the Quasar untilize gives a wrong output when another op ran before it.

The ResNet-50 layer1 conv2 goes wrong on Horizon because its DRAM height-slicing path untilizes the conv input
(to_layout TILE -> ROW_MAJOR) after the previous conv ran, and that untilize output is wrong; the first untilize
on the device is right (test_sliced_conv_steps.py chain_untilize_only).

Each case runs, N times: an earlier op (`prev`), then untilize of the same 1600x64 bfloat16 DRAM interleaved TILE
tensor, and checks the untilize output against the input:
  none    no earlier op
  conv    the model's 3x3 conv2 config on a 16x16x64 input (in L1, no DRAM slicing)
  matmul  a 64x64 by 64x64 matmul
  tilize  a 64x64 tilize
  add     a 64x64 eltwise add
  sliced_conv  the same 3x3 conv on 40x40x64 through DRAM height-slicing (2 slices), from a host-written DRAM input
  padded_slice  the sliced conv's per-slice input gather alone (21 of 40 rows, DRAM ROW_MAJOR -> L1 height-sharded)
  slice_conv  one slice's conv alone: 21x40x64 ROW_MAJOR input in L1, padding top 1, bottom 0
  slice_write  a per-slice output write alone (20x40x64 ROW_MAJOR height-sharded -> DRAM)
  slice_write_tile  the sliced conv's per-slice output write (20x40x64 TILE height-sharded -> TILE DRAM)
  slices_by_hand  the sliced conv's two slices done by hand as op_slicing.cpp does: padded_slice (L1
          height-sharded ROW_MAJOR), the conv with reallocate_halo_output=True, slice_write into a TILE DRAM output
  slices_by_hand_2core  as slices_by_hand with each 840-row input slice spread over 2 cores (448 rows each, the
          last core partly padding)
  dirty_l1  no device op: the host writes random data over nearly all of L1 (an L1 interleaved tensor), then frees it
  reshape  the quasar reshape the sliced conv runs on its DRAM tensors ([1,1,1600,64] -> [1,40,40,64], ROW_MAJOR)
With check=readback the case writes the same tensor from the host and reads it straight back, with no untilize:
that tells a bad untilize apart from DRAM that something else overwrote.

RESNET_REPEAT_N sets the number of calls (default 3).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _hs(device, rows, cores=1):
    return ttnn.create_sharded_memory_config(
        shape=(1, 1, rows, 64),
        core_grid=ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _conv(device, tt_x, size, slice_config=None, padding=(1, 1), height=None, reallocate_halo=False):
    return ttnn.experimental.quasar.conv2d(
        input_tensor=tt_x,
        weight_tensor=ttnn.from_torch(torch.randn((64, 64, 3, 3)).bfloat16() * 0.06, dtype=ttnn.bfloat16),
        bias_tensor=ttnn.from_torch(torch.zeros((1, 1, 1, 64)).bfloat16(), dtype=ttnn.bfloat16),
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=height or size,
        input_width=size,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            reallocate_halo_output=reallocate_halo,
            act_block_h_override=32,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        ),
        compute_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
        ),
        slice_config=slice_config,
        dtype=ttnn.bfloat16,
    )


def _run_prev(device, prev, cores=1):
    if prev == "sliced_conv":
        x = torch.relu(torch.randn((1, 40, 40, 64))).bfloat16()
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
        out = _conv(device, tt_x, 40, ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2))
    elif prev == "padded_slice":
        x = ttnn.from_torch(
            torch.randn((1, 40, 40, 64)).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
        )
        out = ttnn.experimental.quasar.padded_slice(x, [0, 0, 0, 0], [1, 21, 40, 64], memory_config=_hs(device, 864))
        ttnn.deallocate(x)
    elif prev == "slice_conv":
        x = torch.relu(torch.randn((1, 21, 40, 64))).bfloat16()
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        )
        out = _conv(device, tt_x, 40, padding=(1, 0, 1, 1), height=21)
    elif prev == "slice_write":
        src = ttnn.from_torch(torch.randn((1, 20, 40, 64)).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT).to(
            device, _hs(device, 800)
        )
        out = ttnn.from_torch(
            torch.zeros((1, 40, 40, 64)).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
        )
        ttnn.experimental.quasar.slice_write(src, out, [0, 0, 0, 0], [1, 20, 40, 64], [1, 1, 1, 1])
        ttnn.deallocate(src)
    elif prev == "slice_write_tile":
        src = ttnn.from_torch(torch.randn((1, 1, 800, 64)).bfloat16(), layout=ttnn.TILE_LAYOUT).to(
            device, _hs(device, 800)
        )
        src = ttnn.reshape(src, (1, 20, 40, 64))
        out = ttnn.from_torch(
            torch.zeros((1, 40, 40, 64)).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        )
        ttnn.experimental.quasar.slice_write(src, out, [0, 0, 0, 0], [1, 20, 40, 64], [1, 1, 1, 1])
        ttnn.deallocate(src)
    elif prev == "reshape":
        x = ttnn.from_torch(
            torch.randn((1, 1, 1600, 64)).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
        )
        shape = ttnn.Shape([1, 40, 40, 64])
        out = ttnn.experimental.quasar.reshape(x, shape, shape)
    elif prev == "slices_by_hand":
        x = ttnn.from_torch(
            torch.relu(torch.randn((1, 40, 40, 64))).bfloat16(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=DRAM,
        )
        out = ttnn.from_torch(
            torch.zeros((1, 40, 40, 64)).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        )
        # (input rows, conv padding, output rows) for the 2 slices of the 40-row output
        for (r0, r1), pad, (o0, o1) in (((0, 21), (1, 0, 1, 1), (0, 20)), ((19, 40), (0, 1, 1, 1), (20, 40))):
            xs = ttnn.experimental.quasar.padded_slice(
                x, [0, r0, 0, 0], [1, r1, 40, 64], memory_config=_hs(device, 864 if cores == 1 else 448, cores)
            )
            ys = _conv(device, xs, 40, padding=pad, height=r1 - r0, reallocate_halo=True)
            ys = ttnn.reshape(ys, (1, o1 - o0, 40, 64))
            ttnn.experimental.quasar.slice_write(ys, out, [0, o0, 0, 0], [1, o1, 40, 64], [1, 1, 1, 1])
            ttnn.deallocate(ys)
        ttnn.deallocate(x)
    elif prev == "dirty_l1":
        out = ttnn.from_torch(
            torch.randn((1, 1, 40960, 64)).bfloat16(),  # ~2.6 MB per L1 bank: nearly all of it
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        )
    elif prev == "conv":
        x = torch.relu(torch.randn((1, 16, 16, 64))).bfloat16()
        tt_x = ttnn.from_torch(x.reshape(1, 1, 256, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(
            device,
            ttnn.create_sharded_memory_config(
                shape=(1, 1, 256, 64),
                core_grid=ttnn.num_cores_to_corerangeset(1, device.compute_with_storage_grid_size(), True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )
        out = _conv(device, tt_x, 16)
    else:
        a = ttnn.from_torch(
            torch.randn((1, 1, 64, 64)).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        if prev == "matmul":
            out = ttnn.experimental.quasar.matmul(a, a)
        elif prev == "add":
            out = ttnn.experimental.quasar.add(a, a)
        else:  # tilize
            rm = ttnn.from_torch(torch.randn((1, 1, 64, 64)).bfloat16(), dtype=ttnn.bfloat16, device=device)
            out = ttnn.experimental.quasar.tilize(rm)
            ttnn.deallocate(rm)
        ttnn.deallocate(a)
    ttnn.deallocate(out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "prev, check",
    [
        ("none", "untilize"),
        ("conv", "untilize"),
        ("matmul", "untilize"),
        ("tilize", "untilize"),
        ("add", "untilize"),
        ("sliced_conv", "untilize"),
        ("sliced_conv", "readback"),
        ("padded_slice", "untilize"),
        ("slice_conv", "untilize"),
        ("slice_write", "untilize"),
        ("slice_write_tile", "untilize"),
        ("reshape", "untilize"),
        ("dirty_l1", "untilize"),
        ("slices_by_hand", "untilize"),
        ("slices_by_hand_2core", "untilize"),
    ],
)
def test_untilize_after_op(mesh_device, prev, check, monkeypatch):
    device = mesh_device
    torch.manual_seed(0)
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # the model's split path for HS conv2
    n_iter = int(os.environ.get("RESNET_REPEAT_N", "3"))
    flat = torch.relu(torch.randn((1, 1, 1600, 64))).bfloat16()

    pccs = []
    for it in range(n_iter):
        if prev != "none":
            _run_prev(device, prev.removesuffix("_2core"), cores=2 if prev.endswith("_2core") else 1)
        x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        if check == "untilize":
            rm = ttnn.experimental.quasar.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            got = ttnn.to_torch(rm).float().reshape(-1, 64)
            ttnn.deallocate(rm)
        else:
            got = ttnn.to_torch(x).float().reshape(-1, 64)
        ttnn.deallocate(x)
        bad_rows = ((got - flat.float().reshape(-1, 64)).abs().amax(dim=-1) > 0.1).nonzero().flatten()
        pccs.append(_pcc(got, flat.float()))
        logger.info(f"[UNTILIZE] prev={prev} check={check} call={it} pcc={pccs[-1]:.6f} bad_rows={len(bad_rows)}/1600")

    assert min(pccs) >= 0.999, f"an untilize output was wrong: min PCC {min(pccs):.6f}"
