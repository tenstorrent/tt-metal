# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Step-by-step check of the Quasar split conv's DRAM height-slicing path for #60551.

test_halo_tile_repeat.py conv-40-one goes wrong (PCC ~0.6 on a later call, ~0.2 on every call with the program cache
off) only when conv2d.cpp's L1-fit check sends the conv through DRAM height-slicing:
  1. to_memory_config: the height-sharded L1 TILE input -> DRAM interleaved
  2. to_layout: TILE -> ROW_MAJOR
  3. conv2d_DRAM: per slice, padded_slice the input rows into L1, run conv2d_L1 on the slice, slice_write the
     result into a DRAM interleaved output
  4. to_memory_config: DRAM interleaved -> the conv's height-sharded L1 output config
Each case runs one of these pieces N times on the same 40x40x64 input (1 core, as conv-40-one) and checks every call
against torch:
  moves       steps 1, 2 and 4 alone (no conv)
  dram_conv   step 3 alone: conv2d with a DRAM interleaved ROW_MAJOR input and a 2-slice DRAM_HEIGHT slice_config
  slice_conv  one slice's conv alone: the 21-row top slice (padding top 1, bottom 0) on an L1 interleaved input
  padded_slice  the top slice's input gather alone (DRAM ROW_MAJOR -> L1 height-sharded)
  chain_keep  steps 1-4 in order, keeping the L1 input alive during the sliced conv, as conv2d.cpp does
  chain_free  steps 1-4 in order, freeing the L1 input before the sliced conv
  chain_hostin  steps 3-4: the sliced conv on a DRAM ROW_MAJOR input written from the host, then step 4
  chain_noreshard  steps 1-3: the device-made DRAM input and the sliced conv, no step 4
  chain_untilize_only  step 2 then 3: host writes a DRAM TILE input, the device untilizes it, then the sliced conv
  chain_untilize_pause  as chain_untilize_only with a 2 s host pause between the untilize and the sliced conv
  chain_tointerleaved_only  step 1 then 3: host writes an L1 height-sharded ROW_MAJOR input, the device moves it to
              DRAM, then the sliced conv

RESNET_REPEAT_N sets the number of calls (default 3). RESNET_NO_PROGRAM_CACHE=1 turns the program cache off.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

CH = 64
S = 40  # image size; 1 core puts this conv over the L1-fit threshold on Horizon
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _hs_config(device, rows, cores=1):
    return ttnn.create_sharded_memory_config(
        shape=(1, 1, rows // cores, CH),
        core_grid=ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _conv_config():
    # The model's conv2 config, as test_halo_tile_repeat.py.
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=32,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        "moves",
        "dram_conv",
        "slice_conv",
        "padded_slice",
        "chain_keep",
        "chain_free",
        "chain_hostin",
        "chain_noreshard",
        "chain_untilize_only",
        "chain_tointerleaved_only",
        "chain_untilize_pause",
    ],
)
def test_sliced_conv_steps(mesh_device, case, monkeypatch):
    device = mesh_device
    torch.manual_seed(0)
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")
    n_iter = int(os.environ.get("RESNET_REPEAT_N", "3"))
    if os.environ.get("RESNET_NO_PROGRAM_CACHE"):
        device.disable_and_clear_program_cache()

    torch_input = torch.relu(torch.randn((1, CH, S, S))).bfloat16().float()
    torch_weight = (torch.randn((CH, CH, 3, 3)) * (2.0 / (CH * 9)) ** 0.5).bfloat16().float()
    torch_bias = (torch.randn((CH,)) * 0.1).bfloat16().float()
    nhwc = torch_input.permute(0, 2, 3, 1).contiguous()  # [1, S, S, C]
    flat = nhwc.reshape(1, 1, S * S, CH)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    def conv(tt_input, in_h, padding, slice_config=None):
        return ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16),
            bias_tensor=ttnn.from_torch(torch_bias.reshape(1, 1, 1, CH), dtype=ttnn.bfloat16),
            in_channels=CH,
            out_channels=CH,
            batch_size=1,
            input_height=in_h,
            input_width=S,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=padding,
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=_conv_config(),
            compute_config=compute_config,
            slice_config=slice_config,
            dtype=ttnn.bfloat16,
        )

    def golden_conv(x_nchw, padding):  # padding = (top, bottom, left, right)
        top, bottom, left, right = padding
        x = torch.nn.functional.pad(x_nchw, (left, right, top, bottom))
        out = torch.relu(torch.nn.functional.conv2d(x, torch_weight, bias=torch_bias))
        return out[0].permute(1, 2, 0).reshape(-1, CH)  # [H*W, C]

    results = []
    for it in range(n_iter):
        if case == "moves":
            x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(
                device, _hs_config(device, S * S)
            )
            d = ttnn.experimental.quasar.to_memory_config(x, DRAM)
            d = ttnn.experimental.quasar.to_layout(d, ttnn.ROW_MAJOR_LAYOUT)
            got_dram_rm = ttnn.to_torch(d).float().reshape(-1, CH)
            # Step 4 on its own input: a DRAM interleaved TILE tensor back to the height-sharded L1 config.
            t = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
            back = ttnn.experimental.quasar.to_memory_config(t, _hs_config(device, S * S))
            got_back = ttnn.to_torch(back).float().reshape(-1, CH)
            for t_ in (x, d, t, back):
                ttnn.deallocate(t_)
            checks = {
                "to_dram+to_rm": (got_dram_rm, flat.reshape(-1, CH)),
                "dram_to_l1_hs": (got_back, flat.reshape(-1, CH)),
            }
        elif case == "dram_conv":
            x = ttnn.from_torch(
                nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
            )
            out = conv(x, S, (1, 1), ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2))
            got = ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[: S * S, :CH]
            ttnn.deallocate(out)
            checks = {"dram_conv": (got, golden_conv(torch_input, (1, 1, 1, 1)))}
        elif case == "slice_conv":
            # The top slice of a 2-slice split: output rows 0..19 need input rows 0..20, padding top 1, bottom 0.
            rows = S // 2 + 1
            x_slice = nhwc[:, :rows].contiguous()
            x = ttnn.from_torch(
                x_slice, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=L1
            )
            out = conv(x, rows, (1, 0, 1, 1))
            got = ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[: (rows - 1) * S, :CH]
            ttnn.deallocate(out)
            checks = {"slice_conv": (got, golden_conv(torch_input[:, :, :rows], (1, 0, 1, 1)))}
        elif case.startswith("chain"):
            if case == "chain_hostin":
                x = None
                d = ttnn.from_torch(
                    nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
                )
            elif case in ("chain_untilize_only", "chain_untilize_pause"):
                x = ttnn.from_torch(
                    flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
                )
                d = ttnn.reshape(ttnn.experimental.quasar.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, S, S, CH))
                untilize_out = ttnn.to_torch(d).float().reshape(-1, CH)
                if case == "chain_untilize_pause":
                    ttnn.synchronize_device(device)
                    time.sleep(2)
            elif case == "chain_tointerleaved_only":
                x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(
                    device, _hs_config(device, S * S)
                )
                d = ttnn.reshape(ttnn.experimental.quasar.to_memory_config(x, DRAM), (1, S, S, CH))
            else:
                x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(
                    device, _hs_config(device, S * S)
                )
                d = ttnn.experimental.quasar.to_memory_config(x, DRAM)
                d = ttnn.experimental.quasar.to_layout(d, ttnn.ROW_MAJOR_LAYOUT)
                d = ttnn.reshape(d, (1, S, S, CH))
                if case != "chain_keep":
                    ttnn.deallocate(x)
                    x = None
            out = conv(d, S, (1, 1), ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2))
            if case != "chain_noreshard":
                out = ttnn.experimental.quasar.to_memory_config(out, _hs_config(device, S * S))
            got = ttnn.to_torch(out).float().reshape(-1, out.shape[-1])[: S * S, :CH]
            ttnn.deallocate(out)
            if x is not None:
                ttnn.deallocate(x)
            checks = {case: (got, golden_conv(torch_input, (1, 1, 1, 1)))}
            if case in ("chain_untilize_only", "chain_untilize_pause"):
                checks["untilize_out"] = (untilize_out, flat.reshape(-1, CH))
        else:  # padded_slice
            rows = S // 2 + 1
            x = ttnn.from_torch(
                nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
            )
            shard_rows = ((rows * S + 31) // 32) * 32
            out = ttnn.experimental.quasar.padded_slice(
                x, [0, 0, 0, 0], [1, rows, S, CH], memory_config=_hs_config(device, shard_rows)
            )
            got = ttnn.to_torch(out).float().reshape(-1, CH)[: rows * S]
            ttnn.deallocate(out)
            ttnn.deallocate(x)
            checks = {"padded_slice": (got, nhwc[:, :rows].reshape(-1, CH))}

        line = []
        for name, (got, want) in checks.items():
            bad_rows = ((got - want).abs().amax(dim=-1) > 0.1).nonzero().flatten()
            span = f"{int(bad_rows[0])}..{int(bad_rows[-1])}" if len(bad_rows) else "-"
            line.append(f"{name}: pcc={_pcc(got, want):.6f} bad_rows={len(bad_rows)}/{got.shape[0]} ({span})")
            results.append(_pcc(got, want))
        logger.info(f"[STEP] case={case} call={it} " + " | ".join(line))

    assert min(results) >= 0.99, f"a call was wrong: min PCC {min(results):.6f}"
