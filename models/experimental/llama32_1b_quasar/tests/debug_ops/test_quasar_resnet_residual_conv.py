# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone residual convs from the Quasar ResNet50 e2e top-1 drift.

On the Horizon 2x3 emulator (2026-10-13) ``test_resnet50_e2e[pretrained]`` finished with
finite logits at PCC 0.995 but top-1 783 vs golden 610. The same test on emu-quasar-2x3
(2026-08-08, branch bbradel-52085_resnet_qsr) stayed above PCC 0.998 through every block
and matched top-1 610. Stem and max-pool agreed on both runs. The extra error opened at
``layer1_module1`` and then grew a few thousandths of PCC per residual block.

Each case here is one of those convs, run alone against a torch golden, with the e2e
config: bfloat16, LoFi, ``packer_l1_acc``, height-sharded, and
``TT_METAL_QSR_CONV_SPLIT_PROGRAM=1``. ``l1_conv1`` is the 1x1 whose first values still
matched across the two e2e logs. ``l1_conv2`` is the 3x3 where they first split.
``l1_conv3`` and ``l1_downsample`` are the other two convs in that block. The later
``l2``/``l3``/``l4`` cases are the stride-1 3x3 conv2 at each stage, where the e2e gap
kept widening.

A single conv at PCC >= 0.999 matches the August Quasar run. A lower PCC on one of these
cases is the per-op tax; if every case stays at 0.999, the e2e gap is accumulation across
the stack rather than one broken conv.

Run (2x3 emulator, slow dispatch):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \\
    pytest -q models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_resnet_residual_conv.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# name, in_ch, out_ch, spatial, kernel, stride, padding, relu, act_block_h_override,
# reallocate_halo_output, deallocate_activation
# Shapes are ResNet50 batch-1, module stride 1. layer1_module1 is 56x56, 64 in.
RESIDUAL_CONVS = [
    ("l1_conv1", 64, 64, 56, 1, 1, 0, True, None, None, False),
    ("l1_conv2", 64, 64, 56, 3, 1, 1, True, 32, False, True),
    ("l1_conv3", 64, 256, 56, 1, 1, 0, False, None, None, True),
    ("l1_downsample", 64, 256, 56, 1, 1, 0, False, 32, True, True),
    ("l2_conv2", 128, 128, 28, 3, 1, 1, True, 32, False, True),
    ("l3_conv2", 256, 256, 14, 3, 1, 1, True, 32, False, True),
    ("l4_conv2", 512, 512, 7, 3, 1, 1, True, 32, False, True),
]

# August Quasar e2e held 0.998 after the whole residual stack, so one conv should clear this.
PCC = 0.999


def _pcc(dev, gold):
    d = dev.detach().double().reshape(-1)
    g = gold.detach().double().reshape(-1)
    n = min(d.numel(), g.numel())
    d = d[:n]
    g = g[:n]
    good = torch.isfinite(d) & torch.isfinite(g)
    finite_frac = float(good.double().mean()) if good.numel() else 0.0
    d = d[good]
    g = g[good]
    if d.numel() == 0:
        return float("nan"), finite_frac
    d = d - d.mean()
    g = g - g.mean()
    denom = d.norm() * g.norm()
    if denom == 0:
        return (1.0 if d.norm() == 0 and g.norm() == 0 else 0.0), finite_frac
    return float((d @ g) / denom), finite_frac


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "name, in_ch, out_ch, spatial, kernel, stride, padding, relu, act_block_h, reallocate_halo, deallocate_act",
    RESIDUAL_CONVS,
    ids=[c[0] for c in RESIDUAL_CONVS],
)
def test_quasar_resnet_residual_conv(
    mesh_device,
    name,
    in_ch,
    out_ch,
    spatial,
    kernel,
    stride,
    padding,
    relu,
    act_block_h,
    reallocate_halo,
    deallocate_act,
    monkeypatch,
):
    device = mesh_device
    torch.manual_seed(0)
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")

    batch = 1
    kernel_size = (kernel, kernel)
    stride_hw = (stride, stride)
    padding_hw = (padding, padding)

    torch_input = torch.randn((batch, in_ch, spatial, spatial), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_ch, in_ch, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_ch,), dtype=torch.bfloat16).float()
    golden = torch.nn.functional.conv2d(
        torch_input, torch_weight, bias=torch_bias, stride=stride_hw, padding=padding_hw
    )
    if relu:
        golden = torch.relu(golden)

    nhw = batch * spatial * spatial
    flat = torch.permute(torch_input, (0, 2, 3, 1)).reshape(1, 1, nhw, in_ch).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_ch),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_ch), dtype=ttnn.bfloat16)

    conv_config_kwargs = {
        "weights_dtype": ttnn.bfloat16,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "reshard_if_not_optimal": False,
        "deallocate_activation": deallocate_act,
    }
    if relu:
        conv_config_kwargs["activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    if act_block_h is not None:
        conv_config_kwargs["act_block_h_override"] = act_block_h
    if reallocate_halo is not None:
        conv_config_kwargs["reallocate_halo_output"] = reallocate_halo

    out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_ch,
        out_channels=out_ch,
        batch_size=batch,
        input_height=spatial,
        input_width=spatial,
        kernel_size=kernel_size,
        stride=stride_hw,
        padding=padding_hw,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=ttnn.Conv2dConfig(**conv_config_kwargs),
        compute_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            packer_l1_acc=True,
        ),
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    got = ttnn.to_torch(ttnn.from_device(out)).float()
    got = got.reshape(batch, oh, ow, got.shape[-1])[:, :, :, :out_ch]
    got = torch.permute(got, (0, 3, 1, 2))
    pcc, finite_frac = _pcc(got, golden)
    flat_got = got.flatten()
    logger.info(
        f"[residual-conv] {name} {in_ch}->{out_ch} k={kernel} {spatial}x{spatial} "
        f"pcc={pcc:.6f} finite={finite_frac:.6f} "
        f"mean={flat_got.mean().item():.6f} std={flat_got.std().item():.6f} "
        f"first8={[round(v, 4) for v in flat_got[:8].tolist()]}"
    )
    assert finite_frac > 0.999, f"{name} non-finite (finite_frac={finite_frac})"
    assert pcc >= PCC, f"{name} PCC {pcc:.6f} < {PCC}"
