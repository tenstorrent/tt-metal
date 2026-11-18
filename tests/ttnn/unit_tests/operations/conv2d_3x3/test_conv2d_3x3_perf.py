# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for conv2d_3x3: one dispatch per shape, meant for `run_safe_pytest.sh --profile`.
Correctness is still checked (PCC), so a profile of a wrong kernel is not mistaken for a number.

test_conv2d_3x3_perf: DRAM-interleaved inputs on the whole grid (regimes R0 / R3).
test_conv2d_3x3_height_sharded_perf: the 14 Panoptic HEIGHT-sharded layers at their pinned shard
geometry, in their real config (bf8_b weights, bias, ReLU; regime R2).
test_conv2d_3x3_block_sharded_perf: the 9 Panoptic BLOCK-sharded layers (K split + row reduce-scatter,
regime R4) at their pinned grids, TILE bf8_b x bf8_b, fp32 DEST."""

import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

# ((H, W, C_in), C_out, padding)
PERF_SHAPES = [
    ((32, 64, 128), 128, 1),
    ((16, 16, 64), 128, 1),
    ((8, 16, 512), 256, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
]


def _shape_id(case):
    (H, W, c_in), c_out, padding = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{padding}"


LAYOUTS = [
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="rm_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="tile_bf16"),
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="tile_bf8b"),
]


@pytest.mark.parametrize("dtype,layout", LAYOUTS)
@pytest.mark.parametrize("case", PERF_SHAPES, ids=[_shape_id(c) for c in PERF_SHAPES])
def test_conv2d_3x3_perf(case, dtype, layout, device):
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding = case
    kwargs = dict(
        stride=1,
        padding=padding,
        dilation=1,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=padding)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    assert_with_pcc(ref, ttnn.to_torch(tt_output).float(), 0.995 if dtype == ttnn.bfloat16 else 0.99)


def _row_major_cores(num_cores, grid_x):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


# (layer, ((H, W, C_in), C_out, padding, stride), dtype, layout, shard_shape, num_cores on an 11-wide grid)
HEIGHT_SHAPES = [
    ("stem_conv1", ((512, 1024, 8), 64, 1, 2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, [4767, 8], 110),
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 64], 103),
    ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [384, 128], 86),
    ("res3_conv2", ((64, 128, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 128], 86),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 256], 86),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
]


@pytest.mark.parametrize(
    "layer,case,dtype,layout,shard_shape,num_cores", HEIGHT_SHAPES, ids=[c[0] for c in HEIGHT_SHAPES]
)
def test_conv2d_3x3_height_sharded_perf(layer, case, dtype, layout, shard_shape, num_cores, device):
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride = case
    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y or grid.x < 11:
        pytest.skip(f"{layer}: pinned 11-wide {num_cores}-core shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_row_major_cores(num_cores, 11), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=1,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    tt_output = conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        activation="relu",
    )
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(
        torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), stride=stride, padding=padding)
    )
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    assert_with_pcc(ref, ttnn.to_torch(tt_output).float(), 0.99)


# Ablation profiling stubs kernel payloads on purpose, so the output is wrong by construction: with this set,
# the BLOCK harness measures without checking PCC.
ABLATION = os.environ.get("CONV2D_3X3_PERF_ABLATION") == "1"

# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity)
BLOCK_SHAPES = [
    ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, ttnn.MathFidelity.HiFi4, 2),
    ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, ttnn.MathFidelity.LoFi, 1),
    ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, ttnn.MathFidelity.HiFi4, 1),
]


@pytest.mark.parametrize(
    "layer,case,shard_shape,grid_xy,bias_relu,fidelity,stride", BLOCK_SHAPES, ids=[c[0] for c in BLOCK_SHAPES]
)
def test_conv2d_3x3_block_sharded_perf(layer, case, shard_shape, grid_xy, bias_relu, fidelity, stride, device):
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, dilation = case
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid_xy
    if gx > grid.x or gy > grid.y:
        pytest.skip(f"{layer}: pinned {gx}x{gy} shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if bias_relu else None
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs) if bias_relu else None
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    tt_output = conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if bias_relu else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if bias_relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    if not ABLATION:
        assert_with_pcc(ref, ttnn.to_torch(tt_output).float(), 0.99)


# The perf-flagged Panoptic LOOSE_CASES (feature_spec "[perf]" rows), at their exact config: shard pin, dtypes,
# fidelity, bias and ReLU. One dispatch per case, so the profiler's DEVICE KERNEL DURATION row is the op.
def _perf_focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    ids = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cases]
    return cases, ids


_FOCUS_CASES, _FOCUS_IDS = _perf_focus_cases()


@pytest.mark.parametrize("case", _FOCUS_CASES, ids=_FOCUS_IDS)
def test_conv2d_3x3_perf_focus(case, device):
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    tt_output = conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    if ABLATION:
        return
    ref = helpers.pytorch_conv2d_3x3(
        torch_input,
        torch_weight,
        input_height=H,
        input_width=W,
        bias=torch_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    )
    pcc, _ = extras.get("tolerance", (0.99, None))
    assert_with_pcc(ref.float(), ttnn.to_torch(tt_output).float().reshape(ref.shape), pcc)
