# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests op_design.md requires beyond the acceptance test (regime `col_split_stream`, thread plan (1, 1, 1)).

- the thread plan, read off the program artifacts, and every engaged cluster's channel range covered;
- more than one DEST chunk per cluster (grid narrowed), and several blocks per cluster (block cap turned);
- the tile-padding rows of the last position tile row masked whatever they hold (rule R1);
- the op in one device session after another program and before one of tt-metal's own;
- the model's call: width-sharded on two cores, LoFi with approximation, a device compute config (R3, R4);
- refusals by name (window, layout) and the malformed calls.
"""

import pytest
import torch

import ttnn
from ttnn.operations.avg_pool2d import avg_pool2d
from ttnn.operations.avg_pool2d import avg_pool2d_program_artifacts as artifacts

TILE = 32


def _config(fp32=True, fidelity=ttnn.MathFidelity.HiFi4, approx=False):
    return ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=approx)


def _tile(device, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _mean(x, positions):
    c = x.shape[-1]
    return x[0, 0, :positions].double().mean(0).reshape(1, 1, 1, c)


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_artifacts(device, x_dev, positions, channels, config, **kwargs):
    """The op's program on a tensor whose rows beyond `positions` are padding (or a narrowed grid)."""
    out_spec = ttnn.TensorSpec(ttnn.Shape([1, 1, 1, channels]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                               ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM))
    out = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, run_args, indices = artifacts.create_program_artifacts(
        x_dev, out, positions=positions, compute_config=config, **kwargs)
    return ttnn.to_torch(ttnn.generic_op([x_dev, out], spec, run_args, indices)).float()


def test_thread_plan_and_channel_coverage(device):
    """The plan is (1, 1, 1) on every kernel, and every engaged cluster's channel range comes back.

    The input is ramped by channel (constant over positions), so each output channel is its own ramp value: a
    cluster that wrote nothing, or wrote another cluster's columns, shows as a wrong channel range."""
    H, W, C = 7, 7, 2048
    grid = device.compute_with_storage_grid_size()
    _, work = artifacts.split_channel_tiles(C // TILE, grid)
    assert len(work) == min(C // TILE, grid.x * grid.y)

    ramp = (torch.arange(C) % 61 + 1).to(torch.bfloat16)  # small integers: exact in bfloat16
    x = ramp.reshape(1, 1, 1, C).expand(1, 1, H * W, C).contiguous()
    x_dev = _tile(device, x)

    spec, _, _ = artifacts.create_program_artifacts(
        x_dev, x_dev, positions=H * W, compute_config=_config())  # artifacts only, never dispatched
    threads = {k.unique_id: k.num_threads for k in spec.kernels}
    assert (threads["reader"], threads["compute"], threads["writer"]) == artifacts.THREAD_PLAN == (1, 1, 1)

    got = ttnn.to_torch(avg_pool2d(x_dev, 1, H, W, C, (H, W), (1, 1), (0, 0),
                                   output_layout=ttnn.TILE_LAYOUT)).float()
    for core, (ct_start, count, _) in work.items():
        cols = slice(ct_start * TILE, (ct_start + count) * TILE)
        assert torch.allclose(got[0, 0, 0, cols], ramp[cols].float(), rtol=0.01), (core, ct_start, count)


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "fp16dest"])
@pytest.mark.parametrize("grid,cap", [((2, 1), None), ((1, 1), None), ((3, 1), 3)],
                         ids=["2cl_1block", "1cl_1block", "3cl_cap3"])
def test_several_chunks_and_blocks_per_cluster(device, fp32, grid, cap):
    """A narrowed grid gives each cluster more channel tiles than one DEST chunk (4 or 8): the reader's order
    (chunk -> position row -> column) must match the helper's. A block cap of 3 gives several blocks per cluster,
    the last one ragged, and chunks narrower than DEST."""
    H, W, C = 7, 7, 2048
    torch.manual_seed(1)
    x = torch.randn(1, 1, H * W, C, dtype=torch.bfloat16)
    got = _run_artifacts(device, _tile(device, x), H * W, C, _config(fp32),
                         grid_size=ttnn.CoreCoord(*grid), block_ct_tiles_cap=cap)
    ref = _mean(x, H * W)
    assert _pcc(got, ref) > 0.995
    assert (got - ref).abs().max().item() < 0.02


@pytest.mark.parametrize("poison", [1000.0, float("inf"), float("nan")], ids=["big", "inf", "nan"])
@pytest.mark.parametrize("positions,C,grid,cap", [
    (49, 64, None, None), (15, 64, None, None), (196, 64, None, None), (33, 64, None, None),
    (49, 2048, (3, 1), 3),  # several chunks and blocks per cluster, batches cut at the ring's end
], ids=["49", "15", "196", "33", "49_c2048_3cl_cap3"])
def test_padding_rows_do_not_enter_the_mean(device, positions, C, grid, cap, poison):
    """Rule R1: rows beyond H*W in the last position tile row are padding, whatever they hold. The tensor here
    has its padding rows poisoned (large, inf or nan) and is reduced over `positions` rows only."""
    rows = -(-positions // TILE) * TILE
    torch.manual_seed(2)
    x = torch.rand(1, 1, rows, C, dtype=torch.bfloat16)
    x[0, 0, positions:] = poison
    kwargs = {} if grid is None else {"grid_size": ttnn.CoreCoord(*grid), "block_ct_tiles_cap": cap}
    got = _run_artifacts(device, _tile(device, x), positions, C, _config(), **kwargs)
    ref = _mean(x, positions)
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 0.01, (got - ref).abs().max().item()
    # P1: a mean stays inside the range of what it averages.
    valid = x[0, 0, :positions].float()
    assert bool(((got[0, 0, 0] >= valid.min(0).values - 0.01) & (got[0, 0, 0] <= valid.max(0).values + 0.01)).all())


def test_one_device_session_before_and_after_other_programs(device):
    """The op after another program and before a tt-metal program (to_layout to row-major, an untagged
    explicit-sync untilize), in one device session: the harts' transaction tags are handed back."""
    torch.manual_seed(3)
    other = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    back = ttnn.to_torch(ttnn.to_layout(_tile(device, other), ttnn.ROW_MAJOR_LAYOUT)).float()
    assert torch.equal(back, other.float())

    for seed, (H, W, C) in enumerate([(7, 7, 512), (4, 8, 64), (14, 14, 32)]):
        torch.manual_seed(seed)
        x = torch.randn(1, 1, H * W, C, dtype=torch.bfloat16)
        y = avg_pool2d(_tile(device, x), 1, H, W, C, [H, W], [1, 1], [0, 0, 0, 0], output_layout=ttnn.TILE_LAYOUT)
        rm = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT))).float()
        ref = _mean(x, H * W).float()
        assert rm.shape == ref.shape
        assert (rm - ref).abs().max().item() < 0.01


def _width_sharded_two_cores(device, x):
    rows = -(-x.shape[-2] // TILE) * TILE
    mc = ttnn.create_sharded_memory_config(
        shape=(rows, x.shape[-1] // 2), core_grid=ttnn.num_cores_to_corerangeset(2, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    return _tile(device, x, mc)


def _device_config():
    """tt-metal's device compute kernel config as the model builds it (rule R4)."""
    if not hasattr(ttnn, "WormholeComputeKernelConfig"):
        pytest.skip("no device compute kernel config in this ttnn")
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
                                            fp32_dest_acc_en=False, packer_l1_acc=False)


@pytest.mark.parametrize("C", [64, 2048])
def test_model_call_width_sharded_device_config(device, C):
    """The model's call: [1, 1, 49, C] width-sharded on two cores in L1, every argument by keyword as lists, a
    device compute config at LoFi with approximation (R3, R4); the result interleaved in L1 (R9)."""
    torch.manual_seed(4)
    x = torch.rand(1, 1, 49, C, dtype=torch.bfloat16)
    y = avg_pool2d(input_tensor=_width_sharded_two_cores(device, x), batch_size=1, input_h=7, input_w=7, channels=C,
                   kernel_size=[7, 7], stride=[1, 1], padding=[0, 0, 0, 0], output_layout=ttnn.TILE_LAYOUT,
                   dtype=ttnn.bfloat16, compute_kernel_config=_device_config())
    mc = y.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got, ref = ttnn.to_torch(y).float(), _mean(x, 49)
    assert _pcc(got, ref) >= 0.99
    valid = x[0, 0].float()
    assert bool(((got[0, 0, 0] >= valid.min(0).values - 0.01) & (got[0, 0, 0] <= valid.max(0).values + 0.01)).all())


def test_refusals_and_rejections(device):
    x = _tile(device, torch.randn(1, 1, 49, 64, dtype=torch.bfloat16))
    call = dict(batch_size=1, input_h=7, input_w=7, channels=64, kernel_size=(7, 7), stride=(1, 1),
                padding=(0, 0), output_layout=ttnn.TILE_LAYOUT)
    with pytest.raises(NotImplementedError, match="window"):
        avg_pool2d(x, **{**call, "kernel_size": (3, 3)})
    with pytest.raises(NotImplementedError, match="output_layout"):
        avg_pool2d(x, **{**call, "output_layout": ttnn.ROW_MAJOR_LAYOUT})
    with pytest.raises(ValueError):  # X1: 49 rows read as 8 x 8
        avg_pool2d(x, **{**call, "input_h": 8, "input_w": 8, "kernel_size": (8, 8)})
    with pytest.raises(ValueError):  # X2: three padding values
        avg_pool2d(x, **{**call, "padding": (0, 0, 0)})
    with pytest.raises(ValueError):  # X3: stride 0
        avg_pool2d(x, **{**call, "stride": (0, 0)})
