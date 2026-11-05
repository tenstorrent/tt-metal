# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Interleaved rms_norm on inputs with few tile rows and a wide last dimension: on Blackhole these
# split the width across cores instead of giving each core whole tile rows. Covers the split's core-count edge cases
# (the narrowest split, a last core that overhangs the width, one row of cores, a rectangle of
# them), every weight/bias combination and layout, dtypes, and output memory placement, against a
# float64 golden.

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole
from ttnn.device import is_blackhole

pytestmark = pytest.mark.use_module_device

# The split is selected on Blackhole only; elsewhere these shapes run the interleaved factory, which
# test_rms_norm.py covers. test_rms_norm_width_split_arch_gate runs on every arch.
blackhole_only = run_for_blackhole("the width split runs on Blackhole only")


def _golden(x, w, b, eps):
    x = x.double()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.double()
    if b is not None:
        y = y + b.double()
    return y


def _affine(device, t, layout, dtype=ttnn.bfloat16):
    if t is None:
        return None
    W = t.shape[-1]
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.from_torch(t.reshape(1, 1, W // 32, 32), dtype=dtype, layout=layout, device=device)
    return ttnn.from_torch(t.reshape(1, 1, 1, W), dtype=dtype, layout=layout, device=device)


def _selects_split(
    tx, eps, weight=None, bias=None, residual=None, compute_kernel_config=None, output_memory_config=None
):
    # Whether rms_norm takes the width split for these arguments with the default program config.
    # Pass the compute config and output memory config the call uses; when it passes none, rms_norm
    # fills in its own default, which this mirrors.
    params = ttnn.LayerNormParams()
    params.norm_type = ttnn.LayerNormType.RMSNORM
    params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    params.eps = eps
    params.output_mem_config = output_memory_config if output_memory_config is not None else tx.memory_config()
    params.program_config = ttnn.create_layernorm_program_config(None)
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.rmsnorm_default_compute_config(tx.device().arch())
    params.compute_kernel_config = compute_kernel_config
    inputs = ttnn.LayerNormInputs(tx)
    if weight is not None:
        inputs.weight = weight
    if bias is not None:
        inputs.bias = bias
    if residual is not None:
        inputs.residual_input_tensor = residual
    factory = ttnn.LayerNormDeviceOperation.select_program_factory(params, inputs)
    return isinstance(factory, ttnn.LayerNormWidthSplitProgramFactory)


def _check_rows(golden, out, rel_max):
    # Per row: a row normalised with the wrong statistics hides in the norm of the whole tensor.
    W = golden.shape[-1]
    g = golden.reshape(-1, W)
    o = out.double().reshape(-1, W)
    row_err = (o - g).norm(dim=-1) / g.norm(dim=-1).clamp_min(1e-30)
    worst = row_err.argmax().item()
    assert row_err[worst].item() <= rel_max, f"row {worst} rel err {row_err[worst].item():.5f} (max {rel_max})"


def _check(golden, out, pcc_min, rel_max):
    out = out.double().reshape(golden.shape)
    g = golden.flatten()
    o = out.flatten()
    pcc = torch.corrcoef(torch.stack([g, o]))[0, 1].item()
    rel = ((o - g).norm() / g.norm()).item()
    assert pcc >= pcc_min and rel <= rel_max, f"pcc {pcc:.6f} (min {pcc_min}), rel err {rel:.5f} (max {rel_max})"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),  # 4 width tiles: the narrowest split, two cores
        (1, 1, 4096),  # one tile row on a rectangle of cores
        (1, 1, 32, 384),  # 12 width tiles: one row of cores
        (1, 1, 32, 1120),  # 35 width tiles: the last core holds one tile of its two
        (1, 1, 32, 18944),  # 592 width tiles: too many cores at the minimum block, so blocks widen; the last overhangs
        (1, 1, 64, 4096),  # two tile rows
        (8, 1, 4096),  # eight batches, each padded to one tile row
        (1, 1, 256, 2048),  # eight tile rows
    ],
)
# Weight and bias layouts; when both are given they must share one.
@pytest.mark.parametrize(
    "weight_layout, bias_layout",
    [
        (None, None),
        (ttnn.TILE_LAYOUT, None),
        (ttnn.ROW_MAJOR_LAYOUT, None),
        (None, ttnn.TILE_LAYOUT),
        (None, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@blackhole_only
def test_rms_norm_width_split_affine(device, shape, weight_layout, bias_layout):
    torch.manual_seed(0)
    eps = 1e-6
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((W,))).to(torch.bfloat16) if weight_layout is not None else None
    b = (0.1 * torch.randn((W,))).to(torch.bfloat16) if bias_layout is not None else None
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tw = _affine(device, w, weight_layout)
    tb = _affine(device, b, bias_layout)
    assert _selects_split(tx, eps, weight=tw, bias=tb, compute_kernel_config=ckc)
    out = ttnn.rms_norm(tx, epsilon=eps, weight=tw, bias=tb, compute_kernel_config=ckc)
    _check(_golden(x, w, b, eps), ttnn.to_torch(out), pcc_min=0.9999, rel_max=0.02)


@pytest.mark.parametrize("shape", [(1, 1, 32, 384), (1, 1, 32, 4096), (1, 1, 64, 1120), (1, 1, 256, 2048)])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@blackhole_only
def test_rms_norm_width_split_row_scales(device, shape, fp32_dest_acc_en):
    # Every row gets its own scale, so 1/sqrt(mean(x^2)) differs by row: a row normalised with
    # another row's statistics, or left unnormalised, is off by up to 400x.
    torch.manual_seed(0)
    eps = 1e-6
    rows = shape[-2]
    scales = torch.logspace(-1.3, 1.3, rows).reshape(rows, 1)
    x = (torch.randn(shape) * scales).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((shape[-1],))).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    tw = _affine(device, w, ttnn.ROW_MAJOR_LAYOUT)
    assert _selects_split(tx, eps, weight=tw, compute_kernel_config=ckc)
    out = ttnn.rms_norm(tx, epsilon=eps, weight=tw, compute_kernel_config=ckc)
    golden = _golden(x, w, None, eps)
    _check(golden, ttnn.to_torch(out), pcc_min=0.9999, rel_max=0.02)
    _check_rows(golden, ttnn.to_torch(out), rel_max=0.02)


# (shape, first and last column of the hot block). 4096 wide splits into 64 cores of 64 columns;
# 1120 wide into 18 cores, the last holding one 32-column tile.
@pytest.mark.parametrize(
    "shape, hot",
    [
        ((1, 1, 32, 4096), (0, 64)),  # the root's block
        ((1, 1, 32, 4096), (2048, 2112)),  # a middle core's block
        ((1, 1, 32, 4096), (4032, 4096)),  # the last core's block
        ((1, 1, 64, 1120), (1088, 1120)),  # the overhanging last core, two tile rows
        ((1, 1, 32, 1024), (960, 1024)),  # the last of 16 cores: the gather tile's bottom half is unused
    ],
)
@blackhole_only
def test_rms_norm_width_split_hot_block(device, shape, hot):
    # Nearly all of each row's energy sits in one core's block, so a partial that is lost, stale or
    # written to the wrong gather row changes 1/sqrt(mean(x^2)) by a large factor, not by 1/num_cores.
    # Repeated calls give a race in the gather or broadcast more than one chance to show.
    torch.manual_seed(0)
    eps = 1e-6
    x = 1e-3 * torch.randn(shape)
    x[..., hot[0] : hot[1]] = torch.randn(shape[:-1] + (hot[1] - hot[0],))
    x = x.to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert _selects_split(tx, eps)
    golden = _golden(x, None, None, eps)
    for _ in range(4):
        out = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=eps))
        _check_rows(golden, out, rel_max=0.01)


@pytest.mark.parametrize("shape", [(1, 1, 32, 4096), (1, 1, 64, 1120)])
@pytest.mark.parametrize("eps", [0.25, 4.0])
@blackhole_only
def test_rms_norm_width_split_eps(device, shape, eps):
    # mean(x^2) runs from 1e-3 eps to 1e3 eps down the rows, and one row is all zero, so eps decides
    # some rows outright and is negligible in others; any lane that misses eps is caught on its own row.
    torch.manual_seed(0)
    rows = shape[-2]
    scales = (eps * torch.logspace(-3, 3, rows)).sqrt().reshape(rows, 1)
    x = torch.randn(shape) * scales
    x[..., 5, :] = 0
    x = x.to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((shape[-1],))).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = _affine(device, w, ttnn.ROW_MAJOR_LAYOUT)
    assert _selects_split(tx, eps, weight=tw)
    out = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=eps, weight=tw))
    golden = _golden(x, w, None, eps)
    assert out.reshape(golden.shape)[..., 5, :].abs().max().item() == 0
    _check_rows(golden, out, rel_max=0.02)


@pytest.mark.parametrize(
    "shape, with_residual, splits",
    [
        ((1, 1, 32, 160), False, True),  # 5 width tiles: three cores
        ((1, 1, 64, 256), False, True),  # Kt = 8 = 4 * Mt: the narrowest width that splits two tile rows
        ((1, 1, 64, 224), False, False),  # Kt = 7, one width tile short of that: interleaved factory
        ((1, 1, 96, 1536), False, True),  # three tile rows
        ((1, 1, 40, 4096), False, True),  # a height that ends inside its second tile row
        ((3, 1, 4096), False, True),  # three batches, each padded to one tile row
        ((1, 1, 288, 4096), False, False),  # nine tile rows: interleaved factory
        ((1, 1, 32, 4016), False, False),  # width ends inside a tile: the padding would join the mean
        ((1, 1, 32, 4096), True, False),  # residual add: the split has none
    ],
)
@blackhole_only
def test_rms_norm_width_split_boundaries(device, shape, with_residual, splits):
    # Around the split's engagement limits, with tile padding filled with a sentinel so a padding row
    # leaking into a statistic shows. Shapes left to the interleaved factory only check that the
    # fallback runs and is roughly right; test_rms_norm.py covers its accuracy.
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, dtype=torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((shape[-1],))).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tx = ttnn.fill_implicit_tile_padding(tx, -42)
    tw = _affine(device, w, ttnn.TILE_LAYOUT)
    r = torch.randn(shape, dtype=torch.bfloat16) if with_residual else None
    tr = ttnn.from_torch(r, layout=ttnn.TILE_LAYOUT, device=device) if with_residual else None
    assert _selects_split(tx, eps, weight=tw, residual=tr) == splits
    out = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=eps, weight=tw, residual_input_tensor=tr))
    golden = _golden(x + r if with_residual else x, w, None, eps)
    if splits:
        _check(golden, out, pcc_min=0.9999, rel_max=0.02)
        _check_rows(golden, out, rel_max=0.02)
    else:
        _check(golden, out, pcc_min=0.9999, rel_max=0.03)


@pytest.mark.parametrize(
    "layout, dtype, rel_max",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.float32, 0.02),
        (ttnn.TILE_LAYOUT, ttnn.float32, 0.02),
        (ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 0.03),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
@blackhole_only
def test_rms_norm_width_split_affine_dtypes(device, layout, dtype, rel_max, with_bias):
    torch.manual_seed(0)
    eps = 1e-6
    shape = (1, 1, 64, 1120)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    w = 1.0 + 0.1 * torch.randn((W,))
    b = 0.1 * torch.randn((W,)) if with_bias else None
    tw = _affine(device, w, layout, dtype)
    tb = _affine(device, b, layout, dtype)
    # The golden uses the weight and bias as the device holds them.
    w_dev = ttnn.to_torch(tw).float().reshape(-1)[:W]
    b_dev = ttnn.to_torch(tb).float().reshape(-1)[:W] if with_bias else None
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert _selects_split(tx, eps, weight=tw, bias=tb)
    out = ttnn.rms_norm(tx, epsilon=eps, weight=tw, bias=tb)
    _check(_golden(x, w_dev, b_dev, eps), ttnn.to_torch(out), pcc_min=0.9999, rel_max=rel_max)


@pytest.mark.parametrize("shape", [(1, 1, 4096), (1, 1, 128, 1024)])
@pytest.mark.parametrize(
    "dtype, pcc_min, rel_max",
    [
        (ttnn.bfloat16, 0.9999, 0.02),
        (ttnn.float32, 0.9999, 0.02),
        (ttnn.bfloat8_b, 0.999, 0.05),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@blackhole_only
def test_rms_norm_width_split_dtypes(device, shape, dtype, pcc_min, rel_max, fp32_dest_acc_en):
    torch.manual_seed(0)
    eps = 1e-5
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32)
    w = 1.0 + 0.1 * torch.randn((W,))
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_device = ttnn.to_torch(tx).float()
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    tw = _affine(device, w.to(torch.bfloat16), ttnn.TILE_LAYOUT)
    assert _selects_split(tx, eps, weight=tw, compute_kernel_config=ckc)
    out = ttnn.rms_norm(tx, epsilon=eps, weight=tw, compute_kernel_config=ckc)
    _check(_golden(x_device, w.to(torch.bfloat16), None, eps), ttnn.to_torch(out), pcc_min, rel_max)


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@blackhole_only
def test_rms_norm_width_split_memory(device, memory_config, output_memory_config):
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn((1, 1, 4096), dtype=torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((4096,))).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    tw = _affine(device, w, ttnn.ROW_MAJOR_LAYOUT)
    assert _selects_split(tx, eps, weight=tw, output_memory_config=output_memory_config)
    out = ttnn.rms_norm(tx, epsilon=eps, weight=tw, memory_config=output_memory_config)
    expected_memory_config = output_memory_config if output_memory_config is not None else memory_config
    assert out.memory_config().buffer_type == expected_memory_config.buffer_type
    _check(_golden(x, w, None, eps), ttnn.to_torch(out), pcc_min=0.9999, rel_max=0.02)


@blackhole_only
def test_rms_norm_width_split_program_cache(device):
    # Later calls with new tensors of the same spec reuse the cached program; it must pick up the new
    # buffer addresses. The epsilon is unique to this test, so its program is not cached beforehand.
    eps = 3e-6
    entries_before = device.num_program_cache_entries()
    # Kept alive so each iteration's tensors land at new addresses.
    spacers = []
    for seed in range(3):
        spacers.append(ttnn.from_torch(torch.zeros((1, 1, 32, 32 * (seed + 1))), device=device))
        torch.manual_seed(seed)
        x = torch.randn((1, 1, 4096), dtype=torch.bfloat16) * (seed + 1)
        w = (1.0 + 0.1 * torch.randn((4096,))).to(torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tw = _affine(device, w, ttnn.ROW_MAJOR_LAYOUT)
        assert _selects_split(tx, eps, weight=tw)
        out = ttnn.rms_norm(tx, epsilon=eps, weight=tw)
        _check(_golden(x, w, None, eps), ttnn.to_torch(out), pcc_min=0.9999, rel_max=0.02)
    assert device.num_program_cache_entries() == entries_before + 1


@blackhole_only
def test_rms_norm_width_split_largest_block(device):
    # The most a core holds: eight tile rows of four width tiles (Kt = 400 needs 200 and 134 cores
    # at blocks 2 and 3, too many for the grid), in FP32 with weight, bias, FP32 accumulation and
    # L1 tensors, so every block-sized buffer is at its largest.
    torch.manual_seed(0)
    eps = 1e-6
    shape = (1, 1, 256, 12800)
    x = torch.randn(shape)
    w = 1.0 + 0.1 * torch.randn((shape[-1],))
    b = 0.1 * torch.randn((shape[-1],))
    mem = ttnn.L1_MEMORY_CONFIG
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tw = _affine(device, w, ttnn.TILE_LAYOUT, ttnn.float32)
    tb = _affine(device, b, ttnn.TILE_LAYOUT, ttnn.float32)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    assert _selects_split(tx, eps, weight=tw, bias=tb, compute_kernel_config=ckc)
    out = ttnn.to_torch(
        ttnn.rms_norm(tx, epsilon=eps, weight=tw, bias=tb, memory_config=mem, compute_kernel_config=ckc)
    )
    golden = _golden(x, w, b, eps)
    _check(golden, out, pcc_min=0.9999, rel_max=0.02)
    _check_rows(golden, out, rel_max=0.02)


@pytest.mark.parametrize("W", [3072, 5120])
@blackhole_only
def test_rms_norm_width_split_mean_precision(device, W):
    # 1/W is not a short binary fraction at these widths; taking it at bfloat16 precision anywhere
    # biases every output by about 0.2%, far above what FP32 end to end leaves.
    torch.manual_seed(0)
    eps = 1e-6
    shape = (1, 1, 32, W)
    x = torch.randn(shape)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    assert _selects_split(tx, eps, compute_kernel_config=ckc)
    out = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=eps, compute_kernel_config=ckc))
    golden = _golden(x, None, None, eps)
    _check_rows(golden, out, rel_max=1e-3)


@pytest.mark.parametrize(
    "shape, dtype, affine_dtype, with_bias, config",
    [
        ((1, 1, 32, 7040), ttnn.bfloat16, ttnn.bfloat16, False, "hifi2_fp32"),  # full 11x10 grid
        ((1, 1, 32, 7040), ttnn.bfloat16, ttnn.bfloat16, False, "default"),
        ((1, 1, 32, 7072), ttnn.bfloat16, ttnn.bfloat16, False, "hifi2_fp32"),  # last block 1 of 4 tiles
        ((1, 1, 32, 32768), ttnn.bfloat16, ttnn.bfloat16, False, "hifi2_fp32"),
        ((1, 1, 32, 112640), ttnn.bfloat16, ttnn.bfloat16, False, "hifi2_fp32"),  # largest block width
        ((1, 1, 32, 112640), ttnn.float32, ttnn.float32, True, "hifi2_fp32"),
        ((1, 1, 32, 4096), ttnn.float32, ttnn.float32, False, "hifi2_fp32"),
        ((1, 1, 256, 14080), ttnn.float32, ttnn.float32, True, "hifi2_fp32"),
    ],
)
@blackhole_only
def test_rms_norm_width_split_widths_beyond_one_core_l1(device, shape, dtype, affine_dtype, with_bias, config):
    # The interleaved factory keeps a whole tile row per core, so these widths exceed one core's L1
    # there; they run only because the split takes them. A threshold change that hands them back
    # fails here rather than as an L1 allocation error in a model.
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape)
    w = torch.rand((shape[-1],)) + 0.5
    b = 0.1 * torch.randn((shape[-1],)) if with_bias else None
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tw = _affine(device, w, ttnn.ROW_MAJOR_LAYOUT, affine_dtype)
    tb = _affine(device, b, ttnn.ROW_MAJOR_LAYOUT, affine_dtype)
    if config == "default":
        ckc = ttnn.rmsnorm_default_compute_config(device.arch())
    else:
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    assert _selects_split(tx, eps, weight=tw, bias=tb, compute_kernel_config=ckc)
    out = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=eps, weight=tw, bias=tb, compute_kernel_config=ckc))
    golden = _golden(x, w, b, eps)
    _check(golden, out, pcc_min=0.9999, rel_max=0.02)
    _check_rows(golden, out, rel_max=0.02)


@pytest.mark.parametrize("shape", [(1, 1, 4096), (1, 1, 32, 128), (1, 1, 256, 2048)])
def test_rms_norm_width_split_arch_gate(device, shape):
    # Selection only, on every arch: shapes the split takes on Blackhole stay on the interleaved
    # factory elsewhere, where the split has not been measured.
    eps = 1e-6
    tx = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    assert _selects_split(tx, eps) == is_blackhole(device)
