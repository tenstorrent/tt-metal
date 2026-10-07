# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58722, #58725): binary_ng FPU ops for the device profiler (eb_prof_plugin repeats each test
function): sharded no-broadcast add, subtract and multiply (8 tiles per DEST section, 4 with fp32 DEST: the block unpack),
interleaved ones (one tile per section), ops with a fused post activation (the per-chunk init), and the KDA kernels that call
add_block / sub_block / mul_block. Lives outside the tree; the device is opened here."""
import os

import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)


OPS = {
    "add": (lambda a, b, **kw: ttnn.add(a, b, **kw), torch.add),
    "sub": (lambda a, b, **kw: ttnn.subtract(a, b, **kw), torch.sub),
    "mul": (lambda a, b, **kw: ttnn.multiply(a, b, fast_and_approximate_mode=True, **kw), torch.mul),
}
DT = {"bf16": (ttnn.bfloat16, torch.bfloat16), "bfp8": (ttnn.bfloat8_b, torch.bfloat16), "fp32": (ttnn.float32, torch.float32)}


def _check(out, ref, tol=0.02):
    got = ttnn.to_torch(out).float()
    if os.environ.get("EB_R3_PROBE_INIT"):
        return
    assert torch.allclose(got, ref.float(), rtol=tol, atol=tol), (got - ref.float()).abs().max()


def _sharded(shape, strategy, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.create_sharded_memory_config(shape, core_grid=grid, strategy=strategy, orientation=orientation)


# shape, strategy, grid: the height-sharded case of test_add.py (8 cores), a block-sharded SDXL UNet-sized tensor (8x8),
# a width-sharded decode-sized residual (32 cores) and a height-sharded fp32 batch case (test_binary_ng_sharded_fp32_batch)
SHARDED = {
    "hs_1024x1024_8c": ((1, 1, 1024, 1024), ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=4)),
    "bs_4096x1280_8x8": ((1, 1, 4096, 1280), ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=8, x=8)),
    "ws_32x4096_32c": ((1, 1, 32, 4096), ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=4, x=8)),
    "hs_2048x512_16c": ((1, 1, 2048, 512), ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=8)),
}


@pytest.mark.parametrize("case", list(SHARDED))
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("dt", ["bf16", "bfp8", "fp32"])
def test_bng_sharded(device, case, op, dt):
    shape, strategy, grid = SHARDED[case]
    if dt == "fp32" and case == "bs_4096x1280_8x8":
        pytest.skip("fp32 block-sharded tensor does not fit L1 with the CBs")
    ttdt, tdt = DT[dt]
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=tdt) + 0.5
    b = torch.rand(shape, dtype=tdt) + 0.5
    mem = _sharded(shape, strategy, grid)
    ta = ttnn.from_torch(a, dtype=ttdt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tb = ttnn.from_torch(b, dtype=ttdt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    f, g = OPS[op]
    _check(f(ta, tb, memory_config=mem), g(a.float(), b.float()), 0.05 if dt == "bfp8" else 0.02)


@pytest.mark.parametrize("shape", [(1, 1, 1024, 1024), (1, 1, 32, 4096), (1, 1, 4096, 4096)], ids=["1024x1024", "32x4096", "4096x4096"])
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("mem", ["dram", "l1"])
def test_bng_interleaved(device, shape, op, mem):
    if mem == "l1" and shape[-2] * shape[-1] > 1024 * 1024:
        pytest.skip("three 32 MB tensors do not fit L1")
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    b = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    f, g = OPS[op]
    _check(f(ta, tb, memory_config=mc), g(a.float(), b.float()))


# fused post activations: qwen36 GDN dt bias (add + softplus), SDXL UNet time embedding (add + silu), minimax VAE (add + silu),
# and the same on an L1 interleaved tensor where compute can be the limit
ACT = {
    "softplus": (lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)], lambda x: torch.nn.functional.softplus(x)),
    "silu": (lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], torch.nn.functional.silu),
    "gelu": (lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)], torch.nn.functional.gelu),
}


@pytest.mark.parametrize("shape_a, shape_b", [((1, 1, 1024, 1024), (1, 1, 1024, 1024)), ((1, 1, 2048, 64), (1, 1, 2048, 64)), ((1, 1, 32, 1280), (1, 1, 32, 1280))], ids=["1024x1024", "2048x64", "32x1280"])
@pytest.mark.parametrize("act", list(ACT))
@pytest.mark.parametrize("mem", ["dram", "l1"])
def test_bng_post_activation(device, shape_a, shape_b, act, mem):
    torch.manual_seed(0)
    a = torch.rand(shape_a, dtype=torch.bfloat16) - 0.5
    b = torch.rand(shape_b, dtype=torch.bfloat16) - 0.5
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    acts, ref = ACT[act]
    _check(ttnn.add(ta, tb, activations=acts(), memory_config=mc), ref((a.float() + b.float())), 0.03)


@pytest.mark.parametrize("op", ["add", "mul"])
@pytest.mark.parametrize("act", ["silu", "gelu"])
def test_bng_post_activation_sharded(device, op, act):
    shape, strategy, grid = SHARDED["hs_1024x1024_8c"]
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    b = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    mem = _sharded(shape, strategy, grid)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    acts, ref = ACT[act]
    f, g = OPS[op]
    _check(f(ta, tb, activations=acts(), memory_config=mem), ref(g(a.float(), b.float())), 0.03)


# the KDA kernels that call add_block / sub_block / mul_block, at their production cases
def test_kda_prepare_chunk_recurrence(device):
    from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_prepare_chunk_recurrence as m

    m.test_prepare_chunk_recurrence_contract_accuracy_and_determinism(device, m._PRODUCTION_CASE)


def test_kda_recurrent_chunk_scan(device):
    from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_recurrent_chunk_scan as m
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

    m.test_recurrent_chunk_scan_is_device_deterministic(make_actual_start(device, 0), device)


# subtile broadcasts (binary_ng's kernels_ng: unary_bcast into a CB, then the binary op, init per tile for a row broadcast):
# a bias row (ROW_B), a row on the left (ROW_A), a column (COL_B), a scalar tile (SCALAR_B), an attention-mask-like row over
# 8 heads, and fused post activations (bias + gelu, column + silu)
BCAST = {
    "row_b": ((1, 1, 1024, 1024), (1, 1, 1, 1024)),
    "row_a": ((1, 1, 1, 1024), (1, 1, 1024, 1024)),
    "col_b": ((1, 1, 1024, 1024), (1, 1, 1024, 1)),
    "scalar_b": ((1, 1, 1024, 1024), (1, 1, 1, 1)),
    "mask_row_b": ((1, 8, 512, 512), (1, 1, 1, 512)),
}


@pytest.mark.parametrize(
    "case, op, act, mem",
    [
        ("row_b", "add", None, "dram"),
        ("row_b", "mul", None, "dram"),
        ("row_b", "sub", None, "dram"),
        ("row_b", "add", None, "l1"),
        ("row_a", "add", None, "dram"),
        ("col_b", "add", None, "dram"),
        ("col_b", "mul", None, "l1"),
        ("scalar_b", "mul", None, "dram"),
        ("mask_row_b", "add", None, "dram"),
        ("row_b", "add", "gelu", "dram"),
        ("row_b", "add", "gelu", "l1"),
        ("col_b", "add", "silu", "dram"),
    ],
    ids=lambda v: str(v),
)
def test_bng_bcast(device, case, op, act, mem):
    sa, sb = BCAST[case]
    torch.manual_seed(0)
    a = torch.rand(sa, dtype=torch.bfloat16) - 0.5
    b = torch.rand(sb, dtype=torch.bfloat16) - 0.5
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    f, g = OPS[op]
    ref = g(a.float(), b.float())
    kw = {"memory_config": mc}
    if act:
        kw["activations"] = ACT[act][0]()
        ref = ACT[act][1](ref)
    _check(f(ta, tb, **kw), ref, 0.03)


# #58725 and #58726 review: sharded ops that run one tile per DEST section. Native sharded broadcast: a height-sharded on 8
# cores, b interleaved column or scalar, c like a; the row broadcast, a block-sharded a and a mixed no-broadcast pair take
# the TensorAccessor path. Each with add and mul.
SB = {
    "hs8_colb_dram": ((1, 1, 1024, 1024), (1, 1, 1024, 1), "hs8", "dram"),
    "hs8_scalarb_dram": ((1, 1, 1024, 1024), (1, 1, 1, 1), "hs8", "dram"),
    "hs8_rowb_dram": ((1, 1, 1024, 1024), (1, 1, 1, 1024), "hs8", "dram"),
    "bs64_rowb_dram": ((1, 1, 4096, 1280), (1, 1, 1, 1280), "bs64", "dram"),
    "bs64_colb_dram": ((1, 1, 4096, 1280), (1, 1, 4096, 1), "bs64", "dram"),
    "hs8_nob_dram": ((1, 1, 1024, 1024), (1, 1, 1024, 1024), "hs8", "dram"),
    "hs8_nob_hs8": ((1, 1, 1024, 1024), (1, 1, 1024, 1024), "hs8", "hs8"),
    "hs32_colb_dram": ((1, 1, 4096, 1024), (1, 1, 4096, 1), "hs32", "dram"),
    "hs32_scalarb_dram": ((1, 1, 4096, 1024), (1, 1, 1, 1), "hs32", "dram"),
}
SBM = {"hs8": (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=4)), "hs32": (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=4, x=8)), "bs64": (ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=8, x=8))}


@pytest.mark.parametrize("case", list(SB))
@pytest.mark.parametrize("op", ["add", "mul"])
def test_bng_sharded_bcast(device, case, op):
    sa, sb, ma, mb = SB[case]
    torch.manual_seed(0)
    a = torch.rand(sa, dtype=torch.bfloat16) - 0.5
    b = torch.rand(sb, dtype=torch.bfloat16) - 0.5
    mca = _sharded(sa, *SBM[ma])
    mcb = ttnn.DRAM_MEMORY_CONFIG if mb == "dram" else _sharded(sb, *SBM[mb])
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mca)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mcb)
    f, g = OPS[op]
    _check(f(ta, tb, memory_config=mca), g(a.float(), b.float()), 0.03)


# #58725 review: the re-init skip after a post activation alone in the column and scalar broadcast kernels and the scalar
# kernel (a Python scalar b), interleaved and with a sharded operand.
@pytest.mark.parametrize(
    "case, op, act, mem",
    [
        ("col_b", "add", "gelu", "l1"),
        ("col_b", "mul", "softplus", "dram"),
        ("scalar_b", "add", "silu", "dram"),
        ("scalar_b", "mul", "gelu", "l1"),
        ("col_b", "add", "silu", "hs8"),
        ("scalar_b", "mul", "gelu", "hs8"),
        ("py_scalar", "add", "gelu", "dram"),
        ("py_scalar", "mul", "silu", "l1"),
        ("py_scalar", "add", "softplus", "hs8"),
    ],
    ids=lambda v: str(v),
)
def test_bng_bcast_post_activation(device, case, op, act, mem):
    sa = (1, 1, 1024, 1024)
    torch.manual_seed(0)
    a = torch.rand(sa, dtype=torch.bfloat16) - 0.5
    if mem == "hs8":
        mca = _sharded(sa, *SBM["hs8"])
        mcb = ttnn.DRAM_MEMORY_CONFIG
    else:
        mca = mcb = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mca)
    f, g = OPS[op]
    if case == "py_scalar":
        b = torch.tensor(0.375, dtype=torch.bfloat16)
        out = f(ta, 0.375, memory_config=mca, activations=ACT[act][0]())
    else:
        sb = BCAST[case][1]
        b = torch.rand(sb, dtype=torch.bfloat16) - 0.5
        tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mcb)
        out = f(ta, tb, memory_config=mca, activations=ACT[act][0]())
    _check(out, ACT[act][1](g(a.float(), b.float())), 0.03)
