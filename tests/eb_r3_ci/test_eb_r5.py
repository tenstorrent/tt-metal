# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fifth pass. CI only.
test_nat4 (#58726): native routing classes on small grids (1, 2, 8 cores) and the class boundaries.
test_mulact2 (#58723): the per-tile multiply where binary_ng re-runs the init per tile, with a same-program control
(logical_or, an add).
test_one (#58725): one-section ops of the no-broadcast and Python-scalar kernels, alone."""
import zlib

import pytest
import torch
import ttnn

import test_eb_r3_mp as m

U = ttnn.UnaryWithParam
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


SH = {
    "bs1_t64": ((1, 1, 256, 256), 1, 1, ttnn.ShardStrategy.BLOCK),
    "bs2_t64": ((1, 1, 256, 512), 1, 2, ttnn.ShardStrategy.BLOCK),
    "ws2_t64": ((1, 1, 64, 2048), 1, 2, ttnn.ShardStrategy.WIDTH),
    "bs8_t64": ((1, 1, 512, 1024), 2, 4, ttnn.ShardStrategy.BLOCK),
    "ws8_t64": ((1, 1, 256, 2048), 1, 8, ttnn.ShardStrategy.WIDTH),
    "bs16_t64": ((1, 1, 1024, 1024), 4, 4, ttnn.ShardStrategy.BLOCK),
    "bs32_t64": ((1, 1, 1024, 2048), 4, 8, ttnn.ShardStrategy.BLOCK),
    "bs64_t80": ((1, 1, 4096, 1280), 8, 8, ttnn.ShardStrategy.BLOCK),
    "ws16_t32": ((1, 1, 256, 2048), 2, 8, ttnn.ShardStrategy.WIDTH),
    "ws32_t4": ((1, 1, 32, 4096), 4, 8, ttnn.ShardStrategy.WIDTH),
    "bs4_t256": ((1, 1, 1024, 1024), 2, 2, ttnn.ShardStrategy.BLOCK),
    "bs8_t128": ((1, 1, 1024, 1024), 2, 4, ttnn.ShardStrategy.BLOCK),
}


def _mc(name):
    shape, gy, gx, st = SH[name]
    return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=st)


def _run(device, op, shape, mc, da, db, do, kind, act=None, lact=None):
    torch.manual_seed(zlib.crc32(f"{op}{shape}{da}{db}{do}{kind}{act}{lact}".encode()) % 100000)
    a = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
    b_shape = {"col": (shape[0], 1, shape[2], 1), "scalar": (1, 1, 1, 1), "none": shape}[kind]
    b = torch.rand(b_shape, dtype=torch.bfloat16) + 0.5
    if op == "ldexp":
        b = torch.randint(-3, 4, b_shape).bfloat16()
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG if kind != "none" else mc)
    kw = dict(memory_config=mc)
    if do != da:
        kw["dtype"] = DT[do]
    if act:
        kw["activations"] = [U({"relu": ttnn.UnaryOpType.RELU, "gelu": ttnn.UnaryOpType.GELU, "silu": ttnn.UnaryOpType.SILU}[act])]
    if lact:
        kw["input_tensor_a_activations"] = [U(ttnn.UnaryOpType.RELU)]
    fn = {"add": lambda: ttnn.add(ta, tb, **kw), "sub": lambda: ttnn.subtract(ta, tb, **kw),
          "mul": lambda: ttnn.multiply(ta, tb, fast_and_approximate_mode=True, **kw),
          "logical_and": lambda: ttnn.logical_and(ta, tb, **kw), "logical_or": lambda: ttnn.logical_or(ta, tb, **kw),
          "ldexp": lambda: ttnn.ldexp(ta, tb, **kw), "div": lambda: ttnn.divide(ta, tb, **kw), "rsub": lambda: ttnn.rsub(ta, tb, **kw)}[op]
    for _ in range(3):
        out = fn()
    got = ttnn.to_torch(out)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    assert got.shape[-1] == shape[-1]


NAT4 = [(op, mm, k, d) for op in ("add", "mul") for mm in ("bs1_t64", "bs2_t64", "ws2_t64") for k in ("col", "scalar") for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bf16-bf16-fp32")]
NAT4 += [(op, mm, k, d) for op in ("add", "mul") for mm in ("bs8_t64", "ws8_t64", "bs8_t128") for k in ("col", "scalar") for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4", "bf16-bfp8-bf16", "bf16-bf16-fp32")]
NAT4 += [(op, mm, k, d) for op in ("add", "mul") for mm in ("bs16_t64", "bs32_t64") for k in ("col",) for d in ("bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4", "bf16-bfp8-bf16", "bf16-bf16-fp32")]


@pytest.mark.parametrize("op, mem, kind, dts", NAT4, ids=["-".join(c) for c in NAT4])
def test_nat4(device, op, mem, kind, dts):
    shape, mc = _mc(mem)
    da, db, do = dts.split("-")
    _run(device, op, shape, mc, da, db, do, kind)


NAT5 = [(op, mm, k, act) for op in ("add", "mul") for mm in ("bs16_t64", "bs32_t64", "bs64_t80", "ws16_t32", "ws8_t64", "ws32_t4") for k in ("col", "scalar") for act in ("relu", "gelu")]


@pytest.mark.parametrize("op, mem, kind, act", NAT5, ids=["-".join(c) for c in NAT5])
def test_nat5(device, op, mem, kind, act):
    shape, mc = _mc(mem)
    _run(device, op, shape, mc, "bf16", "bf16", "bf16", kind, act=act)


MA = [(op, mm) for op in ("logical_and", "ldexp", "div", "logical_or") for mm in ("bs16_t64", "bs32_t64", "bs64_t80", "ws16_t32", "bs8_t128")]


@pytest.mark.parametrize("op, mem", MA, ids=["-".join(c) for c in MA])
def test_mulact2(device, op, mem):
    shape, mc = _mc(mem)
    _run(device, op, shape, mc, "bf16", "bf16", "bf16", "col")


MAD = [(op, k) for op in ("logical_and", "ldexp", "div", "logical_or") for k in ("col", "none")]


@pytest.mark.parametrize("op, kind", MAD, ids=["-".join(c) for c in MAD])
def test_mulact2_dram(device, op, kind):
    shape = (1, 1, 1024, 1024)
    _run(device, op, shape, ttnn.DRAM_MEMORY_CONFIG, "bf16", "bf16", "bf16", kind)


ONE = [(op, mm, d) for op in ("div", "rsub", "ldexp", "logical_and", "add_arelu", "rsub_s", "add_arelu_s") for mm in ("hs8_t8", "ws32_t4") for d in ("bf16", "bfp8")]


@pytest.mark.parametrize("op, mem, d", ONE, ids=["-".join(c) for c in ONE])
def test_one(device, op, mem, d):
    m.test_mp(device, op, mem, d)
