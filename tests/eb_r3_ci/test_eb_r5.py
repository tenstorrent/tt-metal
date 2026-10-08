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
    "bs1_t128": ((1, 1, 256, 512), 1, 1, ttnn.ShardStrategy.BLOCK),
    "bs2_t128": ((1, 1, 512, 512), 1, 2, ttnn.ShardStrategy.BLOCK),
    "bs4_t128": ((1, 1, 512, 1024), 2, 2, ttnn.ShardStrategy.BLOCK),
    "ws2_t128": ((1, 1, 128, 2048), 1, 2, ttnn.ShardStrategy.WIDTH),
    "bs4_t64": ((1, 1, 512, 512), 2, 2, ttnn.ShardStrategy.BLOCK),
    "bs16_r32": ((1, 1, 4096, 1024), 4, 4, ttnn.ShardStrategy.BLOCK),
    "bs32_t128": ((1, 1, 2048, 2048), 4, 8, ttnn.ShardStrategy.BLOCK),
    "ws8_t4": ((1, 1, 32, 1024), 1, 8, ttnn.ShardStrategy.WIDTH),
    "ws16_t4": ((1, 1, 32, 2048), 2, 8, ttnn.ShardStrategy.WIDTH),
    "bs16_t4": ((1, 1, 128, 512), 4, 4, ttnn.ShardStrategy.BLOCK),
    "ws16_t64": ((1, 1, 512, 2048), 2, 8, ttnn.ShardStrategy.WIDTH),
    "ws8_t32": ((1, 1, 256, 1024), 1, 8, ttnn.ShardStrategy.WIDTH),
    "ws32_t32": ((1, 1, 256, 4096), 4, 8, ttnn.ShardStrategy.WIDTH),
    "ws8_t128": ((1, 1, 512, 2048), 1, 8, ttnn.ShardStrategy.WIDTH),
    "ws4_t64": ((1, 1, 256, 1024), 1, 4, ttnn.ShardStrategy.WIDTH),
    "ws4_r4": ((1, 1, 128, 2048), 1, 4, ttnn.ShardStrategy.WIDTH),
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
        kw["input_tensor_a_activations"] = [U({"relu": ttnn.UnaryOpType.RELU, "silu": ttnn.UnaryOpType.SILU}[lact])]
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


# fifth pass, second native run (#58726): the class boundaries. Scalar b on 1 to 4 cores at 128 tiles per core; column b on 4
# cores, at 32 tile rows on 16 cores and 16 rows on 32; activations at 4 tiles per core on 8 and 16 cores; scalar relu and
# gelu on width 8, 16 and 32 and block 32 at more tiles per core.
NAT6 = [("add", mm, "scalar", d) for mm in ("bs1_t128", "bs2_t128", "bs4_t128", "ws2_t128") for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4", "bf16-bfp8-bf16", "bf16-bf16-fp32")]
NAT6 += [("mul", mm, "scalar", d) for mm in ("bs1_t128", "bs2_t128", "bs4_t128", "ws2_t128") for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8")]
NAT6 += [(op, "bs1_t64", "scalar", d) for op in ("add", "mul") for d in ("bfp4-bfp4-bfp4", "bf16-bfp8-bf16")]
NAT6 += [(op, "bs4_t64", "col", "bf16-bf16-bf16") for op in ("add", "mul")]
NAT6 += [("add", "bs4_t128", "col", d) for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8")] + [("mul", "bs4_t128", "col", "bf16-bf16-bf16")]
NAT6 += [("add", "ws2_t128", "col", d) for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4")]
NAT6 += [(op, "ws2_t64", "col", "bfp4-bfp4-bfp4") for op in ("add", "mul")]
NAT6 += [(op, "bs16_r32", "col", d) for op in ("add", "mul") for d in ("bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4")]
NAT6 += [("add", "bs32_t128", "col", d) for d in ("bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bfp4-bfp4-bfp4")] + [("mul", "bs32_t128", "col", "bf16-bf16-bf16")]


@pytest.mark.parametrize("op, mem, kind, dts", NAT6, ids=["-".join(c) for c in NAT6])
def test_nat6(device, op, mem, kind, dts):
    shape, mc = _mc(mem)
    da, db, do = dts.split("-")
    _run(device, op, shape, mc, da, db, do, kind)


ACT7 = {"add_relu": ("add", "relu", None), "add_gelu": ("add", "gelu", None), "add_silu": ("add", "silu", None),
        "rsub": ("rsub", None, None), "logical_and": ("logical_and", None, None), "add_arelu": ("add", None, "relu"),
        "mul_asilu": ("mul", None, "silu"), "mul_relu": ("mul", "relu", None), "mul_gelu": ("mul", "gelu", None)}
NAT7 = [(a, mm, k) for mm in ("ws8_t4", "ws16_t4", "bs16_t4") for k in ("col", "scalar") for a in ("add_relu", "add_gelu", "add_silu", "rsub", "logical_and", "add_arelu", "mul_asilu")]
NAT7 += [(a, mm, "scalar") for mm in ("ws16_t64", "ws8_t32", "ws32_t32", "bs32_t128") for a in ("add_relu", "add_gelu", "mul_relu", "mul_gelu")]
NAT7 += [(a, mm, "col") for mm in ("ws32_t32", "bs32_t128") for a in ("add_relu", "mul_relu")]


@pytest.mark.parametrize("act, mem, kind", NAT7, ids=["-".join(c) for c in NAT7])
def test_nat7(device, act, mem, kind):
    shape, mc = _mc(mem)
    op, post, lact = ACT7[act]
    _run(device, op, shape, mc, "bf16", "bf16", "bf16", kind, act=post, lact=lact)


# sixth pass (#58725): one-section and multi-section ops with block-float and bf16 operands, bfp4 included, no accuracy check
ONE4_MEMS = {"hs8_t8": ((1, 1, 512, 128), 2, 4, ttnn.ShardStrategy.HEIGHT), "ws32_t4": ((1, 1, 32, 4096), 4, 8, ttnn.ShardStrategy.WIDTH),
             "hs8_t24": ((1, 1, 768, 256), 2, 4, ttnn.ShardStrategy.HEIGHT), "hs8_t32": ((1, 1, 1024, 256), 2, 4, ttnn.ShardStrategy.HEIGHT)}
ONE4 = [(op, mm, d) for op in ("rsub", "add_arelu", "logical_and", "ldexp", "div", "rsub_s") for mm in ONE4_MEMS for d in ("bf16", "bfp8", "bfp4")]


@pytest.mark.parametrize("op, mem, d", ONE4, ids=["-".join(c) for c in ONE4])
def test_one4(device, op, mem, d):
    shape, gy, gx, st = ONE4_MEMS[mem]
    mc = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=st)
    torch.manual_seed(zlib.crc32(f"one4{op}{mem}{d}".encode()) % 100000)
    a = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
    ta = ttnn.from_torch(a, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = None
    if not op.endswith("_s"):
        b = torch.randint(-4, 5, shape).bfloat16() if op == "ldexp" else torch.rand(shape, dtype=torch.bfloat16) + 0.5
        tb = ttnn.from_torch(b, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    fn = {"rsub": lambda: ttnn.rsub(ta, tb, memory_config=mc), "add_arelu": lambda: ttnn.add(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], memory_config=mc),
          "logical_and": lambda: ttnn.logical_and(ta, tb, memory_config=mc), "ldexp": lambda: ttnn.ldexp(ta, tb, memory_config=mc),
          "div": lambda: ttnn.divide(ta, tb, memory_config=mc), "rsub_s": lambda: ttnn.rsub(ta, 0.375, memory_config=mc),
          "add_arelu_s": lambda: ttnn.add(ta, 0.375, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], memory_config=mc)}[op]
    for _ in range(3):
        out = fn()
    got = ttnn.to_torch(out)
    for t in (ta, tb, out):
        if t is not None:
            ttnn.deallocate(t)
    assert got.shape[-1] == shape[-1]


# fifth pass (#58723): logical_and (per-tile multiply, init per tile) off the native path on block grids around 4x4, the same
# tensors, with logical_or (an add, the same program on both sides) as the control
SH8 = {"g2x4": ((1, 1, 1024, 1024), 2, 4), "g4x4": ((1, 1, 1024, 1024), 4, 4), "g2x8": ((1, 1, 1024, 1024), 2, 8),
       "g4x2": ((1, 1, 1024, 1024), 4, 2), "g3x4": ((1, 1, 768, 1024), 3, 4), "g4x8": ((1, 1, 1024, 1024), 4, 8),
       "g4x4n2": ((2, 1, 1024, 1024), 4, 4), "g8x4": ((1, 1, 2048, 1024), 8, 4)}
MA3 = [(op, g, k) for op in ("logical_and", "logical_or") for g in SH8 for k in ("col", "scalar")]


@pytest.mark.parametrize("op, grid, kind", MA3, ids=["-".join(c) for c in MA3])
def test_mulact3(device, op, grid, kind):
    shape, gy, gx = SH8[grid]
    mc = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.BLOCK)
    _run(device, op, shape, mc, "bf16", "bf16", "bf16", kind)



# sixth pass (#58726): the boundaries (bfp4 column on 4 and 8 width cores, scalar relu and gelu on 8 width cores at 32 to 128
# tiles, bfp8 column on 2 and 4 width cores) and the subsets taken (column add with relu on 32 block cores, scalar add with
# gelu on 16 width cores)
NAT8 = [(op, mm, "col", "bfp4-bfp4-bfp4") for op in ("add", "mul") for mm in ("ws4_t64", "ws8_t64", "ws8_t128")]
NAT8 += [(op, mm, "col", "bfp8-bfp8-bfp8") for op in ("add", "mul") for mm in ("ws2_t64", "ws2_t128", "ws4_r4", "ws4_t64", "ws8_t128")]
NAT9 = [(a, mm, "scalar") for mm in ("ws8_t32", "ws8_t64", "ws8_t128") for a in ("add_relu", "mul_relu", "add_gelu", "mul_gelu")]
NAT9 += [(a, mm, "col") for mm in ("bs32_t64", "bs32_t128") for a in ("add_relu", "mul_relu", "sub_relu")]
NAT9 += [(a, mm, "scalar") for mm in ("ws16_t32", "ws16_t64") for a in ("add_gelu", "mul_gelu", "sub_gelu")]
ACT7["sub_relu"] = ("sub", "relu", None)
ACT7["sub_gelu"] = ("sub", "gelu", None)


@pytest.mark.parametrize("op, mem, kind, dts", NAT8, ids=["-".join(c) for c in NAT8])
def test_nat8(device, op, mem, kind, dts):
    shape, mc = _mc(mem)
    da, db, do = dts.split("-")
    _run(device, op, shape, mc, da, db, do, kind)


@pytest.mark.parametrize("act, mem, kind", NAT9, ids=["-".join(c) for c in NAT9])
def test_nat9(device, act, mem, kind):
    shape, mc = _mc(mem)
    op, post, lact = ACT7[act]
    _run(device, op, shape, mc, "bf16", "bf16", "bf16", kind, act=post, lact=lact)


# sixth pass (#58725): the operand pass over eight sections against four, from 64 tiles per core
K8_MEMS = {"hs8_t64": ((1, 1, 2048, 256), 2, 4, ttnn.ShardStrategy.HEIGHT), "hs8_t128": ((1, 1, 1024, 1024), 2, 4, ttnn.ShardStrategy.HEIGHT),
           "bs64_t80": ((1, 1, 4096, 1280), 8, 8, ttnn.ShardStrategy.BLOCK), "hs8_t256": ((1, 1, 2048, 1024), 2, 4, ttnn.ShardStrategy.HEIGHT),
           "hs8_t32": ((1, 1, 1024, 256), 2, 4, ttnn.ShardStrategy.HEIGHT)}
K8 = [(op, mm, d) for op in ("rsub", "add_arelu", "logical_and", "ldexp", "div", "rsub_s", "add_arelu_s") for mm in ("hs8_t64", "hs8_t128", "bs64_t80", "hs8_t32") for d in ("bf16", "bfp8")]
K8 += [(op, "hs8_t256", d) for op in ("rsub", "logical_and", "rsub_s") for d in ("bfp8", "bfp4")]


@pytest.mark.parametrize("op, mem, d", K8, ids=["-".join(c) for c in K8])
def test_k8(device, op, mem, d):
    ONE4_MEMS[mem] = K8_MEMS[mem]
    test_one4(device, op, mem, d)
