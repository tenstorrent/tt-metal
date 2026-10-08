# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass (#58722, #58726, #58727): binary_ng's block unpack, block pack (main's contiguous form or
#58816's), the Python-scalar kernel, post activations in the block section and the broadcast sections, over tiles per core
and formats. CI only; the toggles are in the factory (EB_R3_*)."""
import zlib

import pytest
import torch
import ttnn

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
HS = {4: (256, 128), 8: (512, 128), 16: (1024, 128), 32: (1024, 256), 128: (1024, 1024)}
ACT = {
    "relu": lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
    "gelu": lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)],
    "silu": lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _mem(name):
    if name.startswith("hs8"):
        h, w = HS[int(name.split("_t")[1])]
        shape = (1, 1, h, w)
        return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.HEIGHT)
    if name == "hs32_t128":
        shape = (1, 1, 4096, 1024)
        return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.HEIGHT)
    if name == "ws32_t4":
        shape = (1, 1, 32, 4096)
        return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.WIDTH)
    shape = (1, 1, 4096, 1280)
    return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)


def _fn(op):
    if op == "mul":
        return lambda a, b, **kw: ttnn.multiply(a, b, fast_and_approximate_mode=True, **kw)
    return {"add": ttnn.add, "sub": ttnn.subtract}[op]


def _run(device, op, shape, mc, da, db, do, b_shape=None, act=None, scalar=None, mcb=None):
    torch.manual_seed(zlib.crc32(f"{op}{shape}{da}{db}{do}{b_shape}{act}{scalar}".encode()) % 100000)
    a = torch.randn(shape, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kw = dict(dtype=DT[do], memory_config=mc)
    if act:
        kw["activations"] = ACT[act]()
    if scalar is not None:
        for _ in range(3):
            out = _fn(op)(ta, scalar, **kw)
        ref = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](ttnn.to_torch(ta).float(), scalar)
    else:
        b = torch.randn(b_shape or shape, dtype=torch.bfloat16)
        tb = ttnn.from_torch(b, dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mcb or mc)
        for _ in range(3):
            out = _fn(op)(ta, tb, **kw)
        ref = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float())
    if act == "relu":
        ref = torch.relu(ref)
    elif act == "gelu":
        ref = torch.nn.functional.gelu(ref)
    elif act == "silu":
        ref = torch.nn.functional.silu(ref)
    got = ttnn.to_torch(out).float()
    for t in [ta, out] + ([tb] if scalar is None else []):
        ttnn.deallocate(t)
    tol = 2.0 if do == "bfp4" else (0.3 if "bfp4" in (da, db) else 0.1)
    assert torch.allclose(got, ref, rtol=0.1, atol=tol), float((got - ref).abs().max())


MEMS = ["hs8_t4", "hs8_t8", "hs8_t16", "hs8_t32", "hs8_t128", "ws32_t4", "bs64_t80"]
DT_ADD = [("bf16", "bf16", "bf16"), ("bfp8", "bfp8", "bfp8"), ("bfp8", "bfp8", "bf16"), ("bf16", "bf16", "fp32"), ("bfp4", "bfp4", "bfp4"),
          ("bfp4", "bfp4", "bf16"), ("bf16", "bfp8", "bf16"), ("fp32", "bf16", "fp32"), ("fp32", "fp32", "fp32"),
          ("bf16", "bfp8", "bfp8"), ("bfp8", "bf16", "bf16")]
DT_OTHER = [("bf16", "bf16", "bf16"), ("bfp8", "bfp8", "bfp8"), ("bf16", "bf16", "fp32"), ("bf16", "bfp8", "bf16")]
NOB = [("add", m, *d) for m in MEMS for d in DT_ADD] + [(op, m, *d) for op in ("sub", "mul") for m in MEMS for d in DT_OTHER]
NOB = [c for c in NOB if not ("fp32" in c[2:] and c[1] in ("bs64_t80", "hs8_t128"))]


@pytest.mark.parametrize("op, mem, da, db, do", NOB, ids=["-".join(c) for c in NOB])
def test_blk4_nob(device, op, mem, da, db, do):
    shape, mc = _mem(mem)
    _run(device, op, shape, mc, da, db, do)


POST = [(op, m, "bf16", "bf16", act) for op in ("add", "mul") for m in ("hs8_t8", "hs8_t16", "hs8_t128", "ws32_t4", "bs64_t80") for act in ("relu", "gelu", "silu")]
POST += [("add", m, d, do, "relu") for m in ("hs8_t8", "hs8_t16", "hs8_t128") for (d, do) in (("bfp8", "bf16"), ("bf16", "bfp8"), ("bf16", "fp32"))]
POST += [("sub", m, "bf16", "bf16", "relu") for m in ("hs8_t16", "hs8_t128")]


@pytest.mark.parametrize("op, mem, d, do, act", POST, ids=["-".join(c) for c in POST])
def test_blk4_post(device, op, mem, d, do, act):
    shape, mc = _mem(mem)
    _run(device, op, shape, mc, d, d, do, act=act)


SCAL = [(op, m, d, act) for op in ("add", "mul") for m in ("hs8_t8", "hs8_t16", "hs8_t128", "ws32_t4", "bs64_t80") for d in ("bf16", "bfp8") for act in (None, "relu")]


@pytest.mark.parametrize("op, mem, d, act", SCAL, ids=["-".join(str(x) for x in c) for c in SCAL])
def test_blk4_scalar(device, op, mem, d, act):
    shape, mc = _mem(mem)
    _run(device, op, shape, mc, d, None, d, act=act, scalar=0.375)


BC = [(op, m, k, d, do, act) for op in ("add", "mul") for m in ("hs8_t128", "hs32_t128", "bs64_t80") for k in ("col", "scalar")
      for (d, do) in (("bf16", "bf16"), ("bfp8", "bf16"), ("bf16", "fp32"), ("bfp8", "bfp8")) for act in (None, "gelu", "silu", "relu")]
BC = [c for c in BC if not (c[5] and c[3] != "bf16") and not (c[4] == "fp32" and c[5])]


@pytest.mark.parametrize("op, mem, kind, d, do, act", BC, ids=["-".join(str(x) for x in c) for c in BC])
def test_blk4_bcast(device, op, mem, kind, d, do, act):
    shape, mc = _mem(mem)
    b_shape = (1, 1, shape[2], 1) if kind == "col" else (1, 1, 1, 1)
    _run(device, op, shape, mc, d, d, do, b_shape=b_shape, act=act, mcb=ttnn.DRAM_MEMORY_CONFIG)


# #58725 / #58726: sharded column and scalar broadcasts whose per-tile operand carries an activation (rsub's NEG, the logical
# ops' NEZ), one tile per section on main.
OPACT = [(op, m, k) for op in ("rsub", "logical_and", "logical_or", "add_arelu", "mul_asilu") for m in ("hs8_t128", "hs32_t128", "bs64_t80") for k in ("col", "scalar")]


@pytest.mark.parametrize("op, mem, kind", OPACT, ids=["-".join(c) for c in OPACT])
def test_blk4_opact(device, op, mem, kind):
    shape, mc = _mem(mem)
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    b = torch.rand((1, 1, shape[2], 1) if kind == "col" else (1, 1, 1, 1), dtype=torch.bfloat16) - 0.5
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    U = ttnn.UnaryWithParam
    fn = {"rsub": ttnn.rsub, "logical_and": ttnn.logical_and, "logical_or": ttnn.logical_or,
          "add_arelu": lambda x, y, **kw: ttnn.add(x, y, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], **kw),
          "mul_asilu": lambda x, y, **kw: ttnn.multiply(x, y, input_tensor_a_activations=[U(ttnn.UnaryOpType.SILU)], fast_and_approximate_mode=True, **kw)}[op]
    for _ in range(3):
        out = fn(ta, tb, memory_config=mc)
    af, bf = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    ref = {"rsub": lambda: bf - af, "logical_and": lambda: torch.logical_and(af, bf).float(), "logical_or": lambda: torch.logical_or(af, bf).float(),
           "add_arelu": lambda: torch.relu(af) + bf, "mul_asilu": lambda: torch.nn.functional.silu(af) * bf}[op]()
    got = ttnn.to_torch(out).float()
    assert torch.allclose(got, ref, rtol=0.05, atol=0.05), float((got - ref).abs().max())


# #58723: HiFi3 for a block-float SrcB, every kernel form binary_ng runs a multiply in.
H3_DT = [("bf16", "bfp8", "bf16"), ("bfp8", "bfp8", "bfp8"), ("bfp8", "bfp8", "bf16"), ("bf16", "bfp4", "bf16"), ("bfp4", "bfp4", "bfp4"),
         ("bf16", "bfp8", "fp32")]
H3 = [(m, "none", d) for m in ("hs8_t4", "hs8_t16", "hs8_t128", "ws32_t4", "bs64_t80", "dram") for d in H3_DT]
H3 += [(m, k, d) for m in ("hs8_t128", "dram") for k in ("col", "row", "scalar") for d in (("bf16", "bfp8", "bf16"), ("bfp8", "bfp8", "bfp8"))]


@pytest.mark.parametrize("mem, kind, d", H3, ids=["-".join((m, k) + d) for m, k, d in H3])
def test_blk4_hifi3(device, mem, kind, d):
    da, db, do = d
    if mem == "dram":
        shape, mc = (1, 1, 1024, 1024), ttnn.DRAM_MEMORY_CONFIG
    else:
        shape, mc = _mem(mem)
    b_shape = {"none": None, "col": (1, 1, shape[2], 1), "row": (1, 1, 1, shape[3]), "scalar": (1, 1, 1, 1)}[kind]
    _run(device, "mul", shape, mc, da, db, do, b_shape=b_shape, mcb=None if kind == "none" else ttnn.DRAM_MEMORY_CONFIG)
