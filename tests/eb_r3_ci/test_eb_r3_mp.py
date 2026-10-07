# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fourth pass. test_mp (#58725): sharded binary_ng ops with an operand activation (the FPU divide's
RECIP on b, rsub's NEG on a, ldexp's EXP2 on b, the logical ops' NEZ, the Python-scalar kernel with an activation on a), whose
kernels rerun the binary init per DEST section; toggles EB_R3_PROBE_CHUNK_INIT (one more init per section) and
EB_R3_MULTI_PASS=K (the operand pass over K sections, then one init). test_nat (#58726): a block or width sharded a with a
column, scalar or row b in DRAM, and a height sharded a with a row b, which binary_ng routes off the native sharded path;
toggles EB_R3_NATIVE_BCAST and EB_R3_NATIVE_ROW route them native. CI only."""
import zlib

import pytest
import torch
import ttnn

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
U = ttnn.UnaryWithParam


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _mem(name):
    """(shape, memory config) per layout name: hs<cores>_t<tiles per core>, ws32_t4, bs<cores>_t<tiles per core>."""
    if name == "hs8_t8":
        shape, grid, st = (1, 1, 512, 128), (2, 4), ttnn.ShardStrategy.HEIGHT
    elif name == "hs8_t32":
        shape, grid, st = (1, 1, 1024, 256), (2, 4), ttnn.ShardStrategy.HEIGHT
    elif name == "hs8_t128":
        shape, grid, st = (1, 1, 1024, 1024), (2, 4), ttnn.ShardStrategy.HEIGHT
    elif name == "hs32_t128":
        shape, grid, st = (1, 1, 4096, 1024), (4, 8), ttnn.ShardStrategy.HEIGHT
    elif name == "ws32_t4":
        shape, grid, st = (1, 1, 32, 4096), (4, 8), ttnn.ShardStrategy.WIDTH
    elif name == "ws16_t32":
        shape, grid, st = (1, 1, 256, 2048), (2, 8), ttnn.ShardStrategy.WIDTH
    elif name == "bs64_t80":
        shape, grid, st = (1, 1, 4096, 1280), (8, 8), ttnn.ShardStrategy.BLOCK
    elif name == "bs16_t64":
        shape, grid, st = (1, 1, 1024, 1024), (4, 4), ttnn.ShardStrategy.BLOCK
    elif name == "bs16_n2_t128":
        shape, grid, st = (2, 1, 1024, 1024), (4, 4), ttnn.ShardStrategy.BLOCK
    else:
        raise ValueError(name)
    return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]), strategy=st)


def _seed(*key):
    torch.manual_seed(zlib.crc32("-".join(str(k) for k in key).encode()) % 100000)


MP_OPS = ["div", "rsub", "ldexp", "logical_and", "add_arelu"]
MP_MEMS = ["hs8_t8", "hs8_t32", "hs8_t128", "ws32_t4", "bs64_t80"]
MP = [(op, m, d) for op in MP_OPS for m in MP_MEMS for d in ("bf16", "bfp8")]
MP += [(op, m, d) for op in ("rsub_s", "add_arelu_s") for m in MP_MEMS for d in ("bf16", "bfp8")]


@pytest.mark.parametrize("op, mem, d", MP, ids=["-".join(c) for c in MP])
def test_mp(device, op, mem, d):
    shape, mc = _mem(mem)
    _seed(op, mem, d)
    a = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
    ta = ttnn.from_torch(a, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = None
    if not op.endswith("_s"):
        if op == "div":
            b = (torch.rand(shape, dtype=torch.bfloat16) + 0.5) * torch.where(torch.rand(shape) < 0.5, -1.0, 1.0).bfloat16()
        elif op == "ldexp":
            b = torch.randint(-4, 5, shape).bfloat16()
        else:
            b = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
        tb = ttnn.from_torch(b, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    fn = {
        "div": lambda: ttnn.divide(ta, tb, memory_config=mc),
        "rsub": lambda: ttnn.rsub(ta, tb, memory_config=mc),
        "ldexp": lambda: ttnn.ldexp(ta, tb, memory_config=mc),
        "logical_and": lambda: ttnn.logical_and(ta, tb, memory_config=mc),
        "add_arelu": lambda: ttnn.add(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], memory_config=mc),
        "rsub_s": lambda: ttnn.rsub(ta, 0.75, memory_config=mc),
        "add_arelu_s": lambda: ttnn.add(ta, 0.75, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], memory_config=mc),
    }[op]
    for _ in range(3):
        out = fn()
    af = ttnn.to_torch(ta).float()
    bf = ttnn.to_torch(tb).float() if tb is not None else None
    ref = {
        "div": lambda: af / bf,
        "rsub": lambda: bf - af,
        "ldexp": lambda: af * torch.pow(2.0, bf),
        "logical_and": lambda: torch.logical_and(af, bf).float(),
        "add_arelu": lambda: torch.relu(af) + bf,
        "rsub_s": lambda: 0.75 - af,
        "add_arelu_s": lambda: torch.relu(af) + 0.75,
    }[op]()
    got = ttnn.to_torch(out).float()
    for t in [ta, out] + ([tb] if tb is not None else []):
        ttnn.deallocate(t)
    tol = 0.15 if d == "bfp8" else 0.06
    assert torch.allclose(got, ref, rtol=0.05, atol=tol), float((got - ref).abs().max())


NAT_MEMS = ["bs64_t80", "bs16_t64", "bs16_n2_t128", "ws32_t4", "ws16_t32"]
NAT = [(op, m, k, d, None) for op in ("add", "mul") for m in NAT_MEMS for k in ("col", "scalar", "row") for d in ("bf16", "bfp8")]
NAT += [("sub", m, k, "bf16", None) for m in ("bs64_t80", "ws16_t32") for k in ("col", "scalar", "row")]
NAT += [(op, m, k, "bf16", "relu") for op in ("add", "mul") for m in ("bs64_t80", "ws16_t32") for k in ("col", "scalar", "row")]
NAT += [(op, m, "row", d, None) for op in ("add", "mul") for m in ("hs8_t128", "hs32_t128") for d in ("bf16", "bfp8")]


@pytest.mark.parametrize("op, mem, kind, d, act", NAT, ids=["-".join(str(x) for x in c) for c in NAT])
def test_nat(device, op, mem, kind, d, act):
    shape, mc = _mem(mem)
    _seed(op, mem, kind, d, act)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b_shape = {"col": (shape[0], 1, shape[2], 1), "scalar": (1, 1, 1, 1), "row": (1, 1, 1, shape[3])}[kind]
    b = torch.randn(b_shape, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kw = dict(memory_config=mc)
    if act:
        kw["activations"] = [U(ttnn.UnaryOpType.RELU)]
    f = {"add": ttnn.add, "sub": ttnn.subtract, "mul": lambda x, y, **k: ttnn.multiply(x, y, fast_and_approximate_mode=True, **k)}[op]
    for _ in range(3):
        out = f(ta, tb, **kw)
    af, bf = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    ref = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](af, bf)
    if act:
        ref = torch.relu(ref)
    got = ttnn.to_torch(out).float()
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    tol = 0.3 if d == "bfp8" else 0.1
    assert torch.allclose(got, ref, rtol=0.1, atol=tol), float((got - ref).abs().max())


NAT2_MEMS = ["bs64_t80", "bs16_t64", "bs16_n2_t128", "ws16_t32", "ws32_t4"]
NAT2 = [(op, m, k, act) for op in ("add", "mul") for m in NAT2_MEMS for k in ("col", "scalar") for act in ("relu", "gelu", "silu")]
NAT2 += [(op, m, k, None) for op in ("rsub", "add_arelu", "mul_asilu", "logical_and") for m in NAT2_MEMS for k in ("col", "scalar")]
NAT2 += [(op, m, k, "fp32") for op in ("add", "mul") for m in ("bs64_t80", "ws16_t32") for k in ("col", "scalar")]


@pytest.mark.parametrize("op, mem, kind, act", NAT2, ids=["-".join(str(x) for x in c) for c in NAT2])
def test_nat2(device, op, mem, kind, act):
    """#58726: native routing for a block or width sharded a with a column or scalar b, with fused activations and a Float32
    output, against the current routing."""
    shape, mc = _mem(mem)
    _seed("nat2", op, mem, kind, act)
    a = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
    b_shape = {"col": (shape[0], 1, shape[2], 1), "scalar": (1, 1, 1, 1)}[kind]
    b = torch.rand(b_shape, dtype=torch.bfloat16) * 2 - 1
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kw = dict(memory_config=mc)
    if act in ("relu", "gelu", "silu"):
        kw["activations"] = [U({"relu": ttnn.UnaryOpType.RELU, "gelu": ttnn.UnaryOpType.GELU, "silu": ttnn.UnaryOpType.SILU}[act])]
    if act == "fp32":
        kw["dtype"] = ttnn.float32
    fn = {
        "add": lambda: ttnn.add(ta, tb, **kw),
        "mul": lambda: ttnn.multiply(ta, tb, fast_and_approximate_mode=True, **kw),
        "rsub": lambda: ttnn.rsub(ta, tb, **kw),
        "add_arelu": lambda: ttnn.add(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], **kw),
        "mul_asilu": lambda: ttnn.multiply(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.SILU)], fast_and_approximate_mode=True, **kw),
        "logical_and": lambda: ttnn.logical_and(ta, tb, **kw),
    }[op]
    for _ in range(3):
        out = fn()
    af, bf = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    ref = {"add": lambda: af + bf, "mul": lambda: af * bf, "rsub": lambda: bf - af, "add_arelu": lambda: torch.relu(af) + bf,
           "mul_asilu": lambda: torch.nn.functional.silu(af) * bf, "logical_and": lambda: torch.logical_and(af, bf).float()}[op]()
    if act == "relu":
        ref = torch.relu(ref)
    elif act == "gelu":
        ref = torch.nn.functional.gelu(ref)
    elif act == "silu":
        ref = torch.nn.functional.silu(ref)
    got = ttnn.to_torch(out).float()
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    assert torch.allclose(got, ref, rtol=0.05, atol=0.06), float((got - ref).abs().max())
