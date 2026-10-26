# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

TML = ttnn.TensorMemoryLayout
LAYOUTS = {"HS": TML.HEIGHT_SHARDED, "WS": TML.WIDTH_SHARDED, "BS": TML.BLOCK_SHARDED}


def sharded_config(layout, h, w):
    if layout == TML.BLOCK_SHARDED:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shard = (h // 2, w // 2)
    else:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard = (h, w // 4) if layout == TML.WIDTH_SHARDED else (h // 4, w)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


def output_config(name, h, w):
    if name == "INT":
        return ttnn.L1_MEMORY_CONFIG
    if name.endswith("-specless"):
        return ttnn.MemoryConfig(LAYOUTS[name.split("-")[0]], ttnn.BufferType.L1)
    return sharded_config(LAYOUTS[name.split("-")[0]], h, w)


def assert_close(expected, actual):
    assert torch.allclose(actual.double(), expected.double(), rtol=0.02, atol=1e-3)


WRONG_VALUE_CASES = [
    (op, src, dst)
    for op, src, dsts in [
        ("relu", "INT", ["WS", "BS", "WS-specless", "BS-specless", "WS-prealloc", "BS-prealloc"]),
        ("relu", "HS", ["WS", "BS", "WS-prealloc", "BS-prealloc"]),
        ("relu", "WS", ["INT", "HS", "BS", "HS-prealloc", "BS-prealloc"]),
        ("relu", "BS", ["INT", "HS", "WS", "HS-prealloc", "WS-prealloc"]),
        ("add", "INT", ["WS", "BS", "WS-specless", "BS-specless"]),
        ("add_scalar", "INT", ["WS", "BS", "WS-specless", "BS-specless"]),
    ]
    for dst in dsts
]


@pytest.mark.parametrize("op, src, dst", WRONG_VALUE_CASES, ids=["-".join(c) for c in WRONG_VALUE_CASES])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("h, w", [(64, 128), (128, 256)], ids=["64x128", "128x256"])
def test_row_major_sharded_output_values(device, op, src, dst, dtype, h, w):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    shape = (1, 1, h, w)
    a = (torch.randn(shape) * 2).to(torch_dtype)
    b = (torch.randn(shape) * 2).to(torch_dtype)
    in_config = ttnn.DRAM_MEMORY_CONFIG if src == "INT" else sharded_config(LAYOUTS[src], h, w)
    a_t = ttnn.from_torch(a, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_config)
    b_t = ttnn.from_torch(b, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_config)
    if op == "relu":
        fn, expected = (lambda **kw: ttnn.relu(a_t, **kw)), torch.relu(a.double())
    elif op == "add":
        fn, expected = (lambda **kw: ttnn.add(a_t, b_t, **kw)), a.double() + b.double()
    else:
        fn, expected = (lambda **kw: ttnn.add(a_t, 1.5, **kw)), a.double() + 1.5
    if dst.endswith("-prealloc"):
        out = ttnn.from_torch(
            torch.zeros(shape, dtype=torch_dtype),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_config(dst, h, w),
        )
        fn(output_tensor=out)
    else:
        out = fn(memory_config=output_config(dst, h, w))
    assert_close(expected, ttnn.to_torch(out))


INTERLEAVED_OPERAND_CASES = [
    (buffers, dst, op)
    for buffers, dsts in [
        ("L1+L1", ["HS", "WS", "BS", "HS-specless", "WS-specless", "BS-specless"]),
        ("DRAM+DRAM", ["WS", "BS", "WS-specless", "BS-specless"]),
        ("L1+DRAM", ["WS", "BS", "WS-specless", "BS-specless"]),
    ]
    for dst in dsts
    for op in (["add", "mul", "sub"] + (["relu"] if not dst.endswith("-specless") and dst != "HS" else []))
]


@pytest.mark.parametrize(
    "buffers, dst, op", INTERLEAVED_OPERAND_CASES, ids=["-".join(c) for c in INTERLEAVED_OPERAND_CASES]
)
def test_row_major_interleaved_operands_sharded_output(device, buffers, dst, op):
    torch.manual_seed(0)
    h, w = 64, 128
    shape = (1, 1, h, w)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    configs = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
    a_buf, b_buf = buffers.split("+")
    a_t = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=configs[a_buf]
    )
    b_t = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=configs[b_buf]
    )
    out_config = output_config(dst, h, w)
    if op == "relu":
        out, expected = ttnn.relu(a_t, memory_config=out_config), torch.relu(a.double())
    elif op == "add":
        out, expected = ttnn.add(a_t, b_t, memory_config=out_config), a.double() + b.double()
    elif op == "mul":
        out, expected = ttnn.multiply(a_t, b_t, memory_config=out_config), a.double() * b.double()
    else:
        out, expected = ttnn.subtract(a_t, b_t, memory_config=out_config), a.double() - b.double()
    assert_close(expected, ttnn.to_torch(out))
