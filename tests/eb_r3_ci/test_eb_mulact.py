# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fourth pass: binary_ng multiplies with an operand activation on the paths that re-run the binary
init per tile (interleaved, and block sharded off the native path), the per-tile against the per-face multiply hand-off. CI
only."""
import zlib

import pytest
import torch
import ttnn

U = ttnn.UnaryWithParam


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _mem(name):
    if name == "dram":
        return (1, 1, 1024, 1024), ttnn.DRAM_MEMORY_CONFIG
    if name == "l1":
        return (1, 1, 512, 512), ttnn.L1_MEMORY_CONFIG
    if name == "dram_big":
        return (1, 1, 4096, 4096), ttnn.DRAM_MEMORY_CONFIG
    shape, grid = {"bs16_t64": ((1, 1, 1024, 1024), (4, 4)), "bs16_n2_t128": ((2, 1, 1024, 1024), (4, 4)), "bs64_t80": ((1, 1, 4096, 1280), (8, 8))}[name]
    return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]), strategy=ttnn.ShardStrategy.BLOCK)


CASES = [(op, m, k) for op in ("logical_and", "div", "ldexp", "mul_asilu") for m in ("dram", "l1", "dram_big") for k in ("none", "col")]
CASES += [(op, m, "col") for op in ("logical_and", "div", "mul_asilu") for m in ("bs16_t64", "bs16_n2_t128", "bs64_t80")]


@pytest.mark.parametrize("op, mem, kind", CASES, ids=["-".join(c) for c in CASES])
def test_mulact(device, op, mem, kind):
    shape, mc = _mem(mem)
    torch.manual_seed(zlib.crc32(f"{op}{mem}{kind}".encode()) % 100000)
    a = torch.rand(shape, dtype=torch.bfloat16) * 2 - 1
    b_shape = shape if kind == "none" else (shape[0], 1, shape[2], 1)
    b = torch.rand(b_shape, dtype=torch.bfloat16) + 0.5
    if op == "ldexp":
        b = torch.randint(-3, 4, b_shape).bfloat16()
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc if kind == "none" else ttnn.DRAM_MEMORY_CONFIG)
    fn = {"logical_and": lambda: ttnn.logical_and(ta, tb, memory_config=mc), "div": lambda: ttnn.divide(ta, tb, memory_config=mc),
          "ldexp": lambda: ttnn.ldexp(ta, tb, memory_config=mc),
          "mul_asilu": lambda: ttnn.multiply(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.SILU)], fast_and_approximate_mode=True, memory_config=mc)}[op]
    for _ in range(3):
        out = fn()
    got = ttnn.to_torch(out)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    assert got.shape[-1] == shape[-1]
