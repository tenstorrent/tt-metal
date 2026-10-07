# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass: does binary_ng's block-pack section slow the op that follows it? Each test runs a
sharded bf16 residual add (16 or 128 tiles per core; the block section with main's block pack under the head's rule, main's
program with EB_R3_NO_BLOCK) and then one common op once, as a model's op order does; the follower's program is the same
under both settings and its op code differs from the add's, so the reduce times it alone. CI only."""
import pytest
import torch
import ttnn

from test_eb_blk4 import _mem


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _t(device, shape, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=dtype, layout=layout, device=device, memory_config=mc)


FOLLOWERS = ["relu_l1", "gelu_sharded", "typecast_bfp8_sharded", "layer_norm", "rms_norm", "softmax", "matmul", "matmul_small",
             "unary_dram", "transpose"]
CASES = [(pre, f) for pre in ("hs8_t16", "hs8_t128") for f in FOLLOWERS]


@pytest.mark.parametrize("pre, follower", CASES, ids=[f"{p}-{f}" for p, f in CASES])
def test_seq(device, pre, follower):
    shape, mc = _mem(pre)
    a = _t(device, shape, mc=mc)
    b = _t(device, shape, mc=mc)
    x = None
    if follower == "relu_l1":
        x = _t(device, (1, 1, 256, 1024), mc=ttnn.L1_MEMORY_CONFIG)
    elif follower in ("layer_norm", "rms_norm", "softmax"):
        x = _t(device, (1, 1, 128, 1024))
        w = _t(device, (1, 1, 1, 1024))
    elif follower == "matmul":
        x, y = _t(device, (1, 1, 256, 1024)), _t(device, (1, 1, 1024, 1024))
    elif follower == "matmul_small":
        x, y = _t(device, (1, 1, 32, 1024)), _t(device, (1, 1, 1024, 256))
    elif follower == "unary_dram":
        x = _t(device, (1, 1, 1024, 1024))
    elif follower == "transpose":
        x = _t(device, (1, 1, 256, 512))
    ttnn.synchronize_device(device)
    c = ttnn.add(a, b, memory_config=mc)
    if follower == "relu_l1":
        out = ttnn.relu(x)
    elif follower == "gelu_sharded":
        out = ttnn.gelu(c, memory_config=mc)
    elif follower == "typecast_bfp8_sharded":
        out = ttnn.typecast(c, ttnn.bfloat8_b, memory_config=mc)
    elif follower == "layer_norm":
        out = ttnn.layer_norm(x, weight=w)
    elif follower == "rms_norm":
        out = ttnn.rms_norm(x, weight=w)
    elif follower == "softmax":
        out = ttnn.softmax(x, dim=-1)
    elif follower in ("matmul", "matmul_small"):
        out = ttnn.matmul(x, y)
    elif follower == "unary_dram":
        out = ttnn.silu(x)
    elif follower == "transpose":
        out = ttnn.transpose(x, -2, -1)
    ttnn.synchronize_device(device)
    for t in [a, b, c, out] + ([x] if x is not None else []):
        ttnn.deallocate(t)
