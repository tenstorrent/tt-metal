# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``decode_step_context`` against host references: the integer terms exactly, the RoPE rows within a bf16 ulp
of float64 ``cos``/``sin`` (and equal to the device chain they replace up to its own fp32 angle error).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_step_context.py
"""

import pytest
import torch

import ttnn
from models.experimental.deepseek_v41_flash.tt.decode.step_context import (
    StepContextBuilder,
    eval_term,
    rope_table,
    term,
)


def _inv_freq(rd: int, base: float) -> torch.Tensor:
    half = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    return half.repeat_interleave(2).reshape(1, 1, 1, rd).to(torch.float32)


@pytest.mark.parametrize("pos", (0, 1, 5, 127, 1023, 1024, 4095, 65537, 1048575))
def test_step_context(device, pos):
    rd, max_pos, window, topk = 64, 1 << 20, 128, 512
    pkt_host = [0] * 16
    pkt_host[0], pkt_host[1], pkt_host[2], pkt_host[3], pkt_host[4], pkt_host[5] = 77, pos % window, pos, 3, 640, 9
    pkt = ttnn.from_torch(
        torch.tensor(pkt_host, dtype=torch.int32).reshape(1, 1, 1, 16),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    families = {"main": (_inv_freq(rd, 10000.0), 1.0), "compress": (_inv_freq(rd, 160000.0), 1.25)}
    tables = {
        name: ttnn.from_torch(
            rope_table(inv, scale, max_pos),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for name, (inv, scale) in families.items()
    }
    ints = {
        "token": ([1, 1], [term(0)], ttnn.uint32),
        "sliding": ([1], [term(1)], ttnn.int32),
        "lengths": (
            [1, 1, 1, 3],
            [
                term(2, pre_add=1, clamp_max=window),
                term(2, pre_add=1, mode="div", div=4, clamp_max=topk),
                term(4),
            ],
            ttnn.int32,
        ),
        "win_slot": ([1], [term(2, mode="mod", div=4, add_field=5)], ttnn.int32),
        "entry": ([1], [term(2, mode="div", div=128)], ttnn.int32),
        "index_pos": ([1], [term(2, pre_add=1, mode="div", div=4, mul=4, post_add=-1)], ttnn.int32),
        "dense_entry": ([1], [term(2, mode="div", div=4, add_field=3, add_mul=topk)], ttnn.int32),
        "zero": ([1, 1, 1, 1], [term(-1, mul=0)], ttnn.int32),
    }
    b = StepContextBuilder(pkt, tables)
    for key, (shape, terms, dtype) in ints.items():
        b.add_int(key, shape, terms, dtype)
    b.add_rope("main", "main", 2)
    b.add_rope("compress", "compress", 2)
    b.add_rope("win", "compress", 2, -1)
    out = b.run()

    for key, (shape, terms, dtype) in ints.items():
        got = ttnn.to_torch(out[key]).to(torch.int64).reshape(-1).tolist()
        want = [eval_term(t, pkt_host) for t in terms]
        assert list(out[key].shape) == shape, (key, out[key].shape)
        assert out[key].dtype == dtype, key
        assert got == want, (key, got, want)

    for key, family, p in (("main", "main", pos), ("compress", "compress", pos), ("win", "compress", pos - 1)):
        inv, scale = families[family]
        angle = p * inv.to(torch.float64).reshape(-1)
        cos_want = (torch.cos(angle) * scale).to(torch.bfloat16).to(torch.float64)
        sin_want = (torch.sin(angle) * scale).to(torch.bfloat16).to(torch.float64)
        cos, sin, neg_sin = (ttnn.to_torch(t).to(torch.float64).reshape(-1) for t in out[key])
        ulp = 2.0**-7 * scale
        for got, want, name in ((cos, cos_want, "cos"), (sin, sin_want, "sin"), (-neg_sin, sin_want, "neg_sin")):
            err = (got - want).abs().max().item()
            assert err <= ulp, (key, name, p, err)
        assert torch.equal(neg_sin, -sin), key
        for t in out[key]:
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 and list(t.shape) == [1, 1, 1, rd]
