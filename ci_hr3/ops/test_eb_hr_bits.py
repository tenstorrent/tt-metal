# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Headroom round, eltwise binary (#58818), bits on the merged head: binary_ng's block sections (8x8 block shards, 64 tiles
per core) on every bf16 bit pattern. a holds all 65536 patterns 64 times; b is a seeded permutation of the same patterns per
copy, so each a pattern meets 64 b patterns, and a second b holds the special set (zeros, infinities, NaN, extremes,
denormals). Each output's raw bits are hashed to $EB_DET_OUT; in-process repeats must agree. Not for merge."""
import hashlib
import json
import os

import pytest
import torch
import ttnn

SHAPE = (1, 1, 2048, 2048)
SPECIAL = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F7F, 0xFF7F, 0x0080, 0x8080, 0x0001, 0x8001, 0x007F, 0x807F]


def _bits(t16):
    return t16.view(torch.bfloat16).reshape(SHAPE)


def _a():
    return _bits(torch.arange(65536, dtype=torch.int32).to(torch.int16).repeat(64))


def _b(kind):
    if kind == "perm":
        g = torch.Generator().manual_seed(7)
        return _bits(torch.cat([torch.randperm(65536, generator=g) for _ in range(64)]).to(torch.int16))
    s = torch.tensor(SPECIAL, dtype=torch.int32).to(torch.int16)
    return _bits(s.repeat(SHAPE[2] * SHAPE[3] // len(SPECIAL)))


@pytest.mark.parametrize("bkind", ["perm", "special"])
@pytest.mark.parametrize("op", ["mul", "add", "sub"])
def test_bits_all(device, op, bkind):
    mem = ttnn.create_sharded_memory_config(SHAPE, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)
    ta, tb = _a(), _b(bkind)
    hs = []
    for _ in range(2):
        a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        if op == "mul":
            o = ttnn.multiply(a, b, fast_and_approximate_mode=True, memory_config=mem)
        elif op == "add":
            o = ttnn.add(a, b, memory_config=mem)
        else:
            o = ttnn.subtract(a, b, memory_config=mem)
        hs.append(hashlib.sha1(ttnn.to_torch(o).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16])
        for t in (a, b, o):
            ttnn.deallocate(t)
    with open(os.environ.get("EB_DET_OUT", "/tmp/eb_det.jsonl"), "a") as fh:
        fh.write(json.dumps({"case": f"all_{op}_{bkind}", "hashes": hs, "in_process_diffs": [int(hs[0] != hs[1])]}) + "\n")
