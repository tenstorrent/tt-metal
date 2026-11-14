"""per-op cost of the merge ops (traced, prefix differences) on the 1x8 row, tile-output layout [8*S,32,608] per column"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dk_dev import timed_traced
from dm_block import close_row, open_row, to_mesh

import ttnn

NC, V = 8, 512
full, md = open_row()


def steps_with(st, x, sr, S):
    def put(k, f):
        return lambda: st.__setitem__(k, f())

    yield "reshape", put("xr", lambda: ttnn.reshape(x, [NC, S, 32, 608]))
    yield "concat", put("x9", lambda: ttnn.concat([st["xr"], sr], dim=0))
    yield "slice m", put("m", lambda: ttnn.slice(st["x9"], [0, 0, 0, V], [NC + 1, S, 32, V + 1]))
    yield "slice l", put("l", lambda: ttnn.slice(st["x9"], [0, 0, 0, V + 32], [NC + 1, S, 32, V + 33]))
    yield "max", put("M", lambda: ttnn.max(st["m"], dim=0, keepdim=True))
    yield "sub", put("d", lambda: ttnn.subtract(st["m"], st["M"]))
    yield "exp", put("e", lambda: ttnn.exp(st["d"]))
    yield "mul l*e", put("w", lambda: ttnn.multiply(st["l"], st["e"]))
    yield "mul x9*w", put("xw", lambda: ttnn.multiply(st["x9"], st["w"]))
    yield "sum", put("acc", lambda: ttnn.sum(st["xw"], dim=0, keepdim=True))
    yield "slice den", put("den", lambda: ttnn.slice(st["acc"], [0, 0, 0, V + 64], [1, S, 32, V + 65]))
    yield "recip", put("r", lambda: ttnn.reciprocal(st["den"]))
    yield "slice o", put("o", lambda: ttnn.slice(st["acc"], [0, 0, 0, 0], [1, S, 32, V]))
    yield "mul o*r", put("out", lambda: ttnn.multiply(st["o"], st["r"]))


try:
    for S in (1, 2):
        x = to_mesh(md, torch.randn(8, NC * S, 32, 608).abs() + 0.5, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sr = to_mesh(md, torch.randn(8, S, 32, 608).abs() + 0.5, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        names = [n for n, _ in steps_with({}, x, sr, S)]
        prev = 0
        for k in range(1, len(names) + 1):

            def call(k=k):
                st = {}
                for n, f in list(steps_with(st, x, sr, S))[:k]:
                    f()

            us = timed_traced(md, call)
            print(
                "RES " + json.dumps({"S": S, "step": names[k - 1], "cum_us": us, "delta_us": round(us - prev, 1)}),
                flush=True,
            )
            prev = us
finally:
    close_row(full, md)
