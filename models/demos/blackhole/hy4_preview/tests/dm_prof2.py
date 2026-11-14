"""reduction-over-sources microbench (traced)"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dk_dev import timed_traced
from dm_block import *

import ttnn

full, md = open_row()


def T(shape, dtype=ttnn.bfloat16):
    return to_mesh(md, (torch.randn(8 * shape[0], *shape[1:]).abs() + 0.5), dtype, ttnn.TILE_LAYOUT)


def bench(name, f):
    try:
        us = timed_traced(md, f)
    except Exception as e:
        try:
            ttnn.end_trace_capture(md, 0, cq_id=0)
        except Exception:
            pass
        print("RES " + json.dumps({"op": name, "error": str(e).strip().splitlines()[0][:150]}), flush=True)
        return
    try:
        print("RES " + json.dumps({"op": name, "us": us}), flush=True)
    except Exception as e:
        print("RES " + json.dumps({"op": name, "error": str(e).strip().splitlines()[0][:150]}), flush=True)


try:
    for S in (1,):
        a = T([9, S, 8, 544])
        a32 = T([9, S, 32, 544])
        b = T([S, 9, 8, 544])
        c = T([S, 8, 9, 544])
        w = T([9, S, 8, 1])
        a1 = T([1, 9, 8 * S, 544])
        bench("sum dim0 [9,S,8,544]", lambda: ttnn.sum(a, dim=0, keepdim=True))
        bench("sum dim0 [9,S,32,544]", lambda: ttnn.sum(a32, dim=0, keepdim=True))
        bench("sum dim1 [S,9,8,544]", lambda: ttnn.sum(b, dim=1, keepdim=True))
        bench("sum dim2 [S,8,9,544]", lambda: ttnn.sum(c, dim=2, keepdim=True))
        bench("sum dim1 [1,9,8S,544]", lambda: ttnn.sum(a1, dim=1, keepdim=True))
        bench("max dim0 [9,S,8,1]", lambda: ttnn.max(w, dim=0, keepdim=True))
        bench("max dim0 [9,S,8,544]", lambda: ttnn.max(a, dim=0, keepdim=True))
        bench("moreh_sum dim0", lambda: ttnn.moreh_sum(a, dim=0, keepdim=True))
        bench("moreh_sum dim1 b", lambda: ttnn.moreh_sum(b, dim=1, keepdim=True))
        bench(
            "sum dim0 fp32 acc cfg",
            lambda: ttnn.sum(
                a,
                dim=0,
                keepdim=True,
                compute_kernel_config=ttnn.init_device_compute_kernel_config(
                    md.arch(), math_fidelity=ttnn.MathFidelity.LoFi
                ),
            ),
        )
        # chain of adds over 9 slices pre-split: 8 adds
        parts = [T([1, S, 8, 544]) for _ in range(9)]

        def chain():
            acc = parts[0]
            for p in parts[1:]:
                acc = ttnn.add(acc, p)
            return acc

        bench("8 adds [1,S,8,544]", chain)
        # slice cost variants
        x = T([9, S, 8, 576])
        bench("slice aligned col 512:513", lambda: ttnn.slice(x, [0, 0, 0, 512], [9, S, 8, 513]))
        bench("slice aligned col 512:544", lambda: ttnn.slice(x, [0, 0, 0, 512], [9, S, 8, 544]))
        bench("slice aligned col 544:545", lambda: ttnn.slice(x, [0, 0, 0, 544], [9, S, 8, 545]))
        bench("slice col 513:514", lambda: ttnn.slice(x, [0, 0, 0, 513], [9, S, 8, 514]))
        bench("slice o 0:512", lambda: ttnn.slice(x, [0, 0, 0, 0], [9, S, 8, 512]))
        bench("slice dim0 0:1", lambda: ttnn.slice(x, [0, 0, 0, 0], [1, S, 8, 576]))
        bench("softmax dim0 [9,S,8,1]", lambda: ttnn.softmax(w, dim=0))
        s32 = T([9, S, 8, 32])
        s256 = T([1, S, 8, 256])
        sx = T([1, S, 8, 544])
        bench("softmax dim-1 [9,S,8,32]", lambda: ttnn.softmax(s32, dim=-1))
        bench("softmax dim-1 [S,8,256]", lambda: ttnn.softmax(s256, dim=-1))
        bench("max dim-1 [S,8,256]", lambda: ttnn.max(s256, dim=-1, keepdim=True))
        bench("sum dim-1 [S,8,256]", lambda: ttnn.sum(s256, dim=-1, keepdim=True))
        bench("permute [9,S,8,32]->(1,2,3,0)", lambda: ttnn.permute(s32, (1, 2, 3, 0)))
        bench("permute [1,9,8,544] (0,2,1,3)", lambda: ttnn.permute(a1, (0, 2, 1, 3)))
        bench("transpose [1,S,8,544] -2,-1", lambda: ttnn.transpose(sx, -2, -1))
finally:
    close_row(full, md)
