# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hang-hunt hooks around every routed-experts call (loaded by tt/experts_ag.moe_hooks via GLM_HANG_HOOKS; the
campaign sets it from a queue entry's "hooks" / "hook_args"). GLM_HANG_HOOK_ARGS (JSON):

    {"layers": [20] | null (all), "every": 1 (every n-th call), "from_call": 0,
     "before": [action, ...], "after": [action, ...]}

Actions (enqueued in order; in a traced run the device ones are captured into the trace):
    {"op": "matmul", "m": 2048, "k": 4096, "n": 4096, "mem": "dram" | "l1", "dtype": "bf16" | "bfp8"}
    {"op": "sleep", "ms": 50}                 host gap (the device drains, then idles)
    {"op": "sync"}                            ttnn.synchronize_device
    {"op": "l1_garbage", "kb": 512}           write random data into free L1 (an L1 tensor, freed again)
    {"op": "dram_garbage", "mb": 64}          the same in DRAM
    {"op": "all_gather", "rows": 2048, "cols": 4096}   an extra fabric all-gather (ttnn.all_gather, cluster axis 1)
    {"op": "save_routing", "path": p, "max": 4}  this call's top-k -> p.L<layer>.c<n>.pt (the first max per layer)
    {"op": "py", "code": "..."}               anything: exec'd with ttnn, torch, ctx (mesh, layer, mode, call, x, idx,
                                              wts), y (after only) and a per-process dict S in scope
Each hook's device ops are logged as "[hook-op] layer L before|after: ops a..b" (a hung hook op maps too)."""

import json
import os
import time

import torch

import ttnn

ARGS = json.loads(os.environ.get("GLM_HANG_HOOK_ARGS", "{}"))
S = {}  # (per-process state: preallocated operands, ...)
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


def _rep(mesh, t, mem, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.L1_MEMORY_CONFIG if mem == "l1" else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _act(a, ctx, y):
    mesh, op = ctx["mesh"], a["op"]
    if op == "sleep":
        time.sleep(a.get("ms", 50) / 1e3)
    elif op == "sync":
        ttnn.synchronize_device(mesh)
    elif op == "matmul":
        m, k, n, mem = a.get("m", 2048), a.get("k", 4096), a.get("n", 4096), a.get("mem", "dram")
        key = ("mm", m, k, n, mem, a.get("dtype", "bf16"))
        if key not in S:
            g = torch.Generator().manual_seed(0)
            dt = DT[a.get("dtype", "bf16")]
            S[key] = (
                _rep(mesh, torch.randn(1, 1, m, k, generator=g), mem, dt),
                _rep(mesh, torch.randn(1, 1, k, n, generator=g) / k**0.5, mem, dt),
            )
        A, B = S[key]
        ttnn.deallocate(ttnn.matmul(A, B, memory_config=A.memory_config()))
    elif op in ("l1_garbage", "dram_garbage"):
        nbytes = a.get("kb", 512) * 1024 if op == "l1_garbage" else a.get("mb", 64) << 20
        rows = max(32, nbytes // (2 * 1024) // 32 * 32)
        t = _rep(mesh, torch.randn(1, 1, rows, 1024), "l1" if op == "l1_garbage" else "dram")
        ttnn.deallocate(t)
    elif op == "all_gather":
        key = ("ag", a.get("rows", 2048), a.get("cols", 4096))
        if key not in S:  # (each chip's share of the rows, gathered over the mesh row)
            S[key] = _rep(mesh, torch.randn(1, 1, key[1] // mesh.shape[1], key[2]), "dram")
        ttnn.deallocate(ttnn.all_gather(S[key], dim=2, cluster_axis=1, num_links=1))
    elif (
        op == "save_routing"
    ):  # this call's top-k (all chips' rows, chip order) -> <path>.L<layer>.c<n>.pt, n per layer
        n = S.setdefault(("saved", ctx["layer"]), 0)
        if n < a.get("max", 4):
            S[("saved", ctx["layer"])] = n + 1
            cat = lambda t: torch.cat(
                [ttnn.to_torch(d).reshape(-1, ttnn.to_torch(d).shape[-1]) for d in ttnn.get_device_tensors(t)]
            )  # noqa: E731
            ids, w = cat(ctx["idx"]).long(), cat(ctx["wts"]).float()
            torch.save(
                {"ids": ids, "w": w, "layer": ctx["layer"], "call": ctx["call"]}, f"{a['path']}.L{ctx['layer']}.c{n}.pt"
            )
            cnt = torch.bincount(ids.flatten(), minlength=288)
            print(
                f"[hook] saved layer {ctx['layer']} call {ctx['call']}: max rows/expert {int(cnt.max())}, "
                f"experts > 96 rows {int((cnt > 96).sum())}, rows per chip {cnt.view(8, -1).sum(1).tolist()}",
                flush=True,
            )
    elif op == "py":
        exec(a["code"], {"ttnn": ttnn, "torch": torch, "ctx": ctx, "y": y, "S": S, "time": time})
    else:
        raise ValueError(f"std_hooks: unknown action {op}")


def _run(when, ctx, y=None):
    layers, every = ARGS.get("layers"), ARGS.get("every", 1)
    acts = ARGS.get(when, [])
    if not acts or (layers is not None and ctx["layer"] not in layers):
        return
    if ctx["call"] < ARGS.get("from_call", 0) or ctx["call"] % every:
        return
    op0 = ttnn._ttnn.get_device_operation_id()
    for a in acts:
        _act(a, ctx, y)
    op1 = ttnn._ttnn.get_device_operation_id() - 1
    if op1 >= op0:
        print(f"[hook-op] layer {ctx['layer']} {when}: ops {op0}..{op1}", flush=True)


def before_moe(ctx):
    _run("before", ctx)


def after_moe(ctx, y):
    _run("after", ctx, y)
