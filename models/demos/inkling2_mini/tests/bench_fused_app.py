# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

# Per layer-application residual path, E2 (tt/residual.py + model glue) vs E10 fused, attention / MoE replaced by
# fixed rows. Reports device-op count (graph capture), traced replay time and output agreement per app kind.
# usage: python bench_app.py [out.jsonl]
import collections
import json
import sys
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused.attnres import FusedAttnRes, FusedRowOps
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv
from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate, FusedQKVGate
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm
from models.demos.inkling2_mini.tt.model import _QKVGate

C = 2304
EPS = 1e-6
SKIP = {"ReshapeViewDeviceOperation"}
MOCK = bool(__import__("os").environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
OUT = sys.argv[1] if len(sys.argv) > 1 else None
g = torch.Generator().manual_seed(7)


def rnd(*s, scale=1.0):
    return torch.randn(*s, generator=g) * scale


def weights(is_global):
    w = {
        "attn_res_q": torch.nn.functional.pad(
            rnd(3 if is_global else 1, C, scale=2), (0, 0, 0, 32 - (3 if is_global else 1))
        )[None, None],
        "mlp_res_q": torch.nn.functional.pad(rnd(1, C, scale=2), (0, 0, 0, 31))[None, None],
        "post_attn_conv": rnd(1, 1, 4, C, scale=0.5),
        "post_mlp_conv": rnd(1, 1, 4, C, scale=0.5),
    }
    for gg in ("q", "k", "v", "mlp"):
        w[f"msg_gate.{gg}.down"] = rnd(1, 1, 2 * C, 64, scale=0.05).bfloat16()
        w[f"msg_gate.{gg}.forget"] = rnd(1, 1, 64, C, scale=0.2)
        w[f"msg_gate.{gg}.gain"] = rnd(1, 1, 64, C, scale=0.2)
    return w


def count(gr):
    c = collections.Counter()
    for n in gr:
        if n["node_type"] == "function_start":
            nm = n["params"].get("name", "")
            if (nm.endswith("DeviceOperation") or nm.startswith("GenericOp")) and nm not in SKIP:
                c[nm] += 1
    return c


def host(t):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    rows = R.RowOps(dev)
    frows = FusedRowOps(dev)
    dram = ttnn.DRAM_MEMORY_CONFIG

    def row(t, dtype):
        return ttnn.from_torch(t.reshape(1, 1, -1, C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

    results = []
    for is_global in (False, True):
        w = weights(is_global)
        nq = 3 if is_global else 1
        norms = [
            ttnn.from_torch(1 + rnd(1, 1, 1, C, scale=0.1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(3 if is_global else 1)
        ]
        e2 = dict(
            attn=R.AttnRes(rows, w, "attn_res_q", nq),
            mlp=R.AttnRes(rows, w, "mlp_res_q", 1),
            ca=R.ShortConv(rows, w, "post_attn_conv"),
            cm=R.ShortConv(rows, w, "post_mlp_conv"),
            qkv=_QKVGate(rows, w),
            mg=R.MessageGate(rows, w, "mlp"),
        )
        fu = dict(
            attn=FusedAttnRes(frows, w, "attn_res_q", nq),
            mlp=FusedAttnRes(frows, w, "mlp_res_q", 1),
            ca=FusedShortConv(rows, w, "post_attn_conv"),
            cm=FusedShortConv(rows, w, "post_mlp_conv"),
            qkv=FusedQKVGate(rows, w),
            mg=FusedMessageGate(rows, w, "mlp"),
            norm=FusedRowNorm(dev, norms),
        )
        ring = {k: (e2[k].alloc_state(0), fu[k].alloc_state(0)) for k in ("ca", "cm")}
        depth_states = [rnd(1, C, scale=s) for s in (1.0, 2.0)]  # 2 sealed states -> depth 3 / 4
        h = row(rnd(1, C, scale=1.5), ttnn.bfloat16)
        A = row(rnd(1, C), ttnn.bfloat16)  # attention output stand-in
        Y = row(rnd(1, C), ttnn.bfloat16)  # MoE output stand-in
        msgs = (row(rnd(3, C), ttnn.float32), row(rnd(1, C), ttnn.float32))
        e2_stack = None
        for k, s in enumerate(depth_states):
            e2_stack = rows.seal(e2_stack, row(s, ttnn.bfloat16), k)
            frows.seal(None, row(s, ttnn.bfloat16), k)
        n0 = len(depth_states)

        def rms(x, wt):
            return rows.rms(x, weight=wt, eps=EPS, dtype=ttnn.bfloat16)

        def norm_inputs(x, with_msg):
            if is_global:
                return [rms(ttnn.slice(x, [0, 0, i, 0], [1, 1, i + 1, C]), norms[i]) for i in range(3)]
            if not with_msg:
                return [rms(x, norms[0])]
            xn = rms(x, norms[0])
            return [ttnn.slice(xn, [0, 0, i, 0], [1, 1, i + 1, C]) for i in range(3)]

        def app(M, fused, with_msg):
            stack, n = (frows.stack if fused else e2_stack), n0
            rep = 3 if (fused and with_msg and not is_global) else None
            x = M["attn"](stack, h, n + 1, split=False, **({"repeat_rows": rep} if rep else {}))
            if with_msg:
                if not is_global and not fused:
                    x = ttnn.repeat(x, ttnn.Shape([1, 1, 3, 1]))
                x = M["qkv"](x, msgs[0])
            qkv = M["norm"](x) if fused else norm_inputs(x, with_msg)
            ca, cm = ring["ca"][fused], ring["cm"][fused]
            if fused:
                resid = M["ca"].conv_add(A, ca, 1, h, ttnn.float32)
            else:
                a = ttnn.typecast(M["ca"](A, ca, 1), ttnn.float32)
                resid = ttnn.add(ttnn.typecast(h, ttnn.float32), a, dtype=ttnn.float32)
            m = M["mlp"](stack, resid, n + 1)[0]
            if with_msg:
                m = M["mg"](m, msgs[1])
            if fused:
                out = M["cm"].conv_add(Y, cm, 1, resid, ttnn.bfloat16)
            else:
                y = M["cm"](Y, cm, 1)
                out = ttnn.add(resid, ttnn.typecast(y, ttnn.float32), dtype=ttnn.bfloat16)
            return [out, m] + qkv

        for with_msg in (False, True):
            rec = dict(kind=("global" if is_global else "local") + ("+msg" if with_msg else ""))
            outs = {}
            for fused in (False, True):
                M = fu if fused else e2
                app(M, fused, with_msg)  # compile
                ttnn.synchronize_device(dev)
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                app(M, fused, with_msg)
                c = count(ttnn.graph.end_graph_capture())
                rec[f"ops_{'fused' if fused else 'e2'}"] = sum(c.values())
                if MOCK:  # no command queue: op counts only
                    rec[f"ops_detail_{'fused' if fused else 'e2'}"] = dict(c.most_common(8))
                    continue
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                o = app(M, fused, with_msg)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                first = [host(t) for t in o]
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                second = [host(t) for t in o]
                t0 = time.perf_counter()
                for _ in range(200):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                us = (time.perf_counter() - t0) / 200 * 1e6
                ttnn.release_trace(dev, tid)
                tag = "fused" if fused else "e2"
                rec[f"ops_{tag}"] = sum(c.values())
                rec[f"us_{tag}"] = round(us, 1)
                rec[f"bitequal_{tag}"] = all(torch.equal(a, b) for a, b in zip(first, second))
                rec[f"ops_detail_{tag}"] = dict(c.most_common(8))
                outs[tag] = first
            if not MOCK:
                rec["pcc_out_vs_e2"] = min(pcc(a, b) for a, b in zip(outs["e2"], outs["fused"]))
            print(json.dumps(rec), flush=True)
            results.append(rec)
    ttnn.close_device(dev)
    if OUT:
        with open(OUT, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
