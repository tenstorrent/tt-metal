# SPDX-License-Identifier: Apache-2.0
"""Single-chip probe of the small control-flow ops the device-resident spec round needs (broadcast compare/mul, int ops, mod via bitwise_and, matmul-with-triangular, argmax tie order, typecast, where, embedding lookups)."""
import json
import os
import traceback

import torch

import ttnn

OUT = os.environ.get("PROBE_OUT", "/tmp/devctl.jsonl")


def rec(**kw):
    s = json.dumps(kw, default=str)
    print("DEVCTL " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def tt(dev, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, device=dev, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def host(t):
    return ttnn.to_torch(t).float()


def case(name, fn):
    try:
        ok, detail = fn()
        rec(name=name, ok=bool(ok), detail=detail)
    except Exception as e:  # noqa
        rec(name=name, ok=False, error=str(e).strip().splitlines()[0][:200] if str(e).strip() else repr(e))
        traceback.print_exc()


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        U = 4
        iota = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 1024, 1)
        slot = torch.randint(0, 1024, (U, 1, 1, 8)).float()

        def bc_eq():
            r = ttnn.eq(tt(dev, iota), tt(dev, slot))
            h = host(r)[:, :, :1024, :8]
            ref = (iota == slot).float()
            return torch.equal(h, ref), tuple(h.shape)

        case("bidirectional_broadcast_eq_fp32", bc_eq)

        def bc_mul_col():
            ring = torch.randn(U, 1, 1024, 128)
            keep = (torch.rand(U, 1, 1024, 1) > 0.5).float()
            r = ttnn.multiply(tt(dev, ring), tt(dev, keep))
            return torch.allclose(host(r)[..., :128], ring * keep), "width-1 broadcast multiply"

        case("mul_width1_broadcast", bc_mul_col)

        def bc_add_row():
            s = torch.randn(U, 1, 64, 1056)
            m = torch.randn(U, 1, 1, 1056)
            r = ttnn.add(tt(dev, s), tt(dev, m))
            return torch.allclose(host(r), s + m, atol=1e-5), "row broadcast add"

        case("add_row_broadcast", bc_add_row)

        def mod1024():
            p = torch.randint(0, 70000, (1, 1, 1, 8), dtype=torch.int32)
            r = ttnn.bitwise_and(tt(dev, p, ttnn.int32), 1023)
            return torch.equal(host(r)[..., :8].int(), (p & 1023)), "bitwise_and int32"

        case("bitwise_and_int32", mod1024)

        def floor_mod():
            p = torch.randint(0, 70000, (1, 1, 1, 8)).float()
            f = ttnn.floor(ttnn.multiply(tt(dev, p), 1.0 / 1024))
            r = ttnn.subtract(tt(dev, p), ttnn.multiply(f, 1024.0))
            return torch.equal(host(r)[..., :8], p % 1024), "fp32 floor mod"

        case("fp32_floor_mod", floor_mod)

        def tri():
            ok = (torch.rand(U, 1, 1, 7) > 0.3).float()
            bad = 1 - ok
            Tm = torch.triu(torch.ones(7, 7))
            c = ttnn.matmul(tt(dev, bad), tt(dev, Tm.reshape(1, 1, 7, 7)))
            pref = ttnn.eq(c, 0.0)
            k = ttnn.sum(pref, dim=3, keepdim=True)
            ref = torch.cumprod(ok, -1).sum(-1, keepdim=True)
            h = host(k)[:, :, :1, :1]
            return torch.equal(h, ref), "tri matmul leading-ones count (note padded cols)"

        case("leading_ones_count", tri)

        def tie_argmax():
            v = torch.zeros(1, 1, 1, 8)
            v[..., 2] = 5.0
            v[..., 5] = 5.0
            r = ttnn.argmax(tt(dev, v, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
            return int(ttnn.to_torch(r).flatten()[0]) == 2, int(ttnn.to_torch(r).flatten()[0])

        case("argmax_first_on_tie", tie_argmax)

        def where_():
            c = (torch.rand(1, 1, 32, 32) > 0.5).float()
            a, b = torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)
            r = ttnn.where(tt(dev, c), tt(dev, a), tt(dev, b))
            return torch.allclose(host(r), torch.where(c > 0, a, b), atol=1e-5), "where"

        case("where", where_)

        def emb():
            tab = torch.randn(70000, 128).bfloat16()
            idx = torch.randint(0, 70000, (1, 32), dtype=torch.int32)
            r = ttnn.embedding(
                tt(dev, idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                tt(dev, tab, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                layout=ttnn.TILE_LAYOUT,
            )
            return torch.equal(host(r).reshape(32, 128), tab[idx[0]].float()), "embedding lookup uint32"

        case("embedding_lookup", emb)

        def tc():
            p = torch.randint(0, 70000, (1, 1, 1, 32)).float()
            r = ttnn.typecast(tt(dev, p), ttnn.uint32)
            r2 = ttnn.typecast(tt(dev, p), ttnn.int32)
            return torch.equal(host(r)[..., :32], p) and torch.equal(host(r2)[..., :32], p), "fp32->uint32/int32"

        case("typecast_fp32_to_ints", tc)

        def assign_():
            a = torch.randn(1, 1, 32, 128).bfloat16()
            dst = tt(dev, torch.zeros(1, 1, 32, 128).bfloat16(), ttnn.bfloat16)
            ttnn.assign(tt(dev, a, ttnn.bfloat16), dst)
            return torch.equal(host(dst), a.float()), "assign bf16"

        case("assign_bf16", assign_)

        def assign_f32():
            a = torch.randn(1, 1, 32, 32)
            dst = tt(dev, torch.zeros(1, 1, 32, 32))
            try:
                ttnn.assign(tt(dev, a), dst)
                return torch.allclose(host(dst), a), "assign fp32"
            except Exception as e:  # noqa
                return False, str(e).splitlines()[0][:150]

        case("assign_fp32", assign_f32)

        def rm_assign_int():
            a = torch.arange(32, dtype=torch.int32)
            dst = tt(dev, torch.zeros(32, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            try:
                ttnn.assign(tt(dev, a, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), dst)
                return torch.equal(ttnn.to_torch(dst).int(), a), "assign int32 RM"
            except Exception as e:  # noqa
                return False, str(e).splitlines()[0][:150]

        case("assign_int32_rm", rm_assign_int)

        def scatter_mm():
            S = torch.zeros(U, 1, 1024, 8)
            slots = torch.randint(0, 1024, (U, 8))
            for u in range(U):
                for t in range(8):
                    S[u, 0, slots[u, t], t] = 1.0
            new = torch.randn(U, 1, 8, 128)
            r = ttnn.matmul(tt(dev, S), tt(dev, new))
            return torch.allclose(host(r), S @ new, atol=1e-4), "fp32 scatter matmul"

        case("scatter_matmul_fp32", scatter_mm)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
