"""B50 (oom15) unit on a 1x8 slot: (1) the iHC collapse transient: stock ttnn.sum(dim=1) allocates a tile-padded 402,653,184 B output (512*32*6144*4) = 50.3 MB/bank at n=512, the B49-style 4 slice-mul + 3 adds does not;
(2) HY4_B50_MASKSHARE shared causal mask is bit-exact vs a per-layer mask and holds 1 mask instead of 22; (3) chain guard arithmetic. Prints RES lines. The stock collapse is run LAST (it is expected to OOM in the constrained state)."""
import json
import os
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4IHCGen, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

N, D = 512, 6144
MB = 1 << 20


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def view(md):
    v = dram_view(md)
    return dict(
        alloc_mb=round(v["alloc_per_bank"] / MB, 2),
        free_mb=round(v["free_per_bank"] / MB, 2),
        largest_mb=round(v["largest_free_per_bank"] / MB, 2),
    )


def b49_collapse(x4, pre):
    n, Dd = x4.shape[2], x4.shape[3]
    acc = None
    for i in range(4):
        xi = ttnn.slice(x4, [0, i, 0, 0], [1, i + 1, n, Dd])
        pi = ttnn.slice(pre, [0, i, 0, 0], [1, i + 1, n, 1])
        t = ttnn.multiply(xi, pi)
        if acc is None:
            acc = t
        else:
            a2 = ttnn.add(acc, t)
            ttnn.deallocate(acc)
            ttnn.deallocate(t)
            acc = a2
    return acc


def snap(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def fill_to(md, target_free_mb):
    """allocate DRAM filler (per-device [1,1,32k,4096] bf16 tensors) until free <= target_free_mb per bank; returns the list."""
    keep = []
    while True:
        v = dram_view(md)
        free_mb = v["free_per_bank"] / MB
        if free_mb <= target_free_mb + 0.5:
            break
        step_mb = min(32.0, free_mb - target_free_mb)  # per bank
        rows = max(32, int(step_mb * MB * 8 / (4096 * 2)) // 32 * 32)
        keep.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        )
    return keep


def main():
    full, md = open_row()
    try:
        res(event="start", **view(md))
        g = torch.Generator().manual_seed(1)
        fn8 = torch.randn(8, 4 * D, generator=g) * 0.01
        ihc = Hy4IHCGen(md, fn8, torch.randn(8, generator=g) * 0.1, torch.tensor([0.5, 0.5]), 1e-6, 1e-6)
        x = upload(md, torch.randn(1, 4, N, D, generator=g) * 2.0, ttnn.float32)
        pre, post = ihc.pre_post(x)
        base = view(md)

        # ---- (2) shared mask bit-exact + footprint
        os.environ["HY4_B50_MASKSHARE"] = "1"
        from models.demos.blackhole.hy4_preview.tt import hy4_indexer as IX

        n, T, start = 512, 2048, 1536
        sc = upload(md, torch.randn(1, 1, n, T), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        m = torch.zeros(n, T)
        m[:, start : start + n] = torch.zeros(n, n).masked_fill(
            torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf")
        )
        per_layer = [upload(md, m.reshape(1, 1, n, T), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) for _ in range(22)]
        a_pl = view(md)
        ref = snap(ttnn.add(sc, per_layer[0]))
        for t in per_layer:
            ttnn.deallocate(t)
        b0 = view(md)
        outs = []
        for _ in range(22):
            o = ttnn.add(sc, IX.shared_causal_mask(md, n, T, start))
            outs.append(snap(o))
            ttnn.deallocate(o)
        a_sh = view(md)
        exact = all(torch.equal(o, ref) for o in outs)
        res(
            event="maskshare",
            bit_exact=bool(exact),
            per_layer22_mb=round(a_pl["alloc_mb"] - base["alloc_mb"] - 0.0, 2),
            shared_mb=round(a_sh["alloc_mb"] - b0["alloc_mb"], 3),
            entries=len(IX._SHARED_MASK),
        )
        IX.drop_shared_masks()
        ttnn.deallocate(sc)

        # ---- (3) chain guard arithmetic (CPU stubs)
        from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

        def guard(free, largest, bpt, fixed, cap=12288, reserve="128"):
            import models.demos.blackhole.hy4_preview.tt.hy4_chain as C

            orig = C.dram_view
            C.dram_view = lambda md_: dict(
                free_per_bank=int(free * MB), largest_free_per_bank=int(largest * MB), alloc_per_bank=0
            )
            os.environ["HY4_B50_CHAIN_GUARD"] = "1"
            os.environ["HY4_B50_CHAIN_RESERVE_MB"] = reserve
            st = types.SimpleNamespace(
                hybrid_cap=cap, md=None, pp=types.SimpleNamespace(chain_bpt=bpt, chain_fixed=fixed * MB)
            )
            try:
                Hy4Engine._b50_chain_guard(st)
            finally:
                C.dram_view = orig
            return st.hybrid_cap

        res(
            event="guard",
            free309=guard(309, 308, 9600, 11),
            free187_blocks=guard(187, 186, 9600, 11),
            free100=guard(100, 99, 9600, 11),
            free40=guard(40, 39, 9600, 11),
        )

        # ---- (1) collapse transient under a constrained DRAM
        keep = fill_to(md, 40.0)
        res(event="constrained", **view(md))
        ref_c = None
        h = b49_collapse(x, pre)
        ref_c = snap(h)
        res(event="b49_collapse_ok", **view(md))
        ttnn.deallocate(h)
        try:
            h2 = ihc.collapse(x, pre)
            d = (snap(h2) - ref_c).abs().max().item()
            res(event="stock_collapse", ok=True, maxdiff_vs_b49=d)
        except RuntimeError as e:
            res(
                event="stock_collapse",
                ok=False,
                err=str(e).split("Out of Memory:")[-1][:260].strip().replace("\n", " "),
            )
        for t in keep:
            ttnn.deallocate(t)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
