"""MKA_GDN24 hook: the GDN role (G2, spec §17 / §20 / v1.33) on the 24 reserved GDN cores (rows 8-9, row-major: head
i // 4, value block i % 4). program.py calls gdn_role(b, coords) once at setup; it returns the kernels / CBs to add to
the engine program, the io tensors for generic_op, and `check` -- call it ONCE right after the FIRST engine run
(+ synchronize), before any other run or trace replay (those advance the GDN state): per GDN layer, out / ring / next
window bit-identical vs the stock r13 gdn_spec_step on an identical copy of the layer state.

The step kernels are megakernel/gdn/step_kernels (absolute path; never the tree's ttnn op sources). Bring-up scope:
in-proj / prefix / wo stay engine stand-ins; the role's `pre` input is produced before the pass (gdn/u1 producers).
Interface (from b): dev, gdn_layers (the pass's GDN-type layers, in order), sems (in_ready, attn_kv,
vb0, vb1 on the GDN cores; out_ready on kgrid), out_ready_target (the Q root's NoC xy: it waits for 24 * (g + 1) at GDN
layer g, then resets) or out_rects (MKA_GDN_WAIT: +1 multicast to every engine core), fphys. Users B = V12_R / 4
(MKA_GDN_B overrides). CB ids 0..40 (<= 44); no program semaphores; no per-core runtime args (arg block kind
gdn = 3, its address = common runtime arg slot 3 of the GDN kernels). env: MKA_GDN_KDIR overrides the kernels dir."""
import os
import sys

_GDN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gdn")
KDIR = os.environ.get("MKA_GDN_KDIR", os.path.join(_GDN, "step_kernels"))
os.environ.setdefault("E1_KDIR", KDIR)
sys.path.insert(0, os.path.join(_GDN, "u1"))


def _gb(b, coords):
    """E1_GB=1 with MK_GDN_INPROJ: dict(gin_addr, gin_map = [(GDN core index, slot)] per in-proj tile, dbg) for build_engine."""
    gg = getattr(b, "gb_gin", None)
    if os.environ.get("E1_GB", "0") != "1":
        return None
    assert gg is not None, "E1_GB needs MK_GDN_INPROJ (G-a lands the in-proj tiles in the GDN cores' staging CB 41)"
    ix = {(c.x, c.y): i for i, c in enumerate(coords)}
    tl = gg["tiles"]
    assert [t[0] for t in tl] == list(range(len(tl))), "G-a tile list not contiguous"
    return dict(gin_addr=gg["gin_addr"], gin_map=[(ix[(x, y)], sl) for _, x, y, sl in tl], dbg=gg["dbg"])


def gdn_role(b, coords):
    import ttnn
    import g2_role

    NG = len(b.gdn_layers)
    if NG == 0:
        return dict(kernels=[], cbs=[], io=[])
    dev = b.dev
    grid = dev.compute_with_storage_grid_size()
    ex = {(c.x, c.y) for c in coords}
    # producer cores (separate programs before the pass; any core off the GDN rows)
    prod = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in ex]
    # users of the GDN step = token rows / T (u8: 32 rows -> 8, u1: 4 rows -> 1); b.NU is the attention users per device
    B = int(os.environ.get("MKA_GDN_B", "0")) or max(1, int(os.environ.get("V12_R", "32")) // 4)
    rects = getattr(b, "out_rects", None)
    tgt = list(rects) if rects else b.out_ready_target
    gen = os.path.join(os.getcwd(), f"gdn24gen_b{B}")
    # the stock-op reference (priming + check) only without device-profiler zones: the stock op's program overflows the
    # kernel config buffer with them (MKA_GDN_CHECK forces either way)
    ref = os.environ.get("MKA_GDN_CHECK", "0" if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" else "1") == "1"
    # attn_kv: the Q root raises it at every attention layer's start and SDPA end (MKA_GDN_ATTNKV); the GDN cores drain
    # it to 2 * (attention layers) before their program-end reset
    attn_n = b.NL - NG
    role = g2_role.GdnRole(dev, coords, NG, B, b.sems, tgt, prod, gen, kdir=KDIR, ref=ref, attn_n=attn_n,
                           gdnout=getattr(b, "gdnout", None),   # MKA_GDNOUT: the GDN out_proj via the wo unit
                           gin_k=getattr(b, "gin_k", None),     # MK_GIN_PERCORE: per-core in_ready targets
                           gb=_gb(b, coords))   # E1_GB: the type-10 prefix in the role, from G-a's staging tiles
    # MKA_GDN_HOLD (bench): HOLD users of every step, e.g. "0,1,3,4,6" = 3 live users (2, 5, 7) -- the padded-row shape
    hold = [int(x) for x in os.environ.get("MKA_GDN_HOLD", "").split(",") if x.isdigit() and int(x) < B]
    role.prepare(hold)
    r = role.role
    print(f"[gdn24] G2 role: {NG} GDN layers {list(b.gdn_layers)}, B {B}, out_ready {'multicast' if rects else 'Q root'}, CB ids {r['info']['cb_ids'][0]}.."
          f"{r['info']['cb_ids'][-1]} ({r['info']['cb_bytes'] // 1024} KB L1 per GDN core), arg block "
          f"{r['info']['ab_words']} words, reference {'on' if ref else 'off (profiler)'}, kernels {KDIR}", flush=True)
    return dict(kernels=r["kernels"], cbs=r["cbs"], io=r["io"], check=lambda: role.check("gdn24"), role=role)
