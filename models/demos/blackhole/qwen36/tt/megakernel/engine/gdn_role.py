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
    role = g2_role.GdnRole(dev, coords, NG, B, b.sems, tgt, prod, gen, kdir=KDIR)
    role.prepare()
    r = role.role
    print(f"[gdn24] G2 role: {NG} GDN layers {list(b.gdn_layers)}, B {B}, out_ready {'multicast' if rects else 'Q root'}, CB ids {r['info']['cb_ids'][0]}.."
          f"{r['info']['cb_ids'][-1]} ({r['info']['cb_bytes'] // 1024} KB L1 per GDN core), arg block "
          f"{r['info']['ab_words']} words, kernels {KDIR}", flush=True)
    return dict(kernels=r["kernels"], cbs=r["cbs"], io=r["io"], check=lambda: role.check("gdn24"), role=role)
