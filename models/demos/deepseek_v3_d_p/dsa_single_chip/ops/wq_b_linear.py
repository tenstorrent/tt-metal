# SPDX-License-Identifier: Apache-2.0
"""Q2 wq_b linear as a generic op: [1,1,160,1536] bf16 x [1536,4096] bf8 (DRAM interleaved) -> [1,1,160,4096] bf16
(L1 interleaved).

DRAM-bound (6.7MB of weight). K is split in two halves: grid rows 0-4 take K[0:768), rows 5-9 K[768:1536). Within a
half, the 3072 weight tiles (column-major) are split over the 55 cores (weighted per column, COLW); both data-movement
RISCs of a core claim 4-tile chunks of its share dynamically, each on its own NoC. Partials are summed on the top core
that holds a column's last K tiles. Each core holds its K half of the activation (245KB), multicast in blocks over
four rects ({left, right} column group x {top, bottom}) and two NoC streams per rect.
See kernels/wqb_plan.h, kernels/wqb_in0.h and kernels/wqb_wstream.h."""
import ttnn
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops import _program as P
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops.noc_coords import grid_noc_coords

MT, KT, NT = 5, 48, 128
KH = KT // 2
# per logical column work weight on the p100a: columns 1-5 (middle of the left group) stream DRAM ~8% slower; fitted
# to equalize the per-column weight-stream finish times measured with uniform work
COLW = (0.97, 0.95, 0.955, 0.963, 0.958, 0.951, 1.053, 1.042, 1.056, 1.057, 1.045)


def _column_groups(device, ncols):
    """Logical column ranges whose NoC x coordinates are consecutive (a multicast rect cannot straddle a gap)."""
    coords = grid_noc_coords(device)
    xs = [coords[(i, 0)][0] for i in range(ncols)]
    groups, start = [], 0
    for i in range(1, ncols):
        if xs[i] != xs[i - 1] + 1:
            groups.append((start, i))
            start = i
    groups.append((start, ncols))
    return groups


def _plan_offsets(ncols, hrows, colw):
    """First weight unit (column-major over a K half) of each rank, plus the total: ranks of logical column x get a
    share proportional to colw[x] (uniform by default). Every rank keeps at least KH and at most 64 units."""
    total, ranks = NT * KH, ncols * hrows
    wts = [1.0 if colw is None or len(colw) <= r // hrows else float(colw[r // hrows]) for r in range(ranks)]
    acc, cum = 0.0, [0]
    for wv in wts:
        acc += wv
        cum.append(round(total * acc / sum(wts)))
    counts = [b - a for a, b in zip(cum, cum[1:])]
    assert min(counts) >= KH and max(counts) <= 64, counts
    return cum


def wq_b_linear(x, w, out, device, ncols: int = 11, nb: int = 6, ch: int = 4, win: int = 1, colw=COLW,
                fidelity=ttnn.MathFidelity.HiFi2):
    g = device.compute_with_storage_grid_size()
    assert g.y % 2 == 0 and ncols <= g.x and nb + 4 <= 16
    assert tuple(x.shape) == (1, 1, MT * 32, KT * 32) and tuple(w.shape) == (KT * 32, NT * 32)
    hrows = g.y // 2
    kb = KH // nb
    ncores = ncols * g.y
    cores = P.core_rect(P.split_work(ncores, device, ncores), device)
    coords = grid_noc_coords(device)
    groups = _column_groups(device, ncols)
    assert len(groups) == 2, "expects the grid to straddle one non-worker NoC column"
    col_split = groups[1][0]
    rects, senders = [], []
    for c0, c1 in groups:
        for half in (0, 1):
            y0, y1 = half * hrows, half * hrows + hrows - 1
            (xa, ya), (xb, yb) = coords[(c0, y0)], coords[(c1 - 1, y1)]
            rects += [xa, ya, xb, yb, (c1 - c0) * hrows]
            members = [c * g.y + y for c in range(c0, c1) for y in range(y0, y1 + 1)]
            # a stream's multicast starts at its NoC's rect corner (NOC0: min x/y, NOC1: max x/y); the sender sits
            # there so the packets stay inside the rect instead of wrapping around the torus
            senders += [members[0] if b % 2 == 0 else members[-1] for b in range(nb)]
    table = [coords[(i // g.y, i % g.y)][0] | (coords[(i // g.y, i % g.y)][1] << 16) for i in range(ncores)]
    plan = _plan_offsets(ncols, hrows, colw)
    per = max(b - a for a, b in zip(plan, plan[1:]))
    cbs = [
        P.cb(0, ttnn.bfloat16, MT * KH, cores),
        P.cb(1, ttnn.bfloat8_b, per, cores),
        P.cb(2, ttnn.bfloat16, 3 * MT, cores),
        P.cb(3, ttnn.bfloat16, 4 * MT, cores),
        P.cb(4, ttnn.bfloat16, 9 * MT, cores),
        P.cb(5, ttnn.bfloat8_b, per, cores),
        P.cb(7, ttnn.uint32, 1, cores, page_bytes=256),
    ]
    # semaphores: [0, nb) activation block flags, nb + fi partial counters (fi < 3), nb + 3 weight chunk claim counter
    sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(nb + 4)]
    ct = [KH, NT, ncols, hrows, g.y, ch, MT, nb, kb, col_split, win]
    common = [w.buffer_address(), x.buffer_address(), out.buffer_address()] + rects + senders + table + plan
    rd = P.reader(P.kernel_path("wqb_w_reader.cpp"), cores, ct + P.accessor_args(w) + P.accessor_args(x), [],
                  common=common)
    wr = P.writer(P.kernel_path("wqb_writer.cpp"), cores,
                  ct + P.accessor_args(w) + P.accessor_args(x) + P.accessor_args(out), [], common=common)
    cp = P.compute(P.kernel_path("wqb_compute.cpp"), cores, [KH, NT, ncols, hrows, MT, kb, ch], fidelity=fidelity,
                   common=plan)
    return P.run([x, w, out], P.program([rd, cp, wr], cbs, sems))
