#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One 4x8 mesh open with fabric, one small CCL on each mesh axis, then the close, of this box's own chips after the C9 reset.
The open is the one HyperFlow and H3 run: set_fabric_config(FABRIC_1D, STRICT_INIT, router payload 8192),
open_mesh_device(MeshShape(4, 8), l1_small_size=16384), no mesh graph descriptor (auto discovery). 32 chips open as 4x8 (8 chips
per row, the Galaxy's 8x4 device_topology), any other count (--known-dropped-tray) as 1xN and the CCL runs on its one axis.
The CCL is all_gather_async (global semaphores, Linear, cluster_axis, 2 links) on axis 1 (rows of 8) then axis 0 (columns of 4);
every chip holds one tile of its own index and must read back every index of its row / column. The links phase (diag/links.py)
runs it after its delay and reads the PCIe tree after it.
--beat holds 'open', 'launch 0' (the open), 'done 0', 'launch 1'/'done 1' (axis 1), 'launch 2'/'done 2' (axis 0),
'dbi pass N', 'close' for fl_watch (this process never returns while a chip hangs).
--dbi-wall-s S > 0: after the CCL, on the same open mesh, the DRAM byte/DBI pattern check
(dram_dbi.dbi_check, G3b's check: every chip's DRAM banks written, copied on the device and read back) for S s, so DRAM
is checked right after the C9 reset too, as the Deployment Test does after each reset; its row has event "dbi", its
finding lines (chunk_sched.dbi_lines: NUMERIC FAIL per chip by BDF, tray and worst DRAM bank, NUMERIC WARN when it
raised, one summary line) go to dbi.txt beside --rows.
--noc-wall-s S > 0: after that, on the same open mesh, the seeded per-chip NOC data check
(noc_walk.noc_check): every chip's Tensix cores writing L1 and DRAM destinations and reading them back, for S s. Its
row has event "nocwalk", its finding lines (noc_walk.nocwalk_lines: NUMERIC FAIL per chip by BDF, tray and ASIC, one
common-fault WARN, one summary line) go to noc.txt beside --rows.
Exit 0 opened, gathered and closed, 1 anything failed: the row's error reads 'MESH 4x8 <stage>: <first error line>'."""
import argparse
import json
import os
import sys

# the sibling programs, also when run or imported through a link
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import time

import noc_walk
from chunk_sched import dbi_lines, tray_of
from quiet_clock import pci_id_bdf

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--label", default="open")
ap.add_argument(
    "--map",
    default="",
    help="run.sh's map.txt ('<BDF> <device id>' rows): names a chip reading wrong data by BDF when sysfs does not",
)
ap.add_argument(
    "--dbi-wall-s", type=float, default=0, help="DRAM byte/DBI pattern check after the CCL, wall cap s; 0 off"
)
ap.add_argument(
    "--noc-wall-s", type=float, default=0, help="seeded per-chip NOC data check after the DRAM check, wall cap s; 0 off"
)
a = ap.parse_args()
rows = open(a.rows, "a")


def row(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": a.label, **kw}
    rows.write(json.dumps(kw) + "\n")
    rows.flush()
    os.fsync(rows.fileno())
    print(json.dumps(kw), flush=True)


def chip_id(m, i):
    """(device id D, BDF or ''); raises when the mesh's ids do not read."""
    return pci_id_bdf(list(m.get_device_ids())[i], a.map)  # row-major in the mesh, as get_device_tensors


def chip_name(m, i, r, c):
    """'chip i (mesh rR cC)' plus, when the chip's PCI address reads, '<BDF> (tray T, ASIC A, device id D)'."""
    try:
        did, bdf = chip_id(m, i)
    except Exception:  # noqa: BLE001
        return f"chip {i} (mesh r{r} c{c})"
    if not bdf:
        return f"chip {i} (mesh r{r} c{c}, device id {did}, not in the map)"
    return (
        f"{bdf[5:]} (tray {tray_of(bdf)}, ASIC {int(bdf.split(':')[-2], 16) & 0x0F}, device id {did}, mesh r{r} c{c})"
    )


def chips_of(m, R, C):
    """mesh index -> dict(pci, tray, dev) for every chip in the mesh."""
    chips = {}
    for i in range(R * C):
        did, bdf = chip_id(m, i)
        chips[i] = dict(pci=bdf or None, tray=tray_of(bdf) if bdf else None, dev=did)
    return chips


def beat(s):
    with open(a.beat + ".tmp", "w") as f:
        f.write(f"{s} {time.time():.0f}\n")
    os.replace(a.beat + ".tmp", a.beat)


def open_check(event, label, txt, run, lines_of):
    """One check on the open mesh after the C9 reset (--dbi-wall-s, --noc-wall-s): run(say, torch, chips, nb) returns its
    dict, lines_of(r, chips) its finding lines (written to txt beside --rows). Its own error is a NUMERIC WARN line and
    the row's error, never the open's: the mesh still closes and the open's verdict stands."""
    t1 = time.time()
    beat(f"{event} pass 0")
    say = lambda s: print(f"{time.strftime('%F %T')} C9 {label} {s}", flush=True)  # noqa: E731
    chips = {}
    try:
        import torch

        chips = chips_of(m, R, C)
        g = m.dram_grid_size()  # MeshDevice has no get_devices() in Python
        r = run(say, torch, chips, int(g.x * g.y))
    except Exception as e:  # noqa: BLE001
        r = dict(error=f"{type(e).__name__}: {str(e)[:300]}", chips={})
        say(f"error: {r['error']}")
    r = dict(r, start=round(t1, 3), end=round(time.time(), 3))
    row(event=event, **r)
    try:
        lines = lines_of(r, chips)
        with open(os.path.join(os.path.dirname(os.path.abspath(a.rows)), txt), "w") as f:
            f.write("".join(ln + "\n" for ln in lines))
        for ln in lines:
            print(ln, flush=True)
    except Exception as e:  # noqa: BLE001
        say(f"WARNING: findings not written: {type(e).__name__}: {str(e)[:300]}")


def dbi_run(say, torch, chips, nb):
    """--dbi-wall-s: G3b's DRAM byte/DBI pattern check."""
    import dram_dbi as dd

    say(
        f"DRAM byte/DBI patterns on {len(chips)} chips {noc_walk.WHY}: {dd.ROWS * dd.ROW_B >> 20} MB per chip, "
        f"{dd.HOPS} copies on the device per pass, wall cap {a.dbi_wall_s:g} s, banks {nb}"
    )
    return dd.dbi_check(ttnn, torch, m, chips, a.dbi_wall_s, nb, beat=beat, say=say)


def noc_run(say, torch, chips, nb):
    """--noc-wall-s: the seeded per-chip NOC data check."""
    say(
        f"NOC data check on {len(chips)} chips {noc_walk.WHY}: {noc_walk.SLICE} B per core per pass, "
        f"wall cap {a.noc_wall_s:g} s, banks {nb}"
    )
    return noc_walk.noc_check(ttnn, torch, m, chips, a.noc_wall_s, nb=nb, beat=lambda s: beat(f"noc {s}"))


beat("open")
import ttnn  # noqa: E402

t0 = time.time()
stage = "open"
R = C = 0
m = None
try:
    n = ttnn.get_num_devices()
    R, C = (4, 8) if n == 32 else (1, n)
    beat("launch 0")
    kw = {}
    if hasattr(ttnn, "FabricReliabilityMode"):
        kw["reliability_mode"] = ttnn.FabricReliabilityMode.STRICT_INIT
    else:
        print(
            "WARNING: this ttnn has no FabricReliabilityMode: the fabric opens in its default reliability mode",
            flush=True,
        )
    if hasattr(ttnn, "FabricRouterConfig"):
        rc = ttnn.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = 8192
        kw["router_config"] = rc
    else:
        print(
            "WARNING: this ttnn has no FabricRouterConfig: the fabric payload stays at its default, not 8192",
            flush=True,
        )
    stage = "set_fabric_config"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, **kw)
    stage = f"open_mesh_device {R}x{C}"
    m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C), l1_small_size=16384)
    t_open = time.time() - t0
    beat("done 0")
    ccl = {}
    if n > 1:
        import torch  # noqa: E402

        g = m.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        sems = lambda k: [ttnn.create_global_semaphore(m, cores, 0) for _ in range(k)]  # noqa: E731
        ttnn.synchronize_device(m)
        # one tile per chip holding its own index, sharded [R, C] over the mesh
        full = torch.arange(R * C, dtype=torch.float32).reshape(R, C, 1, 1).expand(R, C, 32, 32).contiguous()
        x = ttnn.from_torch(
            full.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(m, mesh_shape=(R, C), dims=(0, 1)),
        )
        for k, (ax, dim, size) in enumerate(((1, 3, C), (0, 2, R)), start=1):
            if size < 2:
                continue
            stage = f"all_gather_async axis {ax}"
            beat(f"launch {k}")
            t1 = time.time()
            out = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=sems(2),
                barrier_semaphore=sems(1)[0],
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                cluster_axis=ax,
            )
            ttnn.synchronize_device(m)
            beat(f"done {k}")
            stage = f"all_gather_async axis {ax} data"
            bad = []
            for i, d in enumerate(ttnn.get_device_tensors(out)):
                r, c = divmod(i, C)
                v = ttnn.to_torch(d).float()

                def rd(y, x):  # an unreadable element is a wrong value of this chip, not a crash of the check
                    try:
                        return v[0, 0, y, x].item()  # as read: 1.5, NaN or inf compare unequal to the integer wanted
                    except Exception:  # noqa: BLE001
                        return -1

                want = [r * C + j for j in range(C)] if ax == 1 else [j * C + c for j in range(R)]
                # five points of every gathered tile, not only its first element
                for oy, ox in ((0, 0), (31, 31), (0, 31), (31, 0), (15, 17)):
                    got = (
                        [rd(oy, j * 32 + ox) for j in range(C)] if ax == 1 else [rd(j * 32 + oy, ox) for j in range(R)]
                    )
                    if got != want:
                        bad.append((chip_name(m, i, r, c), f"got {got} want {want} (tile element {oy},{ox})"))
                        break
            ttnn.deallocate(out)
            if bad:
                names = ", ".join(b[0] for b in bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
                raise RuntimeError(f"{len(bad)} of {n} chips read wrong data: {names}; first {bad[0][1]}")
            ccl[f"axis{ax}_s"] = round(time.time() - t1, 1)
        ttnn.deallocate(x)
    if a.dbi_wall_s > 0:
        stage = "dbi"
        open_check("dbi", "DRAM", "dbi.txt", dbi_run, lambda r, _: dbi_lines(r, 0, at=noc_walk.WHY, name="C9")[0])
    if a.noc_wall_s > 0:
        stage = "noc"
        open_check("nocwalk", "NOC", "noc.txt", noc_run, noc_walk.nocwalk_lines)
    stage = "close_mesh_device"
    ttnn.close_mesh_device(m)
    m = None
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    beat("close")
except Exception as e:  # noqa: BLE001
    first = next((ln.strip() for ln in str(e).splitlines() if ln.strip()), "")
    row(
        ok=False,
        s=round(time.time() - t0, 1),
        stage=stage,
        mesh=f"{R}x{C}",
        error=f"MESH {R}x{C} {stage}: {type(e).__name__}: {first[:700]}",
    )
    try:  # the verdict row is written: leave the mesh and the fabric closed, not to the interpreter's exit
        if m is not None:
            ttnn.close_mesh_device(m)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:  # noqa: BLE001
        pass
    sys.exit(1)
row(ok=True, chips=n, mesh=f"{R}x{C}", open_s=round(t_open, 1), s=round(time.time() - t0, 1), **ccl)
