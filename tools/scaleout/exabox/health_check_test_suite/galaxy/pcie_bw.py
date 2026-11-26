#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-chip host to chip and chip to host PCIe bandwidth, data checked, each chip on its own 1x1 mesh opened with no fabric.
diag/resets.py run_end runs it as its own step after fabric_open.py has closed its mesh (no submesh of a fabric mesh: the parent's close
throws while a child's command queue is in use). One chip in turn: open_mesh_device(MeshShape(1, 1), physical_device_ids=[i]),
a host to chip write of --mb MB of seeded uint32 into a DRAM tensor (copy_host_to_device_tensor) and the chip to host read
back (from_device), the faster of 2 each way, every word of the read compared with the write, then the mesh is closed.
(A shard of the 4x8 mesh reads back at 0.15 GB/s and to_torch of it returns chip 0's buffer: not a per-chip measurement.)
One 'pcie' row per chip (event, chip, mb, h2d_gbs, d2h_gbs, ok, bad) in --rows; a chip that fails to open or transfer is its
row's 'bad'. A final 'pcie_done' row (ok, chips, s). Then quiet_clock.pcie_lines judges the rows (--floor-gbs,
--warn-frac): its PCIE FAIL/WARN lines and summary printed and written to pcie.txt beside --rows.
--beat holds 'pcie <i>' while chip i runs (a transfer that hangs is the step's HANG, the beat naming the chip) and 'close'.
Exit 0 all rows written."""
import argparse
import json
import os
import sys

# the sibling programs, also when run or imported through a link
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import socket
import time

from quiet_clock import PCIE_FLOOR_GBS, PCIE_WARN_FRAC, bdf_label, pci_id_bdf, pcie_lines

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--mb", type=int, default=64, help="MB written and read back per chip")
ap.add_argument("--floor-gbs", type=float, default=PCIE_FLOOR_GBS, help="FAIL under this GB/s either way")
ap.add_argument("--warn-frac", type=float, default=PCIE_WARN_FRAC, help="WARN under this fraction of the box median")
a = ap.parse_args()
rows = open(a.rows, "a")


def row(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": "pcie", **kw}
    rows.write(json.dumps(kw) + "\n")
    rows.flush()
    os.fsync(rows.fileno())
    print(json.dumps(kw), flush=True)


def beat(s):
    with open(a.beat + ".tmp", "w") as f:
        f.write(f"{s} {time.time():.0f}\n")
    os.replace(a.beat + ".tmp", a.beat)


def chip_name(m, i):
    """'<BDF> (tray T, ASIC A, device id D)' as fabric_open.py names a chip, else 'chip i'."""
    try:
        did, bdf = pci_id_bdf(list(m.get_device_ids())[0], "")  # the 1x1 mesh holds this one chip
    except Exception:  # noqa: BLE001
        return f"chip {i}"
    if not bdf:
        return f"chip {i} (device id {did}, not in the map)"
    return bdf_label(bdf, did)


def first_line(e):
    return next((ln.strip() for ln in str(e).splitlines() if ln.strip()), "")


beat("open")
import torch  # noqa: E402

import ttnn  # noqa: E402

t0 = time.time()
n = ttnn.get_num_devices()
words = max(1024, (a.mb << 20) // 4 // 1024 * 1024)
nb = words * 4
x = torch.randint(
    -(2**31),
    2**31 - 1,
    (1, 1, words // 1024, 1024),
    dtype=torch.int32,
    generator=torch.Generator().manual_seed(4803),
)
host = ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
gb = lambda ts: round(nb / min(ts) / 1e9, 3) if ts else None  # noqa: E731
rs = []
for i in range(n):
    beat(f"pcie {i}")
    m, name, h2d, d2h, bad = None, f"chip {i}", [], [], ""
    try:
        m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=16384, physical_device_ids=[i])
        name = chip_name(m, i)
        # zeros first (untimed), so a timed write that moves nothing fails the check
        dev = ttnn.from_torch(
            torch.zeros_like(x),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(m)
        for _ in range(2):
            t1 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(host, dev)
            ttnn.synchronize_device(m)
            h2d.append(time.perf_counter() - t1)
        for _ in range(2):
            t1 = time.perf_counter()
            back = ttnn.from_device(dev)
            d2h.append(time.perf_counter() - t1)
        got = ttnn.to_torch(back).contiguous().view(torch.int32).reshape(x.shape)  # uint32 compared as int32 bits
        ne = int((got != x).sum())
        if ne:
            at = (got != x).flatten().nonzero()[0].item()
            g, w = got.flatten()[at].item() & 0xFFFFFFFF, x.flatten()[at].item() & 0xFFFFFFFF
            bad = f"read back {ne} of {words} words wrong, first at word {at}: got {g:#010x} want {w:#010x}"
    except Exception as e:  # noqa: BLE001
        bad = f"{type(e).__name__}: {first_line(e)[:300]}"
    close = ""
    try:
        if m is not None:
            ttnn.close_mesh_device(m)
    except Exception as e:  # noqa: BLE001
        close = f"close: {first_line(e)[:300]}"  # one 'pcie' row per chip: a close error folds into that chip's row, never a second one
    bad = "; ".join(x for x in (bad, close) if x)
    rs.append(
        dict(event="pcie", chip=name, mb=round(nb / 2**20), h2d_gbs=gb(h2d), d2h_gbs=gb(d2h), ok=not bad, bad=bad)
    )
    row(**rs[-1])
beat("close")
row(event="pcie_done", ok=True, chips=n, s=round(time.time() - t0, 1))
lines = pcie_lines(rs, a.floor_gbs, a.warn_frac, socket.gethostname().split(".")[0])
with open(os.path.join(os.path.dirname(os.path.abspath(a.rows)), "pcie.txt"), "w") as f:
    f.write("".join(ln + "\n" for ln in lines))
print("\n".join(lines))
sys.stdout.flush()
os._exit(0)  # every mesh is closed: no tensor destructor at the interpreter's exit
