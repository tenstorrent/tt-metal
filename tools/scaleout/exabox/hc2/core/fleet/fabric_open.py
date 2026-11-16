#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One fabric open and close of this box's own chips after the C9 reset,
through the open path the load stages use (chunk_load.py: set_fabric_config FABRIC_1D, open_mesh_device). 32 chips open
as the 4x8 mesh, any other count (--known-dropped-tray) as 1xN. No kernels run. hc/lib/fleet.sh fl_open_after runs it after
its delay (FLEET_OPEN_DELAY_S) and reads the PCIe tree after it; the HSC under-voltage after-read follows it.
--beat holds 'open', 'launch 0' (the open), 'done 0', 'close' for fl_watch (this process never returns while a chip hangs).
Exit 0 opened and closed, 1 the open or close failed (the error is the last line)."""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--label", default="open")
a = ap.parse_args()
rows = open(a.rows, "a")


def row(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": a.label, **kw}
    rows.write(json.dumps(kw) + "\n")
    rows.flush()
    os.fsync(rows.fileno())
    print(json.dumps(kw), flush=True)


def beat(s):
    with open(a.beat + ".tmp", "w") as f:
        f.write(f"{s} {time.time():.0f}\n")
    os.replace(a.beat + ".tmp", a.beat)


beat("open")
import ttnn  # noqa: E402

t0 = time.time()
try:
    n = ttnn.get_num_devices()
    R, C = (4, 8) if n == 32 else (1, n)
    beat("launch 0")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C))
    t_open = time.time() - t0
    beat("done 0")
    ttnn.synchronize_device(m)
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    beat("close")
except Exception as e:  # noqa: BLE001
    row(ok=False, s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)
row(ok=True, chips=n, mesh=f"{R}x{C}", open_s=round(t_open, 1), s=round(time.time() - t0, 1))
