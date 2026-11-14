"""moe-perf: open the 4x8 mesh and hold it HOLD_S seconds (triage workflow test on a healthy device)."""
import os
import time

from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, p

md = open_mesh()
try:
    p("mesh open, holding")
    time.sleep(int(os.environ.get("HOLD_S", "150")))
finally:
    close_mesh(md)
