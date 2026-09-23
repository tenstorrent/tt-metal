"""In-model sweep of the B1/S512 SDPA chunk sizes and grid.

For each variant, rewrites attention.py from a saved copy, times the traced
forward with bench_nomask.py in a new process, and restores the file at the end.

Usage (tt-metal root): python models/demos/wormhole/bge_m3/tt/helper_scripts/sweep_b1_sdpa.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ATT = Path("models/demos/wormhole/bge_m3/tt/attention.py")
BENCH = Path(__file__).with_name("bench_nomask.py")
BACKUP = Path("/tmp/attention_b1_sdpa_sweep.py")

Q_CHUNKS = (32, 64, 128, 256)
K_CHUNKS = (256, 512)
GRIDS = ("8x8", "device")

shutil.copy(ATT, BACKUP)
src = BACKUP.read_text()
assert src.count("_SDPA_B1S512_Q_CHUNK = 64") == 1 and src.count("_SDPA_B1S512_K_CHUNK = 512") == 1
grid_line = "        grid = ttnn.CoreCoord(8, 8)\n"
assert src.count(grid_line) == 1

results = []
try:
    for grid in GRIDS:
        for q in Q_CHUNKS:
            for k in K_CHUNKS:
                s = src.replace("_SDPA_B1S512_Q_CHUNK = 64", "_SDPA_B1S512_Q_CHUNK = %d" % q)
                s = s.replace("_SDPA_B1S512_K_CHUNK = 512", "_SDPA_B1S512_K_CHUNK = %d" % k)
                if grid == "device":
                    s = s.replace(grid_line, "        pass\n")
                ATT.write_text(s)
                out = subprocess.run(
                    [sys.executable, "-u", str(BENCH), "1", "50"], capture_output=True, text=True, errors="replace"
                ).stdout
                m = re.search(r"BENCH .* mean=([0-9.]+)", out)
                res = float(m.group(1)) if m else None
                name = "q%-3d k%-3d %s" % (q, k, grid)
                results.append((name, res))
                print("RESULT %-20s %s" % (name, "%.3f" % res if res else "FAIL"), flush=True)
finally:
    shutil.copy(BACKUP, ATT)
    print("restored attention.py", flush=True)

print("SUMMARY")
for name, res in sorted((r for r in results if r[1]), key=lambda r: r[1])[:6]:
    print("  %-20s %.3f" % (name, res))
