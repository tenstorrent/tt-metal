"""In-model sweep of the S512 SDPA chunk plan and max_cores_per_head_batch for one batch.

For each variant, rewrites attention.py from a saved copy (an early return in
_sdpa_chunks_for_seq_len, and the max_cores_per_head_batch value), times the
traced forward with bench_nomask.py in a new process, and restores the file.

Usage (tt-metal root): python sweep_sdpa_inmodel.py <batch> [mcphb list] [q list] [k list]
  e.g. sweep_sdpa_inmodel.py 8 8 64,128,256 256,512
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ATT = Path("models/demos/wormhole/bge_m3/tt/attention.py")
BENCH = Path(__file__).with_name("bench_nomask.py")
BACKUP = Path("/tmp/attention_sdpa_sweep_%s.py" % sys.argv[1])

batch = int(sys.argv[1])
mcphbs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8").split(",")]
Q_LIST = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "128,256,512").split(",")]
K_LIST = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "128,256,512").split(",")]
CHUNKS = [(q, k) for q in Q_LIST for k in K_LIST]

shutil.copy(ATT, BACKUP)
src = BACKUP.read_text()
head = "def _sdpa_chunks_for_seq_len(seq_len, batch_size=None, data_parallel=False):\n"
mc = '        kwargs["max_cores_per_head_batch"] = 8\n'
clamp = "        k_chunk = min(k_chunk, 256)\n"
assert src.count(head) == 1 and src.count(mc) == 1 and src.count(clamp) == 1

results = []
try:
    for m in mcphbs:
        for q, k in CHUNKS:
            s = src.replace(
                head, head + "    if seq_len == 512 and batch_size == %d:\n        return %d, %d\n" % (batch, q, k)
            )
            s = s.replace(mc, '        kwargs["max_cores_per_head_batch"] = %d\n' % m)
            s = s.replace(clamp, "        pass\n")
            ATT.write_text(s)
            out = subprocess.run(
                [sys.executable, "-u", str(BENCH), str(batch), "30"], capture_output=True, text=True, errors="replace"
            ).stdout
            hit = re.search(r"BENCH .* mean=([0-9.]+)", out)
            res = float(hit.group(1)) if hit else None
            name = "q%-3d k%-3d mcphb%-2d" % (q, k, m)
            results.append((name, res))
            print("RESULT %-20s %s" % (name, "%.3f" % res if res else "FAIL"), flush=True)
finally:
    shutil.copy(BACKUP, ATT)
    print("restored attention.py", flush=True)

print("SUMMARY batch=%d" % batch)
for name, res in sorted((r for r in results if r[1]), key=lambda r: r[1])[:6]:
    print("  %-20s %.3f" % (name, res))
