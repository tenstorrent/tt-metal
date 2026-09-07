"""Perf 3 -- the headline cells, many reads, for the drift-cancelling A/B (ab.sh)."""
import os

os.environ.setdefault("RMS_READS", "9")
os.environ["RMS_GUARD_SUBSET"] = "1"
import runpy  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
src = (HERE / "guard_set.py").read_text()
# keep only the cells this A/B is about
keep = (
    "INT RESIDENT D42  FOCUS",
    "ROW_MAJOR BAND HEIGHT-shrd",
    "INT RESIDENT combine=F",
    "BLOCK native slot tree",
    "WIDTH native flat combine",
)
out, in_cases = [], False
for line in src.split("\n"):
    if line.startswith("CASES = ["):
        in_cases = True
        out.append(line)
        continue
    if in_cases and line.startswith("]"):
        in_cases = False
        out.append(line)
        continue
    if in_cases:
        if any(k in line for k in keep):
            out.append(line)
        continue
    out.append(line)
exec(compile("\n".join(out), str(HERE / "guard_set.py"), "exec"), {"__name__": "__main__", "__file__": str(HERE / "guard_set.py")})
