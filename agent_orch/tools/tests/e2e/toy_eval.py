#!/usr/bin/env python3
"""Toy eval command: 'latency' is the number in src/speed.txt; two cases. Writes $DREAM_RESULT."""
import json, os
from pathlib import Path

v = float(Path("src/speed.txt").read_text())
res = {
    "valid": v > 0,
    "error": None if v > 0 else "speed must be positive",
    "cases": {"small": {"value": v, "check": 1.0}, "large": {"value": v * 2.0 + 0.01, "check": 1.0}},
}
Path(os.environ["DREAM_RESULT"]).write_text(json.dumps(res))
print("toy eval:", res)
