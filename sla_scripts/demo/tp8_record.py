import json
import re
import sys
from pathlib import Path

RUNS = Path.home() / "teja/runs"
tag = sys.argv[1]
log = (RUNS / f"tp8dp4_step_{tag}.log").read_text().splitlines()

outs, tsu, ttft, exit_code = {}, None, None, None
for line in log:
    m = re.search(r"\[User (\d+)\] (.*)", line)
    if m:
        outs[int(m.group(1))] = m.group(2).strip()
    m = re.search(r"128th token decode time: ([\d.]+)ms \[([\d.]+) t/s/u, ([\d.]+) t/s\]", line)
    if m:
        tsu = dict(ms=float(m.group(1)), tsu=float(m.group(2)), ts=float(m.group(3)))
    m = re.search(r"Average Time to First Token \(TTFT\): ([\d.]+)ms", line)
    if m:
        ttft = float(m.group(1))
    m = re.search(r"=== EXIT (\d+)", line)
    if m:
        exit_code = int(m.group(1))

(RUNS / f"outputs_{tag}.json").write_text(json.dumps(outs))
base_tag = sys.argv[2] if len(sys.argv) > 2 else "baseline"
base_file = RUNS / f"outputs_{base_tag}.json"
match = "n/a"
if base_file.exists() and tag != base_tag:
    base = {int(k): v for k, v in json.loads(base_file.read_text()).items()}
    same = sum(base.get(k) == v for k, v in outs.items())
    match = f"{same}/{len(base)}"

row = dict(tag=tag, exit=exit_code, users=len(outs), ttft_ms=ttft, **(tsu or {}), greedy_match_vs_baseline=match)
with open(RUNS / "tp8_steps.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
