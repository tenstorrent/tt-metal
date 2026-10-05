"""Serial real-HF controls at each batch-dependent attention policy boundary."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


def main():
    (DOC / "logs").mkdir(parents=True, exist_ok=True)
    source = ROOT / "models/demos/k2_horizon_7b_qb2/tt/fused_decoder.py"
    result = {"runtime_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "commands": [], "completed": False}
    for capacity, batches in [
        (4096, [8, 32]),
        (4128, [8, 32]),
        (8192, [3, 4]),
        (8224, [3, 4]),
        (16384, [2]),
        (16416, [2]),
    ]:
        name = f"stock_limit_{capacity}"
        cmd = [
            sys.executable,
            "-m",
            "models.demos.k2_horizon_7b_qb2.tests.probe_stock_batches",
            "--context",
            str(capacity),
            "--batches",
            *map(str, batches),
            "--output",
            str(DOC / (name + ".json")),
        ]
        print("START", name, flush=True)
        with (DOC / "logs" / (name + ".log")).open("w") as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        result["commands"].append({"argv": cmd, "exit_code": p.returncode})
        (DOC / "review_boundary_commands.json").write_text(json.dumps(result, indent=2) + "\n")
        if p.returncode:
            raise SystemExit(p.returncode)
        print("END", name, flush=True)
    result["completed"] = True
    (DOC / "review_boundary_commands.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
