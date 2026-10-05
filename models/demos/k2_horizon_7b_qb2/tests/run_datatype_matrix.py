"""Serialize device jobs; stop on runtime errors so evidence can be investigated."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/datatype_sweep")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--qualification", action="store_true")
    args = p.parse_args()
    names = args.ids or json.loads((DOC / "coarse_matrix.json").read_text())
    if args.smoke and args.qualification:
        p.error("Smoke and full-model qualification are distinct regimes")
    suffix = "_smoke" if args.smoke else "_qualification" if args.qualification else ""
    records = []
    for name in names:
        output = DOC / ("qualifications" if args.qualification else "runs") / f"{name}{suffix}.json"
        output.parent.mkdir(exist_ok=True)
        if output.exists() and json.loads(output.read_text()).get("completed_at"):
            print("REUSE", output, flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "models.demos.k2_horizon_7b_qb2.tests.run_datatype_candidate",
            "--config",
            str(DOC / "configs" / f"{name}.json"),
            "--output",
            str(output),
        ]
        if args.smoke:
            command.append("--smoke")
        if args.qualification:
            command.extend(["--continuation", "--quality", "--repeats", "9"])
        start = time.time()
        print("START", name, flush=True)
        with output.with_suffix(".log").open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(config_id=name, command=command, exit_code=proc.returncode, seconds=time.time() - start))
        (DOC / ("execution_" + names[0] + suffix + ".json")).write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, proc.returncode, flush=True)
        if proc.returncode:
            print(output.with_suffix(".log").read_text()[-5000:], flush=True)
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
