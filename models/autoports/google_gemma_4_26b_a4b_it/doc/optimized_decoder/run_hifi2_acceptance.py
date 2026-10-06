# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize real-input stress and maximum-context gates for prefill QKV HiFi2."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
RUNTIME = DOC.parent.parent / "tt/optimized_decoder.py"
EXPECTED = "b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = DOC / "minimal_hifi2_acceptance_v6"
    directory.mkdir(exist_ok=True)
    journal = directory / "commands.json"
    assert not journal.exists()
    assert digest(RUNTIME) == EXPECTED
    records = []
    for phase in ("stress", "long"):
        for layer in (0, 5):
            output = directory / f"{phase}_layer{layer}.json"
            module = "probe_optimized_minimal_advice" + ("_contract" if phase == "long" else "")
            command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_26b_a4b_it.tests." + module,
                "--minimal-advice",
                "hifi2",
                "--defaults",
                "--layer",
                str(layer),
            ]
            if phase == "long":
                command += [
                    "--contract",
                    "long_context",
                    "--length",
                    "262144",
                    "--threads",
                    "4",
                    "--input-fixture",
                    str(DOC / f"actual_text_long/actual_text_layer{layer}_262144_0.pt"),
                    "--reference-file",
                    str(DOC / f"actual_text_long/actual_text_layer{layer}_262144_reference.pt"),
                ]
            else:
                command += [
                    "--length",
                    "1025",
                    "--real",
                    "--input-fixture",
                    str(DOC / f"actual_text_layer{layer}_1025_512.pt"),
                    "--decode",
                    "--steps",
                    "512",
                    "--timing",
                    "--verify-program-cache",
                ]
            command += ["--output", str(output)]
            assert not output.exists()
            record = dict(
                command=command,
                runtime_sha256=EXPECTED,
                driver_sha256=digest(Path(__file__)),
                probe_sha256=digest(DOC.parent.parent / f"tests/{module}.py"),
                phase=phase,
                layer=layer,
            )
            records.append(record)
            journal.write_text(json.dumps(records, indent=2) + "\n")
            print("Starting", phase, layer, flush=True)
            with output.with_suffix(".log").open("w") as log:
                run = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                    timeout=1800,
                )
            record["returncode"] = run.returncode
            if output.exists():
                report = json.loads(output.read_text())
                record.update(
                    output_sha256=digest(output),
                    passed=report.get("passed"),
                    prefill_pcc=report.get("pcc"),
                    decode_min_pcc=report.get("decode", {}).get("min_pcc"),
                )
            journal.write_text(json.dumps(records, indent=2) + "\n")
            assert digest(RUNTIME) == EXPECTED
            print("Completed", phase, layer, run.returncode, record.get("passed"), flush=True)
            # Numerical failures are preserved and do not prevent independent kind controls.


if __name__ == "__main__":
    main()
