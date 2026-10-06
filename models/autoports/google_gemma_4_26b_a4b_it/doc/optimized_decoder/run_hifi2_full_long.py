# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Complete the independent full-attention gate after a journal parser failure."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
OUT = DOC / "minimal_hifi2_acceptance_v6/long_layer5.json"
RUNTIME = DOC.parent.parent / "tt/optimized_decoder.py"
EXPECTED = "b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846"
assert hashlib.sha256(RUNTIME.read_bytes()).hexdigest() == EXPECTED
assert not OUT.exists()
command = [
    sys.executable,
    "-m",
    "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice_contract",
    "--minimal-advice",
    "hifi2",
    "--defaults",
    "--layer",
    "5",
    "--contract",
    "long_context",
    "--length",
    "262144",
    "--threads",
    "4",
    "--input-fixture",
    str(DOC / "actual_text_long/actual_text_layer5_262144_0.pt"),
    "--reference-file",
    str(DOC / "actual_text_long/actual_text_layer5_262144_reference.pt"),
    "--output",
    str(OUT),
]
journal = OUT.parent / "full_long_resume_command.json"
record = dict(
    command=command,
    runtime_sha256=EXPECTED,
    note="Earlier driver parsed the completed sliding decode list as a dict; no rerun or relabeling of that evidence.",
)
journal.write_text(json.dumps(record, indent=2) + "\n")
with OUT.with_suffix(".log").open("w") as log:
    result = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        timeout=1800,
    )
record["returncode"] = result.returncode
assert hashlib.sha256(RUNTIME.read_bytes()).hexdigest() == EXPECTED
if OUT.exists():
    report = json.loads(OUT.read_text())
    record.update(
        passed=report.get("passed"),
        sampled_rows=len(report.get("sampled_row_diagnostics", [])),
        output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
    )
journal.write_text(json.dumps(record, indent=2) + "\n")
print(record, flush=True)
