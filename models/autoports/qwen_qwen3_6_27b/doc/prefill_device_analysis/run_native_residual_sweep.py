# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sequential reduced-stack sweep after native GDN changes the bottleneck."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
env = dict(
    os.environ,
    PYTHONPATH=".",
    HF_HUB_OFFLINE="1",
    QWEN_AUTOPORT_MODEL_ID="Qwen/Qwen3.8-27B",
    QWEN_AUTOPORT_MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    QWEN36_PREFILL_RECURRENCE="native",
)
records = []
for label, flags in [
    ("selected", ["--save", "/tmp/qwen_native_selected.pt"]),
    ("compact", ["--compact-native-output"]),
    ("compact_block8", ["--compact-native-output", "--program-block-limit", "8"]),
    ("compact_block16", ["--compact-native-output", "--program-block-limit", "16"]),
]:
    if label != "selected":
        flags += ["--reference", "/tmp/qwen_native_selected.pt"]
    command = [
        sys.executable,
        str(root / "probe.py"),
        "--candidate",
        "runtime",
        "--sequence",
        "128",
        "--iterations",
        "3",
        "--result",
        str(root / "artifacts" / f"native_residual_{label}.json"),
        *flags,
    ]
    log = Path(f"/tmp/qwen_native_residual_{label}.log")
    with log.open("w") as output:
        completed = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=240)
    records.append({"candidate": label, "returncode": completed.returncode, "log": str(log), "command": command})
    (root / "artifacts/native_residual_sweep.json").write_text(json.dumps(records, indent=2) + "\n")
    print(label, completed.returncode, flush=True)
