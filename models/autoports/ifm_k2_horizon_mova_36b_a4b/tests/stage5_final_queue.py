# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final serialized hardware gates after the prefill CCL synchronization repair."""

import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."
commands = [["sweep_optimized_multichip", str(D / "accepted_prefill_trace_cases.json")]]
commands += [
    ["accept_optimized_multichip", "--group", group]
    for group in ("sdpa_boundary", "contracts", "chunk", "awkward", "stack", "capacity", "full", "watcher")
]
commands += [
    ["sweep_optimized_multichip", str(D / name)]
    for name in ("agmm_distributed_cases.json", "final_selector_lofi_cases.json", "accepted_default_cases.json")
]
commands += [["profile_multichip_decoder", "--artifact-dir", str(D), "--tag", "accepted"]]
for module, *args in commands:
    command = [sys.executable, "-m", MODULE + module, *args]
    print("FINAL_QUEUE", command, flush=True)
    subprocess.run(command, check=True)
