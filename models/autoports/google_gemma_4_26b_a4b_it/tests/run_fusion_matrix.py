# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial numerical rejection matrix; does not profile devices."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder")
    module = "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_precision"
    cases = [
        (
            f"norm_{site}_batch_sliding",
            ["--candidate", "norm_fp32", "--norm-site", site, "--runner", "batched", "--batch", "32", "--layer", "0"],
        )
        for site in ["input", "post", "head", "router"]
    ]
    cases += [
        (
            f"{candidate}_{kind}",
            ["--candidate", candidate, "--runner", "batched", "--batch", "32", "--layer", str(layer)],
        )
        for candidate in ["rope_native_fp32", "rope_hf_fp32", "sdpa_exact"]
        for layer, kind in [(0, "sliding"), (5, "full")]
    ]
    journal = []
    for name, arguments in cases:
        output = root / (name + ".json")
        command = [sys.executable, "-m", module, *arguments, "--output", str(output)]
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        journal.append(dict(command=command, exit_code=result.returncode, output=str(output)))
        (root / "precision_matrix_commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        print(name, result.returncode, flush=True)


if __name__ == "__main__":
    main()
