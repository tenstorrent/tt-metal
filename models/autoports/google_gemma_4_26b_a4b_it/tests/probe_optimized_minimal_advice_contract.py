# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reuse the frozen QKV advice setup with the contract harness's own defaults."""

import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    from models.autoports.google_gemma_4_26b_a4b_it.tests import probe_optimized_minimal_advice, run_optimized_contract

    original = run_optimized_contract.main
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    shim_hash = probe_optimized_minimal_advice.digest(__file__)

    def contract():
        # The advice wrapper requires --defaults, but contract mode constructs
        # selected defaults itself and does not accept that runner-only flag.
        with patch.object(sys, "argv", [value for value in sys.argv if value != "--defaults"]):
            original()

    try:
        with patch.object(run_optimized_contract, "main", contract):
            probe_optimized_minimal_advice.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report["minimal_advice_contract_shim_sha256"] = shim_hash
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
