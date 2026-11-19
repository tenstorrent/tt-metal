# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resume device sweeps after recorded validation errors and serialized recovery."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    art = root / "doc/optimized_multichip_decoder"
    specifications = json.loads(Path(sys.argv[1]).read_text())
    # Round-robin failures so a difficult role cannot consume the entire time box.
    for round_index in range(100):
        pending = False
        for specification in specifications:
            base = specification["label"]
            if (art / "STOP").exists():
                return 75
            if (art / (base + "_COMPLETE")).exists():
                continue
            pending = True
            attempt = next(i for i in range(1000) if not (art / f"{base}_attempt{i:02d}_command.json").exists())
            label = f"{base}_attempt{attempt:02d}"
            job = dict(specification, label=label, continue_on_failure=False)
            path = art / (label + "_job.json")
            path.write_text(json.dumps([job], indent=2) + "\n")
            rc = subprocess.call([sys.executable, "-m", "models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs", str(path)])
            if rc == 0:
                (art / (base + "_COMPLETE")).write_text(label + "\n")
            elif rc in (75, 124, 137):
                return rc
        if not pending:
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
