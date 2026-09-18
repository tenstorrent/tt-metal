# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only driver: block active copies until the staging process is terminated."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "models/demos/deepseek_v3_d_p/scripts"))
import run_kimi_prefill_ci as ci


def slow_copy(source, target):
    Path(target).write_bytes(b"partial")
    (Path(os.environ["MARKERS"]) / Path(source).name).write_text(str(os.getpid()))
    time.sleep(60)


# The spawn child reimports this fixture, installing the same controlled I/O stall.
ci.shutil.copyfile = slow_copy

if __name__ == "__main__":
    sys.exit(ci.main())
