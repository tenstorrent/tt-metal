# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_GDN_STEP_V2=1 selects kernels/compute_v2.cpp; unset (or any other value) keeps kernels/compute.cpp."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "ttnn" / "fused" / "gdn_step" / "kernels"
REPO = ROOT.parents[3]


def _compute_path(value):
    env = {k: v for k, v in os.environ.items() if k != "QWEN38_GDN_STEP_V2"}
    if value is not None:
        env["QWEN38_GDN_STEP_V2"] = value
    env["PYTHONPATH"] = str(REPO)
    code = "from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step; print(gdn_step.COMPUTE)"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "value,name", [(None, "compute.cpp"), ("0", "compute.cpp"), ("", "compute.cpp"), ("1", "compute_v2.cpp")]
)
def test_flag_selects_compute_kernel(value, name):
    assert _compute_path(value).endswith(f"gdn_step/kernels/{name}")


def test_v2_kernel_keeps_the_reference_cb_indices_and_debug_fallback():
    ref, v2 = (KERNELS / "compute.cpp").read_text(), (KERNELS / "compute_v2.cpp").read_text()
    import re

    def consts(text):
        return dict(re.findall(r"\b(CB_[A-Z0-9]+) = (\d+)", text))

    assert consts(v2) == consts(ref)
    assert '#include "compute.cpp"' in v2  # DEBUG_TAPS builds use the reference kernel unchanged
    assert "kernel_main" in v2
