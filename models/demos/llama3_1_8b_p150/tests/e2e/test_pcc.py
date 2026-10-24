# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness gate for the self-contained Llama-3.1-8B-Instruct demo.

PINNED, single-model, single-threshold:
  * Identity hard-coded to Llama-3.1-8B-Instruct (also pinned in conftest.py).
  * ONE fixed accuracy floor: ``LLAMA31_8B_TOP1_MIN`` — no per-model lookup.

The gate REUSES this package's OWN copied ``simple_text_demo`` token-matching
flow (which loads the copied ``Llama-3.1-8B-Instruct.refpt`` and teacher-forces
against it via ``TokenAccuracy``). The on-device run is delegated to the COPIED
``test_demo_text`` node and its reported Top1 token accuracy is checked against
the single floor above.

All device work is confined to the test body, so this file imports and
``pytest --collect-only`` cleanly with no device.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

# --- Pinned identity + single threshold ------------------------------------
os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REF_MODEL_NAME = "Llama-3.1-8B-Instruct"  # -> tests/reference_outputs/<name>.refpt

# The ONE correctness floor for this model. Fixed constant, no lookup table.
LLAMA31_8B_TOP1_MIN = 0.86

# The COPIED (self-contained) token-matching node + selector.
DEMO_NODE = "models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text"
DEMO_SELECTOR = "ci-token-matching and performance"

# Log line emitted by simple_text_demo:
#   " Top1 Accuracy: 87.20%, Top5 Accuracy: 95.40%"
_TOP1_RE = re.compile(r"Top1 Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%")


def _reference_available():
    """Reuse the copied TokenAccuracy loader to confirm the pinned model's
    reference tokens exist and load (host-only; no device)."""
    from models.demos.llama3_1_8b_p150.demo.simple_text_demo import TokenAccuracy

    TokenAccuracy(model_name=REF_MODEL_NAME)


def _on_device_run():
    """Delegate to the copied token-matching demo on device with the identity
    pinned, capture stdout, and return the measured Top1 token accuracy [0,1]."""
    env = dict(os.environ)
    env["HF_MODEL"] = HF_MODEL_ID
    env.setdefault("MESH_DEVICE", "P150")

    cmd = [sys.executable, "-m", "pytest", DEMO_NODE, "-k", DEMO_SELECTOR, "-s"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr

    matches = _TOP1_RE.findall(out)
    assert matches, (
        "could not parse Top1 token accuracy from the delegated "
        f"simple_text_demo run (rc={proc.returncode}). Tail:\n{out[-2000:]}"
    )
    return float(matches[-1]) / 100.0


def test_e2e_pcc():
    """Fixed-identity e2e correctness gate for Llama-3.1-8B-Instruct.

    Runs the copied token-matching flow on device and asserts the Top1 token
    accuracy clears the single pinned floor. Skips (does not fail) when no
    device is available, so this stays collect-clean everywhere.
    """
    assert HF_MODEL_ID == os.environ.get("HF_MODEL"), "model identity must stay pinned"

    if os.environ.get("MESH_DEVICE") is None:
        pytest.skip("no device (MESH_DEVICE unset); e2e correctness gate needs a device")

    _reference_available()
    top1 = _on_device_run()
    print(f"[llama3_1_8b_p150] e2e Top1 token accuracy = {top1:.4f} (floor {LLAMA31_8B_TOP1_MIN})")
    assert top1 >= LLAMA31_8B_TOP1_MIN, f"Top1 token accuracy {top1:.4f} < floor {LLAMA31_8B_TOP1_MIN}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__ + "::test_e2e_pcc", "-svv"]))
