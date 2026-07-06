# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC-test harness fixtures/shims for `meituan-longcat/LongCat-Image`.

The `tt_hw_planner bringup` test template (see `bringup_loop.py:514`) calls the
module-level helper `_captured_submodule_path(COMPONENT_NAME)` inside every
generated `test_<component>.py`, but the generator never emitted the helper's
*definition* into the test files — so each test raised
`NameError: name '_captured_submodule_path' is not defined` before it could
run. That is a harness bug shared by all 25 tests, so the fix lives here (a
single conftest) rather than in any stub or in each test file.

We provide the exact helper the template expects and inject it into `builtins`
so the bare name resolves from every test module's global scope (name lookup
falls through module globals to builtins). This mirrors the canonical
implementation in `scripts/tt_hw_planner/capture_inputs.py:_captured_submodule_path`.
"""

from __future__ import annotations

import builtins
import os

# The LongCat-Image reference is a ~29 GB diffusers pipeline (Qwen2.5-VL text
# encoder + 6.27 B-param MMDiT transformer + VAE). Building the torch reference
# for a PCC test therefore loads tens of GB of safetensors from disk. The test
# template's default per-stage watchdog is only 120 s (`TT_PLANNER_TEST_STAGE_S`),
# which the `build_torch_reference` stage can blow past on a cold page cache —
# the stage times out mid-load and the test flakily fails to build the reference
# at all. Give every stage a generous budget so the load completes deterministically.
# (`setdefault` respects any value the bring-up runner already exported.)
os.environ.setdefault("TT_PLANNER_TEST_STAGE_S", "900")


def _captured_submodule_path(component_name):
    """Read the submodule_path the capture step hooked when it saved
    inputs for this component. Returns the path string or ``None``.

    Capture and test must resolve to the SAME submodule or the recorded
    args/kwargs won't fit the test-resolved module's signature; using the
    manifest's recorded path as the FIRST candidate keeps them aligned.
    When no manifest exists (nothing captured yet), returns ``None`` and the
    test falls back to `_CANDIDATE_SUBMODULE_PATHS`.
    """
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"

    # conftest.py lives at <demo_dir>/tests/pcc/conftest.py, so the demo dir
    # (which holds `_captured/`) is two levels up. Be defensive: walk upward
    # looking for the `_captured` dir in case the layout ever shifts.
    here = _Path(__file__).resolve()
    demo_dir = None
    for parent in here.parents:
        if (parent / "_captured").is_dir():
            demo_dir = parent
            break
    if demo_dir is None:
        demo_dir = here.parents[2] if len(here.parents) > 2 else here.parent

    manifest_p = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        data = _json.loads(manifest_p.read_text())
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


# Expose the helper to every test module's global scope. The generated tests
# call it as a bare name, so builtins injection is the minimal shim that fixes
# all of them without touching each test file.
builtins._captured_submodule_path = _captured_submodule_path
