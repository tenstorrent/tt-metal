# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device PCC: full DiT block under TP (2-way and 4-way) matches the replicate baseline on a
>=4-chip mesh (e.g. BH_QB 2x2). Opt-in + skip-guarded so it never runs (or conflicts with a
session mesh) in the default suite.

Enable:  ACE_STEP_TP_DEVICE_TEST=1 pytest tests/test_tp_dit_device_pcc.py

Reuses the standalone gate in ``perf/tp_phase2c_layer_pcc.py`` (opens its own mesh + fabric,
compares TP-off vs TP-on for the ``TtAceStepDiTLayer``). Host-only TP tests live in
``test_tp_config.py`` / ``test_lm_head_narrow_tp.py``.
"""

from __future__ import annotations

import os

import pytest


def _num_devices() -> int:
    try:
        import ttnn

        return len(ttnn.get_device_ids())
    except Exception:
        return 0


_OPT_IN = os.environ.get("ACE_STEP_TP_DEVICE_TEST", "").strip() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not (_OPT_IN and _num_devices() >= 4),
    reason="Set ACE_STEP_TP_DEVICE_TEST=1 and use a >=4-chip mesh (BH_QB) to run TP device PCC.",
)


@pytest.mark.parametrize("mode", ["on", "4"])
def test_tp_full_dit_block_matches_replicate(mode, monkeypatch):
    """TP (2-way for ``on``, 4-way for ``4``) full DiT block == replicate baseline (PCC gate)."""
    monkeypatch.setenv("TP_ON", mode)
    from models.experimental.ace_step_v1_5.perf.tp_phase2c_layer_pcc import main

    assert main() == 0, f"TP full-DiT-block PCC gate failed for mode={mode}"
