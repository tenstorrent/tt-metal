# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thin loaders for the real HF ACE-Step v1.5 reference modules (trust_remote_code).

We import the modeling module from the local HF cache snapshot so PCC tests compare
against the genuine ACE-Step implementation (never a re-implementation).
"""

import importlib.util
import sys
from pathlib import Path

_HF_HUB = Path.home() / ".cache/huggingface/hub/models--ACE-Step--acestep-v15-base/snapshots"


def _snapshot_dir() -> Path:
    snaps = sorted(_HF_HUB.glob("*/"))
    assert snaps, f"No ACE-Step snapshot found under {_HF_HUB}"
    return snaps[-1]


def _install_apg_stub():
    """apg_guidance.py is absent from the HF snapshot and only powers guidance sampling
    (not the nn.Module classes we PCC-test). Stub the three imported symbols so the
    genuine modeling module imports without pulling in sampling code."""
    if "apg_guidance" in sys.modules:
        return
    stub = type(sys)("apg_guidance")

    def _noop(*args, **kwargs):  # pragma: no cover - never called in module PCC tests
        raise NotImplementedError("apg_guidance is stubbed for reference module tests")

    class MomentumBuffer:  # minimal placeholder
        def __init__(self, *args, **kwargs):
            pass

    stub.adg_forward = _noop
    stub.apg_forward = _noop
    stub.MomentumBuffer = MomentumBuffer
    sys.modules["apg_guidance"] = stub


def load_modeling_module():
    """Import modeling_acestep_v15_base.py from the local HF snapshot as a module."""
    snap = _snapshot_dir()
    # Make sibling imports (configuration_acestep_v15) resolvable.
    if str(snap) not in sys.path:
        sys.path.insert(0, str(snap))
    _install_apg_stub()
    path = snap / "modeling_acestep_v15_base.py"
    spec = importlib.util.spec_from_file_location("modeling_acestep_v15_base", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_acestep_v15_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained("ACE-Step/acestep-v15-base", trust_remote_code=True)
