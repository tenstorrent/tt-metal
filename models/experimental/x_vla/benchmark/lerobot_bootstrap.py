# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Workarounds so `lerobot.policies.xvla` imports cleanly on this host.

This installed lerobot has two upstream issues that prevent loading:
  1. `lerobot.policies/__init__.py` eagerly imports `groot.modeling_groot`
     which fails to define a dataclass (`backbone_cfg` non-default arg follows
     a default arg) and aborts the whole `lerobot.policies` import.
  2. `lerobot.policies.xvla.modeling_florence2` imports
     `is_flash_attn_greater_or_equal_2_10` which was renamed in the local
     transformers (5.4.0) to `is_flash_attn_greater_or_equal`.

Both are isolated upstream bugs unrelated to X-VLA. Fixing them in the venv
itself is out of scope (PROGRAM.md forbids dep changes), so we patch sys.modules
and the transformers symbol table at import time.
"""

from __future__ import annotations

import sys
import types


def install() -> None:
    # 1. Stub lerobot.policies.groot so the module-level import in
    #    lerobot.policies/__init__.py does not blow up.
    for mod_name in (
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["lerobot.policies.groot.configuration_groot"].GrootConfig = type(
        "GrootConfig", (), {}
    )
    sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = type(
        "GrootPolicy", (), {}
    )
    sys.modules["lerobot.policies.groot"].GrootConfig = sys.modules[
        "lerobot.policies.groot.configuration_groot"
    ].GrootConfig
    sys.modules["lerobot.policies.groot"].GrootPolicy = sys.modules[
        "lerobot.policies.groot.modeling_groot"
    ].GrootPolicy

    # 2. Restore the renamed transformers helper. We never use flash-attn here,
    #    so a constant `False` shim is correct.
    import transformers.utils as _tu

    if not hasattr(_tu, "is_flash_attn_greater_or_equal_2_10"):
        _tu.is_flash_attn_greater_or_equal_2_10 = lambda *a, **k: False
