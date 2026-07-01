# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared loader for the pi0.5 production perf flags.

`pi05_production.env` lives next to this loader in models/experimental/pi0_5/common/.
Perf tests call `apply_production_env_defaults()` at module load (before any ttnn /
pi0_5 import) so the full validated production flag set is in place without a manual
`source`. setdefault semantics: an explicitly-set env var always wins.

Resolved RELATIVE TO THIS FILE (not a pardir count from the test file), so it works
regardless of cwd / TT_METAL_HOME and from any tests subdirectory.
"""
from __future__ import annotations

import os
import re

# models/experimental/pi0_5/common/pi05_production.env  (same dir as this loader)
PROD_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pi05_production.env")


def apply_production_env_defaults(verbose: bool = True, skip: set | None = None) -> list:
    """setdefault every `export KEY=VAL` from pi05_production.env into os.environ.

    `skip` is an optional set of keys to ignore (e.g. machine-specific paths a
    caller wants to leave to its own logic). Returns the list of "KEY=VAL" actually
    applied (i.e. not already set). Safe to call multiple times. Does nothing
    (warns) if the env file is missing."""
    skip = skip or set()
    if not os.path.exists(PROD_ENV_PATH):
        if verbose:
            print(f"[pi0.5 prod-env] WARN: {PROD_ENV_PATH} not found; production flags NOT applied", flush=True)
        return []
    applied = []
    with open(PROD_ENV_PATH) as f:
        for line in f:
            m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if not m or m.group(1) in skip:
                continue
            k, v = m.group(1), m.group(2)
            if k not in os.environ:
                os.environ[k] = v
                applied.append(f"{k}={v}")
    if verbose:
        print(f"[pi0.5 prod-env] applied {len(applied)} production defaults from pi05_production.env", flush=True)
    return applied
