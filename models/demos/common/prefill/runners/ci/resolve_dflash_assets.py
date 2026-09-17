#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Print the on-disk assets a DFlash prefill run will open, as shell assignments.

The launcher preflights these on the workers, so it has to name the same paths the runner and producer
will resolve for themselves. Resolving them from the manifest plus the model adapter here -- in the same
order the runner uses, caller environment first -- keeps the launcher from restating paths that would
then drift from the adapter that owns them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--format", choices=("shell", "report"), default="shell")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest_env = json.load(f).get("env") or {}
    # setdefault, not assignment: the runner applies the manifest the same way, so an exported override
    # must win here too or the preflight probes a path the run will not use.
    for key, val in manifest_env.items():
        os.environ.setdefault(key, str(val))

    adapter = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
    if not adapter.supports_dflash:
        print(f"model {adapter.name!r} ships no DFlash drafter", file=sys.stderr)
        return 2

    paths = {
        "HF_MODEL": os.environ.get("PREFILL_HF_MODEL") or adapter.hf_model_default,
        "DFLASH_MODEL": os.environ.get("DFLASH_HF_MODEL") or adapter.dflash_model_default,
        "TRACE_DIR": os.environ.get("PREFILL_TRACE_DIR") or adapter.prefill_trace_default,
        "GOLDEN_KV_DIR": os.environ.get("PREFILL_DFLASH_GOLDEN_KV_DIR") or adapter.dflash_golden_default,
    }
    unset = [k for k, v in paths.items() if not v]
    if unset:
        print(f"model {adapter.name!r} defines no {', '.join(unset)}", file=sys.stderr)
        return 2

    if args.format == "report":
        print(f"manifest : {args.manifest}")
        print(f"adapter  : {adapter.name} ({type(adapter).__name__})")
        for key, val in sorted(manifest_env.items()):
            origin = "manifest" if os.environ.get(key) == str(val) else "environment (manifest overridden)"
            print(f"  {key}={os.environ.get(key)}  <- {origin}")
        for key, val in paths.items():
            print(f"  {key}={val}  [{'present' if os.path.isdir(val) else 'MISSING'}]")
        return 0

    for key, val in paths.items():
        print(f"{key}={shlex.quote(val)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
