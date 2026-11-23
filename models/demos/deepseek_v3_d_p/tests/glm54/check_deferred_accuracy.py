# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the accuracy checks GLM-5.4 device tests deferred (``deferred_accuracy``), host only, after the device lock.

    python -m models.demos.deepseek_v3_d_p.tests.glm54.check_deferred_accuracy <bundle or dir> [...]

Prints ``GLM54_ACCURACY: PASS|FAIL|ERROR <bundle>`` per bundle and deletes a bundle that passed (a failed one stays for
diagnosis). ERROR: the check could not run (a bundle from another checkout HEAD, an import or runtime error); it counts
as failed. Exit status 0: every bundle passed; 1: a bundle failed or errored; 2: no bundle found.

The check functions live in device test modules, whose import queries the device arch (``is_blackhole`` in
decorators). Run as a command, it sets the mock Blackhole galaxy descriptor before any of them loads ttnn, so that
query opens no device.
"""

import importlib
import os
import sys
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tests.glm54.deferred_accuracy import checkout_head

_MOCK_GALAXY = (
    Path(__file__).resolve().parents[5]
    / "tt_metal/third_party/umd/descriptors/cluster_descriptor_examples/blackhole_galaxy.yaml"
)


def _bundles(args: list[str]) -> list[Path]:
    out = []
    for a in map(Path, args):
        out += sorted(a.glob("*.pt")) if a.is_dir() else [a]
    return out


def main(args: list[str]) -> int:
    bundles = _bundles(args)
    if not bundles:
        print(f"GLM54_ACCURACY: no bundles in {args}")
        return 2
    head, failed = checkout_head(), 0
    for path in bundles:
        t0 = time.perf_counter()
        try:
            bundle = torch.load(path, weights_only=True)
            if bundle.get("head") != head:
                raise RuntimeError(f"bundle from HEAD {bundle.get('head')}, checkout is at {head}")
            module, qualname = bundle["check"].split(":")
            check = getattr(importlib.import_module(module), qualname)
            check(**bundle["inputs"])
        except AssertionError as e:
            failed += 1
            print(f"GLM54_ACCURACY: FAIL {path} ({time.perf_counter() - t0:.1f} s): {e}", flush=True)
            continue
        except Exception as e:
            failed += 1
            print(f"GLM54_ACCURACY: ERROR {path}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"GLM54_ACCURACY: PASS {path} (head {head[:11]}, {time.perf_counter() - t0:.1f} s)", flush=True)
        path.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    # Only for the command line: set before any check module imports ttnn. Never on import, which would hand the
    # mock to device tests collected in the same pytest session.
    os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", str(_MOCK_GALAXY))
    sys.exit(main(sys.argv[1:]))
