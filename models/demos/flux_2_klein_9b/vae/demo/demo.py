# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dispatcher over the emitted per-task pipeline demos (repointed by emit-e2e)."""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMOS = ["demo_decode.py", "demo_encode.py", "demo_reconstruct.py"]


def main() -> None:
    if len(sys.argv) > 1:
        pick = sys.argv[1]
        target = pick if pick.endswith(".py") else f"demo_{pick}.py"
        if target in _DEMOS:
            sys.argv = [os.path.join(_HERE, target)] + sys.argv[2:]
            runpy.run_path(os.path.join(_HERE, target), run_name="__main__")
            return
    print(f"This model has {len(_DEMOS)} task demo(s). Run one of:")
    for _n in _DEMOS:
        print(f"  python demo/{_n}")


if __name__ == "__main__":
    main()
