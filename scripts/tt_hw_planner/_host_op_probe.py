# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path


def probe(demo_dir: Path) -> dict:
    sys.path.insert(0, str(demo_dir.parent))
    try:
        mod = __import__("%s.tt.pipeline" % demo_dir.name, fromlist=["pipeline"])
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "reason": "cannot import tt.pipeline: %s" % e}
    fn = getattr(mod, "host_op_selftest", None)
    if fn is None:
        return {"ran": False, "reason": "no host_op_selftest hook"}
    try:
        return {"ran": True, "verdict": fn()}
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "reason": "host_op_selftest raised: %s" % e}


def main(argv) -> int:
    if len(argv) < 2:
        return 0
    print("HOST_OP_PROBE=" + json.dumps(probe(Path(argv[1]).resolve())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
