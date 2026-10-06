# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful active-count-banded grid candidate, isolated from final policy.

4x3's current kernel has a full-M-block slice accounting mismatch. Keep expert
counts<=128 on4x3 and execute every larger expert on validated4x4, preserving
all routes and common region offsets. Counts are tested in device kernels.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import ttnn

from . import multichip_checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8193)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    a.tag = "hybrid_grid"
    a.baseline = False
    os.environ["MC_POLICY"] = json.dumps({"prefill_expert_grid": [4, 3]})
    namespace = ttnn.experimental.deepseek_prefill
    original = namespace.moe_fused_swiglu

    def banded(*args, **kw):
        output = original(*args, **(kw | {"min_active_tokens": 1, "max_active_tokens": 128}))
        if kw["input_m_tiles"] > 4:
            output = original(
                *args,
                **(
                    kw | {"core_grid": ttnn.CoreCoord(4, 4), "min_active_tokens": 129, "max_active_tokens": 2**32 - 1}
                ),
            )
        return output

    namespace.moe_fused_swiglu = banded
    try:
        multichip_checks.run(a)
    finally:
        namespace.moe_fused_swiglu = original
    path = multichip_checks.OUT / f"{a.tag}_{a.layer}_{a.tokens}.json"
    value = json.loads(path.read_text())
    value["candidate_adapter"] = dict(
        path=__file__,
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bands=[dict(grid=[4, 3], minimum=1, maximum=128), dict(grid=[4, 4], minimum=129, maximum=2**32 - 1)],
        shared_output=True,
        host_count_reads=False,
    )
    path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
