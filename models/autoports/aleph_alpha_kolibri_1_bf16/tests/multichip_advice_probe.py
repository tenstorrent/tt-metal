# SPDX-License-Identifier: Apache-2.0
"""Isolated final per-device advice probes under the selected precision policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from . import multichip_checks


class AdviceDecoder(MultichipDecoder):
    mode = ""

    def _sparse_config(self, m, n):
        if n == 256 or not self.mode.startswith("down"):
            return super()._sparse_config(m, n)
        rows = 4 if self.mode == "down40" else 2
        width = 8 // rows
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(10, rows),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=width,
            out_block_h=1,
            out_block_w=width,
            per_core_M=(m + 31) // 32,
            per_core_N=width,
            fuse_batch=False,
            mcast_in0=True,
        )

    def _prefill_linear(self, x, w, dtype, compute):
        if self.mode == "prefill_l1":
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return super()._prefill_linear(x, w, dtype, compute)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["down40", "down20", "prefill_l1", "router6", "link1"])
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    AdviceDecoder.mode = a.mode
    a.tag = "advice_" + a.mode
    a.baseline = False
    options = {"router_grid": [3, 2]} if a.mode == "router6" else {"num_links": 1} if a.mode == "link1" else {}
    os.environ["MC_POLICY"] = json.dumps(options)
    multichip_checks.MultichipDecoder = AdviceDecoder
    multichip_checks.run(a)
    path = multichip_checks.OUT / f"{a.tag}_{a.layer}_{a.tokens}.json"
    value = json.loads(path.read_text())
    value["candidate_adapter"] = dict(
        path=__file__,
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mode=a.mode,
        decoder_sha256=hashlib.sha256((multichip_checks.ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
    )
    path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
