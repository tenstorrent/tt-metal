# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the explicit optimized class with drains at existing prefill boundaries."""
import argparse
import os
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_decoder
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--build-kwargs", default=os.environ.get("QWEN_PROFILE_POLICY", "{}"))
    p.set_defaults(seed=42, chunk_size=None, compare_functional=False, profile=True)
    args = p.parse_args()
    forward = OptimizedDecoder._forward

    def drain(self, *a, **kw):
        out = forward(self, *a, **kw)
        if not kw["decode"]:
            ttnn.synchronize_device(self.mesh_device)
            ttnn.ReadDeviceProfiler(self.mesh_device)
        return out

    with patch.object(run_decoder, "FunctionalDecoder", OptimizedDecoder), patch.object(
        OptimizedDecoder, "_forward", drain
    ):
        run_decoder.run(args)


if __name__ == "__main__":
    main()
