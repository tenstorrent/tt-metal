# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight projection candidates, measured by the existing decoder harness.

This diagnostic subclass is not the delivered runtime default. Packing and dtype
conversion happen only at construction. Every measured forward stays on device.
"""

import argparse

from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


class ProjectionCandidateDecoder(OptimizedDecoder):
    """Expose production policy overrides to the serial diagnostic harness."""


def main():
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_decoder

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-kwargs", default="{}")
    p.add_argument("--activation-file")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.add_argument("--compare-functional", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=30)
    run_decoder.FunctionalDecoder = ProjectionCandidateDecoder
    run_decoder.run(p.parse_args())


if __name__ == "__main__":
    main()
