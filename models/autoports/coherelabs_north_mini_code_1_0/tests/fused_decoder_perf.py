# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed wall/trace harness for the graph-fused North-Mini decoder."""

from models.autoports.coherelabs_north_mini_code_1_0.tests import functional_decoder_perf as harness
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


def main():
    # The functional harness owns identical setup, cache, trace, signpost, and
    # JSON semantics.  Swapping its constructor keeps before/after regimes
    # exactly comparable while ensuring every measured forward is fused.
    harness.FunctionalDecoder = FusedDecoder
    harness.main()


if __name__ == "__main__":
    main()
