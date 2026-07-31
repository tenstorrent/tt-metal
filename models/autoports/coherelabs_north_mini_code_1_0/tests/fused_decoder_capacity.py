# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Long-context capacity probes for the graph-fused North-Mini decoder."""

from models.autoports.coherelabs_north_mini_code_1_0.tests import functional_decoder_capacity as harness
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


def main():
    # Reuse the functional stage's exact allocation and trace probe while
    # selecting the stage-02 runtime constructor.
    harness.FunctionalDecoder = FusedDecoder
    harness.main()


if __name__ == "__main__":
    main()
