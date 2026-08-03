# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Long-context capacity probes for the optimized North-Mini decoder."""

from models.autoports.coherelabs_north_mini_code_1_0.tests import functional_decoder_capacity as harness
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


def main():
    harness.FunctionalDecoder = OptimizedDecoder
    harness.main()


if __name__ == "__main__":
    main()
