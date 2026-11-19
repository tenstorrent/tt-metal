# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated identity-elision probe; production sources and quality gates are unchanged.

--direct-probe selects the existing bitwise-state / PCC .995 standalone probe.
--keep-k-identity retains k*1, isolating elimination of the inverse identity only.
Remaining flags are passed unchanged to the selected existing test harness.
"""

import sys

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_fast_identity as kernel


def main():
    original_argv = sys.argv[:]
    original_elide = kernel.ELIDE_K_IDENTITY
    restore = None
    try:
        if "--keep-k-identity" in sys.argv:
            sys.argv.remove("--keep-k-identity")
            kernel.ELIDE_K_IDENTITY = False
        if "--direct-probe" in sys.argv:
            sys.argv.remove("--direct-probe")
            from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_probe as harness

            restore = (harness, "fused", harness.fused)
            harness.fused = kernel
        else:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
            from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as harness

            restore = (production, "flat_ordered_post_loop", production.flat_ordered_post_loop)
            production.flat_ordered_post_loop = kernel.flat_ordered_post_loop
        return harness.main()
    finally:
        if restore is not None:
            setattr(*restore)
        kernel.ELIDE_K_IDENTITY = original_elide
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
