# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replace scalar RISC stores with NoC-local zero fills; unchanged arithmetic."""
import sys
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_compile_args_kernel as kernel
if __name__ == "__main__":
    if "--direct-probe" in sys.argv:
        sys.argv.remove("--direct-probe")
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_probe as stack
        stack.fused=kernel
    else:
        from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
        production.flat_ordered_post_loop=kernel.flat_ordered_post_loop
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.main()
