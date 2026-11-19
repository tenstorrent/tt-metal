# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run unchanged quantized-control and bitwise gates against explicit RNE state packing."""
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_bf16_state_staged_probe as probe
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_bf16_state_rne_kernel as fused

if __name__ == "__main__":
    probe.fused = fused
    probe.main()
