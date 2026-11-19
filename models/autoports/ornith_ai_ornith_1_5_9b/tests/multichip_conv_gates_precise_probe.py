# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged direct quality gates against the precise gate candidate."""
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_conv_gates_precise as precise
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_conv_gates_probe as probe

if __name__ == "__main__":
    probe.fused = precise
    probe.main()
