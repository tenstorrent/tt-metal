# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unchanged strict conv/gate direct probe against the cumulative reader candidate."""
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_conv_gates_probe as probe
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_fast_zero_conv_gates as fused
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_fast_zero_conv_gates_recurrent as recurrent
from models.autoports.ornith_ai_ornith_1_5_9b.tt import flat_ordered_post_loop as control

if __name__ == "__main__":
    probe.fused = fused
    probe.recurrent = recurrent
    probe.control = control
    probe.main()
