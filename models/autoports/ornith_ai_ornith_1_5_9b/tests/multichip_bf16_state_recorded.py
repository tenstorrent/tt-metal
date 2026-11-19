# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recorded HF activation discriminator for BF16 pooled recurrence precision."""
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf16_state_rne_candidate import BF16State, flat_ordered_post_loop
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_recorded_batch as harness
if __name__ == '__main__':
    production.flat_ordered_post_loop = flat_ordered_post_loop
    harness.GdnFidelityCandidate = BF16State
    harness.main()
