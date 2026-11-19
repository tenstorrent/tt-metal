# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_row_direct_candidate import BF8RowDirect, BF8Collectives
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_recorded_batch as harness
if __name__ == '__main__':
    production.MultichipCollectives = BF8Collectives
    harness.GdnFidelityCandidate = BF8RowDirect
    harness.main()
