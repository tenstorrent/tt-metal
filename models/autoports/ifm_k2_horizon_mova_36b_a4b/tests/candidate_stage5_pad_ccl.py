# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _reduce(self, x, slot=0):
        rows = x.shape[-2]
        if rows > 32:
            alignment = self.options.get("ccl_row_alignment", 64)
            padded = math.ceil(rows / alignment) * alignment
            if padded != rows:
                x = ttnn.pad(x, [(0, 0)] * (len(x.shape) - 2) + [(0, padded - rows), (0, 0)], 0)
                return super()._reduce(x, slot=slot)[..., :rows, :]
        return super()._reduce(x, slot=slot)
