# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic global MLP rounding control matching optimized sum output dtype."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _mlp(self, x, *, valid_tokens=None):
        out = super()._mlp(x, valid_tokens=valid_tokens)
        return ttnn.typecast(out, ttnn.bfloat8_b) if self.sparse else out
