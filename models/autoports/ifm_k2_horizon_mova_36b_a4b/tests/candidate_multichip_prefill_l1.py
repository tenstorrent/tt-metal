# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure perf-report's L1 activation producer advice at TP4 shapes."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_swiglu import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _linear(self, x, name, activation=None):
        if x.shape[-2] > 32:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return super()._linear(x, name, activation)
