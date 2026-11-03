# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check whether value EP also wins for a single prefill tile."""

from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if value and valid_tokens is None:
            return self._experts_ep_value(x, logits)
        return super()._experts(x, logits=logits, value=value, valid_tokens=valid_tokens)
