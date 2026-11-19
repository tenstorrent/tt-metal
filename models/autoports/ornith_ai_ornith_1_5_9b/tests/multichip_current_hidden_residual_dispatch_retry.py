# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Avoid gathering already-full-width scalar-prefill projection inputs twice."""

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_current_hidden_residual_candidate as family
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class HiddenResidualDispatchRetry(family.CurrentHiddenResidualCandidate):
    def decode_linear(self, x, name):
        # Scalar prefill reaches this via generic norm -> linear. That norm has
        # already gathered all hidden columns, unlike packed decode_norm.
        # Preserve scalar fused row projections: generic reduce correctly returns
        # an already-scattered width1024 result unchanged in the sharded contract.
        if self.fused_gather_enabled and name == "packed_input" and x.shape[-1] == 4096:
            return MultichipDecoder.decode_linear(self, x, name)
        return super().decode_linear(x, name)


def main():
    original = family.CurrentHiddenResidualCandidate
    try:
        family.CurrentHiddenResidualCandidate = HiddenResidualDispatchRetry
        family.main()
    finally:
        family.CurrentHiddenResidualCandidate = original


if __name__ == "__main__":
    main()
