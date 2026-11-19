# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read beta/a/z directly from the packed projection in the recurrent launch."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_loop_candidate as base
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_ordered_packed_precise import (
    flat_ordered_packed_precise,
)
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_packed_tiled_conv import packed_tiled_conv


class PackedPreciseCandidate(base.FlatOrderedPostLoopCandidate):
    def pooled_linear_attention_decode_batch(self, packed, states):
        n = len(states)
        conv = packed_tiled_conv(packed, states[0].conv, self.float_taps)
        recurrent = ttnn.reshape(states[0].recurrent, (n * 8, 128, 128))
        return flat_ordered_packed_precise(
            conv,
            packed,
            recurrent,
            self.dt_bias,
            self.decay,
            self.weights["linear_attn.norm.weight"],
            self.config.rms_norm_eps,
            self.wave,
        )


if __name__ == "__main__":
    base.FlatOrderedPostLoopCandidate = PackedPreciseCandidate
    base.main()
