# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer traced decode for this stage's policies, on the decoder stage's harness.

Two things the full-model sweep cannot answer, both of which need the decoder
stage's own per-layer A/B (same warm-up, same 3-round min, same traced replay
count), so the numbers are comparable with the committed ones:

1. **The layer-stack floor under the selected policy.** The floor the
   optimized-full-model stage quotes was measured on *its* precision policy. The
   selection changes the attention weight dtype and the decode CCL payload, so
   quoting that floor for this stage would be quoting a different model's floor.

2. **What the KV-cache dtype is worth as a function of context.** The full-model
   sweep decodes at 128-256 positions, where the paged SDPA reads almost nothing
   and a BFP4 cache measures as worth zero. The decoder stage measured the same
   lever at 131071 and found the *cache dtype* worth 10 % once the SDPA chunking
   was fixed. A rejection of BFP4 KV that only cites the short-context number
   would be hiding that, so the long-context arms are measured here and the
   rejection is stated against both.

Usage::

    python doc/datatype_sweep/bench/layer_ab.py --candidates baseline,selected \\
        --prefill-seq 128 --decode-context 256
    python doc/datatype_sweep/bench/layer_ab.py --candidates baseline,selected,selected_kv4 \\
        --prefill-seq 128 --decode-context 131071
"""

from __future__ import annotations

from dataclasses import replace

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.doc.optimized_multichip_decoder.bench.layer_ab import (  # noqa: F401
    CANDIDATES,
    main,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import DEFAULT_PRECISION

BFP4, BFP8 = ttnn.bfloat4_b, ttnn.bfloat8_b

#: The selected policy's decoder-layer part: BFP4 attention weights at LoFi, BFP4
#: MLP weights at LoFi, BFP8 KV cache, BF16 activations.
_SELECTED_PRECISION = replace(DEFAULT_PRECISION, name="c14-attn4-cclbfp8-kv8", attn_weight_dtype=BFP4)

CANDIDATES.update(
    {
        # ``tp4`` under a name that says what it is here: the carried-forward
        # optimized-full-model policy, i.e. this stage's c00 baseline.
        "baseline": {},
        "baselineb": {},
        # The selected policy: c14.  ``decode_ccl_dtype`` is the second half of
        # it; the attention weight dtype is the first.
        "selected": {"precision": _SELECTED_PRECISION, "decode_ccl_dtype": BFP8},
        "selectedb": {"precision": _SELECTED_PRECISION, "decode_ccl_dtype": BFP8},
        # The KV-cache arms, with and without the selected policy around them, so
        # the cache dtype's value can be read at any context.
        "selected_kv4": {
            "precision": replace(_SELECTED_PRECISION, name="c14+kv4", kv_cache_dtype=BFP4),
            "decode_ccl_dtype": BFP8,
        },
        "baseline_kv4": {"precision": replace(DEFAULT_PRECISION, name="c00+kv4", kv_cache_dtype=BFP4)},
    }
)


if __name__ == "__main__":
    main()
