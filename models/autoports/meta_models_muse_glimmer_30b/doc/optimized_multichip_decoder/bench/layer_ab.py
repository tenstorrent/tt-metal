# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the *optimized* multichip decoder.

Re-uses the multichip stage's harness verbatim -- same warm-up, same 3-round
min, same traced decode replay count, same PCC reference -- and only adds this
stage's candidates, so a number here is directly comparable with a number in
``doc/multichip_decoder/logs/``.

    python .../doc/optimized_multichip_decoder/bench/layer_ab.py \
        --mesh 1x4 --candidates tp4,oproj_c8_bw4 --prefill-seq 8192 --decode-context 2048
    python .../doc/optimized_multichip_decoder/bench/layer_ab.py --list
"""

from __future__ import annotations

from dataclasses import replace

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.doc.multichip_decoder.bench.layer_ab import (  # noqa: F401
    CANDIDATES,
    geometry,
    main,
    mlp_geometry,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import DEFAULT_PRECISION

BF16, BFP8, BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
LOFI, HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2


def precision(name: str, **changes):
    return replace(DEFAULT_PRECISION, name=name, **changes)


#: The best *decode* collective found by ``logs/ab_ccl_async.log`` -- before the
#: watcher showed that its all-gather was missing a mandatory barrier semaphore.
#: Retained so that superseded sweep can be reproduced; it is **not** shipped, and
#: ``ccl_persistent_buffers`` is rejected outright.  See the README.
_CCL_BEST = {
    "decode_ccl_impl": "async",
    "decode_ccl_ag_workers": 1,
    "ccl_persistent_buffers": True,
}

#: The multichip stage's configuration: composite wrappers for **both** modes and
#: a DRAM-interleaved layer boundary.  This is the ``before`` row, and it has to
#: pin ``prefill_ccl_impl`` too -- the optimized default moves prefill to the async
#: primitives, so leaving it out would compare the new prefill against itself.
_BEFORE = {
    "ccl_impl": "wrapper",
    "ccl_persistent_buffers": False,
    "sharded_decode_io": False,
}


CANDIDATES.update(
    {
        # -- OPT-011: a narrower working shard for the `o_proj` input ----------
        # `o_proj`'s per-device K is 1024 = 32 tiles.  On the 16-core boundary
        # grid that is 2 tiles/core, so `in0_block_w <= 2` -- the one decode
        # matmul `tt-perf-report` marks SLOW at 62 % of peak DRAM.  The gated
        # attention tensor that feeds it is *already* written on `o_proj`'s core
        # count (see `decode_forward`), so a narrower grid for the attention
        # gate + `o_proj` pair costs no extra reshard.
        "oproj_c8_bw4": {"decode_matmul": geometry(o_proj__bfp8=(8, 4))},
        "oproj_c4_bw8": {"decode_matmul": geometry(o_proj__bfp8=(4, 8))},
        "oproj_c2_bw16": {"decode_matmul": geometry(o_proj__bfp8=(2, 16))},
        "oproj_c1_bw32": {"decode_matmul": geometry(o_proj__bfp8=(1, 32))},
        "oproj_c8_bw2": {"decode_matmul": geometry(o_proj__bfp8=(8, 2))},
        "oproj_c4_bw4": {"decode_matmul": geometry(o_proj__bfp8=(4, 4))},
        # -- OPT-007: attention-weight precision, on the multichip topology ----
        "attn_bfp4": {"precision": precision("attn-bfp4-mlp-bfp4-kv-bfp8-lofi", attn_weight_dtype=BFP4)},
        "attn_bf16": {"precision": precision("attn-bf16-mlp-bfp4-kv-bfp8-lofi", attn_weight_dtype=BF16)},
        # -- decode fidelity, re-measured on the multichip topology ------------
        "fid_hifi2": {"precision": precision("attn-bfp8-mlp-bfp4-kv-bfp8-hifi2", decode_math_fidelity=HIFI2)},
        # -- activation / residual dtype ---------------------------------------
        "act_bfp8": {"precision": precision("act-bfp8", activation_dtype=BFP8)},
        # -- KV cache dtype (OPT-002) ------------------------------------------
        "kv_bfp4": {"precision": precision("kv-bfp4", kv_cache_dtype=BFP4)},
        "kv_bf16": {"precision": precision("kv-bf16", kv_cache_dtype=BF16)},
        # -- OPT-009: the async CCL primitives, caller-owned semaphores, and the
        # all-gather tuning surface `ttnn.all_gather` does not expose ----------
        "ccl_async": {"decode_ccl_impl": "async"},
        "ccl_async_persist": {"decode_ccl_impl": "async", "ccl_persistent_buffers": True},
        "ccl_async_agw1": {"decode_ccl_impl": "async", "decode_ccl_ag_workers": 1},
        "ccl_async_agw2": {"decode_ccl_impl": "async", "decode_ccl_ag_workers": 2},
        "ccl_async_agw4": {"decode_ccl_impl": "async", "decode_ccl_ag_workers": 4},
        "ccl_async_agw1_persist": {
            "decode_ccl_impl": "async",
            "decode_ccl_ag_workers": 1,
            "ccl_persistent_buffers": True,
        },
        "ccl_best": _CCL_BEST,
        "ccl_best_cps2": {**_CCL_BEST, "ccl_chunks_per_sync": 2},
        "ccl_best_cps10": {**_CCL_BEST, "ccl_chunks_per_sync": 10},
        "ccl_best_cps20": {**_CCL_BEST, "ccl_chunks_per_sync": 20},
        "ccl_best_buf2": {**_CCL_BEST, "ccl_buffers_per_channel": 2},
        "ccl_best_buf8": {**_CCL_BEST, "ccl_buffers_per_channel": 8},
        "ccl_best_links1": {**_CCL_BEST, "ccl_num_links": 1},
        "ccl_best_rsw2": {**_CCL_BEST, "decode_ccl_rs_workers": 2},
        "ccl_best_oproj8": {**_CCL_BEST, "decode_matmul": geometry(o_proj__bfp8=(8, 4))},
        "ccl_async_prefill": {"prefill_ccl_impl": "async"},
        "ccl_async_prefill_agw4": {"prefill_ccl_impl": "async", "prefill_ccl_ag_workers": 4},
        "ccl_async_prefill_agw2": {"prefill_ccl_impl": "async", "prefill_ccl_ag_workers": 2},
        "ccl_async_prefill_rsw4": {"prefill_ccl_impl": "async", "prefill_ccl_rs_workers": 4},
        # -- same-config repeat controls: identical to ``tp4``, run again in the
        # same process, so the prefill noise floor is measured rather than assumed
        "tp4b": {},
        "tp4c": {},
        # -- the shipped multichip decode collective, for the before/after row --
        "ccl_wrapper": {"ccl_impl": "wrapper"},
        "no_sharded_io": {"sharded_decode_io": False},
        "before": dict(_BEFORE),
        "beforeb": dict(_BEFORE),
    }
)


if __name__ == "__main__":
    main()
