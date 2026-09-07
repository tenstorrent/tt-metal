# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the head-GROUPED concat_heads compute the same thing as the single-call one?

TP=1 is the only reason head grouping exists: ``nlp_concat_heads``' src0 circular buffer is
``2 x heads x head_dim/32`` tiles regardless of sequence length, so a Gemma 4 global layer costs
0.5 MB at TP=4 (8 heads x head_dim 512) and 2.0 MB at TP=1 -- past Blackhole's 1.5 MB L1. The fix
splits the heads, concatenates the per-group results on the embedding axis, and is *argued* to be
identical because heads are contiguous there in head order.

That argument is exactly the kind that is right until it is not: get the group ordering wrong and
o_proj receives a permuted embedding, which produces finite, plausible, wrong numbers with nothing
to catch them -- this branch has no PCC gate on the context-parallel prefill path at all. So check
it against the definition rather than against another device run:

    nlp_concat_heads([1, H, S, D])  ==  x.permute(0, 2, 1, 3).reshape(1, 1, S, H*D)

Both the grouped and ungrouped paths are compared to that torch identity, at every head count
Gemma 4-31B actually produces between TP=1 and TP=8. Seconds to run; no weights, no model.

    python models/demos/gemma4/tests/perf/pp4/verify_concat_heads.py
"""

import os
import sys


def main():
    import torch
    import ttnn
    from loguru import logger

    from models.demos.gemma4.tt.attention.operations import _concat_heads_groups, concat_heads

    # (heads, head_dim, label) -- Gemma 4-31B has 32 Q heads, head_dim 256 on sliding layers and
    # 512 on global ones, so this is the whole reachable set from TP=8 down to TP=1.
    CASES = [
        (4, 256, "sliding TP=8"),
        (8, 256, "sliding TP=4"),
        (16, 256, "sliding TP=2"),
        (32, 256, "sliding TP=1"),
        (4, 512, "global  TP=8"),
        (8, 512, "global  TP=4"),
        (16, 512, "global  TP=2"),
        (32, 512, "global  TP=1"),
    ]
    SEQ = 1024  # a CP=8 rank's local Q slab at chunk 8192

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    rc = 1
    try:
        torch.manual_seed(0)
        failures = []
        for heads, head_dim, label in CASES:
            x = torch.randn(1, heads, SEQ, head_dim, dtype=torch.bfloat16)
            groups = _concat_heads_groups(
                ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            )
            tt_in = ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = ttnn.to_torch(concat_heads(tt_in, is_decode_mode=False))
            # The definition nlp_concat_heads implements: heads laid out contiguously along the
            # embedding axis, in head order.
            want = x.permute(0, 2, 1, 3).reshape(1, 1, SEQ, heads * head_dim)
            exact = torch.equal(out.to(torch.bfloat16), want)
            cb_mb = 2 * heads * (head_dim // 32) * 2048 / 2**20
            logger.warning(
                f"[concat] {label}: heads={heads:2d} head_dim={head_dim} -> groups={groups} "
                f"(ungrouped CB would be {cb_mb:.2f} MB) exact={exact}"
            )
            if not exact:
                failures.append(label)

        # The point of the exercise: TP=1 global must be grouped, TP=4 global must not be.
        assert _concat_heads_groups(
            ttnn.from_torch(torch.zeros(1, 32, SEQ, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh)
        ) > 1, "TP=1 global layer is not being grouped; it will not fit L1"
        assert _concat_heads_groups(
            ttnn.from_torch(torch.zeros(1, 8, SEQ, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh)
        ) == 1, "TP=4 global layer is being grouped; the baseline path must stay a single call"

        rc = 0 if not failures else 1
        logger.warning(f"[concat] RESULT {'PASS' if rc == 0 else 'FAIL: ' + ', '.join(failures)}")
    except Exception:
        from loguru import logger as lg

        lg.exception("[concat] FAILED")
    finally:
        ttnn.close_mesh_device(mesh)
    sys.exit(rc)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))
    main()
