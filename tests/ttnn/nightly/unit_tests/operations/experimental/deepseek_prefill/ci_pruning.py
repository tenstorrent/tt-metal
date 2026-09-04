# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Collection-time pruning rules for the deepseek_prefill nightly suites.

Every rule lives here so a change to what CI covers is one edit instead of one
edit per decorator; `conftest.py` owns the `uncollect_if` marker that applies
them. Predicates only prune in CI - a local run still collects everything.
"""

import ttnn


def _ci_only(pred):
    """Wrap `pred` so it prunes in CI and is a no-op everywhere else."""

    def uncollect_if(**params):
        if not (params["is_ci_env"] or params["is_ci_v2_env"]):
            return False
        return pred(**params)

    return uncollect_if


# Whole module: the op has no production counterpart.
no_production_counterpart = _ci_only(lambda **params: True)

# Blackhole prefill always hands the cast a tile-layout input.
bh_row_major_input = _ci_only(lambda **params: params["is_bh"] and params["input_layout"] == ttnn.ROW_MAJOR_LAYOUT)

# Blackhole prefill keeps the scales at fp32.
bh_narrow_scales_to_bf16 = _ci_only(lambda **params: params["is_bh"] and params["narrow_scales_to_bf16"])

# Blackhole prefill reads the scales back from the packed metadata, never precomputed.
bh_scales_not_from_metadata = _ci_only(lambda **params: params["is_bh"] and not params["scales_from_metadata"])

# fp32 logits cover the pre-#51009 host-typecast gate contract; production feeds bf16 directly.
bh_fp32_input = _ci_only(lambda **params: params["is_bh"] and params["input_dtype"] == ttnn.float32)

# The routed expert is always fed a row-major activation in production.
tiled_x_input = _ci_only(lambda **params: not params["x_row_major"])
