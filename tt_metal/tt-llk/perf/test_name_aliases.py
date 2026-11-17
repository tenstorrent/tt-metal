# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Historical perf test names needed by the sparse-checkout perf gate."""

TEST_NAME_ALIASES = {
    "perf_eltwise_binary_fpu": "perf_eltwise_binary",
    "perf_eltwise_typecast": "perf_eltwise_unary_typecast",
    "perf_fast_untilize_baseline_compare": "perf_pack_untilize",
    "perf_math_transpose": "perf_transpose_dest",
    "perf_vif_targets": "perf_vif_equiv_sweep",
}


def canonical_test_name(name):
    return TEST_NAME_ALIASES.get(name, name)
