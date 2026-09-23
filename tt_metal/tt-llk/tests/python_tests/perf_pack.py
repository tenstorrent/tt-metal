# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.perf.relevance import _PACK_BLOCK_RUNTIMES, KEEP_ALL, PerfRelevance
from helpers.test_variant_parameters import DEST_INDEX, RELU_CONFIG
from test_pack import PACK_SWEEP
from test_pack import test_pack as run_pack


class PackRelevance(PerfRelevance):
    """``perf_pack`` / ``pack_test.cpp``, reached through ``test_pack``.

    The block geometry is shared by every mode. ``RELU_CONFIG`` is a packer
    register, so only PACK and L1_CONGESTION observe it -- a ReLU sweep
    therefore reuses UNPACK. MATH is full fidelity (empty TILE_LOOP when
    ``unpack_to_dest``) and is never stored in ``EXECUTE_CACHE``. The production
    perf test intentionally does not attach this policy.
    """

    math_templates = KEEP_ALL
    math_runtimes = KEEP_ALL
    math_runtime_fields = KEEP_ALL
    math_formats = KEEP_ALL
    unpack_runtimes = _PACK_BLOCK_RUNTIMES
    pack_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG, DEST_INDEX})
    cong_runtimes = _PACK_BLOCK_RUNTIMES | frozenset({RELU_CONFIG, DEST_INDEX})


PACK_RELEVANCE = PackRelevance()


@pytest.mark.perf
@parametrize(
    **{**PACK_SWEEP, "dest_index": [0]},
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack(
    perf_report,
    formats,
    dest_acc,
    input_dimensions,
    relu_type,
    dest_sync,
    dest_index,
    run_types,
    loop_factor,
    is_perf,
):
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        # Keep this measurement full-fidelity regardless of the global
        # LLK_DISABLE_PERF_RELEVANCE setting.
        relevance=None,
    )
