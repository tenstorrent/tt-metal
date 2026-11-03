# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestSync,
    TopKSortDirection,
    TopKXLChunkBaseMode,
    TopKXLIndexOp,
    TopKXLSortMode,
)
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_topk_xl import (
    GROUP_ID,
    GROUP_SHIFT,
    _run,
    _run_test_topk,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    num_chunks=[1, 2, 4],
    partial_tail=lambda num_chunks: [False] if num_chunks == 1 else [False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl(
    perf_report, K, num_chunks, partial_tail, run_types, loop_factor, is_perf
):
    _run_test_topk(
        K,
        num_chunks=num_chunks,
        tail_elements=(K // 2) if partial_tail else K,
        num_rows=2,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_input_modes(perf_report, K, run_types, loop_factor, is_perf):
    # signed and random share one TOPK_XL field set; keep the signed stimulus.
    _run_test_topk(
        K,
        num_chunks=2,
        mode="signed",
        compare_index_set=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_all_equal(perf_report, K, run_types, loop_factor, is_perf):
    _run(
        K,
        mode="all_equal",
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    core_id=[0, 1, 31],
    group_shift=lambda core_id: [16, 11, 20] if core_id == 0 else [16],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_separate_indices(
    perf_report, K, core_id, group_shift, run_types, loop_factor, is_perf
):
    _run(
        K,
        index_op=TopKXLIndexOp.Separate,
        group_id=GROUP_ID,
        group_shift=group_shift,
        core_id=core_id,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_remove_msb(perf_report, K, run_types, loop_factor, is_perf):
    _run(
        K,
        index_op=TopKXLIndexOp.RemoveMsb,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    chunk_base_mode=list(TopKXLChunkBaseMode),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_chunk_base(
    perf_report, K, chunk_base_mode, run_types, loop_factor, is_perf
):
    chunk_base = 0x1F800
    _run_test_topk(
        K,
        num_chunks=2,
        num_rows=2,
        chunk_base_mode=chunk_base_mode,
        chunk_base=chunk_base,
        index_offset=chunk_base,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_rebuild_ascending(
    perf_report, K, run_types, loop_factor, is_perf
):
    _run(
        K,
        num_chunks=2,
        sort_direction=TopKSortDirection.Ascending,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], num_chunks=[2, 4], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_fused_reduce(
    perf_report, K, num_chunks, run_types, loop_factor, is_perf
):
    _run(
        K,
        num_chunks=num_chunks,
        fused_reduce=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], num_chunks=[2, 4, 32], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_fused_e2e(
    perf_report, K, num_chunks, run_types, loop_factor, is_perf
):
    mode = f"planted:{K}" if num_chunks * K > 16384 else "positive"
    _run_test_topk(
        K,
        num_chunks=num_chunks,
        fused_e2e=True,
        mode=mode,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048], seg_base_chunks=[32, 96, 160], **PERF_MEASUREMENT_AXES
)
def test_perf_topk_xl_fused_e2e_segment_base(
    perf_report, K, seg_base_chunks, run_types, loop_factor, is_perf
):
    seg_base = seg_base_chunks * K
    _run_test_topk(
        K,
        num_chunks=4,
        fused_e2e=True,
        seg_base=seg_base,
        index_offset=seg_base,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], with_seg_base=[False, True], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_fused_e2e_partial_tail(
    perf_report, K, with_seg_base, run_types, loop_factor, is_perf
):
    seg_base = (32 * K) if with_seg_base else 0
    _run_test_topk(
        K,
        num_chunks=3,
        tail_elements=K // 2 + 17,
        fused_e2e=True,
        seg_base=seg_base,
        index_offset=seg_base,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_fused_e2e_two_rows(perf_report, run_types, loop_factor, is_perf):
    _run_test_topk(
        1024,
        num_chunks=4,
        num_rows=2,
        fused_e2e=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_fused_e2e_single_chunk(
    perf_report, K, run_types, loop_factor, is_perf
):
    _run_test_topk(
        K,
        num_chunks=1,
        fused_e2e=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], partial_tail=[False, True], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_input_float32(
    perf_report, K, partial_tail, run_types, loop_factor, is_perf
):
    _run_test_topk(
        K,
        num_chunks=2,
        tail_elements=(K // 2) if partial_tail else K,
        formats=InputOutputFormat(DataFormat.Float32, DataFormat.UInt32),
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_dest_sync_half(perf_report, K, run_types, loop_factor, is_perf):
    _run_test_topk(
        K,
        num_chunks=2,
        num_rows=2,
        dest_sync=DestSync.Half,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_local_sort_order(perf_report, K, run_types, loop_factor, is_perf):
    # Descending Separate with the default group shift is test_perf_topk_xl_separate_indices.
    _run(
        K,
        index_op=TopKXLIndexOp.Separate,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        sort_direction=TopKSortDirection.Ascending,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_local_sort_generic(
    perf_report, K, run_types, loop_factor, is_perf
):
    _run_test_topk(
        K,
        num_rows=2,
        sort_mode=TopKXLSortMode.Generic,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[1024, 2048],
    sort_direction=list(TopKSortDirection),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_local_sort_early_exit(
    perf_report, K, sort_direction, run_types, loop_factor, is_perf
):
    _run(
        K,
        index_op=TopKXLIndexOp.Separate,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        sort_mode=TopKXLSortMode.EarlyExitK64,
        sort_direction=sort_direction,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    num_chunks=[2, 4],
    sort_direction=list(TopKSortDirection),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_rebuild_columns(
    perf_report, num_chunks, sort_direction, run_types, loop_factor, is_perf
):
    _run(
        1024,
        num_chunks=num_chunks,
        sort_mode=TopKXLSortMode.Columns,
        sort_direction=sort_direction,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    core_id=[0, 31],
    blaze_compat=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_add_lsb_indices_row_major(
    perf_report, K, core_id, blaze_compat, run_types, loop_factor, is_perf
):
    _run(
        K,
        index_op=TopKXLIndexOp.Separate,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        core_id=core_id,
        lsb_row_major=True,
        blaze_compat=blaze_compat,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], core_id=[0, 31], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_remove_msb_row_major(
    perf_report, K, core_id, run_types, loop_factor, is_perf
):
    _run(
        K,
        index_op=TopKXLIndexOp.RemoveMsb,
        core_id=core_id,
        lsb_row_major=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    num_chunks=[2, 4],
    fused=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_reinit_after_copy(
    perf_report, K, num_chunks, fused, run_types, loop_factor, is_perf
):
    if not fused:
        _run_test_topk(
            K,
            num_chunks=num_chunks,
            num_rows=2,
            reinit_after_copy=True,
            **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
        )
        return
    _run(
        K,
        num_chunks=num_chunks,
        fused_reduce=True,
        reinit_after_copy=True,
        group_id=GROUP_ID,
        group_shift=GROUP_SHIFT,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    K=[512, 1024, 2048],
    fused_e2e=[False, True],
    input_fp32=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_topk_xl_blaze_compat(
    perf_report, K, fused_e2e, input_fp32, run_types, loop_factor, is_perf
):
    _run_test_topk(
        K,
        num_chunks=4,
        tail_elements=K // 2,
        num_rows=2,
        mode="signed",
        reinit_after_copy=True,
        fused_e2e=fused_e2e,
        blaze_compat=True,
        formats=InputOutputFormat(
            DataFormat.Float32 if input_fp32 else DataFormat.Float16_b,
            DataFormat.UInt32,
        ),
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(K=[512, 1024, 2048], core_id=[0, 31], **PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_blaze_compat_pack_remove_msb(
    perf_report, K, core_id, run_types, loop_factor, is_perf
):
    _run(
        K,
        num_rows=2,
        index_op=TopKXLIndexOp.RemoveMsb,
        core_id=core_id,
        blaze_compat=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
