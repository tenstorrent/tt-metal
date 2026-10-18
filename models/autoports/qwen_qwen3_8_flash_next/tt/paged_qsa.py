# SPDX-License-Identifier: Apache-2.0
"""Bounded physical pooled-index storage, with one reusable lane scoring buffer."""
from pathlib import Path
from types import SimpleNamespace

import ttnn

from .indexer_prefix import prepare_indexer_prefix
from .optimized_decoder import OptimizedQSA


def mask_scores_per_row(attention, scores, positions, seq):
    """The canonical suffix mask with one original search bound per decode row."""
    if seq != 1 or attention.batch_size not in (4, 8, 16, 32):
        raise ValueError("Per-row selection is qualified only for B4/8/16/32 decode")
    bound = ttnn.empty(
        (attention.batch_size, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=attention.device
    )
    grid = attention.device.compute_with_storage_grid_size()
    workers = grid.x * grid.y
    if attention.batch_size > workers:
        raise ValueError("Per-row TopK requires at least one worker per input row")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    tensors = [scores, positions, bound]
    args = [attention.batch_size, seq, attention.blocks, workers]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "qsa_mask_per_row.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[
            ((c.x, c.y), [*[tensor.buffer_address() for tensor in tensors], i]) for i, c in enumerate(coords)
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=768,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=768)],
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return bound


def transfer(attention, source, output, positions, page_table, *, seq, lane=None):
    """Commit completed pooled rows, or gather one lane's complete prefix.

    Page IDs retain the scheduler's coordinates. Every gather defines the complete
    pooled prefix. The score mask discards all incomplete columns before TopK;
    whole suffix tiles may retain arbitrary data from the preceding lane. The
    next lane reuses the workspace after score consumption on the same queue.
    """
    gather = lane is not None
    work = attention.blocks // 32 * 4 if gather else attention.batch_size * 4
    grid = attention.device.compute_with_storage_grid_size()
    _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, work, True)
    tensors = [source, output, positions, page_table]
    args = [
        int(gather),
        attention.batch_size,
        seq,
        attention.max_context,
        attention.pool_tokens // 32,
        0 if lane is None else lane,
    ]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    coords = ttnn.corerange_to_cores(cores, row_wise=True)
    workers = len(coords)
    if gather:
        attention._pooled_gather_partition = dict(kind="cyclic", work=work, workers=workers)
    runtime, offset = [], 0
    for i, core in enumerate(coords):
        if gather:
            # Spread a short live prefix over workers instead of assigning it
            # to the first contiguous full-capacity range alone.
            begin, count, stride = i, (work + workers - 1 - i) // workers, workers
        else:
            count = work1 if i < group1.num_cores() else work2
            begin, stride = offset, 1
            offset += count
        runtime.append(((core.x, core.y), [*[t.buffer_address() for t in tensors], begin, count, stride]))
    scratch = 2048 + 128 + 128 + attention.max_context // 32 * 4
    cb = ttnn.CBDescriptor(
        total_size=scratch,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(Path(__file__).with_name("kernels") / "qsa_pooled_paged.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


def select_groups(attention, pooled, iq, positions, page_table, seq):
    batched_selection = seq == 1 and attention.batch_size in (4, 8, 16, 32)
    attention._qsa_batched_selection = batched_selection
    transfer(attention, pooled, attention.pooled_cache, positions, page_table, seq=seq)
    qq = ttnn.to_layout(ttnn.concat((attention.indexer_prefix, iq), dim=2), ttnn.TILE_LAYOUT)
    padded = (seq + 62) // 32 * 32
    qq = ttnn.pad(qq, [(0, 0), (0, 0), (0, padded - seq - 31), (0, 0)], 0)
    # Reuse the canonical mask kernel with its actual one-lane shape, without
    # mutating the live attention object's batch size during trace construction.
    mask_context = SimpleNamespace(batch_size=1, device=attention.device, blocks=attention.blocks)
    selected_rows = []
    for lane in range(attention.batch_size):
        transfer(attention, attention.pooled_cache, attention.pooled_keys, positions, page_table, seq=seq, lane=lane)
        prefix_args = {}
        if getattr(attention, "indexer_prefix_cap", False):
            prepare_indexer_prefix(
                attention.device,
                positions,
                attention.indexer_prefix_bound,
                seq=seq,
                lane=lane,
                blocks=attention.blocks,
            )
            prefix_args["valid_end_tensor"] = attention.indexer_prefix_bound
        scores = ttnn.experimental.indexer_score_dsa(
            qq[lane : lane + 1],
            attention.pooled_keys,
            attention.indexer_weights[lane : lane + 1, :, :padded, :],
            chunk_start_idx=attention.blocks - 32,
            key_compression_ratio=1,
            cache_batch_idx=0,
            program_config=ttnn.IndexerScoreProgramConfig(
                q_chunk_size=32,
                k_chunk_size=attention.indexer_chunk,
                head_group_size=4,
            ),
            compute_kernel_config=attention.indexer_compute,
            **prefix_args,
        )
        scores = scores[:, :, 31 : 31 + seq, :]
        scores = ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        if batched_selection:
            selected_rows.append(scores)
            continue
        bound = OptimizedQSA._mask_scores(mask_context, scores, positions[lane : lane + 1], seq)
        selected_rows.append(ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound))
        # Only small logical-ID selections survive the lane. Never concatenate
        # B dense score matrices or return physical pooled row IDs to expansion.
        del scores, bound
    if batched_selection:
        scores = ttnn.concat(selected_rows, dim=0)
        bound = mask_scores_per_row(attention, scores, positions, seq)
        selected = ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound)
        grid = attention.device.compute_with_storage_grid_size()
        attention._qsa_batched_selection_geometry = dict(
            batch=attention.batch_size,
            sequence=seq,
            score_shape=list(scores.shape),
            score_padded_shape=list(scores.padded_shape),
            score_dtype=str(scores.dtype),
            score_layout=str(scores.layout),
            score_memory_config=str(scores.memory_config()),
            bound_shape=list(bound.shape),
            bound_padded_shape=list(bound.padded_shape),
            bound_dtype=str(bound.dtype),
            bound_layout=str(bound.layout),
            bound_memory_config=str(bound.memory_config()),
            output_shape=list(selected.shape),
            output_dtype=str(selected.dtype),
            output_layout=str(selected.layout),
            output_memory_config=str(selected.memory_config()),
            k=512,
            metadata_mode="per_row_one_row_per_core",
            workers=grid.x * grid.y,
        )
        return selected
    return selected_rows[0] if len(selected_rows) == 1 else ttnn.concat(selected_rows, dim=0)
