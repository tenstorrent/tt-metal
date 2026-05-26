# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_bfloat16_packed, float_to_uint32


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_singleton_prefix_shape(shape, expected_last_dim: int) -> bool:
    dims = tuple(int(d) for d in shape)
    if not dims or dims[-1] != expected_last_dim:
        return False
    return all(d == 1 for d in dims[:-1])


class SamplingOp:
    """
    Sampling micro-op entry point.

    Current implementation supports k=1 (argmax fast path) for a single device
    with multi-core local winners reduced on a final core.
    """

    @staticmethod
    def golden(
        scores: torch.Tensor,
        indices: torch.Tensor,
        k: int = 32,
        p: float = 1.0,
        temperature: float = 0.6,
        rand_value: float | None = None,
        return_p_metadata: bool = False,
    ):
        """
        PyTorch reference for sampling.

        For k=1, this is argmax with deterministic tie-break on lowest index.
        For k>1, implements top-k, temperature scaling, softmax, top-p
        cumulative probability filtering, then random selection among the
        surviving candidates.

        The random value used for inverse-CDF selection is determined by:
          - rand_value: used directly if provided (for host-side verification
            against a kernel-returned random value).
          - If neither is given, a non-deterministic random value is used.

        Returns (selected_index, topk_indices) where:
          - selected_index: [1, 1] uint32 tensor with the chosen token index.
          - topk_indices:   [1, k] int64 tensor of the top-k original indices
                            (sorted by descending score).
        """
        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)

        # if k == 1:
        #     max_score = torch.max(scores_f32)
        #     tied_mask = scores_f32 == max_score
        #     selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)
        #     return selected_index.reshape(1, 1), selected_index.reshape(1, 1).to(torch.int64)
        if len(scores_f32) >= 32:
            top32_values, top32_positions = torch.topk(scores_f32, k=32, sorted=True)
            logger.debug(f"Top 32 values: {top32_values}")
            logger.debug(f"Top 32 positions: {top32_positions}")
        actual_k = min(k, len(scores_f32))
        topk_values, topk_positions = torch.topk(scores_f32, k=actual_k, sorted=True)
        topk_indices = indices_i64[topk_positions]

        scaled = (topk_values.to(torch.bfloat16) * torch.tensor(1.0 / temperature, dtype=torch.bfloat16)).float()
        probs = torch.softmax(scaled, dim=-1)
        logger.debug(f"Raw Probabilities: {probs}")

        cum_probs = torch.cumsum(probs, dim=-1)
        logger.debug(f"Cumulative probabilities: {cum_probs}")
        num_kept = int((cum_probs < p).sum().item()) + 1
        num_kept = max(1, min(num_kept, actual_k))

        logger.debug(f"Number of kept tokens: {num_kept}")

        kept_probs = probs[:num_kept]
        kept_indices = topk_indices[:num_kept]
        kept_probs = kept_probs / kept_probs.sum()

        logger.debug(f"Kept probabilities: {kept_probs}")
        logger.debug(f"Kept indices: {kept_indices}")

        if rand_value is not None:
            rand_val = rand_value
        else:
            rand_val = torch.rand(1).item()

        cum_sum = 0.0
        selected_pos = num_kept - 1
        for i in range(num_kept):
            cum_sum += kept_probs[i].item()
            if cum_sum > rand_val:
                selected_pos = i
                break

        selected_index = kept_indices[selected_pos].to(torch.uint32)

        if return_p_metadata:
            # Golden layout for DeepseekMetadata.p_scores / p_indices:
            #   p_scores[0, num_kept) = kept_probs (rescaled, in bf16)
            #   p_scores[num_kept, actual_k) = 0.0 (zeroed by kernel after top-P cutoff)
            #   p_scores[actual_k, 32) = undefined (caller should not assert).
            #   p_indices[0, actual_k) = topk_indices.
            p_scores_golden = torch.zeros(actual_k, dtype=torch.bfloat16)
            p_scores_golden[:num_kept] = kept_probs.to(torch.bfloat16)
            p_indices_golden = topk_indices[:actual_k].to(torch.int64)
            return (
                selected_index.reshape(1, 1),
                topk_indices.reshape(1, -1),
                p_scores_golden,
                p_indices_golden,
            )

        return selected_index.reshape(1, 1), topk_indices.reshape(1, -1)

    @staticmethod
    def op(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        k: int = 32,
        p: float = 1.0,
        temperature: float = 0.6,
        seed: int = 520,
        rand_output_tensor=None,
        final_core_coord=None,
        final_mesh_coord=None,
        global_semaphore=None,
        global_stage2_semaphore=None,
        fabric_scratch_tensor=None,
        scores_scratch_tensor=None,
        indices_scratch_tensor=None,
        metadata_output_tensor=None,
        copy_probabilities: bool = False,
        mesh_axis: str = "x",
        num_internal_iterations: int = 1,
    ):
        """
        Execute sampling.

        Args:
            scores_tensor: [1, shard_width * num_cores] bfloat16, WIDTH_SHARDED with shard shape [1, shard_width].
            indices_tensor: [1, shard_width * num_cores] uint32, WIDTH_SHARDED with shard shape [1, shard_width].
            output_index_tensor: [1, 1] uint32, sharded on final core.
            k: sampling k; currently only k=1 supported.
            p: top-p threshold (unused for k=1).
            final_core_coord: target output core coordinate (validated, optional).
            final_mesh_coord: mesh coordinate containing final output in mesh mode.
            global_semaphore: external global semaphore handle used for stage-1 inter-device sync.
            global_stage2_semaphore: external global semaphore handle used for stage-2 inter-device sync.
            fabric_scratch_tensor: persistent L1 scratch tensor (single-core sharded) for fabric slot exchange (k=1).
            scores_scratch_tensor: persistent L1 scratch tensor for mesh stage scores exchange (k>1).
            indices_scratch_tensor: persistent L1 scratch tensor for mesh stage indices exchange (k>1).
            mesh_axis: reduction axis for first stage; currently only "x" is supported.
        """
        mesh_device = scores_tensor.device()
        mesh_mode_requested = mesh_device.get_num_devices() > 1
        if mesh_mode_requested:
            if global_semaphore is None:
                raise ValueError("global_semaphore is required in mesh mode")
            if global_stage2_semaphore is None:
                raise ValueError("global_stage2_semaphore is required in mesh mode")
            if final_mesh_coord is None:
                raise ValueError("final_mesh_coord is required in mesh mode")
            if mesh_axis != "x":
                raise NotImplementedError("Sampling mesh mode currently supports only mesh_axis='x'")
            if mesh_device.shape[0] >= 2 and mesh_device.shape[1] == 2:
                if scores_scratch_tensor is None or indices_scratch_tensor is None:
                    raise ValueError(
                        "scores_scratch_tensor and indices_scratch_tensor are required in mesh mode for k>1"
                    )
                return SamplingOp._op_mesh_topk(
                    scores_tensor=scores_tensor,
                    indices_tensor=indices_tensor,
                    output_index_tensor=output_index_tensor,
                    k=k,
                    p=p,
                    temperature=temperature,
                    seed=seed,
                    rand_output_tensor=rand_output_tensor,
                    final_core_coord=final_core_coord,
                    final_mesh_coord=final_mesh_coord,
                    global_semaphore=global_semaphore,
                    global_stage2_semaphore=global_stage2_semaphore,
                    scores_scratch_tensor=scores_scratch_tensor,
                    indices_scratch_tensor=indices_scratch_tensor,
                    mesh_axis=mesh_axis,
                    num_internal_iterations=num_internal_iterations,
                    metadata_output_tensor=metadata_output_tensor,
                    copy_probabilities=copy_probabilities,
                )
        else:
            return SamplingOp._op_single_device_topk(
                scores_tensor=scores_tensor,
                indices_tensor=indices_tensor,
                output_index_tensor=output_index_tensor,
                k=k,
                p=p,
                temperature=temperature,
                seed=seed,
                rand_output_tensor=rand_output_tensor,
                final_core_coord=final_core_coord,
                final_mesh_coord=final_mesh_coord,
                num_internal_iterations=num_internal_iterations,
                metadata_output_tensor=metadata_output_tensor,
                copy_probabilities=copy_probabilities,
            )

    @staticmethod
    def _op_single_device_topk(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        k: int = 32,
        p: float = 1.0,
        temperature: float = 0.6,
        seed: int = 520,
        rand_output_tensor=None,
        metadata_output_tensor=None,
        final_core_coord=None,
        final_mesh_coord=None,
        num_internal_iterations: int = 1,
        copy_probabilities: bool = False,
    ):
        """
        Single-device top-K sampling path (k >= 1).

        Each core finds its local top-K, sends them to the final core, which
        reduces across all gathered candidates. Currently outputs the argmax
        as a stepping stone; later phases will add softmax + top-P + random.
        """
        scores_shard_spec = scores_tensor.memory_config().shard_spec
        indices_shard_spec = indices_tensor.memory_config().shard_spec
        output_shard_spec = output_index_tensor.memory_config().shard_spec

        all_cores = scores_shard_spec.grid
        num_cores = all_cores.num_cores()
        assert num_cores >= 1, "Sampling requires at least one active core"
        assert indices_shard_spec.grid == all_cores, "Scores and indices must be sharded on the same core grid"
        assert output_shard_spec.grid.num_cores() == 1, "Output tensor must be sharded on a single final core"
        assert scores_tensor.dtype == ttnn.bfloat16, "Scores tensor must be bfloat16"
        assert indices_tensor.dtype == ttnn.uint32, "Indices tensor must be uint32"
        assert output_index_tensor.dtype == ttnn.uint32, "Output index tensor must be uint32"
        assert scores_shard_spec.shape[0] == 1, f"Expected scores shard height 1, got {scores_shard_spec.shape[0]}"
        assert indices_shard_spec.shape[0] == 1, f"Expected indices shard height 1, got {indices_shard_spec.shape[0]}"
        assert (
            scores_shard_spec.shape[1] == indices_shard_spec.shape[1]
        ), f"Scores/indices shard widths must match, got {scores_shard_spec.shape[1]} and {indices_shard_spec.shape[1]}"
        num_values = int(scores_shard_spec.shape[1])
        assert num_values > 0, "Sampling shard width must be > 0"
        assert k >= 1 and k <= num_values, f"k={k} must be >= 1 and <= num_values={num_values} per core"
        assert _is_singleton_prefix_shape(
            scores_tensor.shape, num_values * num_cores
        ), f"Expected scores shape [1, ..., 1, {num_values * num_cores}], got {scores_tensor.shape}"
        assert _is_singleton_prefix_shape(
            indices_tensor.shape, num_values * num_cores
        ), f"Expected indices shape [1, ..., 1, {num_values * num_cores}], got {indices_tensor.shape}"
        assert tuple(output_index_tensor.shape) == (
            1,
            1,
        ), f"Expected output shape (1, 1), got {output_index_tensor.shape}"

        output_core = output_shard_spec.grid.ranges()[0].start
        if final_core_coord is not None:
            assert (
                final_core_coord.x == output_core.x and final_core_coord.y == output_core.y
            ), "final_core_coord must match output shard core"
        else:
            final_core_coord = output_core

        if final_mesh_coord is not None:
            assert (
                final_mesh_coord[0] == 0 and final_mesh_coord[1] == 0
            ), "Single-device sampling currently expects final_mesh_coord=(0, 0)"

        sender_cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
        loop_mcast_bbox = all_cores.bounding_box()
        loop_mcast_start_worker = scores_tensor.device().worker_core_from_logical_core(loop_mcast_bbox.start)
        loop_mcast_end_worker = scores_tensor.device().worker_core_from_logical_core(loop_mcast_bbox.end)
        loop_mcast_start_x = int(loop_mcast_start_worker.x)
        loop_mcast_start_y = int(loop_mcast_start_worker.y)
        loop_mcast_end_x = int(loop_mcast_end_worker.x)
        loop_mcast_end_y = int(loop_mcast_end_worker.y)
        loop_mcast_logical_width = int(loop_mcast_bbox.end.x) - int(loop_mcast_bbox.start.x) + 1
        loop_mcast_logical_height = int(loop_mcast_bbox.end.y) - int(loop_mcast_bbox.start.y) + 1
        loop_mcast_num_dests = loop_mcast_logical_width * loop_mcast_logical_height - 1
        logger.info(
            f"Loop mcast bbox logical start={loop_mcast_bbox.start}, end={loop_mcast_bbox.end}, "
            f"worker start=({loop_mcast_start_x}, {loop_mcast_start_y}), "
            f"worker end=({loop_mcast_end_x}, {loop_mcast_end_y}), "
            f"logical_size={loop_mcast_logical_width}x{loop_mcast_logical_height}, num_dests={loop_mcast_num_dests}"
        )

        assert any(
            core.x == final_core_coord.x and core.y == final_core_coord.y for core in sender_cores
        ), "final_core_coord must be in scores/indices shard grid"
        final_is_sender = any(core.x == output_core.x and core.y == output_core.y for core in sender_cores)
        expected_remote_incs = num_cores - 1 if final_is_sender else num_cores

        topk_min_alignment = 32
        winner_cb = 0
        softmax_in_cb = 2
        softmax_out_cb = 3
        max_cb = 4
        sum_cb = 5
        scaler_cb = 6
        softmax_exp_cb = 7
        temp_cb = 8
        softmax_sub_cb = 9
        rand_cb = 10
        topk_in_scores_cb = 11
        topk_in_indices_cb = 12
        topk_out_scores_cb = 13
        topk_out_indices_cb = 14
        semaphore_id = 0
        local_ready_semaphore_id = 1
        l1_alignment = 16
        bf16_tile_size = 2 * 32 * 32  # 2048 bytes per bf16 32x32 tile
        uint32_tile_size = 4 * 32 * 32  # 4096 bytes per uint32 32x32 tile

        logger.debug(f"Temperature: {temperature}")
        logger.debug(f"1.0 / temperature: {1.0 / temperature}")
        inv_temp_bf16 = float_to_bfloat16_packed(1.0 / temperature)
        p_uint32_cast = float_to_uint32(p)
        logger.debug(f"Inv temp BF16: {inv_temp_bf16}")
        logger.debug(f"P uint32: {p_uint32_cast}, seed: {seed}")
        # Globally-split gather layout: all scores contiguous, then all indices contiguous.
        # Each per-core region is independently aligned for NOC transfers.
        topk_scores_slot_bytes = _round_up(topk_min_alignment * 2, l1_alignment)
        topk_indices_slot_bytes = _round_up(topk_min_alignment * 4, l1_alignment)
        winner_page_bytes = topk_scores_slot_bytes + topk_indices_slot_bytes

        phase1_tiles = (num_values + 1023) // 1024
        phase2_tiles = (topk_min_alignment * num_cores + 1023) // 1024
        total_input_tiles = phase1_tiles + phase2_tiles
        phase2_scores_byte_offset = phase1_tiles * bf16_tile_size
        phase2_indices_byte_offset = phase1_tiles * uint32_tile_size

        ncrisc_named_compile_time_args = [
            ("sampling_num_values", num_values),
            ("sampling_topk_k", k),
            ("sampling_winner_page_bytes", winner_page_bytes),
            ("sampling_num_senders", num_cores),
            ("sampling_expected_remote_incs", expected_remote_incs),
            ("sampling_winner_cb", winner_cb),
            ("sampling_receiver_semaphore_id", semaphore_id),
            ("sampling_local_ready_semaphore_id", local_ready_semaphore_id),
            ("sampling_mesh_mode", 0),
            ("sampling_stage1_sender", 0),
            ("sampling_stage1_receiver", 0),
            ("sampling_stage2_sender", 0),
            ("sampling_stage2_receiver", 0),
            ("sampling_stage1_slot_base_offset", 0),
            ("sampling_stage1_num_slots", 0),
            ("sampling_stage1_expected_remote_incs", 0),
            ("sampling_stage1_local_slot_offset", 0),
            ("sampling_stage2_slot_base_offset", 0),
            ("sampling_stage2_num_slots", 0),
            ("sampling_stage2_expected_remote_incs", 0),
            ("sampling_stage2_local_slot_offset", 0),
            ("sampling_mesh_send_slot_offset", 0),
            ("sampling_mesh_local_send_slot_offset", 0),
            ("sampling_softmax_in_cb", softmax_in_cb),
            ("sampling_softmax_out_cb", softmax_out_cb),
            ("sampling_softmax_exp_cb", softmax_exp_cb),
            ("sampling_scaler_cb", scaler_cb),
            ("sampling_temp_cb", temp_cb),
            ("sampling_inv_temp_bf16", inv_temp_bf16),
            ("sampling_topk_in_scores_cb", topk_in_scores_cb),
            ("sampling_topk_in_indices_cb", topk_in_indices_cb),
            ("sampling_topk_out_scores_cb", topk_out_scores_cb),
            ("sampling_topk_out_indices_cb", topk_out_indices_cb),
            ("sampling_phase2_scores_byte_offset", phase2_scores_byte_offset),
            ("sampling_phase2_indices_byte_offset", phase2_indices_byte_offset),
            ("sampling_mesh_stage_scores_cb", 0xFFFFFFFF),
            ("sampling_mesh_stage_indices_cb", 0xFFFFFFFF),
            ("sampling_scores_scratch_stage2_offset", 0),
            ("sampling_indices_scratch_stage2_offset", 0),
            ("sampling_scores_scratch_addr", 0),
            ("sampling_indices_scratch_addr", 0),
            ("sampling_loop_mcast_start_x", loop_mcast_start_x),
            ("sampling_loop_mcast_start_y", loop_mcast_start_y),
            ("sampling_loop_mcast_end_x", loop_mcast_end_x),
            ("sampling_loop_mcast_end_y", loop_mcast_end_y),
            ("sampling_loop_num_dests", loop_mcast_num_dests),
            ("sampling_num_internal_iterations", num_internal_iterations),
        ]

        logger.info(f"num_dests {loop_mcast_num_dests}")

        trisc_named_compile_time_args = [
            ("sampling_softmax_in_cb", softmax_in_cb),
            ("sampling_softmax_out_cb", softmax_out_cb),
            ("sampling_softmax_exp_cb", softmax_exp_cb),
            ("sampling_softmax_sub_cb", softmax_sub_cb),
            ("sampling_max_cb", max_cb),
            ("sampling_sum_cb", sum_cb),
            ("sampling_scaler_cb", scaler_cb),
            ("sampling_temp_cb", temp_cb),
            ("sampling_rand_cb", rand_cb),
            ("sampling_seed", seed),
            ("sampling_topk_k", k),
            ("sampling_mesh_mode", 0),
            ("sampling_stage1_receiver", 0),
            ("sampling_stage2_receiver", 0),
            ("sampling_num_values", num_values),
            ("sampling_num_senders", num_cores),
            ("sampling_topk_in_scores_cb", topk_in_scores_cb),
            ("sampling_topk_in_indices_cb", topk_in_indices_cb),
            ("sampling_topk_out_scores_cb", topk_out_scores_cb),
            ("sampling_topk_out_indices_cb", topk_out_indices_cb),
            ("sampling_mesh_stage_scores_cb", 0xFFFFFFFF),
            ("sampling_mesh_stage_indices_cb", 0xFFFFFFFF),
            ("sampling_stage1_row_elements", 0),
            ("sampling_stage1_num_input_tiles", 0),
            ("sampling_stage2_row_elements", 0),
            ("sampling_stage2_num_input_tiles", 0),
            ("sampling_num_internal_iterations", num_internal_iterations),
        ]
        brisc_named_compile_time_args = [
            ("sampling_winner_page_bytes", winner_page_bytes),
            ("sampling_local_ready_semaphore_id", 1),
            ("sampling_topk_k", k),
            ("sampling_softmax_out_cb", softmax_out_cb),
            ("sampling_rand_cb", rand_cb),
            ("sampling_winner_cb", winner_cb),
            ("sampling_p_bf16", p_uint32_cast),
            ("sampling_topk_scores_slot_bytes", topk_scores_slot_bytes),
            ("sampling_mesh_mode", 0),
            ("sampling_stage2_receiver", 0),
            ("sampling_output_addr", int(output_index_tensor.buffer_address())),
            (
                "sampling_rand_output_addr",
                int(rand_output_tensor.buffer_address()) if rand_output_tensor is not None else 0,
            ),
            ("sampling_num_internal_iterations", num_internal_iterations),
            ("sampling_softmax_in_cb", softmax_in_cb),
            ("sampling_temp_cb", temp_cb),
            ("sampling_inv_temp_bf16", inv_temp_bf16),
            ("sampling_enable_metadata", 1 if metadata_output_tensor is not None else 0),
            ("sampling_copy_probabilities", 1 if copy_probabilities else 0),
            (
                "sampling_metadata_address",
                int(metadata_output_tensor.buffer_address()) if metadata_output_tensor is not None else 0,
            ),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                # math_approx_mode=False routes recip_tile through the
                # accurate Newton-Raphson path (~24-bit precision with
                # fp32_dest_acc_en) instead of the ~7-bit LUT path. This is
                # the dominant precision knob for top-P sampling; the
                # approximate path was responsible for the ~3% relative
                # error we were seeing on softmax outputs.
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            ),
            ncrisc_common_runtime_args=[
                int(scores_tensor.buffer_address()),
                int(indices_tensor.buffer_address()),
                int(output_index_tensor.buffer_address()),
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).x),
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).y),
                0,
                0,
            ],
            brisc_common_runtime_args=[
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).x),
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).y),
            ],
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_is_final_core",
                    core_range=final_core_coord,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_mesh_sender_core",
                    core_range=all_cores,
                    value=0,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="sampling_sender_idx",
                    core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
                    other_value=0,
                ),
            ],
        )

        winner_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=winner_cb,
            data_format=ttnn.uint32,
            page_size=winner_page_bytes,
        )
        winner_cb_descriptor = ttnn.CBDescriptor(
            total_size=winner_page_bytes,
            core_ranges=all_cores,
            format_descriptors=[winner_cb_format],
        )
        final_core_crs = ttnn.CoreRangeSet([ttnn.CoreRange(final_core_coord, final_core_coord)])
        softmax_in_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=softmax_in_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )
        softmax_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=softmax_out_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                )
            ],
        )
        max_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=max_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )
        sum_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=sum_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )
        scaler_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=scaler_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )
        softmax_exp_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=softmax_exp_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                )
            ],
        )
        temp_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=temp_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )

        softmax_sub_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=softmax_sub_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                )
            ],
        )
        rand_cb_descriptor = ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=final_core_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=rand_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )

        topk_cbs = [
            ttnn.CBDescriptor(
                total_size=total_input_tiles * bf16_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=topk_in_scores_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                    )
                ],
            ),
            ttnn.CBDescriptor(
                total_size=total_input_tiles * uint32_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=topk_in_indices_cb, data_format=ttnn.uint32, page_size=uint32_tile_size
                    )
                ],
            ),
        ]

        topk_cbs.append(
            ttnn.CBDescriptor(
                total_size=bf16_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=topk_out_scores_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                    )
                ],
            )
        )
        topk_cbs.append(
            ttnn.CBDescriptor(
                total_size=uint32_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=topk_out_indices_cb, data_format=ttnn.uint32, page_size=uint32_tile_size
                    )
                ],
            )
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        full_grid = scores_tensor.device().compute_with_storage_grid_size()
        loop_mcast_bbox_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))]
        )
        local_ready_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=local_ready_semaphore_id,
            core_ranges=loop_mcast_bbox_grid,
            initial_value=0,
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                winner_cb_descriptor,
                softmax_in_cb_descriptor,
                softmax_out_cb_descriptor,
                max_cb_descriptor,
                sum_cb_descriptor,
                scaler_cb_descriptor,
                softmax_exp_cb_descriptor,
                temp_cb_descriptor,
                softmax_sub_cb_descriptor,
                rand_cb_descriptor,
            ]
            + topk_cbs,
            semaphores=[receiver_semaphore_descriptor, local_ready_semaphore_descriptor],
        )

        tensors = [scores_tensor, indices_tensor, output_index_tensor]
        if rand_output_tensor is not None:
            tensors.append(rand_output_tensor)
        ttnn.generic_op(tensors, program_descriptor)
        return output_index_tensor

    @staticmethod
    def _op_mesh_topk(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        rand_output_tensor,
        final_core_coord,
        final_mesh_coord,
        global_semaphore,
        global_stage2_semaphore,
        scores_scratch_tensor,
        indices_scratch_tensor,
        seed: int = 520,
        k: int = 32,
        p: float = 1.0,
        temperature: float = 0.6,
        metadata_output_tensor=None,
        copy_probabilities: bool = False,
        mesh_axis: str = "x",
        num_internal_iterations: int = 1,
    ):
        """
        Mesh (R,2) top-K sampling path (k > 1).

        Each device performs local top-K across its cores, then mesh stages
        merge top-K arrays across devices.  The absolute final device (stage-2
        receiver) runs softmax + top-P + random selection.
        """
        if mesh_axis != "x":
            raise NotImplementedError("Sampling mesh mode currently supports only mesh_axis='x'")

        mesh_device = scores_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = int(mesh_shape[0]), int(mesh_shape[1])
        if mesh_rows < 2 or mesh_cols != 2:
            raise ValueError(f"Sampling mesh mode currently requires an (R,2) mesh with R>=2, got {mesh_shape}")
        target_row, target_col = int(final_mesh_coord[0]), int(final_mesh_coord[1])
        if not (0 <= target_row < mesh_rows and 0 <= target_col < mesh_cols):
            raise ValueError(f"final_mesh_coord {final_mesh_coord} out of bounds for mesh shape {mesh_shape}")

        def _x_axis_link_idx_for_stage1_sender(sender_row: int) -> int:
            linear_distance = abs(int(sender_row) - target_row)
            ring_distance = min(linear_distance, mesh_rows - linear_distance)
            max_ring_distance = mesh_rows // 2
            first_half_threshold = (max_ring_distance + 1) // 2
            return 0 if ring_distance <= first_half_threshold else 1

        scores_per_device = ttnn.get_device_tensors(scores_tensor)
        indices_per_device = ttnn.get_device_tensors(indices_tensor)
        output_per_device = ttnn.get_device_tensors(output_index_tensor)
        scores_scratch_per_device = ttnn.get_device_tensors(scores_scratch_tensor)
        indices_scratch_per_device = ttnn.get_device_tensors(indices_scratch_tensor)
        rand_per_device = ttnn.get_device_tensors(rand_output_tensor) if rand_output_tensor is not None else None

        total_devices = mesh_shape[0] * mesh_shape[1]
        if not (
            len(scores_per_device)
            == len(indices_per_device)
            == len(output_per_device)
            == len(scores_scratch_per_device)
            == len(indices_scratch_per_device)
            == total_devices
        ):
            raise ValueError("All mesh tensors must have one device tensor per mesh coordinate")

        global_sem_addr = int(ttnn.get_global_semaphore_address(global_semaphore))
        global_stage2_sem_addr = int(ttnn.get_global_semaphore_address(global_stage2_semaphore))

        winner_cb = 0
        softmax_in_cb = 2
        softmax_out_cb = 3
        max_cb = 4
        sum_cb = 5
        scaler_cb = 6
        softmax_exp_cb = 7
        temp_cb = 8
        softmax_sub_cb = 9
        rand_cb = 10
        topk_in_scores_cb = 11
        topk_in_indices_cb = 12
        topk_out_scores_cb = 13
        topk_out_indices_cb = 14
        semaphore_id = 0
        local_ready_semaphore_id = 1

        l1_alignment = 16
        bf16_tile_size = 2 * 32 * 32
        uint32_tile_size = 4 * 32 * 32
        topk_min_alignment = 32

        inv_temp_bf16 = float_to_bfloat16_packed(1.0 / temperature)
        p_uint32_cast = float_to_uint32(p)
        topk_scores_slot_bytes = _round_up(topk_min_alignment * 2, l1_alignment)
        topk_indices_slot_bytes = _round_up(topk_min_alignment * 4, l1_alignment)
        winner_page_bytes = topk_scores_slot_bytes + topk_indices_slot_bytes
        stage1_num_slots = mesh_rows
        stage2_num_slots = mesh_cols
        mesh_stage_scores_cb = 15
        mesh_stage_indices_cb = 16

        stage1_mesh_tiles = (stage1_num_slots * topk_min_alignment + 1023) // 1024
        stage2_mesh_tiles = (stage2_num_slots * topk_min_alignment + 1023) // 1024
        total_mesh_stage_tiles = stage1_mesh_tiles + stage2_mesh_tiles
        stage2_scores_scratch_offset = stage1_mesh_tiles * bf16_tile_size
        stage2_indices_scratch_offset = stage1_mesh_tiles * uint32_tile_size
        required_scores_scratch_bytes = total_mesh_stage_tiles * bf16_tile_size
        required_indices_scratch_bytes = total_mesh_stage_tiles * uint32_tile_size

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for row in range(mesh_shape[0]):
            for col in range(mesh_shape[1]):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_shape[1] + col
                is_final_mesh_device = row == target_row and col == target_col

                scores_tensor_device = scores_per_device[device_idx]
                indices_tensor_device = indices_per_device[device_idx]
                output_tensor_device = output_per_device[device_idx]
                scores_scratch_device = scores_scratch_per_device[device_idx]
                indices_scratch_device = indices_scratch_per_device[device_idx]

                scores_shard_spec = scores_tensor_device.memory_config().shard_spec
                indices_shard_spec = indices_tensor_device.memory_config().shard_spec
                output_shard_spec = output_tensor_device.memory_config().shard_spec

                all_cores = scores_shard_spec.grid
                num_cores = all_cores.num_cores()
                if num_cores < 1:
                    raise ValueError("Sampling requires at least one active core")
                if indices_shard_spec.grid != all_cores:
                    raise ValueError("Scores and indices must be sharded on the same core grid")
                if output_shard_spec.grid.num_cores() != 1:
                    raise ValueError("Output tensor must be single-core sharded per device in mesh mode")
                num_values = int(scores_shard_spec.shape[1])

                sender_cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
                output_core = output_shard_spec.grid.ranges()[0].start
                if final_core_coord is not None:
                    if final_core_coord.x != output_core.x or final_core_coord.y != output_core.y:
                        raise ValueError("final_core_coord must match output shard core on all mesh devices")
                else:
                    final_core_coord = output_core

                if not any(c.x == final_core_coord.x and c.y == final_core_coord.y for c in sender_cores):
                    raise ValueError("final_core_coord must be in scores/indices shard grid")

                final_is_sender = any(c.x == final_core_coord.x and c.y == final_core_coord.y for c in sender_cores)
                expected_remote_incs = num_cores - 1 if final_is_sender else num_cores

                is_stage1_sender = row != target_row
                is_stage1_receiver = row == target_row
                is_stage2_sender = row == target_row and col != target_col
                is_stage2_receiver = is_final_mesh_device
                is_mesh_sender_core = is_stage1_sender or is_stage2_sender

                if is_stage1_sender:
                    dest_coord = ttnn.MeshCoordinate(target_row, col)
                    sender_link_idx = _x_axis_link_idx_for_stage1_sender(row)
                elif is_stage2_sender:
                    dest_coord = ttnn.MeshCoordinate(target_row, target_col)
                    sender_link_idx = 0
                else:
                    dest_coord = ttnn.MeshCoordinate(row, col)
                    sender_link_idx = 0

                phase1_tiles_mesh = (num_values + 1023) // 1024
                phase2_tiles_mesh = (topk_min_alignment * num_cores + 1023) // 1024
                total_input_tiles_mesh = phase1_tiles_mesh + phase2_tiles_mesh
                phase2_scores_byte_offset_mesh = phase1_tiles_mesh * bf16_tile_size
                phase2_indices_byte_offset_mesh = phase1_tiles_mesh * uint32_tile_size

                ncrisc_named_compile_time_args = [
                    ("sampling_num_values", num_values),
                    ("sampling_topk_k", k),
                    ("sampling_winner_page_bytes", winner_page_bytes),
                    ("sampling_num_senders", num_cores),
                    ("sampling_expected_remote_incs", expected_remote_incs),
                    ("sampling_winner_cb", winner_cb),
                    ("sampling_receiver_semaphore_id", semaphore_id),
                    ("sampling_local_ready_semaphore_id", local_ready_semaphore_id),
                    ("sampling_mesh_mode", 1),
                    ("sampling_stage1_sender", 1 if is_stage1_sender else 0),
                    ("sampling_stage1_receiver", 1 if is_stage1_receiver else 0),
                    ("sampling_stage2_sender", 1 if is_stage2_sender else 0),
                    ("sampling_stage2_receiver", 1 if is_stage2_receiver else 0),
                    ("sampling_stage1_slot_base_offset", 0),
                    ("sampling_stage1_num_slots", stage1_num_slots),
                    ("sampling_stage1_expected_remote_incs", mesh_rows - 1),
                    ("sampling_stage1_local_slot_offset", row),  # slot index
                    ("sampling_stage2_slot_base_offset", 0),
                    ("sampling_stage2_num_slots", stage2_num_slots),
                    ("sampling_stage2_expected_remote_incs", mesh_cols - 1),
                    ("sampling_stage2_local_slot_offset", col),  # slot index
                    ("sampling_mesh_local_send_slot_offset", 0),  # unused, sender reads from winner CB
                    ("sampling_softmax_in_cb", softmax_in_cb if is_final_mesh_device else 0),
                    ("sampling_softmax_out_cb", softmax_out_cb if is_final_mesh_device else 0),
                    ("sampling_softmax_exp_cb", softmax_exp_cb if is_final_mesh_device else 0),
                    ("sampling_scaler_cb", scaler_cb if is_final_mesh_device else 0),
                    ("sampling_temp_cb", temp_cb if is_final_mesh_device else 0),
                    ("sampling_inv_temp_bf16", inv_temp_bf16),
                    ("sampling_topk_in_scores_cb", topk_in_scores_cb),
                    ("sampling_topk_in_indices_cb", topk_in_indices_cb),
                    ("sampling_topk_out_scores_cb", topk_out_scores_cb),
                    ("sampling_topk_out_indices_cb", topk_out_indices_cb),
                    ("sampling_phase2_scores_byte_offset", phase2_scores_byte_offset_mesh),
                    ("sampling_phase2_indices_byte_offset", phase2_indices_byte_offset_mesh),
                    ("sampling_mesh_stage_scores_cb", mesh_stage_scores_cb),
                    ("sampling_mesh_stage_indices_cb", mesh_stage_indices_cb),
                    ("sampling_scores_scratch_stage2_offset", stage2_scores_scratch_offset),
                    ("sampling_indices_scratch_stage2_offset", stage2_indices_scratch_offset),
                    ("sampling_scores_scratch_addr", int(scores_scratch_device.buffer_address())),
                    ("sampling_indices_scratch_addr", int(indices_scratch_device.buffer_address())),
                    ("sampling_loop_mcast_start_x", 0),
                    ("sampling_loop_mcast_start_y", 0),
                    ("sampling_loop_mcast_end_x", 0),
                    ("sampling_loop_mcast_end_y", 0),
                    ("sampling_loop_num_dests", 0),
                    ("sampling_num_internal_iterations", num_internal_iterations),
                ]
                trisc_named_compile_time_args = [
                    ("sampling_softmax_in_cb", softmax_in_cb if is_final_mesh_device else 0),
                    ("sampling_softmax_out_cb", softmax_out_cb if is_final_mesh_device else 0),
                    ("sampling_softmax_exp_cb", softmax_exp_cb if is_final_mesh_device else 0),
                    ("sampling_softmax_sub_cb", softmax_sub_cb if is_final_mesh_device else 0),
                    ("sampling_max_cb", max_cb if is_final_mesh_device else 0),
                    ("sampling_sum_cb", sum_cb if is_final_mesh_device else 0),
                    ("sampling_scaler_cb", scaler_cb if is_final_mesh_device else 0),
                    ("sampling_temp_cb", temp_cb if is_final_mesh_device else 0),
                    ("sampling_rand_cb", rand_cb if is_final_mesh_device else 0),
                    ("sampling_seed", seed),
                    ("sampling_topk_k", k),
                    ("sampling_mesh_mode", 1),
                    ("sampling_stage1_receiver", 1 if is_stage1_receiver else 0),
                    ("sampling_stage2_receiver", 1 if is_stage2_receiver else 0),
                    ("sampling_num_values", num_values),
                    ("sampling_num_senders", num_cores),
                    ("sampling_topk_in_scores_cb", topk_in_scores_cb),
                    ("sampling_topk_in_indices_cb", topk_in_indices_cb),
                    ("sampling_topk_out_scores_cb", topk_out_scores_cb),
                    ("sampling_topk_out_indices_cb", topk_out_indices_cb),
                    ("sampling_mesh_stage_scores_cb", mesh_stage_scores_cb),
                    ("sampling_mesh_stage_indices_cb", mesh_stage_indices_cb),
                    ("sampling_stage1_row_elements", stage1_num_slots * topk_min_alignment),
                    ("sampling_stage1_num_input_tiles", stage1_mesh_tiles),
                    ("sampling_stage2_row_elements", stage2_num_slots * topk_min_alignment),
                    ("sampling_stage2_num_input_tiles", stage2_mesh_tiles),
                    ("sampling_num_internal_iterations", num_internal_iterations),
                ]
                rand_output_addr = 0
                if rand_per_device is not None and is_final_mesh_device:
                    rand_output_addr = int(rand_per_device[device_idx].buffer_address())
                brisc_named_compile_time_args = [
                    ("sampling_winner_page_bytes", winner_page_bytes),
                    ("sampling_local_ready_semaphore_id", local_ready_semaphore_id),
                    ("sampling_topk_k", k),
                    ("sampling_softmax_out_cb", softmax_out_cb if is_final_mesh_device else 0),
                    ("sampling_rand_cb", rand_cb if is_final_mesh_device else 0),
                    ("sampling_winner_cb", winner_cb),
                    ("sampling_p_bf16", p_uint32_cast),
                    ("sampling_topk_scores_slot_bytes", topk_scores_slot_bytes),
                    ("sampling_mesh_mode", 1),
                    ("sampling_stage2_receiver", 1 if is_stage2_receiver else 0),
                    ("sampling_output_addr", int(output_tensor_device.buffer_address())),
                    ("sampling_rand_output_addr", rand_output_addr),
                    ("sampling_num_internal_iterations", num_internal_iterations),
                    ("sampling_softmax_in_cb", softmax_in_cb if is_final_mesh_device else 0),
                    ("sampling_temp_cb", temp_cb if is_final_mesh_device else 0),
                    ("sampling_inv_temp_bf16", inv_temp_bf16),
                    ("sampling_enable_metadata", 1 if metadata_output_tensor is not None else 0),
                    ("sampling_copy_probabilities", 1 if copy_probabilities else 0),
                    (
                        "sampling_metadata_address",
                        int(metadata_output_tensor.buffer_address()) if metadata_output_tensor is not None else 0,
                    ),
                ]

                per_core_brisc_runtime_args = []
                if is_mesh_sender_core:
                    dest_idx = dest_coord[0] * mesh_shape[1] + dest_coord[1]
                    dest_scores_scratch_base = int(scores_scratch_per_device[dest_idx].buffer_address())
                    dest_indices_scratch_base = int(indices_scratch_per_device[dest_idx].buffer_address())
                    sender_dst_sem_addr = global_sem_addr if is_stage1_sender else global_stage2_sem_addr
                    if is_stage1_sender:
                        dst_scores_addr = dest_scores_scratch_base + row * topk_scores_slot_bytes
                        dst_indices_addr = dest_indices_scratch_base + row * topk_indices_slot_bytes
                    else:
                        dst_scores_addr = (
                            dest_scores_scratch_base + stage2_scores_scratch_offset + col * topk_scores_slot_bytes
                        )
                        dst_indices_addr = (
                            dest_indices_scratch_base + stage2_indices_scratch_offset + col * topk_indices_slot_bytes
                        )
                    per_core_brisc_runtime_args.append(
                        (
                            final_core_coord,
                            [
                                int(mesh_device.get_fabric_node_id(dest_coord).mesh_id),
                                int(mesh_device.get_fabric_node_id(dest_coord).chip_id),
                                dst_scores_addr,
                                dst_indices_addr,
                                sender_dst_sem_addr,
                            ],
                        )
                    )

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
                    core_ranges=all_cores,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        # See single-device path above for rationale.
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                    ),
                    ncrisc_common_runtime_args=[
                        int(scores_tensor_device.buffer_address()),
                        int(indices_tensor_device.buffer_address()),
                        int(output_tensor_device.buffer_address()),
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).x),
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).y),
                        global_sem_addr,
                        global_stage2_sem_addr,
                    ],
                    brisc_common_runtime_args=[
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).x),
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).y),
                    ],
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_active_core",
                            core_range=all_cores,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_final_core",
                            core_range=final_core_coord,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_mesh_sender_core",
                            core_range=final_core_coord,
                            value=1 if is_mesh_sender_core else 0,
                            other_value=0,
                        ),
                    ],
                    per_core_compile_time_descriptors=[
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="sampling_sender_idx",
                            core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        brisc_args=per_core_brisc_runtime_args,
                    ),
                )
                kernel_result = unified_kernel.get_kernel_descriptors()

                winner_cb_descriptor = ttnn.CBDescriptor(
                    total_size=winner_page_bytes,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=winner_cb, data_format=ttnn.uint32, page_size=winner_page_bytes
                        )
                    ],
                )
                topk_in_scores_cb_descriptor = ttnn.CBDescriptor(
                    total_size=total_input_tiles_mesh * bf16_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=topk_in_scores_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                        )
                    ],
                )
                topk_in_indices_cb_descriptor = ttnn.CBDescriptor(
                    total_size=total_input_tiles_mesh * uint32_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=topk_in_indices_cb, data_format=ttnn.uint32, page_size=uint32_tile_size
                        )
                    ],
                )
                receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=semaphore_id,
                    core_ranges=all_cores,
                    initial_value=0,
                )
                local_ready_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=local_ready_semaphore_id,
                    core_ranges=all_cores,
                    initial_value=0,
                )
                cbs = [winner_cb_descriptor, topk_in_scores_cb_descriptor, topk_in_indices_cb_descriptor]

                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=bf16_tile_size,
                        core_ranges=all_cores,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=topk_out_scores_cb, data_format=ttnn.bfloat16, page_size=bf16_tile_size
                            )
                        ],
                    )
                )
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=uint32_tile_size,
                        core_ranges=all_cores,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=topk_out_indices_cb, data_format=ttnn.uint32, page_size=uint32_tile_size
                            )
                        ],
                    )
                )

                if is_stage1_receiver or is_stage2_receiver:
                    final_core_crs = ttnn.CoreRangeSet([ttnn.CoreRange(final_core_coord, final_core_coord)])
                    # Per-device CB total_size MUST equal the number of pages the
                    # device pushes/pops per iteration so the FIFO wraps cleanly
                    # between iterations. Otherwise the CB read pointer drifts
                    # and TRISC's LLK reads stale L1.
                    #
                    # Stage-1-only receivers (target_row, non-target_col):
                    #   push/pop stage1_mesh_tiles per iter.
                    # Final receiver (= stage2_receiver):
                    #   push/pop stage1_mesh_tiles + stage2_mesh_tiles per iter.
                    device_mesh_tiles = stage1_mesh_tiles
                    if is_stage2_receiver:
                        device_mesh_tiles += stage2_mesh_tiles

                    mesh_scores_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        mesh_stage_scores_cb,
                        scores_scratch_device,
                        address_offset=0,
                        total_size=device_mesh_tiles * bf16_tile_size,
                    )
                    mesh_scores_cb_fmt = ttnn.CBFormatDescriptor(
                        buffer_index=mesh_stage_scores_cb,
                        data_format=ttnn.bfloat16,
                        page_size=bf16_tile_size,
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                    mesh_scores_cb_desc.format_descriptors = [mesh_scores_cb_fmt]
                    cbs.append(mesh_scores_cb_desc)

                    mesh_indices_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        mesh_stage_indices_cb,
                        indices_scratch_device,
                        address_offset=0,
                        total_size=device_mesh_tiles * uint32_tile_size,
                    )
                    mesh_indices_cb_fmt = ttnn.CBFormatDescriptor(
                        buffer_index=mesh_stage_indices_cb,
                        data_format=ttnn.uint32,
                        page_size=uint32_tile_size,
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                    mesh_indices_cb_desc.format_descriptors = [mesh_indices_cb_fmt]
                    cbs.append(mesh_indices_cb_desc)

                if is_final_mesh_device:
                    final_core_crs = ttnn.CoreRangeSet([ttnn.CoreRange(final_core_coord, final_core_coord)])
                    for cb_idx, cb_data_format in [
                        (softmax_in_cb, ttnn.bfloat16),
                        (softmax_out_cb, ttnn.bfloat16),
                        (max_cb, ttnn.bfloat16),
                        (sum_cb, ttnn.bfloat16),
                        (scaler_cb, ttnn.bfloat16),
                        (softmax_exp_cb, ttnn.bfloat16),
                        (temp_cb, ttnn.bfloat16),
                        (softmax_sub_cb, ttnn.bfloat16),
                        (rand_cb, ttnn.bfloat16),
                    ]:
                        cbs.append(
                            ttnn.CBDescriptor(
                                total_size=bf16_tile_size,
                                core_ranges=final_core_crs,
                                format_descriptors=[
                                    ttnn.CBFormatDescriptor(
                                        buffer_index=cb_idx, data_format=cb_data_format, page_size=bf16_tile_size
                                    )
                                ],
                            )
                        )

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=cbs,
                    semaphores=[receiver_semaphore_descriptor, local_ready_semaphore_descriptor],
                )

                if is_mesh_sender_core:
                    sender_group = kernel_result.get_group_by_arg("sampling_mesh_sender_core", 1)
                    sender_kernel_idx = sender_group.brisc_kernel_index
                    fabric_rt_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
                        dst_fabric_node_id=mesh_device.get_fabric_node_id(dest_coord),
                        link_idx=sender_link_idx,
                        program_descriptor=program,
                        worker_core=final_core_coord,
                    )
                    program.kernels[sender_kernel_idx].runtime_args[final_core_coord.x][final_core_coord.y].extend(
                        fabric_rt_args
                    )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        tensors = [scores_tensor, indices_tensor, output_index_tensor, scores_scratch_tensor, indices_scratch_tensor]
        if rand_output_tensor is not None:
            tensors.append(rand_output_tensor)
        ttnn.generic_op(tensors, mesh_program_descriptor)
        return output_index_tensor
