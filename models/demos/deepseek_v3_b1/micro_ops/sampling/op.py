# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class SamplingOp:
    """
    Sampling micro-op entry point.

    Current implementation supports k=1 (argmax fast path) for a single device
    with multi-core local winners reduced on a final core.
    """

    @staticmethod
    def golden(scores: torch.Tensor, indices: torch.Tensor, k: int = 1, p: float = 1.0) -> torch.Tensor:
        """
        PyTorch reference for sampling.

        For k=1, this is argmax with deterministic tie-break on lowest index.
        Returns a [1, 1] uint32 tensor containing the selected index.
        """
        if k != 1:
            raise NotImplementedError("Sampling golden currently supports only k=1")

        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)
        max_score = torch.max(scores_f32)
        tied_mask = scores_f32 == max_score
        selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)
        return selected_index.reshape(1, 1)

    @staticmethod
    def op(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        gather_buffer_tensor,
        k: int,
        p: float,
        final_core_coord=None,
        final_mesh_coord=None,
    ):
        """
        Execute sampling.

        Args:
            scores_tensor: [1, 160 * num_cores] bfloat16, WIDTH_SHARDED with shard shape [1, 160].
            indices_tensor: [1, 160 * num_cores] uint32, WIDTH_SHARDED with shard shape [1, 160].
            output_index_tensor: [1, 1] uint32, sharded on final core.
            gather_buffer_tensor: [1, 4 * num_cores] uint32, sharded on final core.
            k: sampling k; currently only k=1 supported.
            p: top-p threshold (unused for k=1).
            final_core_coord: target output core coordinate (validated, optional).
            final_mesh_coord: mesh coordinate containing final output (single-device path, optional).
        """
        if k != 1:
            raise NotImplementedError("Sampling currently supports only k=1 (argmax fast path)")

        # p is intentionally unused in k=1 path, keep for stable API
        _ = p

        scores_shard_spec = scores_tensor.memory_config().shard_spec
        indices_shard_spec = indices_tensor.memory_config().shard_spec
        output_shard_spec = output_index_tensor.memory_config().shard_spec
        gather_shard_spec = gather_buffer_tensor.memory_config().shard_spec

        all_cores = scores_shard_spec.grid
        num_cores = all_cores.num_cores()
        assert num_cores >= 1, "Sampling requires at least one active core"
        assert indices_shard_spec.grid == all_cores, "Scores and indices must be sharded on the same core grid"
        assert output_shard_spec.grid.num_cores() == 1, "Output tensor must be sharded on a single final core"
        assert gather_shard_spec.grid.num_cores() == 1, "Gather buffer must be sharded on a single final core"
        assert (
            output_shard_spec.grid == gather_shard_spec.grid
        ), "Output and gather buffer must share the same final core"
        assert scores_tensor.dtype == ttnn.bfloat16, "Scores tensor must be bfloat16"
        assert indices_tensor.dtype == ttnn.uint32, "Indices tensor must be uint32"
        assert output_index_tensor.dtype == ttnn.uint32, "Output index tensor must be uint32"
        assert gather_buffer_tensor.dtype == ttnn.uint32, "Gather buffer tensor must be uint32"
        assert tuple(scores_shard_spec.shape) == (
            1,
            160,
        ), f"Expected scores shard shape (1, 160), got {scores_shard_spec.shape}"
        assert tuple(indices_shard_spec.shape) == (
            1,
            160,
        ), f"Expected indices shard shape (1, 160), got {indices_shard_spec.shape}"
        assert tuple(scores_tensor.shape) == (
            1,
            160 * num_cores,
        ), f"Expected scores shape (1, {160 * num_cores}), got {scores_tensor.shape}"
        assert tuple(indices_tensor.shape) == (
            1,
            160 * num_cores,
        ), f"Expected indices shape (1, {160 * num_cores}), got {indices_tensor.shape}"
        assert tuple(output_index_tensor.shape) == (
            1,
            1,
        ), f"Expected output shape (1, 1), got {output_index_tensor.shape}"
        assert tuple(gather_buffer_tensor.shape) == (
            1,
            4 * num_cores,
        ), f"Expected gather buffer shape (1, {4 * num_cores}), got {gather_buffer_tensor.shape}"
        assert tuple(gather_shard_spec.shape) == (
            1,
            4 * num_cores,
        ), f"Expected gather shard shape (1, {4 * num_cores}), got {gather_shard_spec.shape}"

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
        assert any(
            core.x == final_core_coord.x and core.y == final_core_coord.y for core in sender_cores
        ), "final_core_coord must be in scores/indices shard grid"
        final_is_sender = any(core.x == output_core.x and core.y == output_core.y for core in sender_cores)
        expected_remote_incs = num_cores - 1 if final_is_sender else num_cores

        winner_cb = 2
        semaphore_id = 0
        winner_page_bytes = 16

        ncrisc_named_compile_time_args = [
            ("sampling_num_values", 160),
            ("sampling_winner_page_bytes", winner_page_bytes),
            ("sampling_num_senders", num_cores),
            ("sampling_expected_remote_incs", expected_remote_incs),
            ("sampling_winner_cb", winner_cb),
            ("sampling_receiver_semaphore_id", semaphore_id),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            ncrisc_common_runtime_args=[
                int(scores_tensor.buffer_address()),
                int(indices_tensor.buffer_address()),
                int(output_index_tensor.buffer_address()),
                int(gather_buffer_tensor.buffer_address()),
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

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[winner_cb_descriptor],
            semaphores=[receiver_semaphore_descriptor],
        )

        ttnn.generic_op(
            [scores_tensor, indices_tensor, output_index_tensor, gather_buffer_tensor],
            program_descriptor,
        )
        return output_index_tensor
