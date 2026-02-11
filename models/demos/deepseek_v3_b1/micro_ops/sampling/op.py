# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class SamplingOp:
    """
    Sampling micro-op entry point.

    Current implementation supports only k=1 (argmax fast path) for a single
    device and a single core.
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
        k: int,
        p: float,
        worker_corecoord=None,
        mesh_coord=None,
    ):
        """
        Execute sampling.

        Args:
            scores_tensor: [1, 160] bfloat16, L1-sharded on one core.
            indices_tensor: [1, 160] uint32, L1-sharded on one core.
            output_index_tensor: [1, 1] uint32, L1-sharded on one core.
            k: sampling k; currently only k=1 supported.
            p: top-p threshold (unused for k=1).
            worker_corecoord: target output core coordinate (validated, optional).
            mesh_coord: mesh coordinate containing final output (validated for single-device path, optional).
        """
        if k != 1:
            raise NotImplementedError("Sampling currently supports only k=1 (argmax fast path)")

        # p is intentionally unused in k=1 path, keep for stable API
        _ = p

        scores_shard_spec = scores_tensor.memory_config().shard_spec
        indices_shard_spec = indices_tensor.memory_config().shard_spec
        output_shard_spec = output_index_tensor.memory_config().shard_spec

        all_cores = scores_shard_spec.grid
        assert all_cores.num_cores() == 1, "Sampling k=1 currently supports only a single core"
        assert indices_shard_spec.grid == all_cores, "Scores and indices must be sharded on the same core"
        assert output_shard_spec.grid == all_cores, "Output must be sharded on the same core as inputs"
        assert scores_tensor.dtype == ttnn.bfloat16, "Scores tensor must be bfloat16"
        assert indices_tensor.dtype == ttnn.uint32, "Indices tensor must be uint32"
        assert output_index_tensor.dtype == ttnn.uint32, "Output index tensor must be uint32"
        assert tuple(scores_tensor.shape) == (1, 160), f"Expected scores shape (1, 160), got {scores_tensor.shape}"
        assert tuple(indices_tensor.shape) == (1, 160), f"Expected indices shape (1, 160), got {indices_tensor.shape}"
        assert tuple(output_index_tensor.shape) == (
            1,
            1,
        ), f"Expected output shape (1, 1), got {output_index_tensor.shape}"

        output_core = output_shard_spec.grid.ranges()[0].start
        if worker_corecoord is not None:
            assert (
                worker_corecoord.x == output_core.x and worker_corecoord.y == output_core.y
            ), "worker_corecoord must match output shard core"

        if mesh_coord is not None:
            assert (
                mesh_coord[0] == 0 and mesh_coord[1] == 0
            ), "Single-device sampling currently expects mesh_coord=(0, 0)"

        ncrisc_named_compile_time_args = [
            ("sampling_num_values", 160),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            ncrisc_common_runtime_args=[
                int(scores_tensor.buffer_address()),
                int(indices_tensor.buffer_address()),
                int(output_index_tensor.buffer_address()),
            ],
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                )
            ],
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[],
        )

        ttnn.generic_op([scores_tensor, indices_tensor, output_index_tensor], program_descriptor)
        return output_index_tensor
