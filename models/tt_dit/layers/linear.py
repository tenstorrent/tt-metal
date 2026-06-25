# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Sequence

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ..utils.matmul import get_1d_matmul_config, get_fused_mmrs_config, get_matmul_config, get_matmul_core_grid
from ..utils.tensor import interleave_swiglu_tiles
from .module import Module, Parameter

MATH_FIDELITY = {
    ttnn.bfloat16: ttnn.MathFidelity.HiFi2,
    ttnn.float32: ttnn.MathFidelity.HiFi4,
}

# Activation strings accepted by Linear / ColParallelLinear `activation_fn`,
# mapped to the values the matmul fused-activation path expects. Each value is
# either a bare ttnn.UnaryOpType (no parameter) or a (UnaryOpType, param0)
# tuple; nanobind's implicit caster handles both forms.
#
# "gelu":      exact GELU (piecewise CDF / FP32 erf), matches F.gelu().
# "gelu_fast": 6-segment piecewise-linear LUT, ~1% absolute error vs exact GELU.
# "gelu_tanh": FP32 tanh approximation, matches F.gelu(approximate="tanh").
_FUSED_GELU_VARIANTS = {
    "gelu": (ttnn.UnaryOpType.GELU, False),
    "gelu_fast": (ttnn.UnaryOpType.GELU, True),
    "gelu_tanh": ttnn.UnaryOpType.GELU_TANH,
}


class Linear(Module):
    """
    Linear layer with replicated weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation_fn=None,
        dtype=ttnn.bfloat16,
        mesh_device=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.fused_activation_fn = None
        self.fuse_swiglu = False
        if self.activation_fn == "swiglu":
            # Double out features for the packed [gate|up] swiglu weight.
            self.out_features = self.out_features * 2
            self.fuse_swiglu = True
            self.activation_fn = None
        elif self.activation_fn in _FUSED_GELU_VARIANTS:
            self.fused_activation_fn = _FUSED_GELU_VARIANTS[self.activation_fn]
            self.activation_fn = None
        self.mesh_device = mesh_device

        """
        NOTE: This is the special config which attains good correctness
        HiFi2 + packer_l1_acc + bf16 acc in a fused linear (matmul + bias) with unfused non-approx activation
        """
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY[dtype],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weight = Parameter(total_shape=[self.in_features, self.out_features], device=mesh_device, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_features], device=mesh_device, dtype=dtype) if bias else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            weight = state["weight"].transpose(0, 1)
            if self.fuse_swiglu:
                weight = interleave_swiglu_tiles(weight, ndev=1)
            state["weight"] = weight
        if "bias" in state:
            bias = state["bias"].reshape(1, -1)
            if self.fuse_swiglu:
                bias = interleave_swiglu_tiles(bias, ndev=1)
            state["bias"] = bias

    def forward(
        self, x: ttnn.Tensor, compute_kernel_config=None, dtype=None, default_block_size=None, use_1d_fallback=False
    ) -> ttnn.Tensor:
        M, K, N = x.padded_shape[-2], x.padded_shape[-1], self.weight.data.padded_shape[-1]
        core_grid = get_matmul_core_grid(self.mesh_device)

        # 1D fallback can't fuse swiglu (plain ttnn.linear), so skip it when fusing.
        if use_1d_fallback and M <= 64 and not self.fuse_swiglu:  # TEMPORARY for FLUX2: 1D mcast_in0 for small M
            program_config = get_1d_matmul_config(M, K, N, core_grid)
            output = ttnn.linear(
                x,
                self.weight.data,
                bias=self.bias.data if self.bias is not None else None,
                program_config=program_config,
                activation=self.fused_activation_fn,
                compute_kernel_config=compute_kernel_config or self.compute_config,
                dtype=dtype,
            )
        else:
            matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)
            output = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.weight.data,
                bias_tensor=self.bias.data if self.bias is not None else None,
                config=matmul_config,
                fused_activation=self.fused_activation_fn,
                compute_kernel_config=compute_kernel_config or self.compute_config,
                dtype=dtype,
                fuse_swiglu=self.fuse_swiglu,
            )

        return _apply_activation_fn(output, self.activation_fn)


def gelu_decomposed(x: ttnn.Tensor) -> ttnn.Tensor:
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # ttnn.gelu is the same, but avoiding for potential issues (see ttnn.layernorm)
    # Use a single scratch buffer that's reused for every intermediate so peak
    # DRAM is x + scratch (2x input) instead of the naive 6x.
    sqrt_2 = math.sqrt(2.0)
    tmp = ttnn.multiply(x, 1.0 / sqrt_2)
    ttnn.erf(tmp, output_tensor=tmp)
    ttnn.add(tmp, 1.0, output_tensor=tmp)
    ttnn.multiply(x, tmp, output_tensor=tmp)
    ttnn.multiply(tmp, 0.5, output_tensor=tmp)
    return tmp


class ColParallelLinear(Module):
    """
    Linear layer with column parallel weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation_fn=None,
        dtype=ttnn.bfloat16,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
        chunks=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.fused_activation_fn = None
        self.fuse_swiglu = False
        if self.activation_fn == "swiglu":
            # Double out features for the packed [gate|up] swiglu weight.
            self.out_features = self.out_features * 2
            self.fuse_swiglu = True
            self.activation_fn = None
        elif self.activation_fn in _FUSED_GELU_VARIANTS:
            self.fused_activation_fn = _FUSED_GELU_VARIANTS[self.activation_fn]
            self.activation_fn = None
        self.mesh_device = mesh_device

        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager
        self.chunks = chunks

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis
            assert self.ccl_manager is not None

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY[dtype],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weight = Parameter(
            total_shape=[self.in_features, self.out_features],
            mesh_axes=[fsdp_mesh_axis, mesh_axis],
            device=mesh_device,
            dtype=dtype,
        )
        self.bias = (
            Parameter(total_shape=[1, self.out_features], mesh_axes=[None, mesh_axis], device=mesh_device, dtype=dtype)
            if bias
            else None
        )

        self._mesh_axis_size = self.mesh_device.shape[self.mesh_axis] if self.mesh_axis is not None else 1

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)

        def permute_for_swiglu(tensor):
            assert self.activation_fn == "swiglu"
            ndev = self._mesh_axis_size
            tensor = tensor.reshape(-1, 2, ndev, tensor.shape[-1] // 2 // ndev)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(-1, self.out_features)
            assert tensor.shape[0] in [1, self.in_features]
            return tensor

        if weight is not None:
            weight = weight.transpose(0, 1)
            if self.fuse_swiglu:
                weight = interleave_swiglu_tiles(weight, ndev=self._mesh_axis_size)
            elif self.activation_fn == "swiglu":
                weight = permute_for_swiglu(weight)
            state["weight"] = weight
        if bias is not None:
            bias = bias.reshape(1, -1)
            if self.fuse_swiglu:
                bias = interleave_swiglu_tiles(bias, ndev=self._mesh_axis_size)
            elif self.activation_fn == "swiglu":
                bias = permute_for_swiglu(bias)
            state["bias"] = bias

    def forward(
        self,
        x: ttnn.Tensor,
        compute_kernel_config=None,
        default_block_size=None,
        parallel_config=None,
        dtype=None,
        core_grid=None,
        use_heuristic_mmcfg=False,
        use_1d_fallback=False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        If chunks is set, returns a list of tensors split along the output dimension.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=self.fsdp_mesh_axis
            )

            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        parallel_config_tp = parallel_config.tensor_parallel.factor if parallel_config is not None else 1
        needs_gather = x.padded_shape[-1] != weight.padded_shape[-2]  # If gathered, switch to non fused AGMM

        if parallel_config_tp > 1 and self.ccl_manager.topology == ttnn.Topology.Ring and needs_gather:
            M, K, N = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = core_grid or ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size, use_heuristic=use_heuristic_mmcfg)

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, -1, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            outputs = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=self.bias.data if self.bias is not None else None,
                config=matmul_config,
                fused_activation=self.fused_activation_fn,
                compute_kernel_config=compute_kernel_config or self.compute_config,
                persistent_output_buffer=ag_persistent_buffer,
                multi_device_global_semaphore=ag_global_semaphores,
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=parallel_config.tensor_parallel.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                chunks=self.chunks if self.chunks is not None else 1,
                dtype=dtype,
                fuse_swiglu=self.fuse_swiglu,
            )

            if self.chunks is not None and (self.chunks > 1):
                return [_apply_activation_fn(o, self.activation_fn) for o in outputs]
            else:
                output = outputs[0]
        else:
            M, K, N = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
            core_grid = get_matmul_core_grid(self.mesh_device)

            # Gather if needed here. Helps cleanup upstream code
            if needs_gather:
                x = self.ccl_manager.all_gather_persistent_buffer(
                    x, dim=-1, mesh_axis=parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
                )

            if self.chunks is not None:
                matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)
                outputs = ttnn.experimental.minimal_matmul_split(
                    x,
                    weight,
                    chunks=self.chunks,
                    dim=-1,
                    bias_tensor=self.bias.data if self.bias is not None else None,
                    fused_activation=self.fused_activation_fn,
                    compute_kernel_config=compute_kernel_config or self.compute_config,
                    config=matmul_config,
                    dtype=dtype,
                    fuse_swiglu=self.fuse_swiglu,
                )
                return [_apply_activation_fn(o, self.activation_fn) for o in outputs]

            # 1D fallback can't fuse swiglu (plain ttnn.linear), so skip it when fusing.
            if use_1d_fallback and M <= 128 and not self.fuse_swiglu:  # TEMPORARY for FLUX2: 1D mcast_in0 for small M
                program_config = get_1d_matmul_config(M, K, N, core_grid)
                output = ttnn.linear(
                    x,
                    weight,
                    bias=self.bias.data if self.bias is not None else None,
                    program_config=program_config,
                    activation=self.fused_activation_fn,
                    compute_kernel_config=compute_kernel_config or self.compute_config,
                )
            else:
                matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)
                output = ttnn.experimental.minimal_matmul(
                    input_tensor=x,
                    weight_tensor=weight,
                    bias_tensor=self.bias.data if self.bias is not None else None,
                    config=matmul_config,
                    fused_activation=self.fused_activation_fn,
                    compute_kernel_config=compute_kernel_config or self.compute_config,
                    dtype=dtype,
                    fuse_swiglu=self.fuse_swiglu,
                )

        return _apply_activation_fn(output, self.activation_fn)

    def forward_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
        parallel_config=None,
        dtype=None,
        core_grid=None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""

        # Handle FSDP weight gathering (mirrors ColParallelLinear.forward)
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=self.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M, K, N = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = core_grid or ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N, core_grid)

            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, -1, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            output = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=self.bias.data if self.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or self.compute_config,
                persistent_output_buffer=ag_persistent_buffer,
                multi_device_global_semaphore=ag_global_semaphores,
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=parallel_config.tensor_parallel.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                scalar=1.0,
                addcmul_input_tensor1=addcmul_residual,
                addcmul_input_tensor2=addcmul_gate,
                dtype=dtype,
            )[0]
        else:
            M, K, N_out = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
            core_grid = self.mesh_device.compute_with_storage_grid_size()
            matmul_config = get_matmul_config(M, K, N_out, core_grid)

            output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x,
                weight,
                1.0,  # scalar
                addcmul_residual,
                addcmul_gate,
                bias_tensor=self.bias.data if self.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or self.compute_config,
                dtype=dtype,
            )

        return output


class RowParallelLinear(Module):
    """
    Linear layer with row parallel weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=ttnn.bfloat16,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY[dtype],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        ndev = self.mesh_device.shape[self.mesh_axis] if self.mesh_axis is not None else 1

        self.weight = Parameter(
            total_shape=[self.in_features, self.out_features],
            mesh_axes=[mesh_axis, fsdp_mesh_axis],
            device=mesh_device,
            dtype=dtype,
        )
        self.bias = (
            Parameter(
                total_shape=[1, self.out_features * ndev], mesh_axes=[None, mesh_axis], device=mesh_device, dtype=dtype
            )
            if bias
            else None
        )

        self._mesh_axis_size = ndev

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)

        bias = state.pop("bias", None)
        if bias is not None:
            bias = bias.reshape(1, -1)
            if self._mesh_axis_size > 1:
                zero_bias = torch.zeros(1, bias.shape[1] * (self._mesh_axis_size - 1))
                bias = torch.cat([bias, zero_bias], dim=-1)
            state["bias"] = bias

    def forward(
        self,
        x: ttnn.Tensor | list[ttnn.Tensor],
        *,
        compute_kernel_config=None,
        use_persistent_buffer: bool = True,
        default_block_size: tuple = None,
        dtype=None,
    ) -> ttnn.Tensor:
        """
        Expects x to be column fractured.
        x may be a 2-element list [prefix, suffix] for virtual concat over K (concat-free).
        Return output fractured on columns.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=3, mesh_axis=self.fsdp_mesh_axis
            )

            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        if isinstance(x, (list, tuple)):
            assert len(x) == 2, f"RowParallelLinear.forward: list x must be [prefix, suffix], got {len(x)}"
            x, x_second = x
            K = weight.padded_shape[-2]
        else:
            x_second = None
            K = x.padded_shape[-1]

        M, N = x.padded_shape[-2], weight.padded_shape[-1]
        core_grid = get_matmul_core_grid(self.mesh_device)
        matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)
        output = ttnn.experimental.minimal_matmul(
            input_tensor=[x, x_second] if x_second is not None else x,
            weight_tensor=weight,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config or self.compute_config,
            dtype=dtype,
        )

        if self._mesh_axis_size > 1:
            output = self.ccl_manager.reduce_scatter(
                output, dim=-1, mesh_axis=self.mesh_axis, use_persistent_buffer=use_persistent_buffer
            )

        return output

    def forward_fused_addcmul(
        self,
        x: ttnn.Tensor | list[ttnn.Tensor],
        addcmul_a: ttnn.Tensor,
        addcmul_b: ttnn.Tensor,
        scalar: float = 1.0,
        *,
        compute_kernel_config=None,
        dtype=None,
    ) -> ttnn.Tensor:
        """Fused RowParallel matmul + reduce-scatter + addcmul at the RS final write step.

        Computes: output = addcmul_a + scalar * rs_result * addcmul_b

        ``x`` may be a single tensor or a 2-element list ``[prefix, suffix]`` for virtual concat over K.
        The weight must be per-segment tile-padded (see ``prepare_weight_for_concatenated_input``).

        Both addcmul_a and addcmul_b must already be at their per-TP-device slice size
        [D/tp]. The RS kernel fuses the addcmul at the final ring write, eliminating
        extra CCL ops entirely.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=3, mesh_axis=self.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        # x: single tensor, or [prefix, suffix] virtually concatenated over K (concat-free).
        if isinstance(x, (list, tuple)):
            assert len(x) == 2, f"forward_fused_addcmul: list x must be exactly [prefix, suffix], got {len(x)}"
            x, x_second = x
        else:
            x_second = None

        # For virtual concat the matmul K spans both halves = the weight's K; x is only the prefix half.
        K = weight.padded_shape[-2] if x_second is not None else x.padded_shape[-1]
        M, N = x.padded_shape[-2], weight.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()

        needs_reshape = len(x.shape) <= 3
        if needs_reshape:
            x = ttnn.unsqueeze(x, 0)
            if x_second is not None:
                x_second = ttnn.unsqueeze(x_second, 0)
        pre_rs_shape = tuple(list(x.shape)[:-1] + [N])
        _, rs_output_buffer = self.ccl_manager.get_rs_ping_pong_buffer(
            pre_rs_shape, 3, self.mesh_axis, return_intermediate=False
        )
        _, output = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tensor=[x, x_second] if x_second is not None else x,
            weight_tensor=weight,
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(self.mesh_axis),
            **get_fused_mmrs_config(M, K, N, core_grid, self.ccl_manager.num_links),
            bias=self.bias.data if self.bias is not None else None,
            memory_config_mm=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            rs_output_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            topology=self.ccl_manager.topology,
            cluster_axis=self.mesh_axis,
            compute_kernel_config=compute_kernel_config or self.compute_config,
            using_persistent_buffers=True,
            optional_rs_output_tensor=rs_output_buffer,
            fused_ternary_scalar=scalar,
            addcmul_input_tensor1=addcmul_a,
            addcmul_input_tensor2=addcmul_b,
            dtype=dtype,
        )
        if needs_reshape:
            output = ttnn.squeeze(output, 0)
        return output


def _apply_activation_fn(t: ttnn.Tensor, activation_fn: str | None) -> ttnn.Tensor:
    if activation_fn is None:
        return t
    if activation_fn == "silu":
        return ttnn.silu(t)
    if activation_fn == "decomposed_gelu":
        return gelu_decomposed(t)
    if activation_fn == "quick_gelu":
        return t * ttnn.sigmoid(1.702 * t)  # quick approx gelu
    if activation_fn == "swiglu":
        t, gate = ttnn.chunk(t, 2, -1)
        return ttnn.multiply_(t, ttnn.silu(gate, output_tensor=gate))

    msg = f"Activation function {activation_fn} not supported"
    raise ValueError(msg)


_TILE_WIDTH = 32


def prepare_weight_for_concatenated_input(
    weight: torch.Tensor,
    sizes: Sequence[int],
    *,
    device_count: int,
    tile_pad_segments: bool = True,
) -> torch.Tensor:
    """Shard weight by device_count per segment and stack.

    tile_pad_segments=True (virtual concat): zero-pad each per-device segment K to a tile boundary
    so minimal_matmul([prefix, suffix], weight) works for any channel count.
    tile_pad_segments=False (materialized concat): contiguous stack, for use with ttnn.concat.
    """
    segments = weight.split(sizes, dim=1)
    padded_segments = []
    for seg in segments:
        unf = seg.unflatten(1, [device_count, -1])  # [out, device_count, K_seg/dev]
        if tile_pad_segments:
            k_per_dev = unf.shape[2]
            k_padded = ((k_per_dev + _TILE_WIDTH - 1) // _TILE_WIDTH) * _TILE_WIDTH
            if k_padded != k_per_dev:
                pad = torch.zeros(*unf.shape[:2], k_padded - k_per_dev, dtype=unf.dtype)
                unf = torch.cat([unf, pad], dim=2)
        padded_segments.append(unf)
    return torch.cat(padded_segments, dim=2).flatten(1, 2)


def prepare_chunked_linear_output(
    state: dict[str, torch.Tensor], *, prefix: str, device_count: int, chunks: int
) -> None:
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    weight = state.get(weight_key)
    bias = state.get(bias_key)

    if weight is not None:
        _, in_dim = weight.shape
        weight = weight.reshape([chunks, device_count, -1, in_dim]).transpose(0, 1).reshape([-1, in_dim])
        state[weight_key] = weight

    if bias is not None:
        bias = state[bias_key].reshape([chunks, device_count, -1]).transpose(0, 1).reshape([-1])
        state[bias_key] = bias


# =====================================================================
# LoRA-aware Linear variants
# =====================================================================
# Each variant subclasses its base Linear + the shared LoRAMixin. The
# mixin offers two execution paths chosen at construction with
# ``lora_mode`` ('fuse' or 'runtime'); see models/tt_dit/layers/lora.py
# for the trade-offs.
from .lora import LoRAMixin  # noqa: E402


class LoRALinear(LoRAMixin, Linear):
    def __init__(self, *args, lora_mode: str = "fuse", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_lora_state(mode=lora_mode)


class LoRAColParallelLinear(LoRAMixin, ColParallelLinear):
    def __init__(self, *args, lora_mode: str = "fuse", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_lora_state(mode=lora_mode)


class LoRARowParallelLinear(LoRAMixin, RowParallelLinear):
    def __init__(self, *args, lora_mode: str = "fuse", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Runtime mode lacks the all-reduce the base path performs via
        # reduce_scatter, so the delta and base sit at different mesh layouts.
        if lora_mode == "runtime" and self._mesh_axis_size > 1:
            raise ValueError(
                "LoRARowParallelLinear with lora_mode='runtime' is unsupported "
                f"at TP>1 (mesh_axis_size={self._mesh_axis_size}); use lora_mode='fuse'"
            )
        self._init_lora_state(mode=lora_mode)
