# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ..utils.matmul import get_fabric_agmm_config, get_fused_mmrs_config, get_matmul_config, get_matmul_core_grid
from ..utils.tensor import prepare_for_fused_swiglu
from .module import Module, Parameter

# Fidelity per weight dtype. Quantized dtypes (e.g. bfloat8_b) are deliberately absent: callers
# look up with .get(dtype, HiFi2). This compute_config is a dead default for any op handed an
# explicit compute_kernel_config (every LTX matmul is), so the fallback only has to construct.
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


def maybe_cast_activation(x: ttnn.Tensor, activation_dtype) -> ttnn.Tensor:
    """Cast an activation that is about to cross the fabric, if a quant config asked for it.

    Must be applied BEFORE the collective, never after: the win is in the bytes the gather moves,
    and the gather's page size is the tile size of the gathered dtype (bfloat8_b tiles are 1088 B
    vs bfloat16's 2048 B). A cast placed after the gather buys nothing and costs a full pass.

    The op-level constraint that makes this legal: the AG-matmul validates the activation and the
    weight dtypes independently, so a bf8 activation composes with the bf16-weight carve-out the
    fused addcmul epilogue requires.
    """
    if activation_dtype is None or x.get_dtype() == activation_dtype:
        return x
    return ttnn.typecast(x, activation_dtype)


def resolve_output_dtype(dtype, x: ttnn.Tensor):
    """Pin a block-float-fed matmul's output back to bf16 unless the caller asked for something else.

    Called only by a linear whose quant config opted in (``pin_output_bf16``), so no other
    model's default output dtype (``output_dtype.value_or(in0.dtype())``) changes. Keyed on the input's
    dtype so it covers an input that arrived block-float from upstream (e.g. the gate projection fed
    the shared bf8 activation), not just one this linear cast itself; without the pin a bf8 activation
    would push downstream into the residual stream and ``DistributedRMSNorm``, which rejects anything
    but bf16.
    """
    if dtype is None and x.get_dtype() in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return ttnn.bfloat16
    return dtype


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
        # Branch addition kept over main: the H3 / Qwen3-VL layers pass an explicit config for
        # the sites that need more precision than the shared default.
        compute_kernel_config=None,
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
        self.compute_config = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY.get(dtype, ttnn.MathFidelity.HiFi2),
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
                weight = prepare_for_fused_swiglu(weight, ndev=1)
            state["weight"] = weight
        if "bias" in state:
            bias = state["bias"].reshape(1, -1)
            if self.fuse_swiglu:
                bias = prepare_for_fused_swiglu(bias, ndev=1)
            state["bias"] = bias

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None, dtype=None, default_block_size=None) -> ttnn.Tensor:
        M, K, N = x.padded_shape[-2], x.padded_shape[-1], self.weight.data.padded_shape[-1]
        core_grid = get_matmul_core_grid(self.mesh_device)
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
        chunk_sizes=None,
        activation_dtype=None,
        # Branch addition kept over main: the H3 / Qwen3-VL layers pass an explicit config for
        # the sites that need more precision than the shared default.
        compute_kernel_config=None,
        pin_output_bf16=False,
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
        # Per-chunk output widths in ELEMENTS (global, pre-TP-shard); None => uniform N/chunks.
        self.chunk_sizes = chunk_sizes

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis
            assert self.ccl_manager is not None

        # Optional cast of the *input* activation, set by a quant config. This is the only Linear
        # variant that honours it, because it is the only one whose input crosses the fabric: at
        # TP>1 the input is the payload of the fused all-gather, and the gather's page size follows
        # the dtype of the gathered tensor. Casting a RowParallel/replicated Linear's input would
        # buy matmul-internal precision only while paying a full typecast pass — and RowParallel's
        # input is the 4x-wide FFN intermediate, so that trade is strictly negative.
        self.activation_dtype = activation_dtype
        # Pin a bf8/bf4-fed output back to bf16 (see resolve_output_dtype). Set by the quant config on
        # the linears on its path; off elsewhere so no other model's default output dtype changes.
        self.pin_output_bf16 = pin_output_bf16

        self.compute_config = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY.get(dtype, ttnn.MathFidelity.HiFi2),
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
                weight = prepare_for_fused_swiglu(weight, ndev=self._mesh_axis_size)
            elif self.activation_fn == "swiglu":
                weight = permute_for_swiglu(weight)
            state["weight"] = weight
        if bias is not None:
            bias = bias.reshape(1, -1)
            if self.fuse_swiglu:
                bias = prepare_for_fused_swiglu(bias, ndev=self._mesh_axis_size)
            elif self.activation_fn == "swiglu":
                bias = permute_for_swiglu(bias)
            state["bias"] = bias

    def _forward_fabric_agmm(self, x, weight, fabric_cfg, parallel_config, compute_kernel_config, dtype) -> ttnn.Tensor:
        """Optimized fabric-bound TP all-gather-matmul via strided_all_gather_minimal_matmul_async.

        The matmul runs on ``fabric_cfg.mm_core_grid`` (lower rows); the strided all-gather workers
        run on the rows starting at ``fabric_cfg.ag_core_grid_offset`` (disjoint region). Returns the
        single (chunks==1) matmul output; the op's first output is the gathered-K scratch.

        Under fused SwiGLU the weight is the packed [gate|up] matrix, so ``fabric_cfg`` blocks on the
        doubled width and the returned tensor is half as wide. The op has no dtype override, so the
        output follows the input/weight dtype rather than the caller's requested one.
        """
        mesh_axis = parallel_config.tensor_parallel.mesh_axis
        # The op gathers on dim 3 and fatals unless padded_shape[0] and [1] are both 1, but model
        # activations are rank 3 ([1, seq, K]), whose [1] is the sequence length. Widen here and
        # restore the caller's rank on the way out so this stays a drop-in for the non-fabric path.
        orig_rank = len(x.padded_shape)
        if orig_rank != 4:
            x = ttnn.unsqueeze_to_4D(x)
        if self.fuse_swiglu:
            # The factory partitions gate/up PAIRS across cores, so a pair must never straddle an
            # N block. N_tiles and N_tiles_per_core are even by construction of the packed weight.
            assert (
                fabric_cfg.N_block_size % 2 == 0
            ), f"fuse_swiglu needs an even N_block_size (in tiles), got {fabric_cfg.N_block_size}"
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=fabric_cfg.M_block_size,
            K_block_size=fabric_cfg.K_block_size,
            N_block_size=fabric_cfg.N_block_size,
            subblock_h=fabric_cfg.subblock_h,
            subblock_w=fabric_cfg.subblock_w,
            compute_with_storage_grid_size=fabric_cfg.mm_core_grid,
        )
        ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(x.shape, 3, mesh_axis, dtype=x.get_dtype())
        ag_global_semaphores = self.ccl_manager.get_strided_ag_mm_semaphore(mesh_axis, fabric_cfg.num_workers_per_link)
        dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        outputs = ttnn.experimental.strided_all_gather_minimal_matmul_async(
            x,
            weight,
            persistent_output_buffer=ag_persistent_buffer,
            dim=3,
            multi_device_global_semaphore=ag_global_semaphores,
            strided_all_gather_core_grid_offset=fabric_cfg.ag_core_grid_offset,
            num_links=self.ccl_manager.num_links,
            memory_config_ag=dram,
            topology=self.ccl_manager.topology,
            cluster_axis=mesh_axis,
            bias=self.bias.data if self.bias is not None else None,
            fused_activation=self.fused_activation_fn,
            config=matmul_config,
            memory_config_mm=dram,
            compute_kernel_config=compute_kernel_config or self.compute_config,
            num_workers_per_link=fabric_cfg.num_workers_per_link,
            num_buffers_per_channel=fabric_cfg.num_buffers_per_channel,
            read_local_slice_from_input=True,
            chunks=1,
            fuse_swiglu=self.fuse_swiglu,
        )
        # Op returns [all_gather_output, matmul_chunk_0]; take the single matmul chunk.
        out = _apply_activation_fn(outputs[1], self.activation_fn)
        if orig_rank != 4:
            out = ttnn.reshape(out, tuple(out.shape)[-orig_rank:])
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        compute_kernel_config=None,
        default_block_size=None,
        parallel_config=None,
        dtype=None,
        addcmul_a=None,
        addcmul_b=None,
        addcmul_scalar: float = 1.0,
        core_grid=None,
        use_heuristic_mmcfg=False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        If chunks is set, returns a list of tensors split along the output dimension.

        `addcmul_a` / `addcmul_b` fuse a gated residual into the matmul epilogue, returning
        `addcmul_a + addcmul_scalar * matmul_result * addcmul_b`. Both must already be at the
        per-TP-device output slice. Only the all-gather-matmul path supports it, so it requires
        `parallel_config`; callers on the unfused path should apply the addcmul themselves.
        """
        if addcmul_a is not None or addcmul_b is not None:
            if (addcmul_a is None) != (addcmul_b is None):
                msg = "addcmul_a and addcmul_b must be given together"
                raise ValueError(msg)
            if parallel_config is None or parallel_config.tensor_parallel.factor <= 1:
                msg = "fused addcmul needs the all-gather-matmul path; pass parallel_config"
                raise ValueError(msg)
            if self.chunks is not None and self.chunks > 1:
                msg = "fused addcmul is not supported alongside chunked output"
                raise ValueError(msg)

        x = maybe_cast_activation(x, self.activation_dtype)
        if self.pin_output_bf16:
            dtype = resolve_output_dtype(dtype, x)

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

            # Fabric-bound path: known shapes route to the optimized strided all-gather-matmul op.
            # N is the weight width, so a fused-SwiGLU layer keys on its packed [gate|up] width.
            # Restricted to chunks==1; other shapes fall through to the all_gather_minimal_matmul_async
            # path below. The op gathers on dim 3 of a rank-4 input, so every dim above the matmul's
            # (M, K) must be unit; _forward_fabric_agmm widens rank-3 activations to satisfy that.
            fabric_cfg = get_fabric_agmm_config(M, K, N, (self.chunks or 1), full_grid)
            has_unit_batch = len(x.padded_shape) <= 4 and all(d == 1 for d in list(x.padded_shape)[:-2])
            if fabric_cfg is not None and self.chunks in (None, 1) and has_unit_batch and addcmul_a is None:
                return self._forward_fabric_agmm(x, weight, fabric_cfg, parallel_config, compute_kernel_config, dtype)

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
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                chunks=self.chunks if self.chunks is not None else 1,
                # Op's N is per-device, so pass per-device widths (each global width is % TP == 0).
                chunk_sizes=(
                    [w // parallel_config.tensor_parallel.factor for w in self.chunk_sizes] if self.chunk_sizes else []
                ),
                dtype=dtype,
                fuse_swiglu=self.fuse_swiglu,
                scalar=addcmul_scalar if addcmul_a is not None else None,
                addcmul_input_tensor1=addcmul_a,
                addcmul_input_tensor2=addcmul_b,
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
        # Branch addition kept over main: the H3 / Qwen3-VL layers pass an explicit config for
        # the sites that need more precision than the shared default.
        compute_kernel_config=None,
        mm_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager
        self.mm_memory_config = mm_memory_config

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        self.compute_config = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=MATH_FIDELITY.get(dtype, ttnn.MathFidelity.HiFi2),
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
        gather_output: bool = False,
    ) -> ttnn.Tensor:
        """
        Expects x to be column fractured.
        x may be a 2-element list [prefix, suffix] for fused concat over K (concat-free).
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
            # Reduce over rows when replicating: N may be too narrow to scatter over the mesh axis.
            dim = -2 if gather_output else -1
            output = self.ccl_manager.reduce_scatter(
                output, dim=dim, mesh_axis=self.mesh_axis, use_persistent_buffer=use_persistent_buffer
            )
            if gather_output:
                output = self.ccl_manager.all_gather(
                    output, dim=dim, mesh_axis=self.mesh_axis, use_hyperparams=True, use_persistent_buffer=True
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

        ``x`` may be a single tensor or a 2-element list ``[prefix, suffix]`` for fused concat over K.
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

        # For fused concat the matmul K spans both halves = the weight's K; x is only the prefix half.
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
        # The MM output is scratch here (only the RS output is returned), so hand it to the RS
        # through the rolling L1 window instead of a DRAM round-trip whenever the blocking config
        # carries a window (the default). The window bounds the resident shard, and the credit
        # array is the RS->MM return path that lets the matmul recycle window slots. A config
        # with mm_window_blocks=None opts out and falls back to self.mm_memory_config (the op
        # rejects an L1 MM output without a window).
        mmrs_params = get_fused_mmrs_config(M, K, N, core_grid, self.ccl_manager.num_links)
        use_l1_handoff = mmrs_params["mm_window_blocks"] is not None
        _, output = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tensor=[x, x_second] if x_second is not None else x,
            weight_tensor=weight,
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(self.mesh_axis),
            **mmrs_params,
            bias=self.bias.data if self.bias is not None else None,
            memory_config_mm=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            if use_l1_handoff
            else self.mm_memory_config,
            rs_intermediate_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
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
            mm_progress_counters=self.ccl_manager.get_mm_progress_counters_buffer(),
            mm_credit_counters=self.ccl_manager.get_mm_credit_counters_buffer() if use_l1_handoff else None,
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
