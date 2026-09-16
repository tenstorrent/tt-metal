# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from .linear import ColParallelLinear, Linear, LoRAColParallelLinear, LoRARowParallelLinear, RowParallelLinear
from .module import Module

#: Bytes one row chunk of a pointwise intermediate may hold. Bounds a SwiGLU's 4x-wide hidden
#: activations by the chunk instead of the volume; the DiffVAE's upsample slabs by the same budget.
CHUNK_BYTES = 1 << 30


def _chunk_rows(width: int, *, dtype_bytes: int = 2) -> int:
    """Rows whose ``width``-wide intermediate fits :data:`CHUNK_BYTES`, tile-aligned."""
    tile = ttnn.TILE_SIZE
    rows = CHUNK_BYTES // (width * dtype_bytes)
    return max(tile, rows // tile * tile)


def pointwise_in_chunks(x: ttnn.Tensor, fn, *, width: int) -> ttnn.Tensor:
    """Apply the pointwise ``fn`` to row chunks of ``x`` along its second-to-last dim. **Consumes** ``x``.

    ``width`` is the widest intermediate ``fn`` builds per row. A single chunk runs whole so short
    inputs pay no concat. Leading dims are kept.
    """
    shape = list(x.shape)
    rows = shape[-2]
    step = _chunk_rows(width)
    if step >= rows:
        out = fn(x)
        ttnn.deallocate(x)
        return out

    parts = []
    for start in range(0, rows, step):
        starts = [0] * len(shape)
        stops = list(shape)
        starts[-2], stops[-2] = start, min(start + step, rows)
        chunk = ttnn.slice(x, starts, stops)
        parts.append(fn(chunk))
        ttnn.deallocate(chunk)
    ttnn.deallocate(x)
    joined = ttnn.concat(parts, dim=-2)
    for part in parts:
        ttnn.deallocate(part)
    return joined


class FeedForward(Module):
    """
    Linear layer with replicated weights
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias

        self.ff1 = Linear(dim, inner_dim, bias=bias, mesh_device=mesh_device, activation_fn=activation_fn)
        self.ff2 = Linear(inner_dim, dim_out, bias=bias, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        ff1_out = self.ff1(x, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, compute_kernel_config=compute_kernel_config)


class ParallelFeedForward(Module):
    """
    Linear layer implementing megatron-style parallelism.
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
        lora_enabled: bool = False,
        ff1_dtype=ttnn.bfloat16,
        ff2_dtype=ttnn.bfloat16,
        activation_dtype=None,
        pin_output_bf16=False,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        ColCls = LoRAColParallelLinear if lora_enabled else ColParallelLinear
        RowCls = LoRARowParallelLinear if lora_enabled else RowParallelLinear

        # ff1 is the ColParallel projection whose input crosses the fabric, so it carries the
        # activation cast + output pin; ff2 (RowParallel) only takes a weight dtype.
        self.ff1 = ColCls(
            dim,
            inner_dim,
            bias=bias,
            dtype=ff1_dtype,
            mesh_device=mesh_device,
            activation_fn=activation_fn,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            activation_dtype=activation_dtype,
            pin_output_bf16=pin_output_bf16,
        )
        self.ff2 = RowCls(
            inner_dim,
            dim_out,
            bias=bias,
            dtype=ff2_dtype,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

    def forward(
        self, x: ttnn.Tensor, compute_kernel_config=None, parallel_config=None, default_block_size=None
    ) -> ttnn.Tensor:
        """
        Expects x to be replicated.
        Return output fractured on columns.

        `default_block_size` is forwarded to ff1 only, for callers that have measured block sizes for
        their ff1 shape; ff2 keeps the generic path.
        """
        ff1_out = self.ff1(
            x,
            compute_kernel_config=compute_kernel_config,
            parallel_config=parallel_config,
            default_block_size=default_block_size,
        )
        return self.ff2(ff1_out, compute_kernel_config=compute_kernel_config)

    def forward_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_a: ttnn.Tensor,
        addcmul_b: ttnn.Tensor,
        scalar: float = 1.0,
        compute_kernel_config=None,
        parallel_config=None,
        default_block_size=None,
        core_grid=None,
    ) -> ttnn.Tensor:
        """Fused FFN forward with addcmul fused at the RS final write step.

        Computes: addcmul_a + scalar * ff2(ff1(x)) * addcmul_b
        Both addcmul_a and addcmul_b are already at their per-TP-device [D/tp] slice —
        no AllGather or scatter matmul is required.

        `default_block_size` is forwarded to ff1 only, as in `forward`.
        """
        ff1_out = self.ff1(
            x,
            compute_kernel_config=compute_kernel_config,
            parallel_config=parallel_config,
            default_block_size=default_block_size,
            core_grid=core_grid,
        )
        return self.ff2.forward_fused_addcmul(
            ff1_out,
            addcmul_a,
            addcmul_b,
            scalar=scalar,
            compute_kernel_config=compute_kernel_config,
        )


class SwiGLU(Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, biasless, as LTX-2.5 ships it.

    ``fused`` packs gate and up into one ``[up | gate]`` GEMM whose epilogue emits
    ``silu(gate) * up``; ``tp_mlp`` makes that GEMM column-parallel and ``w_down`` row-parallel
    over ``tp_axis``. The checkpoint's ``w_gate.weight`` / ``w_up.weight`` are packed at load.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        mesh_device=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tp_axis=None,
        ccl_manager=None,
        fused: bool = False,
        tp_mlp: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.mesh_device = mesh_device
        self.tp_axis = tp_axis
        self.ccl_manager = ccl_manager
        # tp_mlp: gate/up column-parallel and w_down row-parallel over tp_axis, on the packed weight.
        self.tp_mlp = tp_mlp and tp_axis is not None
        # fused: one [up | gate] GEMM whose epilogue emits silu(gate) * up.
        self.fused = fused or self.tp_mlp

        linear = {"bias": False, "mesh_device": mesh_device, "dtype": dtype}
        if self.tp_mlp:
            parallel = {"mesh_axis": tp_axis, "ccl_manager": ccl_manager}
            self.gate_up = ColParallelLinear(dim, hidden_dim, activation_fn="swiglu", **linear, **parallel)
            self.w_down = RowParallelLinear(hidden_dim, dim, **linear, **parallel)
        elif self.fused:
            self.gate_up = Linear(dim, hidden_dim, activation_fn="swiglu", **linear)
            self.w_down = Linear(hidden_dim, dim, **linear)
        else:
            self.w_gate = Linear(dim, hidden_dim, **linear)
            self.w_up = Linear(dim, hidden_dim, **linear)
            self.w_down = Linear(hidden_dim, dim, **linear)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Pack the shipped ``w_gate``/``w_up`` into the fused ``[up | gate]`` weight.

        ``up`` first: ``prepare_for_fused_swiglu``'s default ordering is ``[up (N) | gate (N)]``.
        """
        if not self.fused:
            return
        gate = state.pop("w_gate.weight", None)
        up = state.pop("w_up.weight", None)
        if gate is not None and up is not None:
            state["gate_up.weight"] = torch.cat([up, gate], dim=0)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """**Consumes** ``x``. Row-chunked: the hidden intermediates are 4x the activation."""
        width = self.hidden_dim // self.mlp_shards
        return pointwise_in_chunks(x, self._project, width=width)

    @property
    def mlp_shards(self) -> int:
        return int(list(self.mesh_device.shape)[self.tp_axis]) if self.tp_mlp else 1

    def _project(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.fused:
            hidden = self.gate_up(x)
            if self.tp_mlp:
                # The all_gather below names dim 3 absolutely, so the tensor must be rank 4 here.
                hidden = ttnn.reshape(hidden, (1, 1, hidden.shape[-2], hidden.shape[-1]))
            # use_persistent_buffer=False: RowParallelLinear otherwise returns the CCL manager's
            # cached reduce-scatter buffer, which the deallocate below would destroy under it.
            out = self.w_down(hidden, use_persistent_buffer=False) if self.tp_mlp else self.w_down(hidden)
            ttnn.deallocate(hidden)
            if self.tp_mlp:
                gathered = self.ccl_manager.all_gather(out, dim=3, mesh_axis=self.tp_axis, use_hyperparams=False)
                ttnn.deallocate(out)
                out = ttnn.reshape(gathered, (gathered.shape[-2], self.dim))
            return out

        gate = ttnn.silu(self.w_gate(x))
        up = self.w_up(x)
        product = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.w_down(product)
        ttnn.deallocate(product)
        return out
