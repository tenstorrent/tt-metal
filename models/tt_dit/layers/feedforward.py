# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from .linear import ColParallelLinear, Linear, LoRAColParallelLinear, LoRARowParallelLinear, RowParallelLinear
from .module import Module


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

        self.ff1 = ColCls(
            dim,
            inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            activation_fn=activation_fn,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.ff2 = RowCls(
            inner_dim,
            dim_out,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None, parallel_config=None) -> ttnn.Tensor:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        ff1_out = self.ff1(x, compute_kernel_config=compute_kernel_config, parallel_config=parallel_config)
        return self.ff2(ff1_out, compute_kernel_config=compute_kernel_config)

    def forward_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_a: ttnn.Tensor,
        addcmul_b: ttnn.Tensor,
        scalar: float = 1.0,
        compute_kernel_config=None,
        parallel_config=None,
    ) -> ttnn.Tensor:
        """Fused FFN forward with addcmul fused at the RS final write step.

        Computes: addcmul_a + scalar * ff2(ff1(x)) * addcmul_b
        Both addcmul_a and addcmul_b are already at their per-TP-device [D/tp] slice —
        no AllGather or scatter matmul is required.
        """
        ff1_out = self.ff1(x, compute_kernel_config=compute_kernel_config, parallel_config=parallel_config)
        return self.ff2.forward_fused_addcmul(
            ff1_out,
            addcmul_a,
            addcmul_b,
            scalar=scalar,
            compute_kernel_config=compute_kernel_config,
        )


TILE = ttnn.TILE_SIZE


def _padded_intermediate(intermediate_size: int, tp_factor: int) -> int:
    """Round ``intermediate_size`` up so each TP rank gets a tile-aligned slice.

    Returns ``intermediate_size`` unchanged when it is already tile-aligned for the
    given TP factor.
    """
    per_dev = (intermediate_size + tp_factor - 1) // tp_factor
    per_dev_aligned = ((per_dev + TILE - 1) // TILE) * TILE
    return per_dev_aligned * tp_factor


class GatedMLP(Module):
    """Gated MLP used by Gemma 3 / Gemma 4 (text + vision) families::

        out = down_proj( act_fn(gate_proj(x)) * up_proj(x) )

    Parameters:
        hidden_size:        input/output dim (must be tile-aligned).
        intermediate_size:  inner dim. If not tile-aligned for the TP factor we
                            pad to the next tile-aligned multiple and zero-fill
                            the trailing channels at weight load.
        activation_fn:      Linear ``activation_fn`` keyword (default ``"gelu_tanh"`` —
                            Gemma's ``gelu_pytorch_tanh``).

    Caller is responsible for any surrounding norms (e.g. pre_feedforward_layernorm).
    Padded channels of ``gate * up`` are zero; the corresponding columns of
    ``down_proj`` are zero, so the padded path contributes nothing to the output.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mesh_device,
        ccl_manager,
        parallel_config,
        activation_fn: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        assert hidden_size % TILE == 0, f"hidden_size={hidden_size} must be tile-aligned"

        tp_factor = parallel_config.tensor_parallel.factor
        intermediate_padded = _padded_intermediate(intermediate_size, tp_factor)
        assert (intermediate_padded // tp_factor) % TILE == 0

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_padded = intermediate_padded
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.gate_proj = ColParallelLinear(hidden_size, intermediate_padded, activation_fn=activation_fn, **col_kwargs)
        self.up_proj = ColParallelLinear(hidden_size, intermediate_padded, **col_kwargs)
        self.down_proj = RowParallelLinear(
            intermediate_padded,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state) -> None:
        """Zero-pad gate/up output dim and down input dim when intermediate is padded."""
        import torch  # local import to avoid putting torch in the top-level deps

        if self.intermediate_padded == self.intermediate_size:
            return
        pad_n = self.intermediate_padded - self.intermediate_size

        for name in ("gate_proj", "up_proj"):
            w = state.get(f"{name}.weight")
            if w is not None:
                # HF weight: [intermediate, hidden] → pad dim 0.
                state[f"{name}.weight"] = torch.nn.functional.pad(w, (0, 0, 0, pad_n))

        w = state.get("down_proj.weight")
        if w is not None:
            # HF weight: [hidden, intermediate] → pad dim 1.
            state["down_proj.weight"] = torch.nn.functional.pad(w, (0, pad_n))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Replicated [B, S, hidden] → replicated [B, S, hidden]."""
        gate = self.gate_proj(x, parallel_config=self.parallel_config)
        up = self.up_proj(x, parallel_config=self.parallel_config)
        gated = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.down_proj(gated)
        ttnn.deallocate(gated)
        return out
