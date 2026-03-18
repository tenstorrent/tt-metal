# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-style ParallelStyle classes for tensor parallelism.

Users assign ColwiseParallel / RowwiseParallel to modules by name pattern
via parallelize_module(); no Layout construction in user code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .layout import Layout, Shard, Replicate

if TYPE_CHECKING:
    from ttml.modules import AbstractModuleBase


def _mesh_ndim(mesh_device) -> int:
    mesh_shape = mesh_device.shape
    return mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)


class ParallelStyle(ABC):
    """Contract for how a module should be parallelized.

    Defines _apply for parallelize_module to use.
    """

    @abstractmethod
    def _apply(
        self,
        module: "AbstractModuleBase",
        mesh_device,
        tp_axis: int,
    ) -> "AbstractModuleBase":
        """Apply parallelization to the module. Returns the (mutated) module."""
        ...


class ColwiseParallel(ParallelStyle):
    """Partition a LinearLayer in column-wise fashion.

    Weight is sharded on out_features (dim -2 in TTML's [1,1,out,in] layout).
    Output is sharded on last dim; requires replicated input.

    Example:
        parallelize_module(model, mesh, {"w1": ColwiseParallel()})
    """

    def __init__(self, *, gather_output: bool = False):
        """Args:
        gather_output: If True, all_gather the output (e.g. LM head) and use REPLICATED
            grad type in backward (output grads replicated across TP).
        """
        self.gather_output = gather_output

    def get_layout(self, mesh_device, tp_axis: int) -> Layout:
        """Return the weight layout for this style (for composite module rules)."""
        ndim = _mesh_ndim(mesh_device)
        return Layout(ndim=ndim, axis_placements={tp_axis: Shard(-2)})

    def _apply(
        self,
        module: "AbstractModuleBase",
        mesh_device,
        tp_axis: int,
    ) -> "AbstractModuleBase":
        from ttml.modules import LinearLayer

        if not isinstance(module, LinearLayer):
            raise NotImplementedError(
                "ColwiseParallel currently only supports LinearLayer!"
            )

        from .training import distribute_tensor

        layout = self.get_layout(mesh_device, tp_axis)

        # Distribute weight
        new_w = distribute_tensor(module.weight.tensor, mesh_device, layout)
        module.weight.tensor = new_w
        module.override_tensor(new_w, "weight")

        # Distribute bias (sharded on last dim for column-parallel)
        if module.bias is not None:
            ndim = _mesh_ndim(mesh_device)
            bias_layout = Layout(ndim=ndim, axis_placements={tp_axis: Shard(-1)})
            new_b = distribute_tensor(module.bias.tensor, mesh_device, bias_layout)
            module.bias.tensor = new_b
            module.override_tensor(new_b, "bias")

        import ttml
        from .layout import get_layout, set_layout

        _original_forward = module.forward

        def _wrapped_forward(x):
            # Pre: broadcast input on TP axis (colwise needs replicated input)
            x_bc = ttml.ops.distributed.broadcast(x, cluster_axis=tp_axis)
            out = _original_forward(x_bc)
            # Post: all_gather with replicated grad for LM head
            if self.gather_output:
                out = ttml.ops.distributed.all_gather(
                    out,
                    dim=-1,
                    cluster_axis=tp_axis,
                    grad_output_type=ttml.ops.distributed.GradOutputType.REPLICATED,
                )
                inp_layout = get_layout(x)
                if inp_layout is not None:
                    out_layout = inp_layout.with_placement(tp_axis, Replicate())
                    set_layout(out, out_layout)
            return out

        module.forward = _wrapped_forward

        return module


class RowwiseParallel(ParallelStyle):
    """Partition a LinearLayer in row-wise fashion.

    Weight is sharded on in_features (dim -1 in TTML's [1,1,out,in] layout).
    Output is replicated (via all_reduce); input should be sharded on last dim.

    Example:
        parallelize_module(model, mesh, {"w2": RowwiseParallel()})
    """

    def get_layout(self, mesh_device, tp_axis: int) -> Layout:
        """Return the weight layout for this style (for composite module rules)."""
        ndim = _mesh_ndim(mesh_device)
        return Layout(ndim=ndim, axis_placements={tp_axis: Shard(-1)})

    def _apply(
        self,
        module: "AbstractModuleBase",
        mesh_device,
        tp_axis: int,
    ) -> "AbstractModuleBase":
        from ttml.modules import LinearLayer

        if not isinstance(module, LinearLayer):
            raise NotImplementedError(
                "RowwiseParallel currently only supports LinearLayer!"
            )

        from .training import distribute_tensor

        layout = self.get_layout(mesh_device, tp_axis)

        # Distribute weight
        new_w = distribute_tensor(module.weight.tensor, mesh_device, layout)
        module.weight.tensor = new_w
        module.override_tensor(new_w, "weight")

        # Bias stays replicated for row-parallel
        if module.bias is not None:
            ndim = _mesh_ndim(mesh_device)
            bias_layout = Layout(ndim=ndim)
            new_b = distribute_tensor(module.bias.tensor, mesh_device, bias_layout)
            module.bias.tensor = new_b
            module.override_tensor(new_b, "bias")

        # Post: all_reduce on output (row-parallel partial sums)
        from .layout import get_layout, set_layout

        import ttml

        _original_forward = module.forward

        def _wrapped_forward(x):
            out = _original_forward(x)
            # noop_backward=True when input was already sharded on last dim (avoids double all_reduce)
            inp_layout = get_layout(x)
            input_is_sharded = False
            if inp_layout is not None and inp_layout.is_sharded_on(tp_axis):
                dim = inp_layout.shard_dim(tp_axis)
                input_is_sharded = dim in (-1, 3)
            out = ttml.ops.distributed.all_reduce(
                out,
                cluster_axis=tp_axis,
                noop_backward=input_is_sharded,
            )
            if inp_layout is not None:
                out_layout = inp_layout.with_placement(tp_axis, Replicate())
                set_layout(out, out_layout)
            return out

        module.forward = _wrapped_forward

        return module
