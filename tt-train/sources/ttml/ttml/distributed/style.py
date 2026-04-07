# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-style ParallelStyle classes for tensor parallelism.

Users assign ColwiseParallel / RowwiseParallel to modules by name pattern
via parallelize_module(); no DistributedLayout construction in user code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .layout import DistributedLayout, Shard, Replicate

if TYPE_CHECKING:
    from ttml.modules import AbstractModuleBase


def _mesh_ndim(mesh_device) -> int:
    mesh_shape = mesh_device.shape
    return mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)


class ParallelStyle(ABC):
    """Contract for how a module should be parallelized.

    Subclasses must implement:
    - ``get_layouts`` — return a ``{param_name: Layout}`` dict for materialization.
    - ``_apply`` — apply forward hooks / redistribute for parallelize_module.
    """

    @abstractmethod
    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, DistributedLayout]:
        """Return a mapping of parameter name to DistributedLayout for this parallel style."""
        ...

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

    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, DistributedLayout]:
        ndim = _mesh_ndim(mesh_device)
        return {
            "weight": DistributedLayout(ndim=ndim, axis_placements={tp_axis: Shard(-2)}),
            "bias": DistributedLayout(ndim=ndim, axis_placements={tp_axis: Shard(-1)}),
        }

    def _apply(
        self,
        module: "AbstractModuleBase",
        mesh_device,
        tp_axis: int,
    ) -> "AbstractModuleBase":
        import ttml
        from .layout import get_layout, set_layout

        _original_forward = module.forward

        def _wrapped_forward(x):
            x_bc = ttml.ops.distributed.broadcast(x, cluster_axis=tp_axis)
            out = _original_forward(x_bc)
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
    """Row-parallel: weight sharded on in_features (dim -1)."""

    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, DistributedLayout]:
        ndim = _mesh_ndim(mesh_device)
        return {
            "weight": DistributedLayout(ndim=ndim, axis_placements={tp_axis: Shard(-1)}),
            "bias": DistributedLayout(ndim=ndim),
        }

    def _apply(
        self,
        module: "AbstractModuleBase",
        mesh_device,
        tp_axis: int,
    ) -> "AbstractModuleBase":
        import ttml
        from .layout import get_layout, set_layout

        _original_forward = module.forward

        def _wrapped_forward(x):
            out = _original_forward(x)
            out = ttml.ops.distributed.all_reduce(
                out,
                cluster_axis=tp_axis,
            )
            inp_layout = get_layout(x)
            if inp_layout is not None:
                out_layout = inp_layout.with_placement(tp_axis, Replicate())
                set_layout(out, out_layout)
            return out

        module.forward = _wrapped_forward
        return module
