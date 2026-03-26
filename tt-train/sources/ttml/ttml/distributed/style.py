# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-style ParallelStyle classes for tensor parallelism.

Users assign ColwiseParallel / RowwiseParallel to modules by name pattern
via parallelize_module(); no Layout construction in user code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .layout import Layout, Shard, Replicate

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
    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, Layout]:
        """Return a mapping of parameter name to Layout for this parallel style."""
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


class TpPlan:
    """Tensor-parallel plan: style patterns + axis.

    Bundles a ``{pattern: ParallelStyle}`` dict with the mesh axis
    so callers pass a single object to ``TransformerBase``.

    Usage::

        plan = TpPlan({
            r".*\\.(q_linear|kv_linear)": ColwiseParallel(),
            r".*\\.out_linear": RowwiseParallel(),
        }, tp_axis=1)
        model = Llama(config, mesh_device=mesh, tp_plan=plan)
    """

    __slots__ = ("styles", "tp_axis")

    def __init__(self, styles: dict[str, ParallelStyle], tp_axis: int = 0) -> None:
        self.styles = styles
        self.tp_axis = tp_axis

    def resolve(self, mesh_device) -> dict:
        """Convert styles to a ``{pattern: Layout}`` dict for materialization."""
        resolved = {}
        for pattern, style in self.styles.items():
            for param_name, layout in style.get_layouts(
                mesh_device, self.tp_axis
            ).items():
                resolved[pattern + r"\." + param_name] = layout
        return resolved

    def __len__(self) -> int:
        return len(self.styles)


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

    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, Layout]:
        ndim = _mesh_ndim(mesh_device)
        return {
            "weight": Layout(ndim=ndim, axis_placements={tp_axis: Shard(-2)}),
            "bias": Layout(ndim=ndim, axis_placements={tp_axis: Shard(-1)}),
        }

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
        from .layout import get_layout

        layouts = self.get_layouts(mesh_device, tp_axis)
        layout = layouts["weight"]

        # Skip distribute_tensor if weight is already in the correct layout.
        # This happens when the model was built with lazy init (mesh_device + tp_plan).
        current_layout = None
        try:
            current_layout = get_layout(module.weight.tensor)
        except Exception:
            pass

        if current_layout != layout:
            # Distribute weight
            new_w = distribute_tensor(module.weight.tensor, mesh_device, layout)
            module.weight.tensor = new_w
            module.override_tensor(new_w, "weight")

            # Distribute bias (sharded on last dim for column-parallel)
            if module.bias is not None:
                bias_layout = layouts["bias"]
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

    def get_layouts(self, mesh_device, tp_axis: int) -> dict[str, Layout]:
        ndim = _mesh_ndim(mesh_device)
        return {
            "weight": Layout(ndim=ndim, axis_placements={tp_axis: Shard(-1)}),
            "bias": Layout(ndim=ndim),
        }

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
        from .layout import get_layout

        layouts = self.get_layouts(mesh_device, tp_axis)
        layout = layouts["weight"]

        # Skip distribute_tensor if weight is already in the correct layout.
        # This happens when the model was built with lazy init (mesh_device + tp_plan).
        current_layout = None
        try:
            current_layout = get_layout(module.weight.tensor)
        except Exception:
            pass

        if current_layout != layout:
            # Distribute weight
            new_w = distribute_tensor(module.weight.tensor, mesh_device, layout)
            module.weight.tensor = new_w
            module.override_tensor(new_w, "weight")

            # Bias stays replicated for row-parallel
            if module.bias is not None:
                bias_layout = layouts["bias"]
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
