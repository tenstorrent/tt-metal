# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel wrapper for distributed training.

Mirrors the role of ``torch.distributed.FSDP`` / ``torch.nn.parallel.DistributedDataParallel``
but for the TTML autograd stack.  The wrapper is orthogonal to tensor parallelism: apply
``parallelize_module`` first (TP sharding), then wrap with ``DistributedDataParallel``
for gradient synchronization across the DP (and optionally CP) axes.

Example::

    model = Llama(config)
    model = parallelize_module(model, mesh_device, tp_plan, tp_axis=1)
    model = DistributedDataParallel(model, sync_axes=[ddp_axis])

    # Use directly with SFTTrainer - no special hooks needed
    trainer = SFTTrainer(model, ...)
"""

from __future__ import annotations

from typing import Any, List

import ttml

from .training import sync_gradients


class _GradSyncFunction(ttml.autograd.Function):
    """Custom autograd function that syncs gradients in the backward pass."""

    @staticmethod
    def forward(ctx, output, model, sync_axes):
        ctx.model = model
        ctx.sync_axes = sync_axes
        return output.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        sync_gradients(ctx.model, cluster_axes=ctx.sync_axes)
        return grad_output


class DistributedDataParallel:
    """Wraps a model and synchronizes gradients across the given mesh axes after each backward.

    Args:
        module: The underlying model to wrap.
        sync_axes: Mesh axes to all-reduce gradients across (e.g. ``[ddp_axis]``
            for DP-only, or ``[ddp_axis, cp_axis]`` for DP+CP).
    """

    def __init__(self, module: Any, sync_axes: List[int]):
        self._model = module
        self._sync_axes = sync_axes

    def __call__(self, *args, **kwargs):
        output = self._model(*args, **kwargs)
        return _GradSyncFunction.apply(output, self._model, self._sync_axes)

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def parameters(self):
        return self._model.parameters()

    @property
    def config(self):
        return self._model.config

    @config.setter
    def config(self, value):
        self._model.config = value

    def __getattr__(self, name: str):
        return getattr(self._model, name)
