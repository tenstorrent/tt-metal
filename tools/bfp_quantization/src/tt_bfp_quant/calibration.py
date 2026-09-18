# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Collect reusable input second moments without storing full activations."""
from contextlib import contextmanager

import torch


class HessianAccumulator:
    """H = sum(x.T @ x) / observed_rows; no gradients or labels needed.

    Dense storage uses 4 * width**2 bytes. Memory is allocated on first add;
    device='cpu' avoids retaining every layer's Hessian on an accelerator.
    Pass only real tokens, or supply a boolean mask matching x.shape[:-1].
    """

    def __init__(self, width, *, device="cpu", chunk_rows=2048, max_samples=None):
        if type(width) is not int or width <= 0 or type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("width and chunk_rows must be positive integers")
        if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
            raise ValueError("max_samples must be a positive integer or None")
        self.width, self.device = width, torch.device(device)
        self.chunk_rows, self.max_samples = chunk_rows, max_samples
        self.count = 0
        self._sum = None

    @torch.inference_mode()
    def add(self, x, mask=None):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != self.width:
            raise ValueError("activation last dimension must match accumulator width")
        if mask is not None:
            if mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:-1]):
                raise ValueError("mask must be boolean and match activation leading dimensions")
            x = x.detach().reshape(-1, self.width)[mask.to(x.device).reshape(-1)]
        else:
            x = x.detach().reshape(-1, self.width)
        if self.max_samples is not None:
            x = x[: max(0, self.max_samples - self.count)]
        if not len(x):
            return
        if self._sum is None:
            self._sum = torch.zeros((self.width, self.width), dtype=torch.float32, device=self.device)
        for part in x.split(self.chunk_rows):
            part = part.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(part).all():
                raise ValueError("calibration contains non-finite activations")
            self._sum.addmm_(part.T, part)
            self.count += len(part)

    def value(self):
        if self.count == 0:
            raise ValueError("No calibration activations were observed")
        return (self._sum / self.count).cpu()


@contextmanager
def capture_linear_inputs(model, module_names, **accumulator_options):
    """Observe named torch.nn.Linear inputs; restore hooks even on failure.

    Use eval + inference_mode around the model forwards. The convenience hook
    collects every supplied row: use unpadded examples. For padding, fused
    projections, TTNN or non-Linear modules, call accumulator.add(x, mask)
    directly from your own input-capture adapter instead.
    """
    names = list(module_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("module_names must be nonempty and unique")
    modules = dict(model.named_modules())
    for name in names:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"{name!r} is not a named torch.nn.Linear; use HessianAccumulator directly")
    accumulators = {name: HessianAccumulator(modules[name].in_features, **accumulator_options) for name in names}
    handles = []
    try:
        for name in names:
            accumulator = accumulators[name]

            def hook(module, args, kwargs, accumulator=accumulator):
                x = args[0] if args else kwargs.get("input")
                accumulator.add(x)

            handles.append(modules[name].register_forward_pre_hook(hook, with_kwargs=True))
        yield accumulators
    finally:
        for handle in handles:
            handle.remove()
