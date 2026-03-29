# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Architecture inspection script for ibm-granite/granite-timeseries-ttm-r1.

Usage:
    python -m models.demos.granite_ttm_r1.reference.model_summary

Prints the module tree with input/output shapes, confirmed attribute names,
and a parameter count table.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction

MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r1"
CONTEXT_LENGTH = 512
FORECAST_LENGTH = 96
NUM_CHANNELS = 1
BATCH_SIZE = 1


def main():
    print(f"Loading config from {MODEL_NAME} ...")
    config = TinyTimeMixerConfig.from_pretrained(MODEL_NAME)
    print("Config attributes:")
    for key in sorted(vars(config).keys()):
        val = getattr(config, key)
        if not key.startswith("_"):
            print(f"  {key}: {val!r}")

    print(f"\nLoading model ...")
    model = TinyTimeMixerForPrediction.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()

    print("\nModule tree:")
    print(model)

    # Register forward hooks to capture shapes
    captures: OrderedDict[str, dict] = OrderedDict()

    def make_hook(name):
        def hook(module, args, output):
            in_shape = tuple(args[0].shape) if args and isinstance(args[0], torch.Tensor) else None
            out_shape = (
                tuple(output.shape)
                if isinstance(output, torch.Tensor)
                else (
                    tuple(output[0].shape)
                    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor)
                    else None
                )
            )
            n_params = sum(p.numel() for p in module.parameters())
            captures[name] = {
                "class": type(module).__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "n_params": n_params,
            }

        return hook

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    history = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, NUM_CHANNELS)
    observed_mask = torch.ones(BATCH_SIZE, CONTEXT_LENGTH, NUM_CHANNELS)

    with torch.no_grad():
        try:
            outputs = model(past_values=history, past_observed_mask=observed_mask)
        except TypeError:
            try:
                outputs = model(history)
            except Exception as e:
                print(f"Forward pass failed: {e}")
                return

    for h in hooks:
        h.remove()

    print("\n\nShape table:")
    print(f"{'Module path':<50} {'Class':<30} {'In shape':<30} {'Out shape':<30} {'Params':>10}")
    print("-" * 155)
    for name, info in captures.items():
        print(
            f"{name:<50} {info['class']:<30} {str(info['in_shape']):<30} {str(info['out_shape']):<30} {info['n_params']:>10,}"
        )

    print("\n\nTop-level attribute names:")
    for attr in dir(model):
        if not attr.startswith("_"):
            val = getattr(model, attr, None)
            if isinstance(val, torch.nn.Module):
                print(f"  model.{attr} -> {type(val).__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nOutput keys/type:")
    if hasattr(outputs, "_fields"):
        print(f"  namedtuple fields: {outputs._fields}")
    elif hasattr(outputs, "__dict__"):
        for k, v in vars(outputs).items():
            if isinstance(v, torch.Tensor):
                print(f"  .{k}: {tuple(v.shape)}")
    elif isinstance(outputs, torch.Tensor):
        print(f"  Tensor: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
