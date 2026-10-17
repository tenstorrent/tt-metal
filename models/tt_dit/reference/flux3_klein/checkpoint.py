# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/checkpoint.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

"""Load a safetensors file (bf16) into a module. Upstream also reads torch distributed checkpoints and the FP8 / NVFP4
variants; this copy keeps the safetensors path and the flow-map detection only."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"


def safetensors_file(path: str) -> Path | None:
    path = Path(path)
    if path.suffix == ".safetensors":
        return path
    return path / "model.safetensors" if (path / "model.safetensors").exists() else None


def checkpoint_keys(path: str) -> set[str]:
    if file := safetensors_file(path):
        with safe_open(file, "pt") as f:
            return set(f.keys())
    raise FileNotFoundError(f"{path}: no safetensors checkpoint (this copy does not read distributed checkpoints)")


def is_flow_map(path: str) -> bool:
    return FLOW_MAP_KEY in checkpoint_keys(path)


def load_checkpoint(module: nn.Module, path: str, device: str, dtype: torch.dtype) -> nn.Module:
    """Reads only the keys the module has. A safetensors file must match them exactly."""
    if file := safetensors_file(path):
        return _load_safetensors(module, file, device, dtype)
    raise FileNotFoundError(f"{path}: no safetensors checkpoint (this copy does not read distributed checkpoints)")


def _load_safetensors(module: nn.Module, file: Path, device: str, dtype: torch.dtype) -> nn.Module:
    state = load_file(str(file), device=str(device))
    expected = module.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(f"{file}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    keep = lambda k, v: not v.is_floating_point()
    state = {k: v if keep(k, v) else v.to(dtype) for k, v in state.items()}
    module.load_state_dict(state, strict=True, assign=True)
    return module.to(device).eval()
