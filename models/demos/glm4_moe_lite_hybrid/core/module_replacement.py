# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Module replacement utilities for swapping PyTorch modules with TTNN equivalents.

Adapted from tt-symbiote's utils/module_replacement.py. Walks a HuggingFace model
tree and replaces matching nn.Module instances with TTNNModule subclasses via
their from_torch() classmethod.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Union

from torch import nn

from models.demos.glm4_moe_lite_hybrid.core.module import TTNNModule


def _initialize_module(
    old_module: nn.Module,
    class_map: dict,
    module_names: dict,
    model_config: dict | None,
    exclude: Set[str],
) -> Optional[Union[TTNNModule, nn.Module]]:
    if old_module.__class__ not in class_map:
        return None
    if old_module in module_names and module_names[old_module] in exclude:
        return None
    new_module = class_map[old_module.__class__].from_torch(old_module)
    if isinstance(new_module, TTNNModule):
        if old_module in module_names:
            new_module._unique_name = module_names[old_module]
            new_module.override_children_module_names()
        new_module.set_model_config(model_config)
    return new_module


def _replace_recursive(
    model,
    class_map: dict,
    model_config: dict | None,
    module_names: dict,
    exclude: Set[str],
    result: Dict[str, TTNNModule],
) -> None:
    if isinstance(model, nn.Module):
        for name, module in model._modules.items():
            if module is None:
                continue
            if module.__class__ in class_map:
                new_module = _initialize_module(module, class_map, module_names, model_config, exclude)
                if new_module is not None:
                    model._modules[name] = new_module
                    if isinstance(new_module, TTNNModule):
                        result[new_module.module_name] = new_module
            else:
                _replace_recursive(module, class_map, model_config, module_names, exclude, result)

        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except Exception:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in class_map:
                        new_module = _initialize_module(v, class_map, module_names, model_config, exclude)
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    elif isinstance(v, (nn.Module, TTNNModule)):
                        _replace_recursive(v, class_map, model_config, module_names, exclude, result)
            elif isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, nn.Module) and v.__class__ in class_map:
                        new_module = _initialize_module(v, class_map, module_names, model_config, exclude)
                        if new_module is not None:
                            value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    elif isinstance(v, (nn.Module, TTNNModule)):
                        _replace_recursive(v, class_map, model_config, module_names, exclude, result)

    elif isinstance(model, TTNNModule):
        for attr_name in dir(model):
            if attr_name.startswith("_") or attr_name == "torch_layer":
                continue
            try:
                value = getattr(model, attr_name)
            except Exception:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in class_map:
                        new_module = _initialize_module(v, class_map, module_names, model_config, exclude)
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
            elif isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, nn.Module) and v.__class__ in class_map:
                        new_module = _initialize_module(v, class_map, module_names, model_config, exclude)
                        if new_module is not None:
                            value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module


def register_module_replacement(
    model: nn.Module,
    class_map: dict,
    model_config: dict | None = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, TTNNModule]:
    """Replace PyTorch modules with TTNN equivalents throughout the model tree.

    Args:
        model: HuggingFace model to modify in-place.
        class_map: {OldClass: NewTTNNClass} mapping.
        model_config: Config dict passed to each new module via set_model_config().
        exclude: Set of module names (from named_modules) to skip.

    Returns:
        Dict mapping module_name -> TTNNModule for all replaced modules.
    """
    if exclude is None:
        exclude = set()
    module_names = {module: name for name, module in model.named_modules()}
    result: Dict[str, TTNNModule] = {}
    _replace_recursive(model, class_map, model_config, module_names, exclude, result)
    return result
