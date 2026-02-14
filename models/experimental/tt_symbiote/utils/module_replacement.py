# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Module replacement utilities for converting PyTorch modules to TTNN."""

from typing import Dict, Optional, Set, Union

from torch import nn

from models.experimental.tt_symbiote.core.module import TTNNModule


def initialize_module(
    old_module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement: Optional[Set[str]] = None
) -> Optional[Union[TTNNModule, nn.Module]]:
    """Initialize a new TTNN module from a PyTorch module."""
    if old_module.__class__ in old_class_to_new_class_dict:
        if old_module in module_names and module_names[old_module] in exclude_replacement:
            return None
        new_module = old_class_to_new_class_dict[old_module.__class__].from_torch(old_module)
        if isinstance(new_module, TTNNModule):
            if old_module in module_names:
                new_module._unique_name = module_names[old_module]
            new_module.set_model_config(model_config)
        return new_module
    return None


def register_module_replacement_dict_with_module_names(
    model,
    old_class_to_new_class_dict,
    model_config,
    module_names,
    exclude_replacement: Optional[Set[str]] = None,
    result: Optional[Dict[str, TTNNModule]] = None,
):
    """Recursively replace PyTorch modules with TTNN equivalents."""
    from models.experimental.tt_symbiote.core.module import TTNNModule

    if exclude_replacement is None:
        exclude_replacement = set()
    if result is None:
        result = {}
    assert isinstance(exclude_replacement, set), "exclude_replacement must be a set"
    assert all(isinstance(k, str) for k in exclude_replacement), "All keys in exclude_replacement must be strings"
    if isinstance(model, nn.Module):
        for name, module in model._modules.items():
            if module is None:
                continue
            if module.__class__ in old_class_to_new_class_dict:
                new_module = initialize_module(
                    module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                )
                if new_module is not None:
                    model._modules[name] = new_module
                    if isinstance(new_module, TTNNModule):
                        result[new_module.module_name] = new_module
            else:
                register_module_replacement_dict_with_module_names(
                    module, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                )
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except Exception as e:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
            if isinstance(value, (list, tuple)):
                ls_value = list(value)
                for idx, v in enumerate(ls_value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            ls_value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
                setattr(model, attr_name, type(value)(ls_value))
    elif isinstance(model, TTNNModule):
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except Exception as e:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
            if isinstance(value, (list, tuple)):
                ls_value = list(value)
                for idx, v in enumerate(ls_value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            ls_value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
                setattr(model, attr_name, type(value)(ls_value))


def register_module_replacement_dict(
    model, old_class_to_new_class_dict, model_config=None, exclude_replacement: Optional[Set[str]] = None
) -> Dict[str, TTNNModule]:
    """Register module replacements in the model."""
    module_names = {module: name for name, module in model.named_modules()}
    result: Dict[str, TTNNModule] = {}
    register_module_replacement_dict_with_module_names(
        model, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
    )
    return result
