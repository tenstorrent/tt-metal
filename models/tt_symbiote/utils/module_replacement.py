"""Module replacement utilities for converting PyTorch modules to TTNN."""

from torch import nn


def initialize_module(old_module, old_class_to_new_class_dict, module_names, model_config):
    """Initialize a new TTNN module from a PyTorch module."""
    if old_module.__class__ in old_class_to_new_class_dict:
        new_module = old_class_to_new_class_dict[old_module.__class__].from_torch(old_module)
        if old_module in module_names:
            new_module._unique_name = module_names[old_module]
        new_module.set_model_config(model_config)
        return new_module
    return None


def register_module_replacement_dict_with_module_names(model, old_class_to_new_class_dict, model_config, module_names):
    """Recursively replace PyTorch modules with TTNN equivalents."""
    from models.tt_symbiote.core.module import TTNNModule

    if isinstance(model, nn.Module):
        for name, module in model._modules.items():
            if module is None:
                continue
            if module.__class__ in old_class_to_new_class_dict:
                new_module = initialize_module(module, old_class_to_new_class_dict, module_names, model_config)
                if new_module is not None:
                    model._modules[name] = new_module
            else:
                register_module_replacement_dict_with_module_names(
                    module, old_class_to_new_class_dict, model_config, module_names
                )
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(v, old_class_to_new_class_dict, module_names, model_config)
                        if new_module is not None:
                            value[k] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names
                        )
            if isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(v, old_class_to_new_class_dict, module_names, model_config)
                        if new_module is not None:
                            value[idx] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names
                        )
    elif isinstance(model, TTNNModule):
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(v, old_class_to_new_class_dict, module_names, model_config)
                        if new_module is not None:
                            value[k] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names
                        )
            if isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(v, old_class_to_new_class_dict, module_names, model_config)
                        if new_module is not None:
                            value[idx] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names
                        )


def register_module_replacement_dict(model, old_class_to_new_class_dict, model_config=None):
    """Register module replacements in the model."""
    module_names = {module: name for name, module in model.named_modules()}
    register_module_replacement_dict_with_module_names(model, old_class_to_new_class_dict, model_config, module_names)
