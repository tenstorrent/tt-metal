# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import gc
import os
from glob import glob
from typing import Any, Callable

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights

from models.demos.deepseek_v3.utils.config_helpers import dequantize


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model_uninitialized(model_path: str = os.path.dirname(os.path.dirname(__file__)) + "/reference"):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    torch.set_default_dtype(current_dtype)

    model.eval()
    return model


executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())


def load_model_weights(
    model_path: str, thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
) -> dict[str, torch.Tensor]:
    safetensors_filepaths = sorted(glob(f"{model_path}/*.safetensors"))
    weights_dict = {}
    iterable = (
        map(lambda safetensor_filepath: weights_dict.update(load_file(safetensor_filepath)), safetensors_filepaths)
        if thread_pool_executor is None
        else thread_pool_executor.map(
            lambda safetensor_filepath: weights_dict.update(load_file(safetensor_filepath)), safetensors_filepaths
        )
    )
    list(
        tqdm(
            iterable,
            total=len(safetensors_filepaths),
            desc="Loading weights",
        )
    )

    print("Loaded all weights")
    return weights_dict


def apply_with_names(
    model_name: str,
    module: torch.nn.Module,
    func: Callable[[str, torch.Tensor], Any],
    thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None,
):
    named_params = list(module.named_parameters()) + list(module.named_buffers())
    if thread_pool_executor is None:
        for tensor_name_tensor in named_params:
            func(f"{model_name}{tensor_name_tensor[0]}", tensor_name_tensor[1])
    else:
        list(
            thread_pool_executor.map(
                lambda tensor_name_tensor: func(f"{model_name}{tensor_name_tensor[0]}", tensor_name_tensor[1]),
                named_params,
            )
        )


def load_weight_from_weights_dict(weights_dict: dict[str, torch.Tensor]) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def load_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Loading weight: {name}" + " " * 50, end="\r")
        if name not in weights_dict:
            return tensor
        loaded_weight = weights_dict[name]
        if loaded_weight.dtype == torch.float8_e4m3fn:
            loaded_weight_scale = weights_dict[f"{name}_scale_inv"]
            loaded_weight = dequantize(loaded_weight, loaded_weight_scale, (128, 128))
            del loaded_weight_scale
        tensor.data = loaded_weight
        del loaded_weight
        return tensor

    return load_weight


def unload_weight_from_weights_dict(
    weights_dict: dict[str, torch.Tensor],
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def unload_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in weights_dict:
            return tensor
        tensor.data = torch.empty(0)
        return tensor

    return unload_weight


def add_dynamic_weight_loading_hooks(
    module: torch.nn.Module,
    weights_dict: dict[str, torch.Tensor],
    lazy_modules: list[str] = ["DeepseekV3Attention", "DeepseekV3MLP"],
    model_name: str = "",
    thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None,
):
    is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)
    if not is_lazy and next(module.children(), None) is not None:
        for child_name, child in module.named_children():
            add_dynamic_weight_loading_hooks(
                child, weights_dict, model_name=f"{model_name}{child_name}.", thread_pool_executor=thread_pool_executor
            )
        return
    elif not is_lazy:
        apply_with_names(model_name, module, load_weight_from_weights_dict(weights_dict), thread_pool_executor)
        return
    module.register_forward_pre_hook(
        lambda module, args, kwargs, thread_pool_executor=thread_pool_executor: apply_with_names(
            model_name, module, load_weight_from_weights_dict(weights_dict), thread_pool_executor
        ),
        with_kwargs=True,
    )
    module.register_forward_hook(
        lambda module, args, kwargs, output, thread_pool_executor=thread_pool_executor: apply_with_names(
            model_name, module, unload_weight_from_weights_dict(weights_dict), thread_pool_executor
        ),
        with_kwargs=True,
    )


def add_gc_hooks(
    module: torch.nn.Module,
    lazy_modules: list[str] = ["DeepseekV3Attention", "DeepseekV3MLP"],
    model_name: str = "",
):
    def collect():
        gc.collect(0)

    is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)
    if not is_lazy:
        for child_name, child in module.named_children():
            add_gc_hooks(child, model_name=f"{model_name}{child_name}.")
        return
    module.register_forward_hook(
        lambda module, args, kwargs, output: collect(),
        with_kwargs=True,
    )


# This saves the I/O of selected modules to a torch saved dict from the module path to the tuple of (input args, input kwargs, output).
def create_log_io_hook(
    layer_groups_map: dict[str, str],
) -> tuple[
    Callable[[torch.nn.Module, str, Any, dict[str, Any], Any], Any],
    dict[str, tuple[tuple, dict[str, Any], Any]],
    dict[str, str],
]:
    log_dict: dict[str, tuple[tuple, dict[str, Any], Any]] = {}
    logged_layer_dict: dict[str, str] = {}

    def submodule_hook(
        model: torch.nn.Module,
        name: str,
        args: tuple,
        kwargs: dict[str, Any],
        output: Any,
        log_dict: dict[str, tuple[tuple, dict[str, Any], Any]] = log_dict,
        logged_layer_dict: dict[str, str] = logged_layer_dict,
        layer_groups_map: dict[str, str] = layer_groups_map,
    ):
        layer_group = layer_groups_map[name]
        if layer_group not in log_dict:
            log_dict[layer_group] = (args, kwargs, output)
            logged_layer_dict[layer_group] = name

    return submodule_hook, log_dict, logged_layer_dict


def add_model_io_logging_hooks(
    model: torch.nn.Module,
    model_pre_hook: Callable[[torch.nn.Module, list[str]], Any],
    submodule_hook: Callable[[torch.nn.Module, str, Any, dict[str, Any], Any], Any],
    model_hook: Callable[[torch.nn.Module], Any],
    logged_modules: list[str],
):
    logged_modules_used: dict[str, bool] = {name: False for name in logged_modules}
    logged_submodule_names = []
    for name, submodule in model.named_modules():
        if submodule.__class__.__name__ in logged_modules_used or name in logged_modules_used:
            logged_submodule_names.append(name)
            submodule.register_forward_hook(
                lambda _, args, kwargs, output, model=model, name=name: submodule_hook(
                    model, name, args, kwargs, output
                ),
                with_kwargs=True,
            )
            if submodule.__class__.__name__ in logged_modules_used:
                logged_modules_used[submodule.__class__.__name__] = True
            if name in logged_modules_used:
                logged_modules_used[name] = True
    if any(not used for used in logged_modules_used.values()):
        raise ValueError(
            f"The following modules were not found in the model: {', '.join(name for name, used in logged_modules_used.items() if not used)}"
        )
    model.register_forward_pre_hook(lambda _, args, model=model: model_pre_hook(model, logged_submodule_names))
    model.register_forward_hook(lambda _, args, output, model=model: model_hook(model))
