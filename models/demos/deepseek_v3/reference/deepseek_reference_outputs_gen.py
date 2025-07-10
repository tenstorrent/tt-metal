# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
from glob import glob
from itertools import chain
from typing import Any, Callable

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights

from models.demos.deepseek_v3.utils.config_helpers import dequantize


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model_uninitialized(model_path: str):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    torch.set_default_dtype(current_dtype)

    model.eval()
    return model


def load_model_weights(model_path: str) -> dict[str, torch.Tensor]:
    safetensors_filepaths = sorted(glob(f"{model_path}/*.safetensors"))
    weights_dict = {}
    for safetensor_filepath in tqdm(safetensors_filepaths, total=len(safetensors_filepaths), desc="Loading weights"):
        weights_dict.update(load_file(safetensor_filepath))
    print("Loaded all weights" + " " * 50)
    return weights_dict


def apply_with_names(model_name: str, module: torch.nn.Module, func: Callable[[str, torch.Tensor], Any]):
    for tensor_name, tensor in chain(module.named_parameters(), module.named_buffers()):
        func(f"{model_name}{tensor_name}", tensor)


def load_weight_from_weights_dict(weights_dict: dict[str, torch.Tensor]) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def load_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Loading weight: {name}" + " " * 100, end="\r")
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
    lazy_modules: list[str] = ["DeepseekV3MLP", "DeepseekV3Attention"],
    model_name: str = "",
):
    is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)
    if not is_lazy and next(module.children(), None) is not None:
        for child_name, child in module.named_children():
            add_dynamic_weight_loading_hooks(child, weights_dict, model_name=f"{model_name}{child_name}.")
        return
    elif not is_lazy:
        apply_with_names(model_name, module, load_weight_from_weights_dict(weights_dict))
        return
    module.register_forward_pre_hook(
        lambda module, args, kwargs: apply_with_names(model_name, module, load_weight_from_weights_dict(weights_dict)),
        with_kwargs=True,
    )
    module.register_forward_hook(
        lambda module, args, kwargs, output: apply_with_names(
            model_name, module, unload_weight_from_weights_dict(weights_dict)
        ),
        with_kwargs=True,
    )


# This saves the I/O of selected modules to a torch saved dict from the module path to the tuple of (input args, input kwargs, output).
def log_io_to_torch_hooks(
    filepath: str,
) -> tuple[
    Callable[[torch.nn.Module, list[str]], Any],
    Callable[[torch.nn.Module, str, Any, dict[str, Any], Any], Any],
    Callable[[torch.nn.Module], Any],
]:
    per_model_log: dict[torch.nn.Module, dict[str, tuple[tuple, dict[str, Any], Any]]] = {}
    per_model_submodules_list: dict[torch.nn.Module, list[str]] = {}

    def model_pre_hook(
        model: torch.nn.Module,
        submodules_list: list[str],
        per_model_log: dict[torch.nn.Module, dict[str, tuple[tuple, dict[str, Any], Any]]] = per_model_log,
        per_model_submodules_list=per_model_submodules_list,
    ):
        per_model_log[model] = {}
        per_model_submodules_list[model] = submodules_list[:]  # Copy the list to avoid mutation issues

    def submodule_hook(
        model: torch.nn.Module,
        name: str,
        args: tuple,
        kwargs: dict[str, Any],
        output: Any,
        per_model_log: dict[torch.nn.Module, dict[str, tuple[tuple, dict[str, Any], Any]]] = per_model_log,
        per_model_submodules_list=per_model_submodules_list,
        io_log_filepath: str = filepath,
    ):
        assert name not in per_model_log[model], (
            name,
            per_model_log[model],
            "Logging modules that are invoked multiple times per model run is not supported at the moment",
        )
        assert name in per_model_submodules_list[model]
        per_model_log[model][name] = (args, kwargs, output)
        if len(per_model_log[model]) == len(per_model_submodules_list):
            print(f"Saving the model io log into {io_log_filepath}")
            torch.save(per_model_log[model], io_log_filepath)
            exit(0)

    def model_hook(_: torch.nn.Module):
        raise RuntimeError("The execution should have already exited in the submodule_hook")

    return model_pre_hook, submodule_hook, model_hook


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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A script to trace the IO of the deepseek model.")
    parser.add_argument("local_model_path", type=str, help="Path to the local model directory.")
    parser.add_argument("prompt", type=str, help="Prompt to generate outputs for.")
    parser.add_argument(
        "path_to_model_io_log",
        type=str,
        help="Path to output the selected layers' IO to. The log is a torch-saved dict from submodule names to the tuple (input args, input kwargs, output)",
    )
    parser.add_argument(
        "layers",
        nargs="*",
        type=str,
        help="List of layers to log IO for. Can either be a torch module name or a state-dict-style layer path. Defaults to a hardcoded layer types.",
        default=[
            "DeepseekV3MLP",
            "DeepseekV3MoE",
            "DeepseekV3Attention",
            "DeepseekV3DecoderLayer",
            "DeepseekV3Model",
            "DeepseekV3RMSNorm",
        ],
    )
    return parser


def main():
    # Parse the sysargs
    parser = create_parser()
    args = parser.parse_args()

    # Load the tokenizer
    print("Loading tokenizer")
    tokenizer = load_tokenizer(args.local_model_path)
    print("Tokenizer loaded successfully")

    # Load the model with uninitialized weights
    print("Loading uninitialized model")
    model = load_model_uninitialized(args.local_model_path)
    print("Model loaded successfully")

    # Load the model weights
    print("Loading model weights")
    weights_dict = load_model_weights(args.local_model_path)
    add_dynamic_weight_loading_hooks(model, weights_dict)
    print("Model weights loaded successfully")

    # Set up logging hooks
    print("Setting up model I/O logging hooks")
    module_pre_hook, submodule_hook, module_hook = log_io_to_torch_hooks(args.path_to_model_io_log)
    add_model_io_logging_hooks(model, module_pre_hook, submodule_hook, module_hook, args.layers)

    # Run the model
    model_inputs = tokenizer(args.prompt, return_tensors="pt")
    print("Running the model")
    with torch.no_grad():
        _ = model(model_inputs.input_ids)


if __name__ == "__main__":
    main()
