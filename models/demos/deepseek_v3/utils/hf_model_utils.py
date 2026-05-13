# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import gc
import json
import os
import shutil
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any, Callable

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import no_init_weights

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM

MODEL_INDEX_FILENAME = "model.safetensors.index.json"
DEQUANTIZED_CHECKPOINT_SUFFIX = "-dequantized"
DEQUANTIZED_CHECKPOINT_SCRIPT = "models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py"
DEQUANTIZED_CHECKPOINT_ERROR_GUIDANCE = (
    "Pass a dequantized HF checkpoint. "
    f"Use `{DEQUANTIZED_CHECKPOINT_SCRIPT}` to generate one from the original HF weights."
)


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


def index_model_weights(model_path: str | Path) -> Mapping[str, torch.Tensor]:
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    return LazyStateDict(Path(model_path))


def materialize_model_weights(
    model_path: str | Path, thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
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


def default_dequantized_model_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    if model_path.name.endswith(DEQUANTIZED_CHECKPOINT_SUFFIX):
        return model_path
    return model_path.with_name(f"{model_path.name}{DEQUANTIZED_CHECKPOINT_SUFFIX}")


def _get_weight_block_shape_from_quant_config(quantization_config: Any) -> tuple[int, ...]:
    if not isinstance(quantization_config, dict):
        raise ValueError(
            "Missing DeepSeek quantization_config.weight_block_size. "
            "The source checkpoint config must retain the original quantization metadata."
        )
    block_shape = quantization_config.get("weight_block_size")
    if not isinstance(block_shape, (list, tuple)) or not block_shape:
        raise ValueError(
            "Missing DeepSeek quantization_config.weight_block_size. "
            "The source checkpoint config must retain the original quantization metadata."
        )
    return tuple(int(dim) for dim in block_shape)


def get_weight_block_shape(hf_config: PretrainedConfig) -> tuple[int, ...]:
    return _get_weight_block_shape_from_quant_config(getattr(hf_config, "quantization_config", None))


def get_weight_block_shape_from_model_path(model_path: str | Path) -> tuple[int, ...]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find DeepSeek config at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_obj = json.load(handle)
    return _get_weight_block_shape_from_quant_config(config_obj.get("quantization_config"))


def dequantize_weight_tensor(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    block_shape: tuple[int, ...] | list[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if tensor.ndim != inv_scale.ndim:
        raise ValueError(f"Tensor and inverse scale must have same ndim, got {tensor.ndim} and {inv_scale.ndim}")
    if len(block_shape) != tensor.ndim:
        raise ValueError(
            f"Block shape rank mismatch, got len(block_shape)={len(block_shape)} and tensor.ndim={tensor.ndim}"
        )
    if any(inv_scale.shape[i] * block_shape[i] < tensor.shape[i] for i in range(tensor.ndim)):
        raise ValueError(
            "Inverse scale shape does not cover tensor shape: "
            f"tensor={tuple(tensor.shape)}, inv_scale={tuple(inv_scale.shape)}, block_shape={tuple(block_shape)}"
        )

    original_shape = tuple(tensor.shape)
    padded_shape = tuple(inv_scale.shape[i] * int(block_shape[i]) for i in range(tensor.ndim))
    original_slices = tuple(slice(0, size) for size in original_shape)

    out = tensor.float()
    out = out.clone() if out.data_ptr() == tensor.data_ptr() else out
    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded

    interleaved_shape: list[int] = []
    scale_broadcast_shape: list[int] = []
    for dim, block_dim in enumerate(block_shape):
        blocks = inv_scale.shape[dim]
        interleaved_shape.extend([blocks, int(block_dim)])
        scale_broadcast_shape.extend([blocks, 1])

    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices].to(dtype).contiguous()


def dequantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    hf_config: PretrainedConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    dequantized_state_dict: dict[str, torch.Tensor] = {}
    block_shape = get_weight_block_shape(hf_config)

    for name in sorted(key for key in state_dict.keys() if not key.endswith("_scale_inv")):
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"Expected tensor {name} to exist in state_dict but it was None")

        scale_name = f"{name}_scale_inv"
        if scale_name in state_dict:
            dequantized_state_dict[name] = dequantize_weight_tensor(
                tensor, state_dict[scale_name], block_shape, dtype=dtype
            )
            continue

        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(f"Found float8 tensor '{name}' without matching inverse scale '{scale_name}'.")
        dequantized_state_dict[name] = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()

    return dequantized_state_dict


def _load_model_weight_map(model_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model_path / MODEL_INDEX_FILENAME
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index_obj = json.load(handle)
        return dict(index_obj["weight_map"]), dict(index_obj.get("metadata", {}))

    weight_map: dict[str, str] = {}
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard_path.name
    if not weight_map:
        raise FileNotFoundError(f"No `.safetensors` shards found in {model_path}")
    return weight_map, {}


def _copy_non_weight_artifacts(source_model_path: Path, output_model_path: Path) -> None:
    for source_path in source_model_path.iterdir():
        if source_path.name == MODEL_INDEX_FILENAME or source_path.suffix == ".safetensors":
            continue
        destination = output_model_path / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination)


def _load_tensor_from_shards(
    model_path: Path,
    weight_map: Mapping[str, str],
    file_handles: dict[str, Any],
    key: str,
) -> torch.Tensor:
    shard_name = weight_map[key]
    handle = file_handles.get(shard_name)
    if handle is None:
        handle = safe_open(model_path / shard_name, framework="pt", device="cpu")
        file_handles[shard_name] = handle
    return handle.get_tensor(key)


def _close_shard_handles(file_handles: Mapping[str, Any]) -> None:
    for handle in file_handles.values():
        close_fn = getattr(handle, "close", None)
        if callable(close_fn):
            close_fn()


def save_dequantized_hf_checkpoint(
    source_model_path: str | Path,
    output_model_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    source_model_path = Path(source_model_path).resolve()
    output_model_path = (
        default_dequantized_model_path(source_model_path)
        if output_model_path is None
        else Path(output_model_path).resolve()
    )

    if output_model_path == source_model_path:
        raise ValueError("Output checkpoint path must differ from the source model path.")
    if not source_model_path.is_dir():
        raise FileNotFoundError(f"Source model path does not exist: {source_model_path}")

    if output_model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output checkpoint path already exists: {output_model_path}. "
                "Pass overwrite=True or use --force in the CLI script."
            )
        if output_model_path.is_dir():
            shutil.rmtree(output_model_path)
        else:
            output_model_path.unlink()

    output_model_path.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_artifacts(source_model_path, output_model_path)

    block_shape = get_weight_block_shape_from_model_path(source_model_path)
    weight_map, metadata = _load_model_weight_map(source_model_path)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        shard_to_keys.setdefault(shard_name, []).append(key)

    output_weight_map: dict[str, str] = {}
    total_size = 0
    file_handles: dict[str, Any] = {}
    try:
        for shard_name in sorted(shard_to_keys):
            output_tensors: dict[str, torch.Tensor] = {}
            for key in shard_to_keys[shard_name]:
                if key.endswith("_scale_inv"):
                    continue

                scale_key = f"{key}_scale_inv"
                tensor = _load_tensor_from_shards(source_model_path, weight_map, file_handles, key)
                if scale_key in weight_map:
                    output_tensor = dequantize_weight_tensor(
                        tensor,
                        _load_tensor_from_shards(source_model_path, weight_map, file_handles, scale_key),
                        block_shape,
                        dtype=dtype,
                    )
                else:
                    if tensor.dtype == torch.float8_e4m3fn:
                        raise ValueError(f"Found float8 tensor '{key}' without matching inverse scale '{scale_key}'.")
                    output_tensor = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()

                output_tensors[key] = output_tensor
                output_weight_map[key] = shard_name
                total_size += output_tensor.numel() * output_tensor.element_size()

            if output_tensors:
                logger.info(f"Saving dequantized shard {shard_name} with {len(output_tensors)} tensors")
                save_file(output_tensors, str(output_model_path / shard_name))
    finally:
        _close_shard_handles(file_handles)

    output_index = {"metadata": dict(metadata), "weight_map": output_weight_map}
    output_index["metadata"]["total_size"] = total_size
    with (output_model_path / MODEL_INDEX_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(f"Saved dequantized DeepSeek HF checkpoint to {output_model_path}")
    return output_model_path


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


def load_weight_from_weights_dict(
    weights_dict: Mapping[str, torch.Tensor]
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def load_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Loading weight: {name}" + " " * 50, end="\r")
        if name not in weights_dict:
            return tensor
        loaded_weight = weights_dict[name]
        if loaded_weight.dtype == torch.float8_e4m3fn:
            raise RuntimeError(
                f"Expected already-dequantized bf16 weights for '{name}', but found float8 tensor. "
                f"{DEQUANTIZED_CHECKPOINT_ERROR_GUIDANCE}"
            )
        if loaded_weight.dtype != tensor.dtype:
            loaded_weight = loaded_weight.to(dtype=tensor.dtype)
        tensor.data = loaded_weight
        del loaded_weight
        return tensor

    return load_weight


def unload_weight_from_weights_dict(
    weights_dict: Mapping[str, torch.Tensor],
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def unload_weight(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in weights_dict:
            return tensor
        tensor.data = torch.empty(0, dtype=tensor.dtype)
        evict_fn = getattr(weights_dict, "evict", None)
        if callable(evict_fn):
            evict_fn(name)
        return tensor

    return unload_weight


def add_dynamic_weight_loading_hooks(
    module: torch.nn.Module,
    weights_dict: Mapping[str, torch.Tensor],
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


def prepare_model_state_dict(
    hf_config: PretrainedConfig,
    random_weights: bool = False,
    model_path: str | None = None,
    single_layer: str | None = None,
) -> Mapping[str, torch.Tensor]:
    """
    Prepare model state dict from either random weights or loaded HuggingFace weights.

    Args:
        hf_config: HuggingFace model configuration
        random_weights: If True, generate random weights from reference model. If False, load from model_path.
        model_path: Path to HuggingFace model directory containing safetensors files
        single_layer: Optional single layer name (used for validation with random weights)

    Returns:
        Mapping containing model state dict entries for model components.
        For HF checkpoints this may be a lazy mapping backed by safetensors.
    """
    if random_weights:
        if single_layer and single_layer.lower() == "moe":
            raise NotImplementedError(
                "Random weights with 'moe' single layer is not supported by RowBatchedModel demo yet. Use 'mlp' or disable random mode."
            )
        logger.info("Building random weights from HF reference model (ForCausalLM)...")

        ref_model = DeepseekV3ForCausalLM(hf_config).eval()
        # Ensure parameter/buffer dtype matches downstream expectations (bfloat16).
        # Random mode now follows the same dequantized loading path as real HF weights.
        ref_model = ref_model.to(dtype=torch.bfloat16)
        torch_state = ref_model.state_dict()
        model_state = {
            k: v
            for k, v in torch_state.items()
            if k.startswith("model.embed_tokens.")
            or k.startswith("model.layers.")
            or k.startswith("model.norm.")
            or k.startswith("lm_head.")
        }
    else:
        if model_path is None:
            raise ValueError("model_path must be provided when random_weights is False")
        logger.info(f"Indexing HF weights from {model_path} for lazy loading...")
        model_state = index_model_weights(model_path)
        logger.info("HF weights indexed lazily")

        if "lm_head.weight" not in model_state:
            raise RuntimeError(
                "No HF safetensors found in model path or missing 'lm_head.weight'. "
                "Set DEEPSEEK_V3_HF_MODEL to a directory containing DeepSeek-V3 safetensors, or pass --model-path."
            )
        if any(name.endswith("_scale_inv") for name in model_state):
            raise RuntimeError(
                "Detected quantized HF tensors (*_scale_inv) in model weights. "
                "DeepSeek-V3 TT conversion now only supports already-dequantized bf16 checkpoints. "
                f"{DEQUANTIZED_CHECKPOINT_ERROR_GUIDANCE}"
            )

    return model_state
