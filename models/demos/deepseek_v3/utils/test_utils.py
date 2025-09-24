# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Sequence

import safetensors.torch
import torch
from loguru import logger
from transformers import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.scripts.generate_test_inputs_outputs import __file__ as REFERENCE_IO_SCRIPT_NAME
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE, dequantize, even_int_div
from models.tt_transformers.tt.common import PagedAttentionConfig


def load_state_dict(model_path: Path, module_path: str):
    if module_path:
        module_path += "."  # So that the later matches include the separating dot

    weight_paths = json.load(open(model_path / "model.safetensors.index.json", "r"))["weight_map"]
    per_safetensor_weights = {}

    for weight_name in weight_paths.keys():
        if not weight_name.startswith(module_path):
            continue
        per_safetensor_weights.setdefault(weight_paths[weight_name], []).append(weight_name)

    return {
        weight_name[len(module_path) :]: safetensor_state_dict[weight_name]
        for safetensor_file_path, weight_names in per_safetensor_weights.items()
        for safetensor_state_dict in [safetensors.torch.load_file(model_path / safetensor_file_path)]
        for weight_name in weight_names
    }


def get_quant_scale(tensor: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    assert tensor.ndim == len(block_shape), "Weight tensors must have the same dimensionality as the block shape"
    padded_tensor = torch.nn.functional.pad(
        tensor.float(),
        [
            padding_size
            for tensor_dim, block_dim in reversed(list(zip(tensor.shape, block_shape)))
            for padding_size in [0, -tensor_dim % block_dim]
        ],
    )
    blocked_tensor = padded_tensor.reshape(
        [
            new_tensor_dim
            for tensor_dim, block_dim in zip(padded_tensor.shape, block_shape)
            for new_tensor_dim in [even_int_div(tensor_dim, block_dim), block_dim]
        ]
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return (
        fp8_max
        / blocked_tensor.permute(*torch.arange(0, blocked_tensor.ndim, 2), *torch.arange(1, blocked_tensor.ndim, 2))
        .reshape(*(blocked_tensor.shape[dim * 2] for dim in torch.arange(tensor.ndim)), -1)
        .max(dim=-1)
        .values
    )


def dequantize_state_dict(state_dict, hf_config, dtype=torch.bfloat16):
    dequantized_state_dict = {}

    for name, tensor in state_dict.items():
        if name.endswith("_scale_inv"):
            continue

        if tensor is not None:
            # Look for corresponding scale tensor
            scale_name = name + "_scale_inv"
            if scale_name in state_dict:
                scale_tensor = state_dict[scale_name]
                # Dequantize using the scale
                dequantized_tensor = dequantize(
                    tensor, scale_tensor, hf_config.quantization_config["weight_block_size"]
                )
                dequantized_state_dict[name] = dequantized_tensor.to(dtype)
            else:
                dequantized_state_dict[name] = tensor.to(dtype)

    return dequantized_state_dict


def add_inv_scale_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    block_shape: Sequence[int],
    weight_names: list[str] = [
        "up_proj",
        "down_proj",
        "gate_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ],
) -> dict[str, torch.Tensor]:
    """
    Quantizes specified weights in state_dict and adds inverse scale tensors.

    Args:
        state_dict: original model weights
        block_shape: shape of quantization blocks (e.g., [128, 128])
        weight_names: list of substrings to match in parameter names

    Returns:
        new state_dict with quantized weights and _scale_inv tensors
    """
    output_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if weight_names and not any(name.endswith(weight_name + ".weight") for weight_name in weight_names):
            output_state_dict[name] = tensor
            continue
        assert tensor.ndim == len(block_shape), "Weight tensors must have the same dimensionality as the block shape"

        scale_tensor = get_quant_scale(tensor, block_shape)
        output_state_dict[name] = dequantize(tensor, scale_tensor, block_shape).to(torch.float8_e4m3fn)
        output_state_dict[name + "_scale_inv"] = 1.0 / scale_tensor.float()

    return output_state_dict


def torch_cache_from_paged(
    paged_cache: torch.Tensor,
    mapping: torch.Tensor,
    dp_factor: int,
) -> torch.Tensor:
    """
    Convert a set of concatenated paged cache back to the original cache format using the provided mapping.
    Args:
        paged_cache (torch.Tensor): The paged cache tensor of shape
            (num_devices x cache_blocks, num_heads, block_size, dim).
        mapping (torch.Tensor): A tensor of shape (batches_per_device, blocks_per_batch)
            that maps paged cache blocks to their original positions.
        dp_factor (int): The number of data parallel devices.
    Returns:
        torch.Tensor: The reconstructed cache tensor of shape
            (batch_size, num_heads, seq_len, dim), where
            batch_size = dp_factor x batches_per_device and
            seq_len = blocks_per_batch x block_size.
    Note:
        This function assumes that the paged_cache and mapping tensors are compatible
        and that the mapping tensor contains valid indices for reconstructing the cache.
    """

    # paged_cache.shape = (num_devices x cache_blocks, num_heads, block_size, dim)
    _, num_heads, block_size, dim = paged_cache.shape
    batches_per_device, blocks_per_batch = mapping.shape

    paged_cache = paged_cache.reshape(
        dp_factor, mapping.numel(), num_heads, block_size, dim
    )  # (num_devices, cache_blocks, num_heads, block_size, dim)
    paged_cache = paged_cache[
        :, mapping
    ]  # (num_devices, batches_per_device, blocks_per_batch, num_heads, block_size, dim)
    paged_cache = paged_cache.transpose(2, 3)
    paged_cache = paged_cache.reshape(
        dp_factor * batches_per_device, num_heads, blocks_per_batch * block_size, dim
    )  # (batch_size, num_heads, seq_len, dim)

    return paged_cache


def paged_cache_from_torch(
    torch_cache: torch.Tensor,
    dp_factor: int,
    paged_config: PagedAttentionConfig,
    user_id: int | None,
    mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a torch cache tensor into a paged cache format for the ttn model.

    Args:
        torch_cache (torch.Tensor): The input cache tensor of shape (batch_size, num_heads, seq_len, dim).
        dp_factor (int): The number of data parallel devices.
        paged_config (PagedAttentionConfig): Configuration for the paged cache.
        user_id (int | None): Optional user index. If provided, the cache is placed at the corresponding batch index.
        mapping (torch.Tensor | None, optional): Optional mapping tensor for block assignment. If None, a random permutation is generated.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - paged_cache (torch.Tensor): The paged cache tensor.
            - mapping (torch.Tensor): The mapping tensor used for block assignment.

    Raises:
        AssertionError: If the input tensor does not meet expected shapes or configuration constraints.
    """
    if user_id is not None:
        torch_cache_line = torch_cache
        torch_cache = torch.zeros((MAX_BATCH_SIZE, *torch_cache_line.shape[1:]), dtype=torch_cache_line.dtype)
        torch_cache[user_id : user_id + 1] = torch_cache_line

    batch_size, num_heads, seq_len, dim = torch_cache.shape
    batches_per_device = even_int_div(batch_size, dp_factor)
    blocks_per_batch = even_int_div(paged_config.max_num_blocks, batches_per_device)
    assert num_heads == 1, "Expected the kvpe cache to have only one head"

    if mapping is None:
        mapping = torch.randperm(batches_per_device * blocks_per_batch).reshape(batches_per_device, blocks_per_batch)
    assert mapping.shape == (batches_per_device, blocks_per_batch)

    assert paged_config.block_size * blocks_per_batch >= seq_len
    torch_cache = torch.nn.functional.pad(torch_cache, (0, 0, 0, paged_config.block_size * blocks_per_batch - seq_len))

    torch_cache = torch_cache.reshape(
        dp_factor, batches_per_device, num_heads, blocks_per_batch, paged_config.block_size, dim
    )
    torch_cache = torch_cache.transpose(
        2, 3
    )  # (num_devices, batches_per_device, blocks_per_batch, num_heads, block_size, dim)

    paged_cache = torch.empty(
        (dp_factor, batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim),
        dtype=torch_cache.dtype,
    )
    paged_cache[:, mapping] = torch_cache
    paged_cache = paged_cache.reshape(
        dp_factor * batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim
    )

    return paged_cache, mapping


def paged_caches_from_torch(
    torch_caches: tuple[torch.Tensor, ...],
    dp_factor: int,
    paged_config: PagedAttentionConfig,
    user_id: int | None,
    mappings: tuple[torch.Tensor, ...] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Helper function for calling `paged_cache_from_torch` for several torch caches.
    Please refer to the `paged_cache_from_torch` documentation for details.
    """
    assert mappings == None or len(mappings) == len(torch_caches)
    if mappings is None:
        mappings = (None,) * len(torch_caches)
    paged_caches, mappings = zip(
        *(
            paged_cache_from_torch(torch_cache, dp_factor, paged_config, user_id, mapping)
            for mapping, torch_cache in zip(mappings, torch_caches, strict=True)
        )
    )
    return paged_caches, mappings


def transformers_cache_from_torch(torch_caches: tuple[torch.Tensor, ...]) -> DynamicCache:
    return DynamicCache.from_legacy_cache(
        tuple(
            (torch_cache, torch.empty((*torch_cache.shape[:-1], 0), dtype=torch_cache.dtype))
            for torch_cache in torch_caches
        )
    )


def torch_cache_from_transformers(cache: DynamicCache) -> tuple[torch.Tensor, ...]:
    torch_cache, _ = zip(*cache.to_legacy_cache())
    return torch_cache


def torch_cache_from_transformers_single_layer(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    return cache[layer_idx][0]


def transformers_cache_single_layer_from_torch(torch_cache: torch.Tensor, layer_idx: int) -> DynamicCache:
    return transformers_cache_from_torch(
        (torch.empty((*torch_cache.shape[:-1], 0), dtype=torch_cache.dtype),) * layer_idx + (torch_cache,)
    )


def run_reference_with_attention(
    reference_model: torch.nn.Module,
    activation: torch.Tensor,
    position_ids_or_seq_lens: torch.Tensor,
    layer_idx: int | None,
    hf_config: PretrainedConfig,
    mode: str,
    zeroed_cache: bool,
) -> tuple[torch.Tensor, DynamicCache, DynamicCache]:
    (batch_size,) = position_ids_or_seq_lens.shape
    dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    num_layers = hf_config.num_hidden_layers
    max_position_id_or_seq_len = position_ids_or_seq_lens.max().item()

    if mode == "prefill":
        max_seq_len = position_ids_or_seq_lens.max().item()
        position_ids = torch.arange(max_seq_len).unsqueeze(0).repeat(batch_size, 1)
        mask = torch.triu(
            torch.full(
                (batch_size, 1, max_position_id_or_seq_len, max_position_id_or_seq_len),
                float("-inf"),
                dtype=torch.bfloat16,
            ),
            diagonal=1,
        )
        if layer_idx is not None:
            input_cache = transformers_cache_single_layer_from_torch(
                torch.empty((batch_size, 1, 0, dim), dtype=torch.bfloat16), layer_idx
            )
        else:
            input_cache = transformers_cache_from_torch(
                tuple(torch.empty((batch_size, 1, 0, dim), dtype=torch.bfloat16) for _ in range(num_layers))
            )
    else:
        assert mode == "decode"
        position_ids = position_ids_or_seq_lens.unsqueeze(1)
        max_position_id = position_ids.max().item()

        mask = torch.full((batch_size, 1, 1, max_position_id + 1), float("-inf"), dtype=torch.bfloat16)
        for mask_row, position_id in zip(mask, position_ids_or_seq_lens):
            mask_row[:, :, :position_id] = 0.0
        mask[:, :, :, -1] = 0.0

        cache_gen_function = torch.zeros if zeroed_cache else torch.randn
        if layer_idx is not None:
            input_cache = transformers_cache_single_layer_from_torch(
                cache_gen_function((batch_size, 1, max_position_id, dim), dtype=torch.bfloat16), layer_idx
            )
        else:
            input_cache = transformers_cache_from_torch(
                tuple(
                    cache_gen_function((batch_size, 1, max_position_id, dim), dtype=torch.bfloat16)
                    for _ in range(num_layers)
                )
            )

    kv_arg_name = "past_key_value" if layer_idx is not None else "past_key_values"
    model_output = reference_model(
        activation,
        attention_mask=mask,
        position_ids=position_ids,
        output_attentions=True,
        use_cache=True,
        **{kv_arg_name: deepcopy(input_cache)},
    )
    if isinstance(model_output, BaseModelOutputWithPast):
        return model_output.last_hidden_state, input_cache, model_output.past_key_values
    elif isinstance(model_output, CausalLMOutputWithPast):
        return model_output.logits, input_cache, model_output.past_key_values

    out, _, output_cache = model_output
    return out, input_cache, output_cache


def load_reference_io_tensors_for_module(
    mode: Literal["prefill", "decode"],
    module: str,
    seq_len: int,
    num_expand_rows: int,
    concat_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mode in ["prefill", "decode"], f"Unsupported mode: {mode}"

    reference_io = load_reference_io(mode, module)
    assert all(len(logs) <= 1 for logs in reference_io), f"Expected a non-range module"
    assert all(
        len(logs) > 0 for logs in reference_io
    ), f"Some logs for module {module} {mode} not generated. Please run the {REFERENCE_IO_SCRIPT_NAME} script to create it."
    if mode == "prefill":
        assert len(reference_io) == 1, f"Expected one log for {module} prefill, got {len(reference_io)}"
        (((io_module_path, (torch_input,), _, reference_output),),) = reference_io
        assert io_module_path == module
    else:
        io_module_paths, torch_args, _, reference_outputs = zip(*[logs[0] for logs in reference_io])
        (torch_inputs,) = zip(*torch_args)
        assert set(io_module_paths) == {module}
        torch_input = torch.concat(torch_inputs, dim=concat_dim)
        reference_output = torch.concat(reference_outputs, dim=concat_dim)
    torch_input.unsqueeze_(0)
    reference_output.unsqueeze_(0)
    return pad_or_trim_seq_len(torch_input, mode, seq_len).expand(
        num_expand_rows, *(-1 for _ in range(torch_input.ndim - 1))
    ), reference_output.expand(num_expand_rows, *(-1 for _ in range(reference_output.ndim - 1)))


def load_reference_io(mode: Literal["prefill", "decode"], module_range: str):
    path = (
        Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))
        / f"test_io_cache/{mode}.{module_range}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference IO cache file not found at {path}. Please run the {REFERENCE_IO_SCRIPT_NAME} script to create it. Did you set the 'HF_MODEL' environment variable coorectly?"
        )
    return torch.load(path)


SEQ_LEN_DIM_IDX = 2


def pad_or_trim_seq_len(tensor: torch.Tensor, mode: Literal["prefill", "decode"], seq_len: int) -> torch.Tensor:
    """Changes the tensor's sequence length to match the given seq_len, adding padding if necessary."""
    assert mode in ["prefill", "decode"], f"Unsupported mode: {mode}"

    tensor_seq_len = tensor.shape[SEQ_LEN_DIM_IDX]
    if tensor_seq_len == seq_len:
        return tensor.clone()

    padded_tensor_shape = list(tensor.shape)
    padded_tensor_shape[SEQ_LEN_DIM_IDX] = seq_len
    padded_tensor = torch.zeros(padded_tensor_shape, dtype=tensor.dtype, device=tensor.device)

    padded_tensor_ranges = tuple(
        slice(None) if idx != SEQ_LEN_DIM_IDX else slice(None, min(seq_len, tensor_seq_len))
        for idx in range(tensor.ndim)
    )
    padded_tensor[padded_tensor_ranges] = tensor[padded_tensor_ranges]

    return padded_tensor


def get_model_config(ModuleClass: type[AbstractModule], mode: Literal["prefill", "decode"], *args, **kwargs) -> Any:
    """Get the module config for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.prefill_model_config(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.decode_model_config(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def run_module_forward(ModuleClass: type[AbstractModule], mode: Literal["prefill", "decode"], *args, **kwargs) -> Any:
    """Run the module forward pass for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.forward_prefill(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.forward_decode(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def assert_hidden_dim_pcc(
    tt_output_torch: torch.Tensor, reference_output: torch.Tensor, pcc_required: float = 0.98
) -> float:
    assert (
        tt_output_torch.ndim == reference_output.ndim == 4
    ), f"Both model and reference outputs must be 4D tensors; got {tt_output_torch.ndim}D and {reference_output.ndim}D instead"
    assert (
        tt_output_torch.shape[0] == reference_output.shape[0]
    ), f"Model and reference output shape mismatch on dim 0 ({tt_output_torch.shape[0]} != {reference_output.shape[0]})"
    assert (
        tt_output_torch.shape[1] == reference_output.shape[1]
    ), f"Model and reference output shape mismatch on dim 1 ({tt_output_torch.shape[1]} != {reference_output.shape[1]})"
    assert (
        tt_output_torch.shape[3] == reference_output.shape[3]
    ), f"Model and reference output shape mismatch on dim 3 ({tt_output_torch.shape[3]} != {reference_output.shape[3]})"

    seq_len = min(tt_output_torch.shape[2], reference_output.shape[2])
    tt_output_torch = tt_output_torch[:, :, :seq_len, :]
    reference_output = reference_output[:, :, :seq_len, :]

    passing, pcc = comp_pcc(tt_output_torch, reference_output, pcc_required)
    logger.info(f"PCC: {pcc}")
    assert passing, f"Pearson Correlation Coefficient {pcc} is below required {pcc_required}."
    return pcc
