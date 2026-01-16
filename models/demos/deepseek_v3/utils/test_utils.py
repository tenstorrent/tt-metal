# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from loguru import logger
from transformers import DynamicCache
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.scripts.generate_test_inputs_outputs import __file__ as REFERENCE_IO_SCRIPT_NAME
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, dequantize, even_int_div
from models.demos.deepseek_v3.utils.weight_config import get_weight_config
from models.tt_transformers.tt.common import PagedAttentionConfig


def load_state_dict(model_path: Path, module_path: str):
    # Lazily load HF weights: only access tensors when keys are used.
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    lazy = LazyStateDict(model_path)
    if module_path:
        # Ensure dot suffix so that keys are trimmed properly in the view
        return lazy.view_with_prefix(module_path + ".")
    return lazy


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

    # Avoid materializing any unneeded tensors by iterating over keys and filtering
    for name in {k for k in state_dict.keys() if not k.endswith("_scale_inv")}:
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"Expected tensor {name} to exist in state_dict but it was None")

        # Look for corresponding scale tensor
        scale_name = name + "_scale_inv"
        if scale_name in state_dict:
            scale_tensor = state_dict[scale_name]
            # Dequantize using the scale
            dequantized_tensor = dequantize(tensor, scale_tensor, hf_config.quantization_config["weight_block_size"])
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
    mesh_shape: tuple[int, int],
    paged_config: PagedAttentionConfig,
    user_id: int | None,
    mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a torch cache tensor into a paged cache format for the ttn model.

    Args:
        torch_cache (torch.Tensor): The input cache tensor of shape (batch_size, num_heads, seq_len, dim).
        dp_factor (int): The number of data parallel devices.
        batch_size (int): The total batch size.
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
        torch_cache = torch.zeros(
            (mesh_shape[0] * USERS_PER_ROW, *torch_cache_line.shape[1:]), dtype=torch_cache_line.dtype
        )
        torch_cache[user_id : user_id + 1] = torch_cache_line

    batch_size, num_heads, seq_len, dim = torch_cache.shape
    batches_per_device = even_int_div(batch_size, mesh_shape[0] * mesh_shape[1])
    blocks_per_batch = even_int_div(paged_config.max_num_blocks, batches_per_device)
    assert num_heads == 1, "Expected the kvpe cache to have only one head"

    if mapping is None:
        mapping = torch.randperm(batches_per_device * blocks_per_batch).reshape(batches_per_device, blocks_per_batch)
    assert mapping.shape == (batches_per_device, blocks_per_batch)

    assert paged_config.block_size * blocks_per_batch >= seq_len
    torch_cache = torch.nn.functional.pad(torch_cache, (0, 0, 0, paged_config.block_size * blocks_per_batch - seq_len))

    torch_cache = torch_cache.reshape(
        mesh_shape[0] * mesh_shape[1], batches_per_device, num_heads, blocks_per_batch, paged_config.block_size, dim
    )
    torch_cache = torch_cache.transpose(
        2, 3
    )  # (num_devices, batches_per_device, blocks_per_batch, num_heads, block_size, dim)

    paged_cache = torch.empty(
        (mesh_shape[0] * mesh_shape[1], batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim),
        dtype=torch_cache.dtype,
    )
    paged_cache[:, mapping] = torch_cache
    paged_cache = paged_cache.reshape(
        mesh_shape[0] * mesh_shape[1] * batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim
    )

    return paged_cache, mapping


def paged_caches_from_torch(
    torch_caches: tuple[torch.Tensor, ...],
    mesh_shape: tuple[int, int],
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
            paged_cache_from_torch(torch_cache, mesh_shape, paged_config, user_id, mapping)
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
    """
    Run reference model with attention, using memory optimizations for large sequences.

    For long sequences, the code splits processing into chunks to limit peak memory usage.
    All model calls are wrapped with torch.no_grad() to avoid building computation graphs and storing gradients.
    Intermediate tensors are explicitly freed between chunks using del.
    Attention weights are not stored by setting output_attentions=False, since they scale quadratically with sequence length.
    """
    (batch_size,) = position_ids_or_seq_lens.shape
    dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    num_layers = hf_config.num_hidden_layers
    max_position_id_or_seq_len = position_ids_or_seq_lens.max().item()
    mask = None

    # For sequences longer than the chunk size, use chunked processing.
    # Auto-cap chunk size so the causal mask stays within a safe memory budget.
    base_chunk_size = 8192
    bytes_per_elem = torch.tensor([], dtype=torch.bfloat16).element_size()
    target_mask_bytes = 128 * 1024**2
    mask_denominator = batch_size * max_position_id_or_seq_len * bytes_per_elem
    if mask_denominator > 0:
        max_chunk_size = target_mask_bytes // mask_denominator
        chunk_size = max(1, min(base_chunk_size, int(max_chunk_size)))
    else:
        max_chunk_size = base_chunk_size
        chunk_size = base_chunk_size
    use_chunked_processing = mode == "prefill" and max_position_id_or_seq_len > chunk_size
    if mode == "prefill":
        logger.info(
            f"Reference attention config: seq_len={max_position_id_or_seq_len} "
            f"chunk_size={chunk_size} use_chunked_processing={use_chunked_processing}"
        )

    if mode == "prefill":
        max_seq_len = position_ids_or_seq_lens.max().item()
        position_ids = torch.arange(max_seq_len).unsqueeze(0).repeat(batch_size, 1)

        if not use_chunked_processing:
            if max_position_id_or_seq_len > 16384:
                mask = torch.triu(
                    torch.full(
                        (batch_size, 1, max_position_id_or_seq_len, max_position_id_or_seq_len),
                        float("-inf"),
                        dtype=torch.bfloat16,
                        device="cpu",
                    ),
                    diagonal=1,
                )
            else:
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
    deepcopied_cache = deepcopy(input_cache)

    def extract_output_and_cache(model_output) -> tuple[torch.Tensor, DynamicCache]:
        if isinstance(model_output, tuple):
            cache_idx = 2 if len(model_output) == 3 else 1
            return model_output[0], model_output[cache_idx]
        if hasattr(model_output, "logits"):
            return model_output.logits, model_output.past_key_values
        if hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state, model_output.past_key_values
        raise AttributeError(f"Model output has neither 'last_hidden_state' nor 'logits': {type(model_output)}")

    if use_chunked_processing:
        device = activation.device
        num_chunks = (max_position_id_or_seq_len + chunk_size - 1) // chunk_size

        output_chunks = []
        current_cache = deepcopy(deepcopied_cache)

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, max_position_id_or_seq_len)
                chunk_size_actual = end_idx - start_idx

                # Extract chunk from activation and position_ids
                if activation.ndim == 2:
                    activation_chunk = activation[:, start_idx:end_idx].contiguous()
                else:
                    activation_chunk = activation[:, start_idx:end_idx, :].contiguous()
                position_ids_chunk = position_ids[:, start_idx:end_idx].contiguous()

                # Determine current cache length to properly construct mask
                if layer_idx is not None:
                    cache_tensor = current_cache.key_cache[layer_idx]
                    current_cache_length = cache_tensor.shape[2]
                else:
                    legacy_cache = current_cache.to_legacy_cache()
                    first_layer_cache = legacy_cache[0][0]
                    current_cache_length = first_layer_cache.shape[2] if first_layer_cache.numel() > 0 else 0

                kv_seq_len = current_cache_length + chunk_size_actual
                mask_bytes = batch_size * chunk_size_actual * kv_seq_len * bytes_per_elem
                logger.info(
                    f"Reference chunk {chunk_idx + 1}/{num_chunks}: start={start_idx} end={end_idx} "
                    f"kv_seq_len={kv_seq_len} mask_mb={mask_bytes / (1024 ** 2):.1f}"
                )

                # Create causal mask for this chunk
                # Tokens can attend to: (1) all cached tokens, (2) previous tokens in current chunk
                mask_chunk = torch.full(
                    (batch_size, 1, chunk_size_actual, kv_seq_len),
                    float("-inf"),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for i in range(chunk_size_actual):
                    mask_chunk[:, :, i, :current_cache_length] = 0.0  # Attend to cached tokens
                    mask_chunk[
                        :, :, i, current_cache_length : current_cache_length + i + 1
                    ] = 0.0  # Causal mask within chunk

                # Set output_attentions=False to avoid storing attention weights that scale quadratically with sequence length
                chunk_output = reference_model(
                    activation_chunk,
                    attention_mask=mask_chunk,
                    position_ids=position_ids_chunk,
                    output_attentions=False,
                    use_cache=True,
                    **{kv_arg_name: current_cache},
                )

                chunk_out, current_cache = extract_output_and_cache(chunk_output)

                output_chunks.append(chunk_out)

                # Free intermediate tensors to reduce memory usage
                del activation_chunk, position_ids_chunk, mask_chunk, chunk_output

            # Concatenate all chunk outputs
            model_output_tensor = torch.cat(output_chunks, dim=1)

            # Clean up chunk list
            del output_chunks

            out = model_output_tensor
            output_cache = current_cache
    else:
        # Standard processing for shorter sequences or decode mode
        if mask is not None and mask.device.type == "cpu":
            mask = mask.to(activation.device)

        # Use torch.no_grad() to prevent gradient accumulation
        with torch.no_grad():
            # Set output_attentions=False to save memory
            model_output_raw = reference_model(
                activation,
                attention_mask=mask,
                position_ids=position_ids,
                output_attentions=False,
                use_cache=True,
                **{kv_arg_name: deepcopied_cache},
            )

            out, output_cache = extract_output_and_cache(model_output_raw)

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
    tt_output_torch = tt_output_torch.cpu().float()

    assert (
        all(
            d1 == d2
            for d1, d2 in itertools.zip_longest(tt_output_torch.shape[:-2], reference_output.shape[:-2], fillvalue=1)
        )
        and tt_output_torch.shape[-1] == reference_output.shape[-1]
    ), (
        "Model and reference output shape must match on all dimensions except for the second to last one "
        f"(module leading singleton dimensions); got {tt_output_torch.shape=} and {reference_output.shape=} "
    )

    seq_len_or_batch_size = min(tt_output_torch.shape[-2], reference_output.shape[-2])
    tt_output_torch = tt_output_torch[..., :seq_len_or_batch_size, :]
    reference_output = reference_output[..., :seq_len_or_batch_size, :]

    # For very large sequences, `comp_pcc` can OOM due to its internal clones + numpy conversion.
    # If the full PCC is estimated to exceed a memory threshold, process the sequence dimension in chunks.
    hidden_dim = tt_output_torch.shape[-1]
    estimated_memory_gb = (seq_len_or_batch_size * hidden_dim * 4 * 2 * 2) / (1024**3)

    MAX_MEMORY_GB = 50  # Switch to chunking if the estimated full-tensor PCC exceeds this
    CHUNK_SIZE = 8192  # Compare up to 8K sequence positions per chunk

    if estimated_memory_gb > MAX_MEMORY_GB and seq_len_or_batch_size > CHUNK_SIZE:
        num_chunks = (seq_len_or_batch_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_pccs: list[float] = []
        failed_chunks: list[int] = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, seq_len_or_batch_size)

            tt_chunk = tt_output_torch[..., start_idx:end_idx, :]
            ref_chunk = reference_output[..., start_idx:end_idx, :]

            passing, pcc = comp_pcc(tt_chunk, ref_chunk, pcc_required)
            chunk_pccs.append(pcc)

            if not passing:
                failed_chunks.append(chunk_idx + 1)
                logger.error(
                    f"PCC chunk {chunk_idx + 1}/{num_chunks} failed: pcc={pcc} < required={pcc_required} "
                    f"(seq_range=[{start_idx}:{end_idx}])"
                )

        min_pcc = min(chunk_pccs)
        avg_pcc = sum(chunk_pccs) / len(chunk_pccs)
        logger.info(f"PCC (chunked): min={min_pcc:.6f}, avg={avg_pcc:.6f}")

        assert not failed_chunks, (
            "Not all chunks passed PCC check. "
            f"Min PCC: {min_pcc:.6f} (required: {pcc_required}), "
            f"Failed chunks: {failed_chunks}"
        )
        return min_pcc

    passing, pcc = comp_pcc(tt_output_torch, reference_output, pcc_required)
    logger.info(f"PCC: {pcc}")
    assert passing, f"Pearson Correlation Coefficient {pcc} is below required {pcc_required}."
    return pcc


def get_test_weight_config(
    ModuleClass: type[AbstractModule],
    hf_config: PretrainedConfig,
    state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
    cache_path: Path,
    mesh_device: ttnn.Device,
    force_recalculate: bool,
) -> Any:
    """Get the weight config, either by loading from cache or recalculating."""
    per_test_weight_cache_path = cache_path / "tests_cache" / os.environ.get("PYTEST_CURRENT_TEST")
    return get_weight_config(
        ModuleClass, hf_config, state_dicts, per_test_weight_cache_path, mesh_device, force_recalculate
    )


def get_rope_tensors(
    hf_config: PretrainedConfig,
    batch_size_per_row: int,
    seq_len: int,
    position_ids: torch.Tensor | None,
    mesh_device: ttnn.MeshDevice,
) -> dict[str, ttnn.Tensor]:
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size_per_row=batch_size_per_row,
        hf_config=hf_config,
    )
    if position_ids is None:
        return rope_setup.get_rot_mats_table(seq_len)
    return rope_setup.get_rot_mats(position_ids)


# Mapping of system names to their corresponding mesh shapes
SYSTEM_NAME_TO_MESH_SHAPE: dict[str, tuple[int, int]] = {
    "TG": (4, 8),
    "DUAL": (8, 8),
    "QUAD": (16, 8),
    "T3K": (1, 8),
    "N300": (1, 2),
    "N150": (1, 1),
}


def get_valid_system_names() -> tuple[str, ...]:
    return tuple(SYSTEM_NAME_TO_MESH_SHAPE.keys())


def system_name_to_mesh_shape(system_name: str) -> ttnn.MeshShape:
    if system_name not in SYSTEM_NAME_TO_MESH_SHAPE:
        valid_system_names = get_valid_system_names()
        raise ValueError(
            f"Unsupported system name: {system_name}. Supported values are {', '.join(valid_system_names)}."
        )

    rows, cols = SYSTEM_NAME_TO_MESH_SHAPE[system_name]
    return ttnn.MeshShape(rows, cols)
