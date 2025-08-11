# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple

import safetensors.torch
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.scripts.generate_test_inputs_outputs import __file__ as REFERENCE_IO_SCRIPT_NAME
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import SEQ_LEN_CHUNK_SIZE, dequantize
from models.utility_functions import comp_pcc

# Constant for testing
MAX_START_POS = 512


def load_state_dict(model_path: Path, module_path: str):
    if not not module_path:
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


def add_inv_scale_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    block_shape: Sequence[int],
    weight_names: list[str] = ["up_proj", "down_proj", "gate_proj"],
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

        dequant_scale = torch.randn(
            (
                *tensor.shape[: -len(block_shape)],
                *(
                    (tensor.shape[-len(block_shape) + idx] + block_dim - 1) // block_dim
                    for idx, block_dim in enumerate(block_shape)
                ),
            ),
            dtype=torch.float32,
        )

        tensor_quant = dequantize(tensor.to(torch.float8_e4m3fn), 1.0 / dequant_scale, block_shape)
        output_state_dict[name] = tensor_quant.to(torch.float8_e4m3fn)
        output_state_dict[name + "_scale_inv"] = dequant_scale

    return output_state_dict


def quantize_fp8_blockwise(tensor: torch.Tensor, block_shape: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 using blockwise quantization as done in DeepSeek-V3/HuggingFace.

    Args:
        tensor: Input tensor to quantize (typically float32 or bfloat16)
        block_shape: Shape of quantization blocks (e.g., [128, 128] for DeepSeek-V3)

    Returns:
        Tuple of (quantized_tensor, inverse_scale_tensor)
        - quantized_tensor: FP8 quantized tensor (torch.float8_e4m3fn)
        - inverse_scale_tensor: Inverse scale factors for dequantization (float32)
    """
    assert len(block_shape) == tensor.ndim

    # Convert to float32 for computation if needed
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Calculate the number of blocks in each dimension
    # This handles non-divisible cases by using ceil division
    scale_shape = tuple((tensor.shape[i] + block_shape[i] - 1) // block_shape[i] for i in range(tensor.ndim))

    # Initialize outputs
    quantized = torch.zeros(tensor.shape).to(torch.float8_e4m3fn)
    inv_scale = torch.ones(scale_shape, dtype=torch.float32, device=tensor.device)

    # FP8 E4M3 maximum value (448.0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Process each block
    for block_indices in product(*[range(s) for s in scale_shape]):
        # Calculate the actual slice bounds for this block
        slices = []
        for i, block_idx in enumerate(block_indices):
            start = block_idx * block_shape[i]
            end = min(start + block_shape[i], tensor.shape[i])
            slices.append(slice(start, end))
        slices = tuple(slices)

        # Extract the block
        block = tensor[slices]

        # Find the maximum absolute value in this block
        absmax = torch.abs(block).max()

        if absmax > 0:
            # Calculate scale factor to map to FP8 range
            scale = fp8_max / absmax
            inv_scale[block_indices] = 1.0 / scale

            # Quantize the block
            quantized[slices] = (block * scale).to(torch.float8_e4m3fn)
        else:
            # Block is all zeros
            inv_scale[block_indices] = 1.0
            quantized[slices] = block.to(torch.float8_e4m3fn)

    return quantized, inv_scale


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
    return pad_tensor(torch_input, mode, seq_len).expand(
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


def pad_tensor(tensor: torch.Tensor, mode: Literal["prefill", "decode"], seq_len: int) -> torch.Tensor:
    assert mode in ["prefill", "decode"], f"Unsupported mode: {mode}"

    tensor_seq_len = tensor.shape[-2]
    seq_len = min(seq_len, tensor_seq_len)
    if mode == "decode" or seq_len < SEQ_LEN_CHUNK_SIZE:
        return tensor[..., :seq_len, :].clone()

    padded_seq_len = ttnn.core.roundup(seq_len, SEQ_LEN_CHUNK_SIZE)
    padded_tensor_shape = list(tensor.shape)
    padded_tensor_shape[-2] = padded_seq_len

    padded_tensor = torch.zeros(padded_tensor_shape, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., : min(padded_seq_len, tensor_seq_len), :] = tensor

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
