# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
from glob import glob

import torch
import tqdm
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights


def convert_bf16_to_fp32(weights_dict):
    """
    Converts all bfloat16 tensors in a dictionary to float32 tensors.

    Args:
        weights_dict (dict): A dictionary, typically a model's state_dict,
                             where keys are parameter names and values are tensors.

    Returns:
        dict: A new dictionary with all bfloat16 tensors converted to float32.
              Other tensor types remain unchanged.
    """
    converted_dict = {}
    for key, value in tqdm(weights_dict.items()):
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            converted_dict[key] = value.to(torch.float32)
            # print(f"Converted tensor '{key}' from bfloat16 to float32.")
        else:
            converted_dict[key] = value
    return converted_dict


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model_uninitialized(model_path: str):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        model = model.to(torch.float32)  # Ensure model is in float32
    # torch.set_default_dtype(current_dtype)

    model.eval()
    return model


def load_model_weights(
    model_path: str, thread_pool_executor: concurrent.futures.ThreadPoolExecutor | None = None
) -> dict[str, torch.Tensor]:
    safetensors_filepaths = sorted(glob(f"{model_path}/*.safetensors"))

    if thread_pool_executor is None:
        # Sequential loading
        weights_dict = {}
        for filepath in tqdm(safetensors_filepaths, desc="Loading weights"):
            weights_dict.update(load_file(filepath))
    else:
        # Parallel loading with proper thread safety
        def load_single_file(filepath):
            return load_file(filepath)

        # Load all files in parallel
        futures = [thread_pool_executor.submit(load_single_file, fp) for fp in safetensors_filepaths]

        # Combine results sequentially
        weights_dict = {}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(safetensors_filepaths), desc="Loading weights"
        ):
            weights_dict.update(future.result())

    print("Loaded all weights")
    return weights_dict
