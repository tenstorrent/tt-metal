# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import os
from glob import glob

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights


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
