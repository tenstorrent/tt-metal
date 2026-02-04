# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

from loguru import logger

CACHE_DICT_FILE = "cache_dict.json"


def config_id(parallel_config):
    config_id = ""
    for n, v in parallel_config._asdict().items():
        if v is not None:
            config_id += f"{''.join([w[0].upper() for w in n.split('_')])}{v.factor}_{v.mesh_axis}_"
    return config_id


def cache_dir_is_set() -> bool:
    return "TT_DIT_CACHE_DIR" in os.environ


def get_cache_path(model_name, subfolder, parallel_config, mesh_shape, dtype="bf16", is_fsdp=False):
    cache_dir = os.environ.get("TT_DIT_CACHE_DIR")
    assert cache_dir is not None, "TT_DIT_CACHE_DIR environment variable must be set if using caching."

    model_path = os.path.join(os.path.abspath(cache_dir), model_name)
    model_path = os.path.join(model_path, subfolder)
    parallel_name = f"{config_id(parallel_config)}mesh{mesh_shape[0]}x{mesh_shape[1]}_{dtype}" + (
        "_FSDP" if is_fsdp else ""
    )
    cache_path = os.path.join(model_path, parallel_name) + os.sep

    return cache_path


def get_and_create_cache_path(model_name, subfolder, parallel_config, mesh_shape, dtype="bf16", is_fsdp=False):
    cache_path = get_cache_path(model_name, subfolder, parallel_config, mesh_shape, dtype, is_fsdp)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def save_cache_dict(cache_dict, cache_path):
    with open(os.path.join(cache_path, CACHE_DICT_FILE), "w") as f:
        json.dump(cache_dict, f)


def load_cache_dict(cache_path):
    with open(os.path.join(cache_path, CACHE_DICT_FILE), "r") as f:
        return json.load(f)


def cache_dict_exists(cache_path):
    return os.path.exists(os.path.join(cache_path, CACHE_DICT_FILE))


def initialize_from_cache(
    tt_model, torch_state_dict, model_name, subfolder, parallel_config, mesh_shape, dtype="bf16", is_fsdp=False
):
    if cache_dir_is_set():
        cache_path = get_and_create_cache_path(
            model_name=model_name,
            subfolder=subfolder,
            parallel_config=parallel_config,
            mesh_shape=mesh_shape,
            dtype=dtype,
            is_fsdp=is_fsdp,
        )
        if cache_dict_exists(cache_path):
            logger.info(f"loading {subfolder} from cache... {cache_path}")
            tt_model.from_cached_state_dict(load_cache_dict(cache_path))
        elif torch_state_dict is not None:
            logger.info(
                f"Cache does not exist. Creating cache: {cache_path} and loading {subfolder} from PyTorch state dict"
            )
            tt_model.load_torch_state_dict(torch_state_dict)
            save_cache_dict(tt_model.to_cached_state_dict(cache_path), cache_path)
        else:
            return False
        return True
    return False
