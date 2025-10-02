# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

from loguru import logger

CACHE_DICT_FILE = "cache_dict.json"


def cache_dir_is_set() -> bool:
    return "TT_DIT_CACHE_DIR" in os.environ


def get_cache_path(model_name, subfolder, parallel_config, dtype="bf16"):
    cache_dir = os.environ.get("TT_DIT_CACHE_DIR")
    assert cache_dir is not None, "TT_DIT_CACHE_DIR environment variable must be set if using caching."

    model_path = os.path.join(cache_dir, model_name)
    model_path = os.path.join(model_path, subfolder)
    tp_factor, tp_mesh_axis = (
        (parallel_config.tensor_parallel.factor, parallel_config.tensor_parallel.mesh_axis)
        if hasattr(parallel_config, "tensor_parallel") and parallel_config.tensor_parallel is not None
        else (None, None)
    )
    sp_factor, sp_mesh_axis = (
        (parallel_config.sequence_parallel.factor, parallel_config.sequence_parallel.mesh_axis)
        if hasattr(parallel_config, "sequence_parallel") and parallel_config.sequence_parallel is not None
        else (None, None)
    )
    parallel_name = f"tp{tp_factor}_{tp_mesh_axis}_sp{sp_factor}_{sp_mesh_axis}_{dtype}"
    cache_path = os.path.join(model_path, parallel_name) + os.sep

    return cache_path


def get_and_create_cache_path(model_name, subfolder, parallel_config, dtype="bf16"):
    cache_path = get_cache_path(model_name, subfolder, parallel_config, dtype)
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


def initialize_from_cache(tt_model, torch_model, model_name, subfolder, parallel_config, dtype="bf16"):
    if cache_dir_is_set():
        cache_path = get_and_create_cache_path(
            model_name=model_name,
            subfolder=subfolder,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        if cache_dict_exists(cache_path):
            logger.info(f"loading {subfolder} from cache...")
            tt_model.from_cached_state_dict(load_cache_dict(cache_path))
        else:
            logger.info(
                f"Cache does not exist. Creating cache: {cache_path} and loading {subfolder} from PyTorch state dict"
            )
            tt_model.load_state_dict(torch_model.state_dict())
            save_cache_dict(tt_model.to_cached_state_dict(cache_path), cache_path)
        return True
    return False
