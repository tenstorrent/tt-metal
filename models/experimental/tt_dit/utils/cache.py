# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import json

CACHE_DICT_FILE = "cache_dict.json"


def get_cache_path(model_name, subfolder, parallel_config, dtype="bf16"):
    cache_dir = os.environ.get("TT_DIT_CACHE_DIR")
    assert cache_dir is not None, "TT_DIT_CACHE_DIR environment variable must be set if using caching."

    model_path = os.path.join(cache_dir, model_name)
    model_path = os.path.join(model_path, subfolder)
    parallel_name = f"tp{parallel_config.tensor_parallel.factor}_{parallel_config.tensor_parallel.mesh_axis}_sp{parallel_config.sequence_parallel.factor}_{parallel_config.sequence_parallel.mesh_axis}_{dtype}"
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
