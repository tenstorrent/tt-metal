# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib

from models.utility_functions import torch2tt_tensor, pad_by_zero


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    tt_layout=tt_lib.tensor.Layout.TILE,
):
    if padzero:
        assert tt_layout == tt_lib.tensor.Layout.TILE, "padding by zero currently only uses TILE layout"

    """Load cached weights and duplicate per device. Store if not cached."""
    if (
        not overwrite
        and (tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin").exists()
    ):
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(
            str(tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin")
        )
    else:
        if padzero:
            weights_host = pad_by_zero(
                weights_to_cache,
                device=None,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )[0]
        else:
            weights_host = torch2tt_tensor(
                weights_to_cache,
                tt_device=None,
                tt_layout=tt_layout,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )
        # Store weights
        tt_lib.tensor.dump_tensor(
            str(tt_cache_path / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}.bin"),
            weights_host,
        )

    # Duplicate weights on all devices
    weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]

    return weights
