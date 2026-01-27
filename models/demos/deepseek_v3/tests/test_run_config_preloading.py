# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, SavedWeight
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    create_run_config,
    preload_weights_parallel,
)


def test_preload_weights_parallel_behavior():
    """
    Test that preload_weights_parallel correctly populates the cache
    and that subsequent create_run_config calls utilize that cache.
    """
    # Mock mesh device
    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (1, 1)

    # Mock weight and configs
    weight_path = Path("dummy/path/weight.bin")
    saved_weight = SavedWeight(path=weight_path, memory_config=None)

    model_config = {"layer": FromWeightConfig(mesh_device=mock_mesh_device)}
    weight_config = {"layer": saved_weight}
    # Provide a dummy model state that contains the mesh device
    model_state = {MESH_DEVICE_STATE_DICT_KEY: mock_mesh_device}

    cached_ttnn_weights = {}

    # 1. Test that preload_weights_parallel loads weights into cache
    with patch("models.demos.deepseek_v3.utils.run_config.load_weight") as mock_load_weight:
        mock_tensor = MagicMock(spec=ttnn.Tensor)
        mock_load_weight.return_value = mock_tensor

        preload_weights_parallel(model_config, weight_config, model_state, cached_ttnn_weights=cached_ttnn_weights)

        assert weight_path in cached_ttnn_weights
        assert cached_ttnn_weights[weight_path] == mock_tensor
        mock_load_weight.assert_called_once_with(saved_weight, mock_mesh_device)

    # 2. Test that weights present in the cache are skipped by the preloader
    with patch("models.demos.deepseek_v3.utils.run_config.load_weight") as mock_load_weight:
        preload_weights_parallel(model_config, weight_config, model_state, cached_ttnn_weights=cached_ttnn_weights)
        # Should not be called again because it's already in cache
        mock_load_weight.assert_not_called()

    # 3. Test that subsequent create_run_config uses preloaded weights
    with patch("models.demos.deepseek_v3.utils.run_config.load_weight") as mock_load_weight:
        run_config = create_run_config(
            model_config, weight_config, model_state, cached_ttnn_weights=cached_ttnn_weights
        )

        assert run_config["layer"] == mock_tensor
        # verify load_weight was NOT called by create_run_config because it utilized the cache
        mock_load_weight.assert_not_called()


def test_preload_weights_parallel_with_multiple_weights():
    """
    Test preloading with multiple unique weights and verify they are all cached.
    """
    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (1, 1)

    w1_path = Path("weight1.bin")
    w2_path = Path("weight2.bin")
    sw1 = SavedWeight(path=w1_path, memory_config=None)
    sw2 = SavedWeight(path=w2_path, memory_config=None)

    model_config = {
        "layers": [
            FromWeightConfig(mesh_device=mock_mesh_device),
            FromWeightConfig(mesh_device=mock_mesh_device),
        ]
    }
    weight_config = {"layers": [sw1, sw2]}
    model_state = {MESH_DEVICE_STATE_DICT_KEY: mock_mesh_device}

    cached_ttnn_weights = {}

    with patch("models.demos.deepseek_v3.utils.run_config.load_weight") as mock_load_weight:

        def load_weight_side_effect(sw, dev):
            return f"tensor_{sw.path}"

        mock_load_weight.side_effect = load_weight_side_effect

        preload_weights_parallel(model_config, weight_config, model_state, cached_ttnn_weights=cached_ttnn_weights)

        assert w1_path in cached_ttnn_weights
        assert w2_path in cached_ttnn_weights
        assert cached_ttnn_weights[w1_path] == f"tensor_{w1_path}"
        assert cached_ttnn_weights[w2_path] == f"tensor_{w2_path}"
        assert mock_load_weight.call_count == 2
