# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared Base/SFT encoder: all 32 layer outputs and normalized conditioning."""
import pytest
import ttnn

from models.experimental.moss_image.tests.assets import check
from models.experimental.moss_image.tests.conftest import verify_weights
from models.experimental.moss_image.tt.text_encoder import TTTextEncoder


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("branch", ["positive", "negative"])
def test_text_encoder(moss_assets, mesh_device, branch, record_property):
    checkpoint, capture = moss_assets
    verify_weights(checkpoint, capture, ["text_encoder", "tokenizer"])
    inputs = capture.load(f"text/{branch}/inputs")
    length = inputs["input_ids"].shape[1]
    observed = []

    def observe(name, value):
        if name.startswith("layer_") and name.endswith(".output"):
            layer = int(name.split(".")[0].split("_")[1])
            expected = capture.load(f"text/{branch}/layers/language_model.layers.{layer}")["output"]
        elif name == "last_hidden_state":
            expected = capture.load(f"text/{branch}/output")
        else:
            return
        actual = ttnn.to_torch(value).float()[:, :length]
        values = check(actual, expected, lambda key, val: record_property(f"{name}.{key}", val))
        observed.append(values)

    TTTextEncoder(checkpoint, mesh_device).encode(inputs["input_ids"], observe=observe)
    assert len(observed) == 33, "All 32 layers and final normalization must be compared"
