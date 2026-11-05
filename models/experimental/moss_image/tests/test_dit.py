# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn

from models.experimental.moss_image.tests.assets import check
from models.experimental.moss_image.tests.conftest import verify_weights
from models.experimental.moss_image.tt.dit import TTMossDiT


@pytest.mark.parametrize("moss_assets", ["Base", "SFT"], indirect=True)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), {"l1_small_size": 131072}, id="1-card"),
        pytest.param((1, 2), {"l1_small_size": 131072, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2-cards"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("case", ["block-0", "block-21", "block-41", "step"])
def test_dit(moss_assets, mesh_device, case, record_property):
    checkpoint, capture = moss_assets
    verify_weights(checkpoint, capture, ["transformer"])
    model = TTMossDiT(checkpoint, mesh_device)
    for branch in ("positive", "negative"):
        if case.startswith("block-"):
            layer = int(case.split("-")[1])
            values = capture.load(f"blocks/{branch}/{layer:02d}/inputs")
            modulation = tuple(model.upload(value.unsqueeze(1)) for value in values["modulation"].unbind(1))
            actual = model.block(model.upload(values["x"]), layer, values["rope"], modulation)
            expected = capture.load(f"blocks/{branch}/{layer:02d}/output")
        else:
            latent = capture.load("steps/000/input_latents")
            _, _, height, width = latent.shape
            cond = capture.load("conditioning/" + branch)
            context = cond["hidden_states"][:, cond["attention_mask"][0].bool()]
            actual = model(
                model.upload(latent.flatten(2).transpose(1, 2), dtype=ttnn.float32),
                float(capture.load("steps/000/timestep").reshape(-1)[0]),
                model.upload(context),
                height,
                width,
                skip_layers=cond["skip_layers"],
            )
            expected = capture.load(f"steps/000/{branch}_prediction").flatten(2).transpose(1, 2)
        check(
            model.host(actual),
            expected,
            lambda key, value: record_property(f"{branch}.{key}", value),
            min_pcc=0.99,
            max_relative_rms=0.02,
        )
