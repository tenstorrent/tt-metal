# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn

from models.experimental.moss_image.tests.assets import check
from models.experimental.moss_image.tests.conftest import verify_weights
from models.experimental.moss_image.tt.vae import TTMossVAE


@pytest.mark.parametrize("moss_assets", ["Base", "SFT"], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 131072}], indirect=True)
@pytest.mark.parametrize("case", ["preprocess", "decoder"])
def test_vae(moss_assets, mesh_device, case, record_property):
    checkpoint, capture = moss_assets
    verify_weights(checkpoint, capture, ["vae"])
    model = TTMossVAE(checkpoint, mesh_device)
    last = max(step["index"] for step in capture.manifest["steps"])
    latent = capture.load(f"steps/{last:03d}/updated_latents")
    packed = latent.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
    raw = capture.load("vae/raw_output")
    height, width = raw.shape[-2:]
    if case == "preprocess":
        expected = capture.load("vae/input")
        value = model.preprocess(model.upload(packed), height, width)
    else:
        expected = raw.clamp(-1, 1)
        value = model.decode(model.upload(packed), height, width)
    n, channels, height, width = expected.shape
    actual = model.collect(ttnn.reshape(value, (n, height, width, channels)))
    check(actual, expected.float(), record_property)
