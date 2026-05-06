# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.rvc.torch_impl.crepe import CrepePredictor as TorchCrepePredictor
from models.demos.rvc.tt_impl.crepe import CrepePredictor as TTCrepePredictor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_crepe_model(device):
    torch.manual_seed(0)
    model_name = "tiny"
    batch_size = 1024
    pcc = 0.96

    torch_predictor = TorchCrepePredictor(model=model_name)
    tt_predictor = TTCrepePredictor(model=model_name, device=device)

    sample_rate = 16000
    hop_length = 160
    audio_num_samples = batch_size * hop_length
    audio = torch.randn(1, audio_num_samples, dtype=torch.float32)

    torch_output = (
        torch_predictor.predict(
            audio,
            sample_rate=sample_rate,
            hop_length=hop_length,
            batch_size=batch_size,
            confidence_threshold=0.7,
        )
        .detach()
        .to(torch.float32)
    )
    tt_output = (
        tt_predictor.predict(
            audio,
            sample_rate=sample_rate,
            hop_length=hop_length,
            batch_size=batch_size,
            confidence_threshold=0.7,
        )
        .detach()
        .to(torch.float32)
    )

    assert tuple(torch_output.shape) == tuple(tt_output.shape)
    assert_with_pcc(torch_output, tt_output, pcc=pcc)
