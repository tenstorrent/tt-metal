# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.experimental.mamba.tt.full_model import MambaTT
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull


class MambaPytorch(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers=None):
        super().__init__()
        self.embedding = hf_reference_model.embedding

        if num_layers is None:
            self.layers = hf_reference_model.layers
        else:
            self.layers = hf_reference_model.layers[:num_layers]

        self.norm_f = hf_reference_model.norm_f
        self.lm_head = hf_reference_model.lm_head

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.lm_head(x)
        return x


@pytest.mark.skip(reason="Test failing, see issue #7551")
@skip_for_grayskull("Not supported on Grayskull")
@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-370m",
            1,
            0.99,
        ),
    ),
)
def test_mamba_model_inference(
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
    device: ttnn.experimental.device,
):
    torch.manual_seed(10)

    reference_model = MambaDecode.from_pretrained(model_version, batch_size=batch)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]

    reference_output = MambaPytorch(reference_model)(input_ids)

    mamba_model = MambaTT(reference_model, device)

    tt_output = mamba_model(input_ids)

    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
