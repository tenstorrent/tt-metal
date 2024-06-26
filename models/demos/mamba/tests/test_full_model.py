# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer
from typing import Optional
import ttnn
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.tt.full_model import MambaTT
from models.demos.mamba.tt import model_config
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


def run_inference(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
    num_layers: int,
    iterations: int,
    cache_dir: Optional[str],
):
    torch.manual_seed(10)

    reference_model = MambaDecode.from_pretrained(model_version, batch_size=batch)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    input_ids = input_ids.repeat(batch, 1)

    mamba_model_pytorch = MambaPytorch(reference_model, num_layers)
    mamba_model_pytorch.eval()
    for _ in range(iterations):
        with torch.no_grad():
            reference_output = mamba_model_pytorch(input_ids)

    config = model_config.create_model_config(batch, reference_model.args.d_model)
    mamba_model_tt = MambaTT(reference_model, device, config, tt_cache_path=cache_dir, num_layers=num_layers)

    for _ in range(iterations):
        tt_output = mamba_model_tt(input_ids)

    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba output failed")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Not supported on Grayskull")
@pytest.mark.parametrize(
    "model_version, batch, pcc, num_layers, iterations",
    (
        (
            "state-spaces/mamba-2.8b",
            32,
            0.94,
            64,
            1,
        ),
    ),
)
def test_inference(
    device: ttnn.Device,
    use_program_cache,
    get_tt_cache_path,
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
    num_layers: int,
    iterations: int,
):
    run_inference(
        device,
        use_program_cache,
        model_version,
        batch,
        pcc,
        num_layers,
        iterations,
        cache_dir=get_tt_cache_path(model_version),
    )


@skip_for_grayskull("Not supported on Grayskull")
@pytest.mark.parametrize(
    "iterations",
    (1, 2),
)
def test_device_perf(
    device: ttnn.Device,
    use_program_cache,
    get_tt_cache_path,
    iterations,
    model_version="state-spaces/mamba-2.8b",
    batch=32,
    pcc=0.97,
    num_layers=1,
):
    run_inference(
        device,
        use_program_cache,
        model_version,
        batch,
        pcc,
        num_layers,
        iterations,
        cache_dir=get_tt_cache_path(model_version),
    )
