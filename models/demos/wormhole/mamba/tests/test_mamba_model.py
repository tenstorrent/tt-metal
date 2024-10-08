# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer
from typing import Optional
import ttnn
from models.demos.wormhole.mamba.reference.decode_model import MambaPretrainedModelName
from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba
from models.demos.wormhole.mamba.reference.args import ModelMode
from models.demos.wormhole.mamba.tt.mamba_model import MambaTT
from models.demos.wormhole.mamba.tt import model_config
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
    mode: ModelMode,
    batch: int,
    seq_len: int,
    pcc: float,
    num_layers: int,
    iterations: int,
    cache_dir: Optional[str],
):
    torch.manual_seed(0)

    reference_model = Mamba.from_pretrained(model_version, batch_size=batch)
    reference_model.args.mode = mode

    input_sentence = "Climate change refers to long-term shifts in temperatures and weather patterns. \
Such shifts can be natural, due to changes in the sun's activity or large volcanic eruptions. \
But since the 1800s, human activities have been the main driver of climate change, primarily \
due to the burning of fossil fuels like coal, oil and gas. Burning fossil fuels generates \
greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping \
the sun's heat and raising temperatures. The main greenhouse gases that are causing \
climate change include carbon dioxide and methane. These come from using gasoline \
for driving a car or coal for heating a building, for example. Clearing land and \
cutting down forests can also release carbon dioxide. Agriculture, oil and gas \
operations are major sources of methane emissions. Energy, industry, transport, \
buildings, agriculture and land use are among the main sectors causing greenhouse gases."

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    input_ids = tokenizer(input_sentence, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, :seq_len]
    input_ids = input_ids.repeat(batch, 1)

    mamba_model_pytorch = MambaPytorch(reference_model, num_layers)
    mamba_model_pytorch.eval()
    for idx in range(iterations):
        with torch.no_grad():
            reference_output = mamba_model_pytorch(input_ids)

    config = model_config.create_model_config(batch, reference_model.args.d_model, mode=mode, seq_len=seq_len)

    logger.info(f"Using tensor cache path: '{cache_dir}'")

    start = time.time()
    mamba_model_tt = MambaTT(reference_model, device, config, tt_cache_path=cache_dir, num_layers=num_layers)
    logger.info(f"Finished initializing Mamba (took {time.time() - start:.3f} sec)")

    for _ in range(iterations):
        tt_output = mamba_model_tt(input_ids)

    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba output failed")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull("Not supported on Grayskull")
@pytest.mark.parametrize(
    "model_version, mode, batch, seq_len, num_layers, iterations, pcc",
    (
        (
            "state-spaces/mamba-2.8b",
            ModelMode.PREFILL,
            1,
            32,
            64,
            1,
            0.9759,
        ),
        (
            "state-spaces/mamba-2.8b",
            ModelMode.PREFILL,
            1,
            128,
            64,
            1,
            0.9604,
        ),
        (
            "state-spaces/mamba-2.8b",
            ModelMode.DECODE,
            32,
            1,
            64,
            1,
            0.9649,
        ),
        (
            "state-spaces/mamba-2.8b",
            ModelMode.DECODE,
            32,
            1,
            1,
            1,
            0.9955,
        ),
    ),
)
def test_inference(
    device: ttnn.Device,
    use_program_cache,
    get_tt_cache_path,
    model_version: MambaPretrainedModelName,
    mode: ModelMode,
    batch: int,
    seq_len: int,
    num_layers: int,
    iterations: int,
    pcc: float,
):
    run_inference(
        device,
        use_program_cache,
        model_version,
        mode,
        batch,
        seq_len,
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
        ModelMode.DECODE,
        batch,
        1,
        pcc,
        num_layers,
        iterations,
        cache_dir=get_tt_cache_path(model_version),
    )
