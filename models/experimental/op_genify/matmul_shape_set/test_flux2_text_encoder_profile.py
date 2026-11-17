# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the stock FLUX.2-dev text encoder (tt_dit Mistral3Encoder) for the #65148 matmul shape set.

Same encoder class, mesh and parallel settings as the QB2 (2, 2) preset of
`models/tt_dit/pipelines/flux2/pipeline_flux2.py` (encoder TP on mesh axis 1, Linear, 2 links) and the
pipeline's 512-token prompt. The FLUX.2-dev checkpoint is gated, so the config comes from the public
mistralai/Mistral-Small-3.1-24B-Instruct-2503 (same text dims: hidden 5120, FFN 32768, 32 Q / 8 KV heads
x 128) and the weights are random. Two layers stand in for the 30 the pipeline runs (it reads layers 10,
20, 30); every layer has the same shapes.

Run (needs a Tracy build and MISTRAL3_CONFIG_DIR holding that config.json):
    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_flux2_text_encoder_profile.py
"""

import os

import pytest
import torch
import transformers
from tracy import signpost

import ttnn
from models.tt_dit.encoders.mistral3.model_mistral3 import Mistral3Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.flux2.device_params import prompt_encoder_params_flux2
from models.tt_dit.utils import cache, tensor

NUM_LAYERS = 2
SEQUENCE_LENGTH = 512


@pytest.mark.parametrize(
    "mesh_device, encoder_tp_axis, num_links, topology, device_params",
    [[(2, 2), 1, 2, ttnn.Topology.Linear, prompt_encoder_params_flux2]],
    ids=["bh_2x2_linear_pipeline"],
    indirect=["mesh_device", "device_params"],
)
def test_flux2_text_encoder_matmul_profile(mesh_device, encoder_tp_axis, num_links, topology):
    os.environ.pop("TT_DIT_CACHE_DIR", None)  # random weights: never touch a weight cache
    hf_config = transformers.AutoConfig.from_pretrained(os.environ["MISTRAL3_CONFIG_DIR"])
    hf_config.text_config.num_hidden_layers = NUM_LAYERS
    hf_config.vision_config.num_hidden_layers = 1

    torch.manual_seed(0)
    torch_model = transformers.Mistral3ForConditionalGeneration(hf_config).to(torch.bfloat16).eval()

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=int(mesh_device.shape[encoder_tp_axis]), mesh_axis=encoder_tp_axis)
    )
    encoder = Mistral3Encoder(
        Mistral3Encoder.config_from_hf(hf_config),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=CCLManager(mesh_device, num_links=num_links, topology=topology),
    )
    cache.load_model(
        encoder,
        model_name="FLUX.2-dev-text-encoder-random-weights-profile",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=lambda: Mistral3Encoder.convert_state(torch_model.state_dict()),
        create_cache=False,
    )

    tokens = torch.randint(0, hf_config.text_config.vocab_size, [1, SEQUENCE_LENGTH])
    mask = torch.ones([1, SEQUENCE_LENGTH], dtype=torch.int64)
    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_mask = tensor.from_torch(mask, device=mesh_device)

    def encode():
        return encoder.forward(
            tt_tokens,
            mask=tt_mask,
            skip_final_linear=True,
            output_hidden_states=list(range(1, NUM_LAYERS + 1)),
            traced=False,
        )

    encode()  # warm-up: compile every program once
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("TEXT_ENCODER")
    states = encode()
    ttnn.synchronize_device(mesh_device)
    signpost("TEXT_ENCODER_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    assert len(states) == NUM_LAYERS
