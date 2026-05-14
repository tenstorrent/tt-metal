# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import ttnn

from ....encoders.wav2vec2 import Wav2Vec2Config, Wav2Vec2Encoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import to_torch
from ....utils.test import line_params


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 2), 1, 1, line_params, ttnn.Topology.Linear, id="2x2_tp1"),
        pytest.param((2, 4), 1, 1, line_params, ttnn.Topology.Linear, id="2x4_tp1"),
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_4x8_tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "audio_samples",
    [
        # ~1.024s at 16 kHz. Short clip keeps the test fast while exercising the
        # full feature extractor + all 12 encoder layers.
        pytest.param(16000, id="1s"),
        pytest.param(32000, id="2s"),
    ],
)
def test_wav2vec2_encoder(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    audio_samples: int,
) -> None:
    model_id = "facebook/wav2vec2-base-960h"

    # 1. Load HF reference (CPU, float32 for the golden).
    hf_model = Wav2Vec2Model.from_pretrained(model_id).eval()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    config = Wav2Vec2Config.from_hf(hf_model.config)

    # 2. Build a deterministic synthetic waveform and normalize via the HF
    # processor so the TT path sees the same input distribution as the
    # production pipeline.
    torch.manual_seed(0)
    waveform = torch.randn(audio_samples)
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values  # [1, T_raw]

    # 3. Golden forward.
    with torch.no_grad():
        golden = hf_model(input_values).last_hidden_state  # [1, T_features, 768]
    logger.info(f"Golden hidden shape: {tuple(golden.shape)}")

    # 4. Build TT model.
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    tt_model = Wav2Vec2Encoder(
        config=config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_torch_state_dict(hf_model.state_dict())
    tt_model.bind_cpu_modules(hf_model)

    # 5. Forward — feature extractor + feature_projection + 12 transformer
    # layers on device; pos-conv + initial LayerNorm on CPU
    # (see Wav2Vec2Encoder docstring for the kernel constraint that requires this).
    tt_hidden = tt_model(input_values)
    tt_hidden_torch = to_torch(tt_hidden)  # replicated → just take the first shard

    logger.info(f"TT hidden shape: {tuple(tt_hidden_torch.shape)}")
    assert_quality(tt_hidden_torch, golden, pcc=0.99)
