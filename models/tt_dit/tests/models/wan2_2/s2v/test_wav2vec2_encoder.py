# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test for wav2vec2-large-xlsr-53-english (production S2V audio encoder).

Uses the weights bundled in ``Wan-AI/Wan2.2-S2V-14B``. Exercises the
``feat_extract_norm="layer"`` + ``do_stable_layer_norm=True`` code paths.

Skips if the bundled HF cache is not present. PCC ≥ 0.99 against the HF
``Wav2Vec2Model.last_hidden_state`` reference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import ttnn

from .....encoders.wav2vec2.config_wav2vec2 import Wav2Vec2Config
from .....encoders.wav2vec2.model_wav2vec2 import Wav2Vec2Encoder
from .....parallel.config import EncoderParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.tensor import to_torch
from .....utils.test import line_params, ring_params


def _bundled_wav2vec2_path() -> Path | None:
    base = Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.2-S2V-14B/snapshots"
    if not base.exists():
        return None
    snaps = list(base.iterdir())
    if not snaps:
        return None
    candidate = snaps[0] / "wav2vec2-large-xlsr-53-english"
    return candidate if candidate.exists() else None


_BUNDLED_PATH = _bundled_wav2vec2_path()


@pytest.mark.skipif(
    _BUNDLED_PATH is None,
    reason="Wan-AI/Wan2.2-S2V-14B bundled wav2vec2 weights not found. Run `hf download Wan-AI/Wan2.2-S2V-14B`.",
)
@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 2, line_params, ttnn.Topology.Linear, id="bh_2x4_tp0"),
        pytest.param((4, 8), 0, 2, ring_params, ttnn.Topology.Ring, id="bh_4x8_tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "audio_samples",
    [
        pytest.param(16000, id="1s"),
        pytest.param(32000, id="2s"),
    ],
)
def test_wav2vec2_encoder_s2v(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    audio_samples: int,
) -> None:
    model_path = str(_BUNDLED_PATH)
    logger.info(f"Loading wav2vec2-large-xlsr-53 from bundled S2V path: {model_path}")

    hf_model = Wav2Vec2Model.from_pretrained(model_path).eval()
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    config = Wav2Vec2Config.from_hf(hf_model.config)
    assert config.do_stable_layer_norm, "Expected do_stable_layer_norm=True for large-xlsr-53"
    assert config.feat_extract_norm == "layer", "Expected feat_extract_norm='layer' for large-xlsr-53"

    torch.manual_seed(0)
    waveform = torch.randn(audio_samples)
    input_values = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        golden = hf_model(input_values).last_hidden_state
    logger.info(f"Golden hidden shape: {tuple(golden.shape)}")

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

    tt_hidden = tt_model(input_values)
    tt_hidden_torch = to_torch(tt_hidden)
    logger.info(f"TT hidden shape: {tuple(tt_hidden_torch.shape)}")
    assert_quality(tt_hidden_torch.float(), golden, pcc=0.99)
