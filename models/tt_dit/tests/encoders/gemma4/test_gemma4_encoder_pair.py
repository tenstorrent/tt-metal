# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The LTX-2.5 text path end to end: prompt in, video/audio embeddings out.

Covers the plumbing the 2.3 pair cannot: config, tokenizer and Gemma weights out of the
packed text-encoder file, connectors out of the transformer checkpoint. The arithmetic
inside is covered by the parity and feature-extractor tests.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import line_params_req_exact_devices

LTX25 = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5")
TEXT_ENCODER = os.environ.get(
    "GEMMA4_CHECKPOINT", f"{LTX25}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
TRANSFORMER = os.environ.get(
    "LTX25_TRANSFORMER", f"{LTX25}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)

SEQUENCE_LENGTH = 1024
VIDEO_DIM = 4096
AUDIO_DIM = 2048


def test_tokenizer_prepends_bos_and_pads_left():
    """Host-only. Gemma-4's tokenizer emits no BOS of its own, so the pair adds it."""
    if not Path(TEXT_ENCODER).exists():
        pytest.skip(f"no LTX-2.5 text encoder at {TEXT_ENCODER}")

    pair = Gemma4TokenizerEncoderPair.__new__(Gemma4TokenizerEncoderPair)
    from models.tt_dit.encoders.gemma4.encoder_pair import Gemma4Assets

    pair.assets = Gemma4Assets(TEXT_ENCODER)
    pair._sequence_length = 128
    pair.tokenizer = pair.assets.build_tokenizer(128)

    prompt = "a cat playing a piano in a jazz bar"
    assert pair.tokenizer(prompt).input_ids[0] != pair.tokenizer.bos_token_id, "tokenizer gained a BOS post-processor"

    input_ids, attention_mask = pair.tokenize(prompt)
    assert input_ids.shape == (1, 128)
    real = int(attention_mask.sum())
    assert attention_mask[0, :-real].sum() == 0, "padding is not on the left"
    assert input_ids[0, -real] == pair.tokenizer.bos_token_id, "first real token is not BOS"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_encoder_pair_encodes_a_prompt(*, mesh_device):
    for path in (TEXT_ENCODER, TRANSFORMER):
        if not Path(path).exists():
            pytest.skip(f"missing {path}")

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    pair = Gemma4TokenizerEncoderPair(
        TEXT_ENCODER,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        parallel_config=parallel_config,
        transformer_checkpoint=TRANSFORMER,
        sequence_length=SEQUENCE_LENGTH,
    )
    assert pair.config.num_hidden_layers == 48
    assert pair.config.hidden_size == 3840

    pair.ensure_loaded()
    assert pair.is_loaded()

    (video, audio) = pair.encode(["a cat playing a piano in a jazz bar"])[0]

    logger.info(f"video {tuple(video.shape)} std {video.std():.4f} | audio {tuple(audio.shape)} std {audio.std():.4f}")
    for name, embeds, dim in (("video", video, VIDEO_DIM), ("audio", audio, AUDIO_DIM)):
        assert embeds.shape[-1] == dim
        assert torch.isfinite(embeds).all(), f"{name} embeddings are not finite"
        assert embeds.std() > 0.01, f"{name} embeddings are degenerate"
