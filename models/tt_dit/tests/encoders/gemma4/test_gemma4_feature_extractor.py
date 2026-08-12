# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The 2.3 feature extractor against LTX-2.5's projection weights.

LTX-2.3 and LTX-2.5 both aggregate 49 Gemma hidden states of width 3840 into video and
audio features, so ``GemmaFeatureExtractor`` should carry over untouched — what changes in
2.5 is only that ``text_embedding_projection.*`` ships inside the packed text-encoder file
rather than the monolithic LTX checkpoint. This test holds that claim to the real weights.

The torch reference here is a transcription of ``FeatureExtractorV2`` from ltx-core, checked
bit-exact against the upstream module before being inlined.
"""

import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma3.feature_extractor import GemmaFeatureExtractor
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

CHECKPOINT = os.environ.get(
    "GEMMA4_CHECKPOINT",
    os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
)

HIDDEN = 3840
NUM_STATES = 49  # the embedding output plus 48 decoder layers
SEQ_LEN = 128
VIDEO_DIM = 4096
AUDIO_DIM = 2048


def _projection_weights():
    if not Path(CHECKPOINT).exists():
        pytest.skip(f"no Gemma-4 checkpoint at {CHECKPOINT}")
    weights = {}
    with safe_open(CHECKPOINT, "pt") as handle:
        for axis in ("video", "audio"):
            prefix = f"text_embedding_projection.{axis}_aggregate_embed."
            for part in ("weight", "bias"):
                weights[f"{axis}.{part}"] = handle.get_tensor(prefix + part).float()
    return weights


def _reference(hidden_states, mask, weights):
    """FeatureExtractorV2: per-token RMS norm over the hidden dim of each state, D-major
    concat, padding zeroed, then a rescaled projection per axis."""
    encoded = torch.stack(hidden_states, dim=-1)
    variance = encoded.pow(2).mean(dim=2, keepdim=True)
    normed = (encoded * torch.rsqrt(variance + 1e-6)).reshape(1, SEQ_LEN, HIDDEN * NUM_STATES)
    normed = torch.where(mask.bool().unsqueeze(-1), normed, torch.zeros_like(normed))

    out = {}
    for axis, dim in (("video", VIDEO_DIM), ("audio", AUDIO_DIM)):
        rescaled = normed * math.sqrt(dim / HIDDEN)
        out[axis] = torch.nn.functional.linear(rescaled, weights[f"{axis}.weight"], weights[f"{axis}.bias"])
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_feature_extractor_matches_reference_on_ltx25_weights(*, mesh_device):
    weights = _projection_weights()
    assert weights["video.weight"].shape == (VIDEO_DIM, HIDDEN * NUM_STATES)
    assert weights["audio.weight"].shape == (AUDIO_DIM, HIDDEN * NUM_STATES)

    torch.manual_seed(0)
    hidden_states = [torch.randn(1, SEQ_LEN, HIDDEN) for _ in range(NUM_STATES)]
    mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    mask[:, -17:] = 0  # an odd, non-tile-aligned pad so masking cannot pass by luck

    reference = _reference(hidden_states, mask, weights)

    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    extractor = GemmaFeatureExtractor(
        input_dim=HIDDEN * NUM_STATES,
        embedding_dim=HIDDEN,
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        parallel_config=parallel_config,
    )
    # The device concatenates the normed states layer-major, so the checkpoint's D-major
    # columns are permuted to match rather than permuting activations at runtime.
    extractor.load_torch_state_dict(
        {
            f"{axis}_aggregate_embed.{part}": (
                GemmaFeatureExtractor._weight_to_layer_major(weights[f"{axis}.{part}"], HIDDEN, NUM_STATES)
                if part == "weight"
                else weights[f"{axis}.{part}"]
            )
            for axis in ("video", "audio")
            for part in ("weight", "bias")
        }
    )

    tt_hidden = [
        ttnn.from_torch(hs, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16) for hs in hidden_states
    ]
    video, audio = extractor(tt_hidden, extractor.build_mask(mask))

    for axis, tt_out, dim in (("video", video, VIDEO_DIM), ("audio", audio, AUDIO_DIM)):
        got = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float().reshape(1, SEQ_LEN, dim)
        want = reference[axis]
        pcc = torch.corrcoef(torch.stack([got.flatten().double(), want.flatten().double()]))[0, 1].item()
        logger.info(f"{axis}: PCC {pcc * 100:.4f} %")
        assert pcc >= 0.99, f"{axis} PCC {pcc * 100:.4f} %"
