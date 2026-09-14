# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS speaker encoder against the CPU reference.

Block boundary: log-mel [1, T, 128] (mel front-end on host) -> embedding [1, 2048].

Input is a deterministic synthetic voiced clip pushed through the reference mel front-end.
The reference is the vendored upstream encoder loaded from the same checkpoint, so no
golden files are involved.

The gate sits at 0.999 rather than the usual 0.99, and that is deliberate. Upstream pads
every convolution in reflect mode, which `ttnn.conv1d` cannot do, so this port builds the
mirrored columns by hand. Replacing that with plain zero padding still scores **0.9961**,
which a 0.99 gate would wave through. Measured PCC with the padding correct is 0.999996, so
0.999 separates the two with room to spare.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts.reference.qwen3_speaker_ref import SpeakerReference, speaker_mel
from models.demos.audio.qwen3_tts.tests.reference_helpers import speaker_reference, synthetic_voiced_clip
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_speaker import TtSpeakerEncoder, preprocess_speaker_parameters

TARGET_PCC = 0.999

# Per-block gate. Every stage is measured above 0.9999; 0.999 leaves headroom for a
# different card or a compiler change without going slack enough to hide a real break.
BLOCK_PCC = 0.999

# The reference works channel-first, this port time-major, so every intermediate needs one
# transpose before comparison.
INTERMEDIATES = ["blocks.0", "blocks.1", "blocks.2", "blocks.3", "mfa", "asp", "fc"]


def _to_reference_layout(tensor):
    return ttnn.to_torch(tensor).to(torch.float32).permute(0, 2, 1).contiguous()


def _embed(model, device, mel):
    mel_tt = ttnn.from_torch(mel, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_torch(model(mel_tt)).to(torch.float32).reshape(1, -1)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_speaker_pcc(device):
    """Every block, then the embedding, against the reference."""
    references = speaker_reference()
    model = TtSpeakerEncoder(device, preprocess_speaker_parameters(device))

    mel_tt = ttnn.from_torch(references["mel"], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    embedding_tt, intermediates = model(mel_tt, return_intermediates=True)

    failures = []
    for name in INTERMEDIATES:
        got = _to_reference_layout(intermediates[name])
        want = references["intermediates"][name].to(torch.float32)
        assert got.shape == want.shape, f"{name}: ttnn {tuple(got.shape)} vs reference {tuple(want.shape)}"
        passed, message = comp_pcc(want, got, pcc=BLOCK_PCC)
        print(f"  [{name:10s}] {tuple(got.shape)}  {message}")
        if not passed:
            failures.append(f"{name}: {message}")
    assert not failures, "blocks below PCC gate: " + "; ".join(failures)

    embedding = ttnn.to_torch(embedding_tt).to(torch.float32).reshape(1, -1)
    passed, message = comp_pcc(references["embedding"], embedding, pcc=TARGET_PCC)
    print(f"embedding {tuple(embedding.shape)}  {message}")
    assert passed, f"speaker embedding below PCC {TARGET_PCC}: {message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("pad", [2, 3, 4])
def test_reflect_padding_matches_torch(device, pad):
    """The hand-built mirror, against torch's own reflect padding.

    PCC on the whole encoder is a blunt instrument for this: zero padding still scores
    0.9961 there. This compares the padding directly and demands an exact match.
    """
    signal = torch.arange(1, 13, dtype=torch.float32).reshape(1, 12, 1).repeat(1, 1, 32)
    model = TtSpeakerEncoder(device, {"config": {"enc_dim": 0}})

    signal_tt = ttnn.from_torch(signal, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    padded = ttnn.to_torch(model._reflect_pad(signal_tt, pad)).to(torch.float32)

    # torch pads the last dimension, so reflect over time means transposing around it.
    expected = torch.nn.functional.pad(signal.transpose(1, 2), (pad, pad), mode="reflect").transpose(1, 2)
    assert padded.shape == expected.shape, f"{tuple(padded.shape)} vs {tuple(expected.shape)}"
    assert torch.equal(padded, expected), f"got {padded[0, :, 0].tolist()}, want {expected[0, :, 0].tolist()}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_embedding_separates_two_clips(device):
    """The encoder must respond to its input, and respond the way the reference does.

    A single-input PCC check cannot tell a working encoder from one that has collapsed onto
    a constant. This compares the angle between a low, dark voice and a high, bright one on
    device against the same angle on the reference, which measures 0.95: far enough from 1
    to be a real separation, and the number the device has to reproduce.
    """
    model = TtSpeakerEncoder(device, preprocess_speaker_parameters(device))
    reference = SpeakerReference()

    embeddings, reference_embeddings = [], []
    for voice, seed, seconds in (("low", 0, 3.0), ("high", 7, 2.5)):
        mel = speaker_mel(synthetic_voiced_clip(seconds=seconds, seed=seed, voice=voice))
        embeddings.append(_embed(model, device, mel))
        reference_embeddings.append(reference(mel).to(torch.float32))

    on_device = torch.nn.functional.cosine_similarity(*embeddings).item()
    on_host = torch.nn.functional.cosine_similarity(*reference_embeddings).item()
    print(f"cosine between voices: device {on_device:.4f}, reference {on_host:.4f}")

    assert on_device < 0.98, "the two voices produced near-identical embeddings; is the input reaching the graph?"
    assert abs(on_device - on_host) < 0.005, f"device angle {on_device:.4f} does not track reference {on_host:.4f}"
