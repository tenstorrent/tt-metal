# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the Qwen3-TTS speaker encoder, and the mel front-end that feeds it.

Wraps the vendored upstream module (`reference/qwen/speaker_encoder.py`) without touching
its forward pass: intermediates come from forward hooks, so the oracle runs exactly the
code the model ships.

Block boundary for the TTNN port:

    waveform (24 kHz mono) --[mel front-end, host]--> mel [1, T, 128]
    mel [1, T, 128] --[speaker encoder, device]--> embedding [1, 2048]

The encoder transposes to [1, 128, T] on entry and works channel-first throughout, so every
intermediate here is the transpose of its TTNN counterpart. The PCC test aligns them.

The embedding is not normalised: `Qwen3TTSSpeakerEncoder.forward` ends at the output
projection, and the talker consumes the raw vector.
"""

import functools

import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen.speaker_encoder import (
    Qwen3TTSSpeakerEncoder,
    Qwen3TTSSpeakerEncoderConfig,
    mel_spectrogram,
)

# Hooked module -> the name the PCC test compares against. Ordered as the encoder runs.
INTERMEDIATES = {
    "blocks.0": "blocks.0",
    "blocks.1": "blocks.1",
    "blocks.2": "blocks.2",
    "blocks.3": "blocks.3",
    "mfa": "mfa",
    "asp": "asp",
    "fc": "fc",
}


def speaker_mel(waveform, sample_rate=None):
    """Waveform [N] or [1, N] at 24 kHz -> log-mel [1, T, 128].

    Mirrors `Qwen3TTSForConditionalGeneration.extract_speaker_embedding`, including its
    trailing transpose. Runs on host: an STFT is not a TTNN op.
    """
    expected = weights.SPEAKER_MEL["sampling_rate"]
    if sample_rate is not None and sample_rate != expected:
        raise ValueError(f"speaker encoder wants {expected} Hz audio, got {sample_rate}")

    audio = torch.as_tensor(waveform, dtype=torch.float32)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2 or audio.shape[0] != 1:
        raise ValueError(f"expected mono audio shaped [N] or [1, N], got {tuple(audio.shape)}")

    return mel_spectrogram(audio, **weights.SPEAKER_MEL).transpose(1, 2)


class SpeakerReference:
    """The upstream encoder, loaded from the checkpoint and frozen in eval mode."""

    def __init__(self, config=None, state=None, dtype=torch.float32):
        self.config = dict(config or weights.speaker_encoder_config())
        self.model = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**self.config))
        state = weights.load_speaker_state(dtype=dtype) if state is None else state
        self.model.load_state_dict(state, strict=True)
        self.model.to(dtype).eval()

    @torch.inference_mode()
    def __call__(self, mel, return_intermediates=False):
        """mel [1, T, 128] -> embedding [1, 2048], optionally with per-block outputs."""
        if not return_intermediates:
            return self.model(mel)

        captured = {}
        handles = []

        def capture(name):
            def hook(_module, _args, output):
                captured[name] = output.detach().clone()

            return hook

        modules = dict(self.model.named_modules())
        for module_name, key in INTERMEDIATES.items():
            handles.append(modules[module_name].register_forward_hook(capture(key)))
        try:
            embedding = self.model(mel)
        finally:
            for handle in handles:
                handle.remove()
        return embedding, captured


@functools.lru_cache(maxsize=1)
def reference_model(dtype=torch.float32):
    """One loaded reference per process; the checkpoint read is the expensive part."""
    return SpeakerReference(dtype=dtype)
