# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import av
import numpy as np
import torch

_RVC_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIRECTORY = os.path.join(_RVC_BASE_DIR, "data")
SPEECH_DIRECTORY = os.path.join(BASE_DIRECTORY, "speech")
BATCH_AUDIO_SIZE = 8


def transcode_audio(input_stream, output_stream, output_format, sample_rate):
    """Transcode audio between formats using PyAV."""
    inp = av.open(input_stream, "r")
    out = av.open(output_stream, "w", format=output_format)
    codec = output_format
    if codec == "ogg":
        codec = "libvorbis"
    if codec == "f32le":
        codec = "pcm_f32le"

    ostream = out.add_stream(codec, rate=sample_rate)
    try:
        ostream.layout = "mono"
    except (ValueError, AttributeError):
        pass

    try:
        for frame in inp.decode(audio=0):
            for p in ostream.encode(frame):
                out.mux(p)
        # Flush the encoder so any delayed packets are emitted (otherwise the
        # transcoded output can be truncated for codecs that buffer frames).
        for p in ostream.encode(None):
            out.mux(p)
    finally:
        out.close()
        inp.close()


def _decode_audio(f, sr):
    with BytesIO() as out:
        transcode_audio(f, out, "f32le", sr)
        audio = np.frombuffer(out.getvalue(), np.float32).flatten().copy()
        return torch.from_numpy(audio)


def _load_fixed_audio_batch(sr):
    audio_list = []
    for i in range(BATCH_AUDIO_SIZE):
        path = os.path.abspath(os.path.join(SPEECH_DIRECTORY, f"sample-speech-{i}.wav"))
        with open(path, "rb") as f:
            audio = _decode_audio(f, sr)
            audio_list.append(audio)
    return audio_list


def load_audio(sr):
    path = os.path.abspath(os.path.join(SPEECH_DIRECTORY, "sample-speech-0.wav"))
    with open(path, "rb") as f:
        audio = _decode_audio(f, sr)
    return audio


def load_audio_batch(sr):
    audio_list = _load_fixed_audio_batch(sr)
    max_length = max(audio.shape[0] for audio in audio_list)
    batch = torch.zeros((len(audio_list), max_length), dtype=torch.float32)
    for idx, audio in enumerate(audio_list):
        batch[idx, : audio.shape[0]] = audio.to(torch.float32)
    return batch