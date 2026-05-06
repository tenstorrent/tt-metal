# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import av
import numpy as np
import torch

BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SPEECH_DIRECTORY = os.path.join(BASE_DIRECTORY, "speech")
BATCH_AUDIO_SIZE = 8


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, rate=sr)
    try:
        ostream.layout = "mono"
    except Exception:
        pass

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def _get_audio_path(file_index):
    file_name = f"sample-speech-{file_index}.wav"
    file = os.path.abspath(os.path.join(SPEECH_DIRECTORY, file_name))
    if os.path.commonpath([SPEECH_DIRECTORY, file]) != SPEECH_DIRECTORY:
        raise RuntimeError(f"Audio file must be located under {SPEECH_DIRECTORY}")
    if not os.path.exists(file):
        raise RuntimeError(f"Audio file does not exist: {file}")
    return file


def _load_audio_index(file_index, sr):
    file = _get_audio_path(file_index)
    with open(file, "rb") as f:
        with BytesIO() as out:
            audio2(f, out, "f32le", sr)
            audio = np.frombuffer(out.getvalue(), np.float32).flatten().copy()
            return torch.from_numpy(audio)


def load_audio(sr):
    return _load_audio_index(0, sr)


def load_audio_batch(sr):
    audio_list = [_load_audio_index(file_index, sr) for file_index in range(BATCH_AUDIO_SIZE)]
    max_length = max(audio.shape[0] for audio in audio_list)
    batch = torch.zeros((len(audio_list), max_length), dtype=torch.float32)
    for idx, audio in enumerate(audio_list):
        batch[idx, : audio.shape[0]] = audio.to(torch.float32)
    return batch
